"""LoRA fine-tuning of a pretrained latent diffusion model (Stable Diffusion) on your own images.

    from sd_lora_trainer import SDLoraTrainer, SDLoraHparams

    # fine-tune
    trainer = SDLoraTrainer("checkpoints/lora_output", SDLoraHparams(max_steps=2000), data_root="datasets/art-nouveau")
    trainer.finetune()          # or trainer.finetune(resume_from="checkpoints/lora_output/training_state.pt")
    images = trainer.generate(["an illustration in art nouveau style, a cat"], [7])

    # later / elsewhere: generate from the saved LoRA (no dataset needed)
    trainer = SDLoraTrainer("checkpoints/lora_output")          # no data_root => inference mode
    trainer.generate_grid("an illustration in art nouveau style, a cat", scales=[0, 0.5, 1.0], out="compare.png")

Command line (same file):
    python sd_lora_trainer.py finetune --data datasets/art-nouveau --out checkpoints/lora_output
    python sd_lora_trainer.py generate "an illustration in art nouveau style, a cat" --out-dir checkpoints/lora_output

Outputs written to `output_dir`:
    pytorch_lora_weights.safetensors   the LoRA (load with load_lora_pipeline)
    training_state.pt                  optimizer/scheduler/step for resuming
    samples/*.png                      validation grids, baseline, before_after
    loss.png
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.figure import Figure
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
    StableDiffusionPipeline, UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate slightly truncated files
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- data
def list_images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    return files


class ImageCaptionDataset(torch.utils.data.Dataset):
    """Images + captions. Caption = same-named .txt file if present, else `default_caption`."""

    def __init__(self, files, resolution, default_caption, random_flip=True):
        self.files = files
        self.default_caption = default_caption
        tfms = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
        ]
        if random_flip:
            tfms.append(transforms.RandomHorizontalFlip())
        tfms += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(tfms)

    def caption_for(self, path):
        txt = path.with_suffix(".txt")
        if txt.exists():
            text = txt.read_text().strip()
            if text:
                return text
        return self.default_caption

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        for offset in range(len(self.files)):          # skip unreadable files
            path = self.files[(i + offset) % len(self.files)]
            try:
                image = Image.open(path).convert("RGB")
                return {"pixel_values": self.transform(image), "caption": self.caption_for(path)}
            except Exception:
                continue
        raise RuntimeError("No readable images found")


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


# --------------------------------------------------------------------------- helpers
def image_grid(images, ncols=None):
    """Concatenate PIL images (same size) into one grid image."""
    ncols = ncols or len(images)
    nrows = (len(images) + ncols - 1) // ncols
    w, h = images[0].size
    grid = Image.new("RGB", (ncols * w, nrows * h))
    for i, im in enumerate(images):
        grid.paste(im, ((i % ncols) * w, (i // ncols) * h))
    return grid


def load_lora_pipeline(model_id, lora_dir, device=None, dtype=None):
    """Fresh base pipeline + trained LoRA, for inference. Use `cross_attention_kwargs={"scale": s}` to change strength."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.load_lora_weights(lora_dir)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# --------------------------------------------------------------------------- trainer
@dataclass
class SDLoraHparams:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    default_caption: str = "an illustration in art nouveau style"

    resolution: int = 512            # SD 1.5 was trained at 512; use 384/256 if GPU memory is short
    batch_size: int = 1
    grad_accum: int = 4              # effective batch = batch_size * grad_accum
    max_steps: int = 2000            # optimizer steps
    lr: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    rank: int = 16                   # LoRA rank (higher = more capacity, bigger file)
    snr_gamma: float | None = 5.0    # min-SNR loss weighting; None to disable
    max_grad_norm: float = 1.0
    random_flip: bool = True
    grad_checkpointing: bool = True  # big memory saving, ~20% slower
    num_workers: int = 2
    seed: int = 1337

    validation_prompts: list = field(default_factory=lambda: [
        "an illustration in art nouveau style, portrait of a woman with flowing hair",
        "an illustration in art nouveau style, portrait of a man in a suit",
        "an illustration in art nouveau style, a garden with flowers and a peacock",
    ])
    validation_seeds: list = field(default_factory=lambda: [11, 22, 33])
    validation_every: int = 500
    checkpoint_every: int = 500

    allow_cpu: bool = False          # only for smoke tests; real training needs a GPU


class SDLoraTrainer:
    def __init__(self, output_dir="checkpoints/lora_output", hparams: SDLoraHparams | None = None,
                 data_root=None, device=None):
        """With `data_root` -> training mode (dataset, LoRA layers, optimizer).
        Without it -> inference mode: loads the base model and the LoRA saved in `output_dir`."""
        self.hp = hparams or SDLoraHparams()
        self.output_dir = output_dir
        self.training_mode = data_root is not None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and self.training_mode and not self.hp.allow_cpu:
            raise RuntimeError("No GPU found. Use a GPU machine (e.g. Kaggle T4) or set allow_cpu=True for a smoke test.")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.use_amp = self.device == "cuda"

        random.seed(self.hp.seed); np.random.seed(self.hp.seed); torch.manual_seed(self.hp.seed)
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

        self._load_models()
        self._freeze_and_move()
        if self.training_mode:
            self._build_data(data_root)
            self._add_lora()
            self._build_optimization()
        else:
            self._build_sample_pipe()
            self.sample_pipe.load_lora_weights(output_dir)
            print(f"Loaded LoRA from {output_dir}")

        self.global_step = 0
        self.loss_history: list[float] = []
        self.baseline_images = None
        print(f"Device: {self.device} | dtype: {self.dtype}")

    # ---- setup
    def _build_data(self, data_root):
        files = list_images(data_root)
        self.dataset = ImageCaptionDataset(files, self.hp.resolution, self.hp.default_caption, self.hp.random_flip)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.hp.batch_size, shuffle=True,
            num_workers=self.hp.num_workers, collate_fn=collate_fn, drop_last=True,
        )
        print(f"{len(self.dataset)} images under {data_root}")
        print("Example caption:", self.dataset.caption_for(files[0]))

    def _load_models(self):
        mid = self.hp.model_id
        self.tokenizer = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(mid, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(mid, subfolder="scheduler")
        print("prediction type:", self.noise_scheduler.config.prediction_type,
              "| train timesteps:", self.noise_scheduler.config.num_train_timesteps)

    def _freeze_and_move(self):
        """Freeze everything and move to the device in half precision (on GPU)."""
        for m in (self.vae, self.text_encoder, self.unet):
            m.requires_grad_(False)
        self.vae.to(self.device, dtype=self.dtype).eval()
        self.text_encoder.to(self.device, dtype=self.dtype).eval()
        self.unet.to(self.device, dtype=self.dtype)

    def _add_lora(self):
        """Add trainable LoRA layers to the UNet."""
        if self.hp.grad_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.unet.add_adapter(LoraConfig(
            r=self.hp.rank, lora_alpha=self.hp.rank, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],      # attention projections
        ))
        if self.dtype == torch.float16:
            cast_training_params(self.unet, dtype=torch.float32)      # train LoRA weights in fp32

        self.lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in self.lora_params)
        n_total = sum(p.numel() for p in self.unet.parameters())
        print(f"Trainable LoRA parameters: {n_train/1e6:.2f}M of {n_total/1e6:.1f}M UNet parameters ({100*n_train/n_total:.2f}%)")

    def _build_optimization(self):
        hp = self.hp
        self.optimizer = torch.optim.AdamW(self.lora_params, lr=hp.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
        self.lr_scheduler = get_scheduler(hp.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=hp.warmup_steps, num_training_steps=hp.max_steps)
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.use_amp)
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self._build_sample_pipe()

    def _build_sample_pipe(self):
        """Pipeline sharing the live UNet (LoRA included), used for all sampling."""
        self.sample_pipe = StableDiffusionPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        )
        self.sample_pipe.set_progress_bar_config(disable=True)

    def _autocast(self):
        return torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self.use_amp)

    # ---- loss
    def compute_loss(self, pixel_values, captions):
        """image -> latents, add noise, predict it, (min-SNR weighted) MSE."""
        ns = self.noise_scheduler
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(self.device, dtype=self.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            ids = self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding="max_length",
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
            text_emb = self.text_encoder(ids)[0]

        noise = torch.randn_like(latents)
        t = torch.randint(0, ns.config.num_train_timesteps, (latents.size(0),), device=self.device).long()
        noisy = ns.add_noise(latents, noise, t)

        if ns.config.prediction_type == "epsilon":
            target = noise
        elif ns.config.prediction_type == "v_prediction":
            target = ns.get_velocity(latents, noise, t)
        else:
            raise ValueError(ns.config.prediction_type)

        with self._autocast():
            pred = self.unet(noisy, t, text_emb).sample

        loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
        if self.hp.snr_gamma is not None:
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            denom = snr + 1 if ns.config.prediction_type == "v_prediction" else snr
            loss = loss * snr.clamp(max=self.hp.snr_gamma) / denom
        return loss.mean()

    # ---- sampling
    @torch.no_grad()
    def generate(self, prompts=None, seeds=None, steps=25, guidance=7.5, lora_scale=1.0, resolution=None):
        prompts = prompts or self.hp.validation_prompts
        seeds = seeds or self.hp.validation_seeds
        was_training = self.unet.training
        self.unet.eval()
        images = []
        for prompt, seed in zip(prompts, seeds):
            gen = torch.Generator(device=self.device).manual_seed(seed)
            with self._autocast():
                out = self.sample_pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen,
                                       height=resolution or self.hp.resolution, width=resolution or self.hp.resolution,
                                       cross_attention_kwargs={"scale": lora_scale})
            images.append(out.images[0])
        self.unet.train(was_training)
        return images

    def generate_grid(self, prompt, scales=(1.0,), n=1, seed=7, steps=30, guidance=7.5, resolution=None, out=None):
        """One prompt, several LoRA strengths (rows) x n seeds (columns). Scale 0 = original model. Saves to `out` if given."""
        images = []
        for s in scales:
            images += self.generate([prompt] * n, [seed + k for k in range(n)], steps=steps,
                                    guidance=guidance, lora_scale=s, resolution=resolution)
        grid = image_grid(images, ncols=n)
        if out:
            grid.save(out)
            print(f"Saved {len(images)} image(s) ({len(scales)} scale(s) x {n}) -> {out}")
        return grid

    def _save_samples(self, name, images):
        path = os.path.join(self.output_dir, "samples", name)
        image_grid(images).save(path)
        return path

    # ---- checkpoints
    def save_lora(self):
        """Writes pytorch_lora_weights.safetensors to output_dir (load with load_lora_pipeline)."""
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))
        StableDiffusionPipeline.save_lora_weights(save_directory=self.output_dir, unet_lora_layers=state, safe_serialization=True)

    def save_state(self):
        torch.save({
            "lora": get_peft_model_state_dict(self.unet),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "losses": self.loss_history,
        }, os.path.join(self.output_dir, "training_state.pt"))

    def load_state(self, path):
        state = torch.load(path, map_location=self.device)
        set_peft_model_state_dict(self.unet, state["lora"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step, self.loss_history = state["step"], state["losses"]
        print(f"Resumed from step {self.global_step}")

    def checkpoint(self):
        self.save_lora()
        self.save_state()

    # ---- training
    def finetune(self, resume_from=None):
        """Run LoRA fine-tuning; checkpoints, sample grids, loss plot and before/after grid go to output_dir."""
        if not self.training_mode:
            raise RuntimeError("Create the trainer with data_root=... to fine-tune.")
        hp = self.hp
        if resume_from and os.path.isfile(resume_from):
            self.load_state(resume_from)

        if self.global_step == 0:                       # LoRA starts as a no-op => these are the original model's samples
            self.baseline_images = self.generate()
            print("Baseline samples:", self._save_samples("baseline.png", self.baseline_images))

        data_iter = self._infinite(self.dataloader)
        self.unet.train()
        pbar = tqdm(total=hp.max_steps, initial=self.global_step, desc="Fine-tuning")
        try:
            while self.global_step < hp.max_steps:
                step_loss = 0.0
                for _ in range(hp.grad_accum):
                    batch = next(data_iter)
                    loss = self.compute_loss(batch["pixel_values"], batch["captions"]) / hp.grad_accum
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss; try a lower learning rate.")
                    self.scaler.scale(loss).backward()
                    step_loss += loss.item()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.lora_params, hp.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                self.loss_history.append(step_loss)
                pbar.update(1)
                pbar.set_postfix(loss=f"{np.mean(self.loss_history[-50:]):.4f}", lr=f"{self.lr_scheduler.get_last_lr()[0]:.1e}")

                if self.global_step % hp.validation_every == 0 or self.global_step == hp.max_steps:
                    self._save_samples(f"step_{self.global_step:06d}.png", self.generate())
                if self.global_step % hp.checkpoint_every == 0:
                    self.checkpoint()
        except KeyboardInterrupt:
            print("Interrupted; saving what we have.")
        finally:
            pbar.close()
            self.checkpoint()
            print(f"Saved LoRA weights to {self.output_dir}/ at step {self.global_step}")

        self.plot_losses()
        self.save_before_after()
        return self.loss_history

    fit = finetune   # alias

    @staticmethod
    def _infinite(dl):
        while True:
            for batch in dl:
                yield batch

    # ---- reporting
    def plot_losses(self, window=50):
        losses = np.array(self.loss_history)
        if len(losses) == 0:
            return
        window = min(window, len(losses))
        fig = Figure(figsize=(10, 4)); ax = fig.subplots()
        ax.plot(losses, alpha=0.3, label="loss per step")
        ax.plot(range(window - 1, len(losses)), np.convolve(losses, np.ones(window) / window, mode="valid"),
                label=f"moving average ({window})")
        ax.set_xlabel("optimizer step"); ax.set_ylabel("loss"); ax.legend()
        path = os.path.join(self.output_dir, "loss.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        return path

    def save_before_after(self):
        """Same prompts/seeds: top row = original model (only if this run started from step 0), bottom row = fine-tuned."""
        after = self.generate()
        rows = ([self.baseline_images] if self.baseline_images else []) + [after]
        path = os.path.join(self.output_dir, "samples", "before_after.png")
        image_grid([im for row in rows for im in row], ncols=len(after)).save(path)
        return path


# --------------------------------------------------------------------------- command line
def _parse_args():
    import argparse
    d = SDLoraHparams()
    ap = argparse.ArgumentParser(description="LoRA fine-tune / sample Stable Diffusion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("finetune", help="Fine-tune on a folder of images (needs a GPU)")
    f.add_argument("--data", required=True, help="Folder of images (searched recursively)")
    f.add_argument("--out", default="checkpoints/lora_output")
    f.add_argument("--model-id", default=d.model_id)
    f.add_argument("--caption", default=d.default_caption, help="Caption for images without a .txt sidecar")
    f.add_argument("--resolution", type=int, default=d.resolution)
    f.add_argument("--batch-size", type=int, default=d.batch_size)
    f.add_argument("--grad-accum", type=int, default=d.grad_accum)
    f.add_argument("--steps", type=int, default=d.max_steps, help="Optimizer steps")
    f.add_argument("--lr", type=float, default=d.lr)
    f.add_argument("--warmup", type=int, default=d.warmup_steps)
    f.add_argument("--rank", type=int, default=d.rank)
    f.add_argument("--snr-gamma", type=float, default=d.snr_gamma, help="min-SNR weighting; <=0 disables")
    f.add_argument("--no-flip", action="store_true")
    f.add_argument("--no-grad-checkpointing", action="store_true")
    f.add_argument("--workers", type=int, default=d.num_workers)
    f.add_argument("--seed", type=int, default=d.seed)
    f.add_argument("--validation-every", type=int, default=d.validation_every)
    f.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    f.add_argument("--validation-prompt", action="append", default=None, help="Repeatable")
    f.add_argument("--resume", default=None, help="Path to training_state.pt")
    f.add_argument("--allow-cpu", action="store_true", help="Smoke test only")

    g = sub.add_parser("generate", help="Generate images from a saved LoRA")
    g.add_argument("prompt")
    g.add_argument("--out-dir", default="checkpoints/lora_output", help="Folder with pytorch_lora_weights.safetensors")
    g.add_argument("--model-id", default=d.model_id)
    g.add_argument("--scales", type=float, nargs="+", default=[1.0], help="LoRA strengths; one row per value")
    g.add_argument("--n", type=int, default=1, help="Images per scale (seeds seed, seed+1, ...)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--steps", type=int, default=30)
    g.add_argument("--guidance", type=float, default=7.5)
    g.add_argument("--resolution", type=int, default=512)
    g.add_argument("--out", default="generated.png")
    return ap.parse_args()


def main():
    a = _parse_args()
    if a.cmd == "finetune":
        hp = SDLoraHparams(
            model_id=a.model_id, default_caption=a.caption, resolution=a.resolution,
            batch_size=a.batch_size, grad_accum=a.grad_accum, max_steps=a.steps, lr=a.lr,
            warmup_steps=a.warmup, rank=a.rank, snr_gamma=a.snr_gamma if a.snr_gamma > 0 else None,
            random_flip=not a.no_flip, grad_checkpointing=not a.no_grad_checkpointing,
            num_workers=a.workers, seed=a.seed, validation_every=a.validation_every,
            checkpoint_every=a.checkpoint_every, allow_cpu=a.allow_cpu,
        )
        if a.validation_prompt:
            hp.validation_prompts = a.validation_prompt
            hp.validation_seeds = [11 * (i + 1) for i in range(len(a.validation_prompt))]
        SDLoraTrainer(a.out, hp, data_root=a.data).finetune(resume_from=a.resume)
    else:
        trainer = SDLoraTrainer(a.out_dir, SDLoraHparams(model_id=a.model_id, resolution=a.resolution))
        trainer.generate_grid(a.prompt, scales=a.scales, n=a.n, seed=a.seed, steps=a.steps,
                              guidance=a.guidance, out=a.out)


if __name__ == "__main__":
    main()
