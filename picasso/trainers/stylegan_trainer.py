# ——————————————————————————————————————————————————————————————
# Imports
import os
import time
import copy
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageFile

import torch
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from picasso.models.stylegan import Generator, Critic
from picasso.models.augment import AugmentPipe, ALL_OPS

# Tolerate slightly truncated (but readable) image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ——————————————————————————————————————————————————————————————
# Dataset and training functions
def find_dataset_root(relative_path=os.path.join("..", "datasets", "Event", "Event")):
    """
    Finds the dataset folder by walking upward from the current working directory.

    :param relative_path: dataset path relative to some ancestor directory
    :type relative_path: str

    :return: path of the dataset root
    :rtype: str
    """
    search_dir = os.getcwd()
    while True:
        candidate = os.path.join(search_dir, relative_path)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            raise FileNotFoundError(
                f"Could not locate dataset folder '{relative_path}' from {os.getcwd()}"
            )
        search_dir = parent


class SafeImageFolder(dset.ImageFolder):
    """
    ImageFolder that skips unreadable/corrupt images at load time instead of crashing,
    so no slow upfront validation scan is needed.
    """
    def __getitem__(self, index):
        for offset in range(len(self)):
            try:
                return super().__getitem__((index + offset) % len(self))
            except Exception:
                continue
        raise RuntimeError("No readable images found")


def gradient_penalty(
    critic,
    real,
    fake
):
    """
    WGAN-GP: penalizes the critic's gradient norm on random interpolates for deviating from 1.

    :param real: real images in shape (B, 3, H, W)
    :param fake: generated images in shape (B, 3, H, W)

    :return: scalar penalty
    :rtype: torch.Tensor
    """
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores.sum(), inputs=interp, create_graph=True
    )[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def critic_step(
    critic,
    netG,
    opt_c,
    augment,
    real,
    nz,
    gp_weight=10.0,
    drift_weight=0.001,
    mixing_prob=0.0
):
    """
    Executes one critic update: min D(fake) - D(real) + gp_weight * GP + drift * D(real)^2.
    Real and fake images go through the same ADA augmentation before the critic sees them.

    :return: dict with loss, wdist, gp, d_real (mean), d_fake (mean)
    :rtype: dict
    """
    b_size = real.size(0)
    opt_c.zero_grad(set_to_none=True)

    with torch.no_grad():
        noise = torch.randn(b_size, nz, device=real.device)
        fake = netG(noise, mixing_prob=mixing_prob)

    real_aug = augment(real)
    fake_aug = augment(fake)
    d_real = critic(real_aug).view(-1)
    d_fake = critic(fake_aug).view(-1)

    gp = gradient_penalty(critic, real_aug.detach(), fake_aug.detach())
    loss = d_fake.mean() - d_real.mean() + gp_weight * gp + drift_weight * d_real.pow(2).mean()
    loss.backward()
    opt_c.step()

    return {
        "loss": loss.item(),
        "wdist": (d_real.mean() - d_fake.mean()).item(),
        "gp": gp.item(),
        "d_real": d_real.mean().item(),
        "d_fake": d_fake.mean().item(),
    }


def generator_step(
    critic,
    netG,
    opt_g,
    augment,
    b_size,
    nz,
    mixing_prob=0.0
):
    """
    Executes one generator update: min -D(G(z)).

    :return: generator loss
    :rtype: float
    """
    opt_g.zero_grad(set_to_none=True)
    noise = torch.randn(b_size, nz, device=device)
    fake = netG(noise, mixing_prob=mixing_prob)
    loss = -critic(augment(fake)).mean()
    loss.backward()
    opt_g.step()
    return loss.item()


@torch.no_grad()
def update_ema(
    g_ema,
    netG,
    decay,
    n_updates
):
    """
    Exponential moving average of generator weights (used for sampling).
    """
    beta = min(decay, (1 + n_updates) / (10 + n_updates))  # warm-up
    for p_ema, p in zip(g_ema.parameters(), netG.parameters()):
        p_ema.mul_(beta).add_(p.detach(), alpha=1 - beta)
    for b_ema, b in zip(g_ema.buffers(), netG.buffers()):
        b_ema.copy_(b)


# ——————————————————————————————————————————————————————————————
# WGAN-StyleGAN Hyperparameters
class StyleWGANHparams:
    """
    Hyperparameters for the style-based WGAN-GP with adaptive discriminator augmentation.
    """
    IMAGE_SIZE = 64         # power of two >= 8 (64 fits ~3 GB at batch 16; 128 needs batch <= 8 on 8 GB)
    NZ = 256                # latent / w size
    NGF = 32                # generator width (channels = NGF*512/res, capped at NGF*8)
    NDF = 32                # critic width
    N_MLP = 4               # mapping network depth
    LR = 0.0001
    BETA1 = 0.0
    BETA2 = 0.99
    N_CRITIC = 5            # critic updates per generator update
    GP_WEIGHT = 10.0
    DRIFT_WEIGHT = 0.001
    MIXING_PROB = 0.9       # style-mixing regularization
    EMA_DECAY = 0.999
    MBSTD_GROUP = 0         # minibatch-stddev in critic (0 = off, recommended with WGAN-GP)
    # --- Adaptive Discriminator Augmentation
    ADA = True
    ADA_P_INIT = 0.0        # starting (or fixed, if ADA=False) augmentation probability
    ADA_TARGET = 0.6        # target overfitting heuristic r_v
    ADA_INTERVAL = 4        # adjust p every N critic steps
    ADA_KIMG = 100          # thousand images the critic must see for p to move from 0 to 1
    VAL_FRAC = 0.05         # held-out fraction of images used to measure overfitting
    AUGMENTATIONS = ALL_OPS

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        nz=NZ,
        ngf=NGF,
        ndf=NDF,
        n_mlp=N_MLP,
        lr=LR,
        beta1=BETA1,
        beta2=BETA2,
        n_critic=N_CRITIC,
        gp_weight=GP_WEIGHT,
        drift_weight=DRIFT_WEIGHT,
        mixing_prob=MIXING_PROB,
        ema_decay=EMA_DECAY,
        mbstd_group=MBSTD_GROUP,
        ada=ADA,
        ada_p_init=ADA_P_INIT,
        ada_target=ADA_TARGET,
        ada_interval=ADA_INTERVAL,
        ada_kimg=ADA_KIMG,
        val_frac=VAL_FRAC,
        augmentations=AUGMENTATIONS
    ):
        self.image_size = image_size
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.n_mlp = n_mlp
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.mixing_prob = mixing_prob
        self.ema_decay = ema_decay
        self.mbstd_group = mbstd_group
        self.ada = ada
        self.ada_p_init = ada_p_init
        self.ada_target = ada_target
        self.ada_interval = ada_interval
        self.ada_kimg = ada_kimg
        self.val_frac = val_frac
        self.augmentations = tuple(augmentations)


# Hparams that change tensor shapes; a checkpoint must match these to be loadable
ARCH_KEYS = ("image_size", "nz", "ngf", "ndf", "n_mlp", "mbstd_group")


# ——————————————————————————————————————————————————————————————
# WGAN-StyleGAN Trainer
class StyleWGANTrainer:
    """
    Trainer for a StyleGAN2-style generator against a WGAN-GP critic, with ADA.
    """
    def __init__(
        self,
        hparams=None,
        datapath=None,
        batch_size=16,
        ckpt_path="checkpoints/stylewgan_ckpt.pt",
        sample_dir="samples",
        load_from_checkpoint=False,
        n_samples=None,
        num_workers=0,
        seed=999,
        deterministic=False,
    ):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.ckpt_path = ckpt_path
        self.sample_dir = sample_dir
        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # Reproducibility. Note: grid_sample (used by ADA) has no deterministic CUDA
        # backward, so deterministic mode only works with the augmentations disabled.
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)

        if hparams is None:
            hparams = vars(StyleWGANHparams())
        elif isinstance(hparams, dict):
            hparams = vars(StyleWGANHparams(**hparams))
        else:
            raise ValueError("hparams must be a dict or None")
        self.hparams = hparams

        if datapath is None:
            datapath = find_dataset_root()
        self.datapath = datapath

        ds = SafeImageFolder(
            root=datapath,
            transform=transforms.Compose([
                transforms.Resize(hparams["image_size"]),
                transforms.CenterCrop(hparams["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            allow_empty=True
        )
        if n_samples is not None:
            ds = torch.utils.data.Subset(ds, torch.randperm(len(ds))[:n_samples].tolist())

        # Hold out a small validation split: ADA uses it to measure critic overfitting
        n_val = int(round(hparams["val_frac"] * len(ds))) if hparams["ada"] else 0
        if len(ds) - n_val < batch_size:
            n_val = 0
        if len(ds) < batch_size:
            raise ValueError(f"Need at least batch_size={batch_size} images, found {len(ds)}")
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed)
        )
        self.adaptive = bool(hparams["ada"]) and n_val >= 2
        if hparams["ada"] and not self.adaptive:
            print("Warning: too few images for a validation split; ADA keeps p fixed at ada_p_init.")
        print(
            f"There are {len(ds)} samples in the data, "
            f"{len(self.train_ds)} training samples and {len(self.val_ds)} validation samples"
        )

        self.train_dl = torch.utils.data.DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        )
        self.val_dl = torch.utils.data.DataLoader(
            self.val_ds, batch_size=max(1, min(batch_size, n_val)), shuffle=True,
            num_workers=num_workers
        ) if self.adaptive else None
        self._val_iter = None

        self.netG = Generator(
            hparams["nz"], hparams["ngf"], hparams["image_size"], hparams["n_mlp"]
        ).to(device)
        self.g_ema = copy.deepcopy(self.netG).eval().requires_grad_(False)
        self.critic = Critic(
            hparams["ndf"], hparams["image_size"], hparams["mbstd_group"]
        ).to(device)

        betas = (hparams["beta1"], hparams["beta2"])
        self.optG = optim.Adam(self.netG.parameters(), lr=hparams["lr"], betas=betas)
        self.optC = optim.Adam(self.critic.parameters(), lr=hparams["lr"], betas=betas)

        self.augment = AugmentPipe(p=hparams["ada_p_init"], ops=hparams["augmentations"])

        # Fixed noise to visualize generator progress across epochs
        self.fixed_noise = torch.randn(64, hparams["nz"], device=device)

        self.G_losses = []
        self.C_losses = []
        self.w_dists = []
        self.ada_ps = []
        self.epochs_trained = 0
        self.critic_steps = 0
        self.g_steps = 0

        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            print(f"Loading checkpoint from {self.ckpt_path}...")
            self.load()

    def save(self, ckpt_path=None):
        """
        Saves a training checkpoint.
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = {
            "netG_state_dict": self.netG.state_dict(),
            "g_ema_state_dict": self.g_ema.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optG_state_dict": self.optG.state_dict(),
            "optC_state_dict": self.optC.state_dict(),
            "G_losses": self.G_losses,
            "C_losses": self.C_losses,
            "w_dists": self.w_dists,
            "ada_ps": self.ada_ps,
            "ada_p": self.augment.p,
            "epochs_trained": self.epochs_trained,
            "critic_steps": self.critic_steps,
            "g_steps": self.g_steps,
            "fixed_noise": self.fixed_noise.cpu(),
            "hparams": self.hparams,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
        }

        torch.save(ckpt, self.ckpt_path)
        return

    def load(self, ckpt_path=None):
        """
        Loads a training checkpoint.
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = torch.load(self.ckpt_path, map_location=device)

        mismatch = {
            k: (ckpt["hparams"].get(k), self.hparams[k])
            for k in ARCH_KEYS if ckpt["hparams"].get(k) != self.hparams[k]
        }
        if mismatch:
            raise ValueError(
                f"Checkpoint architecture differs from current hparams (checkpoint, current): {mismatch}"
            )

        print("Loading the models...", end="")
        print(self.netG.load_state_dict(ckpt["netG_state_dict"]))
        self.g_ema.load_state_dict(ckpt["g_ema_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.optG.load_state_dict(ckpt["optG_state_dict"])
        self.optC.load_state_dict(ckpt["optC_state_dict"])

        self.G_losses = ckpt["G_losses"]
        self.C_losses = ckpt["C_losses"]
        self.w_dists = ckpt["w_dists"]
        self.ada_ps = ckpt["ada_ps"]
        self.augment.p = ckpt["ada_p"]
        self.epochs_trained = ckpt["epochs_trained"]
        self.critic_steps = ckpt["critic_steps"]
        self.g_steps = ckpt["g_steps"]
        self.fixed_noise = ckpt["fixed_noise"].to(device)

        return

    def _next_val_batch(self):
        try:
            batch = next(self._val_iter)
        except (TypeError, StopIteration):
            self._val_iter = iter(self.val_dl)
            batch = next(self._val_iter)
        return batch[0].to(device)

    def update_ada(self, m_train, m_fake):
        """
        Adjusts the augmentation probability p from the overfitting heuristic
        r_v = (E[D(train)] - E[D(val)]) / (E[D(train)] - E[D(fake)]).
        r_v -> 0 means no overfitting, r_v -> 1 means the critic memorizes the training set.

        :return: r_v
        :rtype: float
        """
        with torch.no_grad():
            m_val = self.critic(self.augment(self._next_val_batch())).mean().item()
        r_v = (m_train - m_val) / max(m_train - m_fake, 1e-3)

        hp = self.hparams
        step = np.sign(r_v - hp["ada_target"]) * (self.batch_size * hp["ada_interval"]) / (hp["ada_kimg"] * 1000)
        self.augment.p = float(min(max(self.augment.p + step, 0.0), 1.0))
        return r_v

    def save_samples(self, name):
        """
        Saves a grid of generated images (EMA generator) from the fixed noise.
        """
        with torch.no_grad():
            fake = self.g_ema(self.fixed_noise).cpu()
        path = os.path.join(self.sample_dir, f"{name}.png")
        vutils.save_image(fake, path, padding=2, normalize=True, value_range=(-1, 1))
        return path

    def generate(self, n=64, truncation_psi=1.0, use_ema=True):
        """
        Generates n images from random noise.

        :param truncation_psi: <1 gives cleaner but less diverse images
        :return: images in shape (n, 3, H, W), roughly in [-1, 1]
        :rtype: torch.Tensor
        """
        model = self.g_ema if use_ema else self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.hparams["nz"], device=device)
            fake = model(noise, truncation_psi=truncation_psi).cpu()
        self.netG.train()
        return fake

    def plot_losses(self, path=None):
        """
        Plots generator/critic loss, Wasserstein estimate and the ADA probability.
        """
        path = path or os.path.join(self.sample_dir, "losses.png")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(self.C_losses, label="critic", alpha=0.6)
        axes[0].set_title("Critic loss")
        axes[0].set_xlabel("critic steps")
        axes[0].legend()
        axes[1].plot(self.w_dists)
        axes[1].set_title("Wasserstein estimate  E[D(real)] - E[D(fake)]")
        axes[1].set_xlabel("critic steps")
        axes[2].plot(self.ada_ps)
        axes[2].set_title("ADA probability p")
        axes[2].set_xlabel("critic steps")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_real_vs_fake(self, path=None, truncation_psi=1.0):
        """
        Plots a batch of real images next to generated ones.
        """
        path = path or os.path.join(self.sample_dir, "real_vs_fake.png")
        real_batch = next(iter(self.train_dl))
        fake = self.generate(64, truncation_psi=truncation_psi)

        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(
            vutils.make_grid(real_batch[0][:64], padding=5, normalize=True), (1, 2, 0)
        ))
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(
            vutils.make_grid(fake, padding=2, normalize=True, value_range=(-1, 1)), (1, 2, 0)
        ))
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    def fit(
        self, 
        epochs, 
        ckpt_path=None,
        log_every=50
    ):
        """
        Trains for the requested number of additional epochs.
        """
        hp = self.hparams
        G_losses, W_dists = [], []
        start = time.time()

        print("Beginning training...")
        print(time.strftime("%Y-%m-%d %H:%M"))

        try:
            epoch_iterator = tqdm(range(epochs), desc=f"Training Epoch: 0/{epochs}")
            for epoch in epoch_iterator:
                epoch_iterator.set_description(f"Training Epoch: {epoch + 1}/{epochs}")
                epoch_G, epoch_W = [], []
                ada_real, ada_fake = [], []
                r_v = float("nan")

                self.netG.train()
                self.critic.train()
                for i, data in enumerate(self.train_dl):
                    real = data[0].to(device)

                    c = critic_step(
                        self.critic, self.netG, self.optC, self.augment, real,
                        hp["nz"], hp["gp_weight"], hp["drift_weight"], hp["mixing_prob"]
                    )
                    self.critic_steps += 1
                    self.C_losses.append(c["loss"])
                    self.w_dists.append(c["wdist"])
                    self.ada_ps.append(self.augment.p)
                    epoch_W.append(c["wdist"])

                    if self.adaptive:
                        ada_real.append(c["d_real"])
                        ada_fake.append(c["d_fake"])
                        if self.critic_steps % hp["ada_interval"] == 0:
                            r_v = self.update_ada(np.mean(ada_real), np.mean(ada_fake))
                            ada_real, ada_fake = [], []

                    if self.critic_steps % hp["n_critic"] == 0:
                        g_loss = generator_step(
                            self.critic, self.netG, self.optG, self.augment,
                            real.size(0), hp["nz"], hp["mixing_prob"]
                        )
                        self.g_steps += 1
                        update_ema(self.g_ema, self.netG, hp["ema_decay"], self.g_steps)
                        self.G_losses.append(g_loss)
                        epoch_G.append(g_loss)

                    if i % log_every == 0:
                        print(
                            "[%d/%d][%d/%d]\tLoss_C: %.4f\tW-dist: %.4f\tGP: %.4f\tD(x): %.3f\tD(G(z)): %.3f\tp: %.3f\tr_v: %.3f"
                            % (epoch + 1, epochs, i, len(self.train_dl), c["loss"], c["wdist"],
                               c["gp"], c["d_real"], c["d_fake"], self.augment.p, r_v)
                        )

                self.epochs_trained += 1
                G_losses.append(sum(epoch_G) / max(len(epoch_G), 1))
                W_dists.append(sum(epoch_W) / len(epoch_W))

                print(
                    f"Epoch: {self.epochs_trained} - Time: {round(time.time() - start, 2)} seconds - "
                    f"G Loss: {G_losses[-1]} - W-dist: {W_dists[-1]} - ADA p: {self.augment.p:.3f}"
                )
                start = time.time()

                self.save_samples(f"epoch_{self.epochs_trained:04d}")
                self.save(ckpt_path=ckpt_path)

        except KeyboardInterrupt:
            pass

        print("Checkpointing...")
        self.save(ckpt_path=ckpt_path)
        self.plot_losses()
        print("Done")
        print(time.strftime("%Y-%m-%d %H:%M"))

        return G_losses, W_dists


# ——————————————————————————————————————————————————————————————
# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a style-based WGAN-GP with ADA")
    parser.add_argument("--datapath", default=None, help="Image folder root (default: auto-find)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=StyleWGANHparams.IMAGE_SIZE)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--no_ada", action="store_true", help="Disable adaptive augmentation")
    parser.add_argument("--ckpt_path", default="checkpoints/stylewgan_ckpt.pt")
    parser.add_argument("--sample_dir", default="samples")
    parser.add_argument("--resume", action="store_true", help="Resume from ckpt_path")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    trainer = StyleWGANTrainer(
        hparams={"image_size": args.image_size, "ada": not args.no_ada},
        datapath=args.datapath,
        batch_size=args.batch_size,
        ckpt_path=args.ckpt_path,
        sample_dir=args.sample_dir,
        load_from_checkpoint=args.resume,
        n_samples=args.n_samples,
        num_workers=args.workers,
    )
    trainer.fit(args.epochs)
    trainer.plot_real_vs_fake()
