"""Generate art from a text prompt with a trained checkpoint.

    python generate.py "oil painting of a stormy sea" -c saved_models/text2art_latest.pt -n 8
"""
import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from dcgan_model import Generator
from text_encoder import CLIPTextEncoder


def load_generator(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    g = Generator(channels=3, noise_dim=cfg["noise_dim"],
                  embed_dim=cfg["embed_dim"], embed_out_dim=cfg["embed_out_dim"]).to(device)
    g.load_state_dict(ckpt["generator"])
    return g.eval(), cfg


@torch.no_grad()
def generate(prompts, ckpt_path, n=8, out_path="generated.png", scale=4, seed=None, encoder=None):
    """One row of `n` samples per prompt. Returns the image grid tensor in [0, 1]."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, cfg = load_generator(ckpt_path, device)
    encoder = encoder or CLIPTextEncoder(cfg["encoder"], device=device)
    if seed is not None:
        torch.manual_seed(seed)

    rows = []
    for p in ([prompts] if isinstance(prompts, str) else prompts):
        emb = encoder.encode([p]).to(device).repeat(n, 1)
        noise = torch.randn(n, cfg["noise_dim"], 1, 1, device=device)
        img = g(noise, emb)                                   # [-1, 1]
        if scale > 1:                                         # smooth upscale; no new detail
            img = F.interpolate(img, scale_factor=scale, mode="bicubic", align_corners=False)
        rows.append(img.clamp(-1, 1))
    grid = vutils.make_grid(torch.cat(rows), nrow=n, normalize=True, value_range=(-1, 1), padding=2)
    if out_path:
        vutils.save_image(grid, out_path)
    return grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("-c", "--ckpt", default="saved_models/text2art_latest.pt")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("-o", "--out", default="generated.png")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    generate(a.prompt, a.ckpt, a.n, a.out, a.scale, a.seed)
    print("saved", a.out)
