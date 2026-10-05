"""
Differentiable augmentation pipeline for Adaptive Discriminator Augmentation (ADA).

Each transform is applied to each image independently with probability `p`.
Everything is differentiable, so gradients reach the generator through the augmented fakes.
"""
import math

import torch
import torch.nn.functional as F

GEOMETRIC = ("xflip", "rotate90", "translate", "scale", "rotate", "aniso")
COLOR = ("brightness", "contrast", "saturation")
ALL_OPS = GEOMETRIC + COLOR


def _eye(b, device):
    return torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)


def _scale_mat(sx, sy):
    m = _eye(sx.shape[0], sx.device)
    m[:, 0, 0] = sx
    m[:, 1, 1] = sy
    return m


def _translate_mat(tx, ty):
    m = _eye(tx.shape[0], tx.device)
    m[:, 0, 2] = tx
    m[:, 1, 2] = ty
    return m


def _rotate_mat(theta):
    m = _eye(theta.shape[0], theta.device)
    m[:, 0, 0] = torch.cos(theta)
    m[:, 0, 1] = -torch.sin(theta)
    m[:, 1, 0] = torch.sin(theta)
    m[:, 1, 1] = torch.cos(theta)
    return m


class AugmentPipe:
    """
    :param p: probability of applying each transform to each image (adjusted by ADA)
    :param ops: names of the enabled transforms (see ALL_OPS)
    """
    def __init__(self, p=0.0, ops=ALL_OPS):
        unknown = set(ops) - set(ALL_OPS)
        if unknown:
            raise ValueError(f"Unknown augmentations: {sorted(unknown)}; choose from {ALL_OPS}")
        self.p = float(p)
        self.ops = tuple(ops)

    def __call__(self, x):
        if self.p <= 0 or len(self.ops) == 0:
            return x
        b, device = x.shape[0], x.device
        gate = lambda: (torch.rand(b, device=device) < self.p)
        ones = torch.ones(b, device=device)
        zeros = torch.zeros(b, device=device)

        # ---- geometric: build one affine matrix per image
        m = _eye(b, device)
        if "xflip" in self.ops:
            flip = gate() & (torch.rand(b, device=device) < 0.5)
            m = m @ _scale_mat(torch.where(flip, -ones, ones), ones)
        if "rotate90" in self.ops:
            k = torch.randint(0, 4, (b,), device=device).float()
            m = m @ _rotate_mat(torch.where(gate(), k * math.pi / 2, zeros))
        if "translate" in self.ops:
            t = (torch.rand(b, 2, device=device) * 2 - 1) * 0.25  # +-12.5% of image size
            g = gate()
            m = m @ _translate_mat(torch.where(g, t[:, 0], zeros), torch.where(g, t[:, 1], zeros))
        if "scale" in self.ops:
            s = torch.exp2(torch.randn(b, device=device) * 0.2)
            s = torch.where(gate(), s, ones)
            m = m @ _scale_mat(s, s)
        if "rotate" in self.ops:
            theta = (torch.rand(b, device=device) * 2 - 1) * math.pi
            m = m @ _rotate_mat(torch.where(gate(), theta, zeros))
        if "aniso" in self.ops:
            a = torch.exp2(torch.randn(b, device=device) * 0.2)
            a = torch.where(gate(), a, ones)
            m = m @ _scale_mat(a, 1 / a)

        if any(op in self.ops for op in GEOMETRIC):
            grid = F.affine_grid(m[:, :2], list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        # ---- color
        if "brightness" in self.ops:
            delta = torch.where(gate(), torch.randn(b, device=device) * 0.2, zeros)
            x = x + delta.view(b, 1, 1, 1)
        if "contrast" in self.ops:
            c = torch.where(gate(), torch.exp2(torch.randn(b, device=device) * 0.5), ones)
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) * c.view(b, 1, 1, 1) + mean
        if "saturation" in self.ops:
            s = torch.where(gate(), torch.exp2(torch.randn(b, device=device) * 1.0), ones)
            gray = x.mean(dim=1, keepdim=True)
            x = gray + (x - gray) * s.view(b, 1, 1, 1)
        return x
