"""
StyleGAN2-style generator + WGAN critic.

Generator: mapping network -> style-modulated convolutions (weight demodulation),
           per-pixel noise injection, skip-architecture RGB outputs, truncation trick.
Critic:    plain residual-free conv stack with no normalization layers (required by the
           WGAN-GP gradient penalty), equalized learning rate, scalar unbounded output.
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2)


def channels_at(res, base, cap_mult=8):
    """
    Number of feature maps at a given resolution (more channels at low resolution).
    """
    return min(base * 256 // res, base * cap_mult)


# ——————————————————————————————————————————————————————————————
# Building blocks
class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class LeakyReLU(nn.Module):
    """Leaky ReLU with gain sqrt(2) to keep activation variance stable."""
    def forward(self, x):
        return F.leaky_relu(x, 0.2) * SQRT2


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_dim, out_dim, bias_init=0.0, lr_mul=1.0, activation=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_dim,), float(bias_init)))
        self.scale = lr_mul / math.sqrt(in_dim)
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)
        if self.activation:
            out = F.leaky_relu(out, 0.2) * SQRT2
        return out


class EqualConv2d(nn.Module):
    """Conv layer with equalized learning rate."""
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated convolution with optional weight demodulation and 2x upsampling."""
    def __init__(self, in_ch, out_ch, kernel_size, w_dim, demodulate=True, upsample=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.modulation = EqualLinear(w_dim, in_ch, bias_init=1.0)

    def forward(self, x, w):
        # Non-fused modulation: scale the input activations, run one ordinary (non-grouped)
        # convolution, then demodulate the output. Mathematically identical to modulating the
        # weights per sample, but much faster (grouped convs are very slow on CPU).
        b = x.size(0)
        s = self.modulation(w)  # (b, in)
        weight = self.weight[0] * self.scale  # (out, in, k, k)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        out = F.conv2d(x * s.view(b, self.in_ch, 1, 1), weight, padding=self.kernel_size // 2)
        if self.demodulate:
            w_sq = weight.pow(2).sum(dim=(2, 3))  # (out, in)
            demod = torch.rsqrt(s.pow(2) @ w_sq.t() + 1e-8)  # (b, out)
            out = out * demod.view(b, self.out_ch, 1, 1)
        return out


class NoiseInjection(nn.Module):
    """Adds learned-strength per-pixel Gaussian noise."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, kernel_size=3, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel_size, w_dim, upsample=upsample)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = LeakyReLU()

    def forward(self, x, w):
        return self.act(self.noise(self.conv(x, w)) + self.bias)


class ToRGB(nn.Module):
    def __init__(self, in_ch, w_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv = ModulatedConv2d(in_ch, 3, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, w, skip=None):
        out = self.conv(x, w) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


def _log2_size(image_size):
    log_size = int(math.log2(image_size))
    if 2 ** log_size != image_size or image_size < 8:
        raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
    return log_size


# ——————————————————————————————————————————————————————————————
# Generator
class Generator(nn.Module):
    def __init__(
        self,
        nz,
        ngf,
        image_size=128,
        n_mlp=4
    ) -> None:
        """
        Initializes the StyleGAN2-style Generator.

        Args:
            nz: Size of z latent vector (also used as the size of the intermediate w space).
            ngf: Base feature-map width in generator (channels = ngf*512/res, capped at ngf*8).
            image_size: Output resolution, a power of two >= 8.
            n_mlp: Number of layers in the mapping network.
        """
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.image_size = image_size
        self.log_size = _log2_size(image_size)
        self.num_blocks = self.log_size - 1  # 4x4, 8x8, ..., image_size

        # Mapping network z -> w (reduced learning rate, as in StyleGAN)
        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(EqualLinear(nz, nz, lr_mul=0.01, activation=True))
        self.mapping = nn.Sequential(*layers)
        self.register_buffer("w_avg", torch.zeros(nz))  # for the truncation trick

        ch = lambda res: channels_at(res, ngf)
        self.const = nn.Parameter(torch.randn(1, ch(4), 4, 4))
        self.conv1 = StyledConv(ch(4), ch(4), nz)
        self.to_rgb1 = ToRGB(ch(4), nz, upsample=False)

        self.up_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            in_ch, out_ch = ch(2 ** (i - 1)), ch(2 ** i)
            self.up_convs.append(StyledConv(in_ch, out_ch, nz, upsample=True))
            self.convs.append(StyledConv(out_ch, out_ch, nz))
            self.to_rgbs.append(ToRGB(out_ch, nz))

    def forward(self, z, truncation_psi=1.0, mixing_prob=0.0):
        """
        :param z: latent vectors in shape (B, nz)
        :param truncation_psi: <1 trades diversity for quality (w = w_avg + psi*(w - w_avg))
        :param mixing_prob: training-time style mixing probability (uses a second random z)
        :return: images in shape (B, 3, image_size, image_size), roughly in [-1, 1]
        """
        z = z.view(z.size(0), -1)
        w = self.mapping(z)
        if self.training:
            with torch.no_grad():
                self.w_avg.lerp_(w.detach().mean(dim=0), 0.01)

        if self.training and mixing_prob > 0 and random.random() < mixing_prob:
            w2 = self.mapping(torch.randn_like(z))
            cut = random.randint(1, self.num_blocks - 1)
            ws = [w if i < cut else w2 for i in range(self.num_blocks)]
        else:
            ws = [w] * self.num_blocks

        if truncation_psi != 1.0:
            ws = [self.w_avg + truncation_psi * (x - self.w_avg) for x in ws]

        x = self.const.expand(z.size(0), -1, -1, -1)
        x = self.conv1(x, ws[0])
        rgb = self.to_rgb1(x, ws[0])
        for i, (up_conv, conv, to_rgb) in enumerate(zip(self.up_convs, self.convs, self.to_rgbs)):
            x = up_conv(x, ws[i + 1])
            x = conv(x, ws[i + 1])
            rgb = to_rgb(x, ws[i + 1], skip=rgb)
        return rgb


# ——————————————————————————————————————————————————————————————
# Critic (WGAN)
def minibatch_std(x, group_size=4):
    b, c, h, w = x.shape
    g = min(group_size, b)
    while b % g:
        g -= 1
    y = x.view(g, b // g, c, h, w)
    y = (y - y.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)
    y = (y + 1e-8).sqrt().mean(dim=(1, 2, 3))
    y = y.view(1, -1, 1, 1, 1).expand(g, -1, 1, h, w).reshape(b, 1, h, w)
    return torch.cat([x, y], dim=1)


class Critic(nn.Module):
    def __init__(
        self,
        ndf,
        image_size=128,
        mbstd_group=0
    ) -> None:
        """
        Initializes the WGAN Critic (no sigmoid, no normalization layers).

        Args:
            ndf: Base feature-map width in critic (channels = ndf*512/res, capped at ndf*8).
            image_size: Input resolution, a power of two >= 8.
            mbstd_group: Minibatch-stddev group size. 0 disables it (recommended with WGAN-GP,
                since it makes each sample's score depend on the rest of the batch).
        """
        super().__init__()
        log_size = _log2_size(image_size)
        ch = lambda res: channels_at(res, ndf)

        layers = [EqualConv2d(3, ch(image_size), 1), LeakyReLU()]
        for i in range(log_size, 2, -1):
            in_ch, out_ch = ch(2 ** i), ch(2 ** (i - 1))
            layers += [
                EqualConv2d(in_ch, in_ch, 3, padding=1), LeakyReLU(),
                EqualConv2d(in_ch, out_ch, 3, padding=1), LeakyReLU(),
                nn.AvgPool2d(2),
            ]
        self.convs = nn.Sequential(*layers)

        self.mbstd_group = mbstd_group
        self.final_conv = nn.Sequential(
            EqualConv2d(ch(4) + (1 if mbstd_group > 0 else 0), ch(4), 3, padding=1), LeakyReLU()
        )
        self.final_linear = nn.Sequential(
            EqualLinear(ch(4) * 16, ch(4), activation=True),
            EqualLinear(ch(4), 1),
        )

    def forward(self, x):
        x = self.convs(x)
        if self.mbstd_group > 0:
            x = minibatch_std(x, self.mbstd_group)
        x = self.final_conv(x)
        return self.final_linear(x.flatten(1))  # (B, 1), unbounded
