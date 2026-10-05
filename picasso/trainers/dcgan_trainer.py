# ——————————————————————————————————————————————————————————————
# Imports
import os
import time
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from picasso.models.dcgan import Generator, Discriminator, weights_init

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

    :return: absolute-ish path of the dataset root
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
        # Try the requested image, then fall forward until one loads
        for offset in range(len(self)):
            try:
                return super().__getitem__((index + offset) % len(self))
            except Exception:
                continue
        raise RuntimeError("No readable images found")


def train_step(
    netD,
    netG,
    optimizerD,
    optimizerG,
    criterion,
    real,
    nz,
    real_label=1.0,
    fake_label=0.0
):
    """
    Executes one optimization step for both D and G.

    :param real: batch of real images in shape (B, 3, H, W)
    :type real: torch.Tensor

    :return: dict with errD, errG, D_x, D_G_z1, D_G_z2
    :rtype: dict
    """
    b_size = real.size(0)

    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ## Train with all-real batch
    netD.zero_grad()
    label = torch.full((b_size,), real_label, dtype=torch.float, device=real.device)
    output = netD(real).view(-1)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    noise = torch.randn(b_size, nz, 1, 1, device=real.device)
    fake = netG(noise)
    label.fill_(fake_label)
    output = netD(fake.detach()).view(-1)
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    # (2) Update G network: maximize log(D(G(z)))
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    return {
        "errD": errD.item(),
        "errG": errG.item(),
        "D_x": D_x,
        "D_G_z1": D_G_z1,
        "D_G_z2": D_G_z2,
    }


def unwrap(model):
    """
    Returns the underlying module when wrapped in nn.DataParallel.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


# ——————————————————————————————————————————————————————————————
# DCGAN Hyperparameters
class DCGANHparams:
    """
    Hyperparameters for DCGAN model and training.
    """
    IMAGE_SIZE = 64
    NZ = 100
    NGF = 64
    NDF = 64
    LR = 0.0002
    BETA1 = 0.5

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        nz=NZ,
        ngf=NGF,
        ndf=NDF,
        lr=LR,
        beta1=BETA1
    ):
        self.image_size = image_size
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.beta1 = beta1


# ——————————————————————————————————————————————————————————————
# DCGAN Trainer
class DCGANTrainer:
    """
    Trainer for DCGAN (Generator + Discriminator).
    """
    def __init__(
        self,
        hparams=None,
        datapath=None,
        batch_size=32,
        ckpt_path="checkpoints/dcgan_ckpt.pt",
        sample_dir="samples",
        load_from_checkpoint=False,
        n_samples=None,
        num_workers=0,
        ngpu=1,
        seed=999,
        deterministic=True,
    ):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.ngpu = ngpu if torch.cuda.is_available() else 0
        self.ckpt_path = ckpt_path
        self.sample_dir = sample_dir
        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # Reproducibility
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)

        if hparams is None:
            hparams = vars(DCGANHparams())
        elif isinstance(hparams, dict):
            hparams = vars(DCGANHparams(**hparams))
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
        if len(ds) < 1:
            raise ValueError(f"No images found in {datapath}")
        self.ds = ds
        print(f"There are {len(ds)} samples in the data")

        self.dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )

        self.netG = Generator(self.ngpu, hparams["nz"], hparams["ngf"]).to(device)
        self.netD = Discriminator(self.ngpu, hparams["ndf"]).to(device)
        if device.type == "cuda" and self.ngpu > 1:
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], 0.999)
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], 0.999)
        )

        # Fixed noise to visualize generator progress across epochs
        self.fixed_noise = torch.randn(64, hparams["nz"], 1, 1, device=device)

        self.G_losses = []
        self.D_losses = []
        self.epochs_trained = 0

        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()

    def save(self, ckpt_path=None):
        """
        Saves a training checkpoint.
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = {
            "netG_state_dict": unwrap(self.netG).state_dict(),
            "netD_state_dict": unwrap(self.netD).state_dict(),
            "optimizerG_state_dict": self.optimizerG.state_dict(),
            "optimizerD_state_dict": self.optimizerD.state_dict(),
            "G_losses": self.G_losses,
            "D_losses": self.D_losses,
            "epochs_trained": self.epochs_trained,
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

        self.hparams = ckpt["hparams"]
        print("Loading the models...", end="")
        print(unwrap(self.netG).load_state_dict(ckpt["netG_state_dict"]))
        print(unwrap(self.netD).load_state_dict(ckpt["netD_state_dict"]))
        self.optimizerG.load_state_dict(ckpt["optimizerG_state_dict"])
        self.optimizerD.load_state_dict(ckpt["optimizerD_state_dict"])

        self.G_losses = ckpt["G_losses"]
        self.D_losses = ckpt["D_losses"]
        self.epochs_trained = ckpt["epochs_trained"]
        self.fixed_noise = ckpt["fixed_noise"].to(device)

        return

    def save_samples(self, name):
        """
        Saves a grid of generated images from the fixed noise.
        """
        unwrap(self.netG).eval()
        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
        unwrap(self.netG).train()
        path = os.path.join(self.sample_dir, f"{name}.png")
        vutils.save_image(fake, path, padding=2, normalize=True)
        return path

    def generate(self, n=64):
        """
        Generates n images from random noise.

        :return: images in shape (n, 3, H, W), values in [-1, 1]
        :rtype: torch.Tensor
        """
        unwrap(self.netG).eval()
        with torch.no_grad():
            noise = torch.randn(n, self.hparams["nz"], 1, 1, device=device)
            fake = self.netG(noise).cpu()
        unwrap(self.netG).train()
        return fake

    def plot_losses(self, path=None):
        """
        Plots generator and discriminator loss.
        """
        path = path or os.path.join(self.sample_dir, "losses.png")
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    def plot_real_vs_fake(self, path=None):
        """
        Plots a batch of real images next to generated ones.
        """
        path = path or os.path.join(self.sample_dir, "real_vs_fake.png")
        real_batch = next(iter(self.dataloader))
        fake = self.generate(64)

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
            vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)
        ))
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    def fit(self, epochs, log_every=50):
        """
        Trains the DCGAN for the requested number of additional epochs.
        """
        G_losses = []
        D_losses = []
        start = time.time()

        print("Beginning training...")
        print(time.strftime("%Y-%m-%d %H:%M"))

        try:
            epoch_iterator = tqdm(range(epochs), desc=f"Training Epoch: 0/{epochs}")
            for epoch in epoch_iterator:
                epoch_iterator.set_description(f"Training Epoch: {epoch + 1}/{epochs}")
                epoch_G, epoch_D = [], []

                self.netG.train()
                self.netD.train()
                for i, data in enumerate(self.dataloader):
                    real = data[0].to(device)
                    stats = train_step(
                        self.netD,
                        self.netG,
                        self.optimizerD,
                        self.optimizerG,
                        self.criterion,
                        real,
                        self.hparams["nz"]
                    )
                    epoch_G.append(stats["errG"])
                    epoch_D.append(stats["errD"])

                    if i % log_every == 0:
                        print(
                            "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                            % (epoch + 1, epochs, i, len(self.dataloader),
                               stats["errD"], stats["errG"],
                               stats["D_x"], stats["D_G_z1"], stats["D_G_z2"])
                        )

                self.G_losses.extend(epoch_G)
                self.D_losses.extend(epoch_D)
                G_losses.append(sum(epoch_G) / len(epoch_G))
                D_losses.append(sum(epoch_D) / len(epoch_D))
                self.epochs_trained += 1

                print(
                    f"Epoch: {self.epochs_trained} - Time: {round(time.time() - start, 2)} seconds - "
                    f"G Loss: {G_losses[-1]} - D Loss: {D_losses[-1]}"
                )
                start = time.time()

                self.save_samples(f"epoch_{self.epochs_trained:04d}")
                self.save()  # checkpoint every epoch so crashes lose little

        except KeyboardInterrupt:
            pass

        print("Checkpointing...")
        self.save()
        self.plot_losses()
        print("Done")
        print(time.strftime("%Y-%m-%d %H:%M"))

        return G_losses, D_losses


# ——————————————————————————————————————————————————————————————
# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DCGAN")
    parser.add_argument("--datapath", default=None, help="Image folder root (default: auto-find)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--ckpt_path", default="checkpoints/dcgan_ckpt.pt")
    parser.add_argument("--sample_dir", default="samples")
    parser.add_argument("--resume", action="store_true", help="Resume from ckpt_path")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ngpu", type=int, default=1)
    args = parser.parse_args()

    trainer = DCGANTrainer(
        datapath=args.datapath,
        batch_size=args.batch_size,
        ckpt_path=args.ckpt_path,
        sample_dir=args.sample_dir,
        load_from_checkpoint=args.resume,
        n_samples=args.n_samples,
        num_workers=args.workers,
        ngpu=args.ngpu,
    )
    trainer.fit(args.epochs)
    trainer.plot_real_vs_fake()