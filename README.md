# Picasso

Picasso is a project for learning and generating **Art
Nouveau** artwork with several generative-model approaches. It demonstrates
the progression from training a GAN from scratch, to a style-based WGAN, to
adapting a pretrained latent diffusion model with parameter-efficient
fine-tuning.

## What this project demonstrates

| Model | Training approach | Notebook |
| --- | --- | --- |
| DCGAN | Convolutional generator and discriminator trained from scratch | [`notebooks/dcgan.ipynb`](./notebooks/dcgan.ipynb) |
| StyleGAN | Style-based generator with WGAN-GP and adaptive discriminator augmentation (ADA) | [`notebooks/stylegan.ipynb`](./notebooks/stylegan.ipynb) |
| Latent diffusion | Stable Diffusion 1.5 fine-tuned with LoRA adapters on Art Nouveau images | [`notebooks/finetune_latent_diffusion.ipynb`](./notebooks/finetune_latent_diffusion.ipynb) |

## Highlights

- Trains a baseline DCGAN at low resolution and saves resumable weights.
- Trains a style-based WGAN with gradient penalty, style mixing, and optional
  ADA to improve stability on a relatively small image collection.
- Fine-tunes only low-rank LoRA adapters in the UNet attention layers of
  Stable Diffusion 1.5; the base model, VAE, and text encoder remain frozen.
- Compares baseline and fine-tuned diffusion samples using the same prompts
  and seeds, then reloads the saved LoRA weights for inference.
- Includes reusable model, augmentation, data, and trainer modules under
  [`picasso`](./picasso).

## Repository structure

```text
picasso/
├── datasets/
│   └── art-nouveau/             # Art Nouveau image collection
├── notebooks/
│   ├── dcgan.ipynb              # DCGAN training and sampling
│   ├── stylegan.ipynb           # Style-based WGAN-GP + ADA
│   └── finetune_latent_diffusion.ipynb
├── picasso/
│   ├── models/                  # GAN, augmentation, and diffusion modules
│   └── trainers/                # DCGAN and StyleGAN training loops
├── checkpoints/                 # Example GAN checkpoints
└── samples/                     # Training/sample visualizations
```

## Getting started

The core GAN experiments support Python 3.10–3.13. Install the project
dependencies with Poetry:

```bash
poetry install
```

Or install the pinned requirements directly:

```bash
pip install -r requirements.txt
```

Launch Jupyter from the repository root:

```bash
poetry run jupyter lab
```

Then open one of the notebooks in [`notebooks`](./notebooks). The GAN
notebooks can run on CPU, although training is substantially faster with a
CUDA-capable GPU.

### Latent diffusion fine-tuning

Open [`notebooks/finetune_latent_diffusion.ipynb`](./notebooks/finetune_latent_diffusion.ipynb)
in Kaggle, Colab, or another GPU environment. The notebook:

1. Downloads or reuses the `art-nouveau` image folder.
2. Loads Stable Diffusion 1.5 from the Hugging Face Hub.
3. Adds LoRA adapters to the UNet attention layers.
4. Generates baseline samples, fine-tunes the adapters, and compares results.
5. Saves checkpoints and demonstrates loading the LoRA weights for inference.

The default configuration targets 512px images, batch size 1, mixed precision,
and gradient checkpointing. A GPU with approximately 6–8 GB of memory is
recommended; reduce the resolution if necessary. The diffusion notebook
installs its additional `diffusers`, `peft`, `accelerate`, and `safetensors`
dependencies in the notebook environment.
