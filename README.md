# Picasso

Picasso is a project for learning and generating **Art
Nouveau** artwork with several generative-model approaches. It demonstrates
the progression from training a GAN from scratch, to a style-based WGAN, to
adapting a pretrained latent diffusion model with parameter-efficient
fine-tuning.

<div align="center">
  <img src="./static/logo.png" alt="Picasso Logo" width="200"/>

  ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
  ![Pytorch](https://img.shields.io/badge/Pytorch-FF8C00?logo=pytorch&logoColor=white)
</div>  


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
├── notebooks/
│   ├── dcgan.ipynb              # DCGAN training and sampling
│   ├── stylegan.ipynb           # Style-based WGAN-GP + ADA
│   └── finetune_latent_diffusion.ipynb
├── picasso/
│   ├── models/                  # GAN, augmentation, and diffusion modules
│   └── trainers/                # DCGAN and StyleGAN training loops
└── samples/                     # Training/sample visualizations
```

## Getting started

The core experiments support Python 3.10–3.13. Install the project
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

Then open one of the notebooks in [`notebooks`](./notebooks).

## Available Models

### Deep Convolutional Generative Adversarial Network (DCGAN)

### StyleGAN2-ADA (Adaptive Discriminator Augmentation)

<img src="./samples/style_gan_output.png" alt="Sample 1"/>


### Finetuned Stable Diffusion

The [`finetune_latent_diffusion.ipynb`](./notebooks/finetune_latent_diffusion.ipynb)
notebook adapts Stable Diffusion 1.5 to the Art Nouveau collection with
parameter-efficient LoRA fine-tuning. It does not train a diffusion model from
scratch: the original model weights remain frozen and only small, trainable
low-rank adapters are added to the UNet attention layers.

The notebook follows this process:

1. **Set up the runtime.** Install `diffusers`, `peft`, `transformers`,
   `accelerate`, and `safetensors`, then run on a CUDA GPU. The default
   configuration targets a 512 x 512 image size, batch size 1, gradient
   accumulation of 4, and 2,000 optimizer steps.
2. **Download and prepare the data.** It downloads the `art-nouveau` folder
   from the `StyleGender/StyleGender-Dataset` Hugging Face dataset (or reuses
   an existing local copy). Images are discovered recursively, resized so the
   short side is 512 pixels, center-cropped, optionally flipped, and normalized
   to `[-1, 1]`. Each image uses a same-name `.txt` sidecar caption when
   available; otherwise the default caption, `an illustration in art nouveau
   style`, is used.
3. **Load the pretrained components.** Stable Diffusion's VAE, UNet, CLIP
   tokenizer, text encoder, and noise scheduler are loaded from the Hub. The
   VAE converts images into latent representations and the text encoder
   provides prompt conditioning; the VAE and text encoder are kept frozen.
4. **Attach LoRA and define the objective.** LoRA adapters (rank 16 by
   default) are attached to the UNet. For each batch, the notebook encodes the
   images into latents, samples a random diffusion timestep and Gaussian noise,
   adds that noise to the latents, and trains the UNet to predict the noise
   using a weighted mean-squared-error loss. Mixed precision, gradient
   checkpointing, gradient accumulation, and min-SNR weighting reduce memory
   use and improve training efficiency.
5. **Compare before and after.** Baseline images are generated before training
   with fixed prompts and seeds. During training, validation images are
   regenerated every 500 steps, and the loss history is plotted so the style
   adaptation can be compared without changing the sampling conditions.
6. **Save and reuse the adapter.** The notebook saves
   `lora_output/pytorch_lora_weights.safetensors` and a resumable
   `training_state.pt` checkpoint every 500 steps and at the end of training.
   For inference, it loads a fresh Stable Diffusion 1.5 pipeline and applies
   the saved LoRA weights; the LoRA scale can be adjusted from the base model
   (`0`) toward the full adapted style (`1`).

Prompt: an illustration in art nouveau style, a young woman holding a lantern among flowers 

<img src="./samples/stable_diffusion_output_3.png" alt="Sample 1"/>


Prompt: an illustration in art nouveau style, a cat in a forest with flowers and butterflies

<img src="./samples/stable_diffusion_output_2.png" alt="Sample 2"/>
