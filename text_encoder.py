"""Frozen CLIP text encoder: prompt string -> fixed-size embedding for the GAN."""
import torch
import torch.nn.functional as F


class CLIPTextEncoder:
    """
    Turns prompts into L2-normalised embeddings (512-d for ViT-B/32).
    Replaces the pre-computed skip-thought vectors that the birds.hdf5 pipeline used.
    CLIP only reads the first 77 tokens, so style words at the start of a prompt
    (e.g. "expressionist painting ...") are what matter most.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device).eval()
        self.embed_dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode(self, prompts, batch_size=256):
        prompts = [str(p) for p in prompts]
        out = []
        for i in range(0, len(prompts), batch_size):
            tok = self.tokenizer(
                prompts[i:i + batch_size],
                padding=True, truncation=True, max_length=77, return_tensors="pt",
            ).to(self.device)
            emb = self.model(**tok).text_embeds
            out.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(out)
