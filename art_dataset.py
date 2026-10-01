"""Parquet shards (prompt + image bytes) -> cached uint8 image tensor + cached text embeddings.

`source` can be a single .parquet file, a folder (searched recursively for *.parquet),
a glob pattern ("data/train-*.parquet"), or a list of any of these.
Each shard is processed and cached separately, so:
  * only one shard is ever decoded in memory at a time,
  * an interrupted run resumes where it stopped,
  * adding new shards later only processes the new ones.
"""
import glob
import io
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x=None, **kw):
        return x


# ----------------------------------------------------------------------------- helpers
def find_shards(source):
    """Resolve a file / folder / glob / list into a sorted list of parquet paths."""
    if isinstance(source, (list, tuple)):
        paths = []
        for s in source:
            paths += find_shards(s)
        return sorted(dict.fromkeys(paths))  # de-duplicate, keep sorted

    source = str(source)
    if os.path.isdir(source):
        paths = glob.glob(os.path.join(source, "**", "*.parquet"), recursive=True)
    elif any(c in source for c in "*?["):
        paths = glob.glob(source, recursive=True)
    else:
        paths = [source]
    paths = sorted(p for p in paths if os.path.isfile(p))
    if not paths:
        raise FileNotFoundError(f"No parquet files found for: {source}")
    return paths


def _raw_bytes(cell):
    # Hugging Face parquet files often store images as {"bytes": ..., "path": ...}
    return cell["bytes"] if isinstance(cell, dict) else cell


def _decode(raw, size, crop):
    img = Image.open(io.BytesIO(raw))
    img.draft("RGB", (size * 4, size * 4))  # fast JPEG downscale for huge images
    img = img.convert("RGB")
    if crop == "center":  # keep aspect ratio, cut the middle square
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    # crop == "squash": resize the whole image into a square (keeps composition, distorts shapes)
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def _safe_decode(raw, size, crop):
    try:
        return _decode(raw, size, crop)
    except Exception:
        return None  # corrupt row -> skipped


def _process_shard(path, size, crop, limit, threads):
    """Stream one shard in small batches (low RAM), decode images with a thread pool."""
    pf = pq.ParquetFile(path)
    imgs, prompts = [], []
    bar = tqdm(total=min(pf.metadata.num_rows, limit or 10**12),
               desc=os.path.basename(path), leave=False)
    with ThreadPoolExecutor(threads) as ex:
        for batch in pf.iter_batches(batch_size=256, columns=["prompt", "image"]):
            d = batch.to_pydict()
            raws = [_raw_bytes(c) for c in d["image"]]
            decoded = list(ex.map(lambda r: _safe_decode(r, size, crop), raws))
            for p, arr in zip(d["prompt"], decoded):
                if arr is not None:
                    imgs.append(arr)
                    prompts.append(str(p))
            if bar is not None and hasattr(bar, "update"):
                bar.update(len(raws))
            if limit and len(imgs) >= limit:
                break
    if bar is not None and hasattr(bar, "close"):
        bar.close()
    if limit:
        imgs, prompts = imgs[:limit], prompts[:limit]
    if not imgs:
        return torch.empty(0, 3, size, size, dtype=torch.uint8), []
    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    return images, prompts


# ----------------------------------------------------------------------------- main API
def prepare_data(source, encoder, image_size=64, crop="center",
                 cache_dir="./cache", max_samples=None, num_threads=8):
    """
    Returns (images uint8 [N,3,S,S], embeds float [N,D], prompts list[str]) over all shards.

    max_samples: stop after this many images in total (handy for smoke tests).
    Needs RAM for all images at once: ~12 KB each at 64x64 (100k images ~ 1.2 GB).
    """
    shards = find_shards(source)
    os.makedirs(cache_dir, exist_ok=True)
    enc_name = encoder.model_name.replace("/", "-")
    print(f"{len(shards)} parquet shard(s) found")

    all_images, all_embeds, all_prompts = [], [], []
    total = 0
    for path in tqdm(shards, desc="shards"):
        remaining = None if max_samples is None else max_samples - total
        if remaining is not None and remaining <= 0:
            break

        # cache key includes the shard's parent folder name (train/test/...) to avoid clashes
        stem = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        tag = f"{parent}_{stem}_{image_size}_{crop}_n{remaining}"
        img_cache = os.path.join(cache_dir, f"{tag}_images.pt")
        emb_cache = os.path.join(cache_dir, f"{tag}_{enc_name}_embeds.pt")

        if os.path.exists(img_cache):
            d = torch.load(img_cache)
            images, prompts = d["images"], d["prompts"]
        else:
            images, prompts = _process_shard(path, image_size, crop, remaining, num_threads)
            torch.save({"images": images, "prompts": prompts}, img_cache)
        if len(images) == 0:
            continue

        if os.path.exists(emb_cache) and len(torch.load(emb_cache)) == len(prompts):
            embeds = torch.load(emb_cache)
        else:
            embeds = encoder.encode(prompts)
            torch.save(embeds, emb_cache)

        all_images.append(images)
        all_embeds.append(embeds)
        all_prompts += prompts
        total += len(prompts)

    if not all_images:
        raise RuntimeError("No images could be loaded from the given source.")
    return torch.cat(all_images), torch.cat(all_embeds), all_prompts


class Text2ArtDataset(Dataset):
    """In-memory dataset. Images are scaled to [-1, 1] to match the generator's Tanh."""

    def __init__(self, images, embeds, flip=True):
        self.images, self.embeds, self.flip = images, embeds, flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].float().div(127.5).sub(1.0)
        if self.flip and random.random() < 0.5:
            img = img.flip(-1)
        return {"image": img, "embed": self.embeds[i]}
