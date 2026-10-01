"""Download data/samples (train/test/validation) from the text-to-art-database dataset.

    python download_data.py                      # everything (~17.5 GB!)
    python download_data.py --splits train --max-files 3     # just 3 train shards
    python download_data.py --dry-run            # list files and sizes only

Requires: pip install huggingface_hub   (poetry add huggingface_hub)
Downloads are resumable: re-run the same command if it gets interrupted.
"""
import argparse
import os

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "quchenyuan/text-to-art-database"
PREFIX = "data/samples/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../datasets", help="local folder (default: ../datasets)")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "validation"])
    ap.add_argument("--max-files", type=int, default=None, help="max files PER split")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    infos = api.list_repo_tree(REPO_ID, path_in_repo=PREFIX.rstrip("/"),
                               repo_type="dataset", recursive=True)
    files = sorted((i for i in infos if hasattr(i, "size")), key=lambda i: i.path)

    selected = []
    for split in a.splits:
        in_split = [f for f in files if f.path.startswith(f"{PREFIX}{split}/")]
        selected += in_split[: a.max_files]

    total_gb = sum(f.size or 0 for f in selected) / 1e9
    print(f"{len(selected)} files, {total_gb:.2f} GB total")
    for f in selected:
        print(f"  {f.path}  ({(f.size or 0) / 1e6:.0f} MB)")
    if a.dry_run:
        return

    for n, f in enumerate(selected, 1):
        print(f"[{n}/{len(selected)}] {f.path}")
        hf_hub_download(REPO_ID, f.path, repo_type="dataset", local_dir=a.out)
    print("done ->", os.path.abspath(os.path.join(a.out, PREFIX)))


if __name__ == "__main__":
    main()
