"""Download files from any Hugging Face Hub repo (dataset, model or space).

    python download_data.py quchenyuan/text-to-art-database --prefix data/samples --splits train test validation
    python download_data.py quchenyuan/text-to-art-database --prefix data/samples --splits train --max-files 3
    python download_data.py username/some-dataset --include "*.parquet" --dry-run
    python download_data.py username/some-model --repo-type model --include "*.safetensors" "*.json"

Requires: pip install huggingface_hub
Downloads are resumable: re-run the same command if it gets interrupted.
Private/gated repos: pass --token or run `huggingface-cli login` first.
"""
import argparse
import fnmatch
import os

from huggingface_hub import HfApi, hf_hub_download


def parse_args():
    ap = argparse.ArgumentParser(description="Download files from a Hugging Face Hub repo")
    ap.add_argument("repo_id", help="Repo id, e.g. 'user/name'")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    ap.add_argument("--revision", default=None, help="Branch, tag or commit (default: main)")
    ap.add_argument("--prefix", default="", help="Only files under this folder in the repo (default: whole repo)")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Subfolders of --prefix to download (e.g. train test). Default: all files under prefix")
    ap.add_argument("--include", nargs="+", default=None, help="Glob patterns to keep, e.g. '*.parquet' '*.json'")
    ap.add_argument("--exclude", nargs="+", default=None, help="Glob patterns to skip")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Max files per split (or in total if --splits is not given)")
    ap.add_argument("--out", default="../datasets", help="Local folder (default: ../datasets)")
    ap.add_argument("--token", default=None, help="HF access token (default: cached login / HF_TOKEN)")
    ap.add_argument("--dry-run", action="store_true", help="List files and sizes only")
    return ap.parse_args()


def matches(path, patterns):
    """True if the path or its filename matches any glob pattern."""
    name = os.path.basename(path)
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p) for p in patterns)


def list_files(api, a):
    prefix = a.prefix.strip("/")
    infos = api.list_repo_tree(
        a.repo_id,
        path_in_repo=prefix,
        repo_type=a.repo_type,
        revision=a.revision,
        recursive=True,
    )
    files = [i for i in infos if hasattr(i, "size")]  # skip folders
    if a.include:
        files = [f for f in files if matches(f.path, a.include)]
    if a.exclude:
        files = [f for f in files if not matches(f.path, a.exclude)]
    return sorted(files, key=lambda f: f.path)


def select(files, a):
    prefix = a.prefix.strip("/")
    base = f"{prefix}/" if prefix else ""

    if not a.splits:
        return files[: a.max_files]

    selected = []
    for split in a.splits:
        in_split = [f for f in files if f.path.startswith(f"{base}{split}/")]
        if not in_split:
            print(f"warning: no files found for split '{split}'")
        selected += in_split[: a.max_files]
    return selected


def main():
    a = parse_args()
    api = HfApi(token=a.token)

    selected = select(list_files(api, a), a)
    total_gb = sum(f.size or 0 for f in selected) / 1e9
    print(f"{len(selected)} files, {total_gb:.2f} GB total")
    for f in selected:
        print(f"  {f.path}  ({(f.size or 0) / 1e6:.0f} MB)")
    if a.dry_run or not selected:
        return

    for n, f in enumerate(selected, 1):
        print(f"[{n}/{len(selected)}] {f.path}")
        hf_hub_download(
            a.repo_id,
            f.path,
            repo_type=a.repo_type,
            revision=a.revision,
            token=a.token,
            local_dir=a.out,
        )
    print("done ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()