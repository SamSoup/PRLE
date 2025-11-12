import argparse
import os
from huggingface_hub import snapshot_download
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(
        description="Download a Hugging Face model repo to a local directory."
    )
    ap.add_argument(
        "--repo",
        required=True,
        help="HF repo id, e.g. meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    )
    ap.add_argument(
        "--out_dir", required=True, help="Local directory to store files"
    )
    ap.add_argument(
        "--cache_dir",
        default=None,
        help="Optional shared cache dir (defaults to HF_HOME if set)",
    )
    ap.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional patterns to limit download",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional patterns to exclude",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Downloading {args.repo} -> {args.out_dir}")
    print(
        "Tip: set HF_HUB_ENABLE_HF_TRANSFER=1 for faster multi-connection downloads."
    )
    snapshot_download(
        repo_id=args.repo,
        local_dir=args.out_dir,
        cache_dir=args.cache_dir,  # None -> uses HF_HOME if defined
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.exclude,
        tqdm_class=tqdm,
    )
    print("Done.")


if __name__ == "__main__":
    main()
