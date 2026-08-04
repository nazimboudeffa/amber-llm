import argparse
import os
import sys
import urllib.request

REPO_ID = "severo/RedPajama-Tiny"
FILES = [
    "data/train-00000-of-00001.parquet",
    "arxiv_sample.jsonl",
    "book_sample.jsonl",
    "c4_sample.jsonl",
    "cc_2023-06_sample.jsonl",
    "github_sample.jsonl",
    "stackexchange_sample.jsonl",
    "wikipedia_sample.jsonl",
]


def download_file(url, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser(
        description=f"Download the {REPO_ID} dataset from Hugging Face."
    )
    parser.add_argument(
        "--out_root",
        default="datasets",
        help="Directory to store the downloaded files (default: datasets)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=FILES,
        help=f"Files to download (default: all of {FILES})",
    )
    args = parser.parse_args()

    for f in args.files:
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{f}"
        dest = os.path.join(args.out_root, f)
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)
            return 1
    print(f"Done. Files saved under {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
