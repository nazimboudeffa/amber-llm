import argparse
import json
import os
import warnings

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def parse_source(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.removesuffix("_sample")


def build_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = int(1e30)
    return tokenizer


def concat_and_tokenize(lines, tokenizer, max_length, bos_text, eos_text):
    bos_tokens = tokenizer(bos_text, add_special_tokens=False)["input_ids"]
    eos_tokens = tokenizer(eos_text, add_special_tokens=False)["input_ids"]
    buffer = []
    for line in lines:
        sample = json.loads(line)
        iids = tokenizer(
            sample["text"], truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        buffer = buffer + bos_tokens + iids + eos_tokens
        while len(buffer) >= max_length:
            yield buffer[:max_length]
            buffer = buffer[max_length:]
    if len(buffer) > 0:
        warnings.warn(
            f"Dropping {len(buffer)} trailing tokens that do not fill a sequence"
        )


def process_source(in_path, out_path, tokenizer, max_length, bos_text, eos_text, truncate):
    print(f"Processing {in_path}...")
    if os.path.exists(out_path):
        print(f"Skipping {out_path} because it already exists...")
        return
    os.makedirs(out_path, exist_ok=False)
    source = parse_source(in_path)

    out_filename = os.path.join(out_path, "train.jsonl")
    n_samples = 0
    with open(in_path, encoding="utf-8") as fin, open(out_filename + ".tmp", "w", encoding="utf-8") as fout:
        for token_ids in tqdm(
            concat_and_tokenize(fin, tokenizer, max_length, bos_text, eos_text),
            desc=source,
        ):
            json_sample = {"token_ids": token_ids, "source": source}
            fout.write(json.dumps(json_sample) + "\n")
            n_samples += 1
            if truncate is not None and n_samples == truncate:
                break
    os.rename(out_filename + ".tmp", out_filename)
    print(f"Wrote {n_samples} samples to {out_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize and concatenate the RedPajama Tiny per-source jsonl files "
        "into the jsonl format used by the amber-data-prep pipeline."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="datasets",
        help="Folder containing the *_sample.jsonl files (default: datasets)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="refined",
        help="Output folder, one subfolder per source (default: refined)",
    )
    parser.add_argument(
        "--concat_tokens", type=int, default=2049, help="Tokens per sequence"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="huggyllama/llama-7b",
        help="HF tokenizer to use (default: huggyllama/llama-7b)",
    )
    parser.add_argument("--bos_text", type=str, default="")
    parser.add_argument("--eos_text", type=str, default="</s>")
    parser.add_argument(
        "--truncate_num_samples",
        type=int,
        default=None,
        help="Stop each source after this many samples (useful for quick tests)",
    )
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer)
    input_files = sorted(
        os.path.join(args.input_root, f)
        for f in os.listdir(args.input_root)
        if f.endswith("_sample.jsonl")
    )
    if not input_files:
        raise FileNotFoundError(
            f"No *_sample.jsonl files found in {args.input_root}. "
            "Run download.py first."
        )

    for in_path in input_files:
        source = parse_source(in_path)
        out_path = os.path.join(args.out_root, source)
        process_source(
            in_path,
            out_path,
            tokenizer,
            args.concat_tokens,
            args.bos_text,
            args.eos_text,
            args.truncate_num_samples,
        )


if __name__ == "__main__":
    main()
