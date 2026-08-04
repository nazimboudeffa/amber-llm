"""
Merge the refined RedPajama Tiny jsonl files (one per source) into a train/valid
split, distributed round-robin across `num_split` chunk files named train_0.jsonl,
train_1.jsonl, ... to match the naming expected by amber-train/main.py.

Shuffle is NOT done by this script (amber-train shuffles on load).
"""

import json
import os
import random
import fire
from tqdm import tqdm

SUBFOLDERS = 'arxiv,book,c4,cc_2023-06,github,stackexchange,wikipedia'


def main(input_root='refined',
         output_root='merged',
         subfolders=SUBFOLDERS,
         num_split=1,
         num_valid_samples_per_subfolder=10):
    os.makedirs(output_root, exist_ok=False)
    train_dir = os.path.join(output_root, 'train')
    valid_dir = os.path.join(output_root, 'valid')
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(valid_dir, exist_ok=False)

    for subfolder in subfolders.split(','):
        subfolder = subfolder.strip()
        subfolder_input_file = os.path.join(input_root, subfolder, 'train.jsonl')
        if not os.path.exists(subfolder_input_file):
            print(f"Skipping {subfolder}: {subfolder_input_file} not found")
            continue

        print(f"Counting lines in: {subfolder}")
        total_lines = 0
        with open(subfolder_input_file) as fin:
            for _ in tqdm(fin):
                total_lines += 1
        if total_lines == 0:
            print(f"Skipping {subfolder}: no samples")
            continue

        num_valid_samples = min(num_valid_samples_per_subfolder, total_lines)
        num_train_samples = total_lines - num_valid_samples
        num_train_per_split = num_train_samples // num_split if num_split else 0
        valid_sample_idx = set(random.sample(range(total_lines), num_valid_samples))
        assert len(valid_sample_idx) == num_valid_samples

        valid_filename = os.path.join(valid_dir, subfolder + '.jsonl')
        print(f'Num samples per train split: {num_train_per_split}')
        print(f'Will write {num_valid_samples} valid samples to: {valid_filename}')

        output_file_pool = [
            open(os.path.join(train_dir, f'train_{output_split_idx}.jsonl'), 'a')
            for output_split_idx in range(num_split)
        ]

        num_written_train = 0
        num_written_valid = 0
        with open(subfolder_input_file) as fin, open(valid_filename, 'w') as out_valid_file:
            for line_no, line in tqdm(enumerate(fin), desc=subfolder, total=total_lines):
                if line_no in valid_sample_idx:
                    out_valid_file.write(line)
                    num_written_valid += 1
                    continue
                output_split_idx = num_written_train % num_split
                output_file_pool[output_split_idx].write(line)
                num_written_train += 1

        for out_file in output_file_pool:
            out_file.close()

        print("Num train samples:", num_written_train)
        valid_percent = num_written_valid / total_lines * 100
        print("Num valid samples:", num_written_valid, f"({valid_percent:.3f}%)")
        print(f"Completed {subfolder}")
        print("===================")


if __name__ == "__main__":
    fire.Fire(main)
