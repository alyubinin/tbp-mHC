"""
WikiText-103 (raw) dataset preparation script.

To use with local parquet files:
1. Download parquet files from:
   https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-raw-v1
2. Place them in a 'parquet' subfolder next to this script:
   data/wikitext-103-raw-v1/parquet/*.parquet
3. Run: python prepare.py

To download from HuggingFace directly:
   python prepare.py --remote
"""

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
NUM_PROC = 8

# Global encoder - initialized once per process (required for Windows multiprocessing)
_enc = None
_eot = None

def _get_encoder():
    global _enc, _eot
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
        _eot = _enc.eot_token
    return _enc, _eot

def process(example):
    """Tokenize a single example."""
    enc, eot = _get_encoder()
    ids = enc.encode_ordinary(example['text'])
    ids.append(eot)
    return {
        'ids': ids,
        'len': len(ids),
        'bytes_len': len(example['text'].encode('utf-8')),
    }


def load_wikitext_dataset(data_dir, parquet_dir, remote=False):
    if remote:
        print("Loading WikiText-103-raw-v1 from HuggingFace...")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", num_proc=NUM_PROC)
        return dataset

    if not os.path.exists(parquet_dir):
        print(f"Error: Parquet directory not found: {parquet_dir}")
        print("\nPlease either:")
        print(f"  1. Download parquet files to: {parquet_dir}")
        print("     From: https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-raw-v1")
        print("  2. Or run with --remote flag to download via Python")
        raise SystemExit(1)

    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"Error: No parquet files found in {parquet_dir}")
        raise SystemExit(1)

    print(f"Loading {len(parquet_files)} parquet files from {parquet_dir}...")
    return load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(parquet_dir, "train-*.parquet"),
            "val": os.path.join(parquet_dir, "validation-*.parquet"),
            "test": os.path.join(parquet_dir, "test-*.parquet"),
        },
        num_proc=NUM_PROC,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare WikiText-103-raw-v1 dataset")
    parser.add_argument("--remote", action="store_true",
                        help="Download from HuggingFace instead of using local parquet files")
    parser.add_argument("--parquet-dir", type=str, default="parquet",
                        help="Directory containing parquet files (default: parquet)")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_dir = os.path.join(data_dir, args.parquet_dir)

    dataset = load_wikitext_dataset(data_dir, parquet_dir, remote=args.remote)

    if "validation" in dataset:
        dataset["val"] = dataset.pop("validation")

    print("Dataset splits:")
    for split, dset in dataset.items():
        print(f"  {split}: {len(dset):,} examples")

    print("Tokenizing...")
    tokenized = {}
    split_stats = {}

    for split, dset in dataset.items():
        cols_to_remove = [c for c in dset.column_names if c != 'text']
        tokenized[split] = dset.map(
            process,
            remove_columns=['text'] + cols_to_remove,
            desc=f"tokenizing {split}",
            num_proc=NUM_PROC,
        )

    for split, dset in tokenized.items():
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16
        batch_size = 100000

        dset_np = dset.with_format('numpy')
        token_count = 0
        byte_count = 0

        with open(filename, 'wb') as f:
            for i in tqdm(range(0, len(dset), batch_size), desc=f'writing {split}'):
                batch = dset_np[i:i+batch_size]
                arr = np.concatenate(batch['ids']).astype(dtype)
                f.write(arr.tobytes())
                token_count += len(arr)
                byte_count += int(batch['bytes_len'].sum())

        print(f'Saved {filename}: {token_count:,} tokens')
        split_stats[split] = {
            'total_tokens': token_count,
            'total_bytes': byte_count,
            'tokens_per_byte': token_count / byte_count,
        }

    meta = {
        'vocab_size': 50304,
        'train_total_tokens': split_stats['train']['total_tokens'],
        'train_total_bytes': split_stats['train']['total_bytes'],
        'train_tokens_per_byte': split_stats['train']['tokens_per_byte'],
        'val_total_tokens': split_stats['val']['total_tokens'],
        'val_total_bytes': split_stats['val']['total_bytes'],
        'val_tokens_per_byte': split_stats['val']['tokens_per_byte'],
    }
    if 'test' in split_stats:
        meta.update({
            'test_total_tokens': split_stats['test']['total_tokens'],
            'test_total_bytes': split_stats['test']['total_bytes'],
            'test_tokens_per_byte': split_stats['test']['tokens_per_byte'],
        })

    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved {meta_path}")

    print("Done!")
    print(f"Files saved in: {data_dir}")
    print("  - train.bin")
    print("  - val.bin")
    if 'test' in split_stats:
        print("  - test.bin")
    print("  - meta.pkl")
    print("\nUse in training with:")
    print("  dataset = 'wikitext-103-raw-v1'")
