"""
OpenWebText dataset preparation script.

To use with local parquet files:
1. Download parquet files from:
   https://huggingface.co/datasets/Skylion007/openwebtext/tree/main
2. Place them in a 'parquet' subfolder next to this script:
   data/openwebtext/parquet/*.parquet
3. Run: python prepare.py

To download from HuggingFace directly (slower):
   python prepare.py --remote

Reference: https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
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
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(eot)  # add the end of text token, e.g. 50256 for gpt2 bpe
    return {
        'ids': ids,
        'len': len(ids),
        'bytes_len': len(example['text'].encode('utf-8')),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare OpenWebText dataset")
    parser.add_argument("--remote", action="store_true",
                        help="Download from HuggingFace instead of using local parquet files")
    parser.add_argument("--parquet-dir", type=str, default="parquet",
                        help="Directory containing parquet files (default: parquet)")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_dir = os.path.join(data_dir, args.parquet_dir)

    if args.remote:
        # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        print("Loading OpenWebText dataset from HuggingFace...")
        dataset = load_dataset("openwebtext", num_proc=NUM_PROC)
        dataset = dataset["train"]  # owt only contains the 'train' split
    else:
        if not os.path.exists(parquet_dir):
            print(f"Error: Parquet directory not found: {parquet_dir}")
            print(f"\nPlease either:")
            print(f"  1. Download parquet files to: {parquet_dir}")
            print(f"     From: https://huggingface.co/datasets/Skylion007/openwebtext/tree/main")
            print(f"  2. Or run with --remote flag to download via Python (slower)")
            exit(1)

        parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        if not parquet_files:
            print(f"Error: No parquet files found in {parquet_dir}")
            exit(1)

        print(f"Loading {len(parquet_files)} parquet files from {parquet_dir}...")
        dataset = load_dataset(
            "parquet",
            data_files=os.path.join(parquet_dir, "*.parquet"),
            split="train",
            num_proc=NUM_PROC,
        )

    print(f"Loaded {len(dataset):,} examples")

    # create a test split
    print("Splitting dataset...")
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

    # DatasetDict({
    #     train: Dataset({ features: ['text'], num_rows: 8009762 })
    #     val: Dataset({ features: ['text'], num_rows: 4007 })
    # })

    print("Tokenizing...")
    cols_to_remove = [c for c in split_dataset['train'].column_names if c != 'text']

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'] + cols_to_remove,
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    split_stats = {}

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16  # can do since enc.max_token_value == 50256 is < 2**16
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
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved {meta_path}")

    print("Done!")
    print(f"Files saved in: {data_dir}")
    print("  - train.bin (~17GB, ~9B tokens)")
    print("  - val.bin (~8.5MB, ~4M tokens)")
    print("  - meta.pkl")

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
