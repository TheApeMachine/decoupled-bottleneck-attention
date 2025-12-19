#!/usr/bin/env python3
"""
prepare_fineweb.py

Download and prepare FineWeb-Edu dataset for training.

This script downloads a subset of the FineWeb-Edu dataset from Hugging Face
and tokenizes it using tiktoken (GPT-2 encoding).

Usage:
    python3.12 prepare_fineweb.py --tokens 100M    # 100 million tokens
    python3.12 prepare_fineweb.py --tokens 1B     # 1 billion tokens
    python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.tokens
    python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy

Requirements:
    pip install datasets tiktoken tqdm numpy
"""

import argparse
import os
import gc
from pathlib import Path
from typing import Optional
import numpy as np

# Check dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: 'datasets' not installed. Run: pip install datasets")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Error: 'tiktoken' not installed. Run: pip install tiktoken")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x


def parse_token_count(s: str) -> int:
    """Parse token count string like '100M' or '1B'."""
    s = s.upper().strip()
    if s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)


def download_and_tokenize(
    target_tokens: int,
    output_path: str,
    sample_name: str = "sample-10BT",  # FineWeb-Edu 10BT sample
    output_format: str = "auto",
):
    """
    Download FineWeb-Edu and tokenize to target token count.
    
    Args:
        target_tokens: Number of tokens to collect
        output_path: Output file path
        sample_name: Which FineWeb-Edu sample to use
        output_format: 'text', 'npy', or 'auto'
    """
    if not HAS_DATASETS or not HAS_TIKTOKEN:
        print("Missing dependencies. Please install: pip install datasets tiktoken")
        return False
    
    # Determine format if auto
    if output_format == "auto":
        output_format = "npy" if output_path.endswith(".npy") else "text"
    
    print(f"Target: {target_tokens:,} tokens")
    print(f"Output: {output_path} (format={output_format})")
    print(f"Dataset: HuggingFaceFW/fineweb-edu ({sample_name})")
    print("-" * 60)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Stream dataset
    print("Loading dataset (streaming mode)...")
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=sample_name,
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative: fineweb-edu-score-2 subset...")
        try:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu-score-2",
                split="train",
                streaming=True,
            )
        except Exception as e2:
            print(f"Error loading alternative: {e2}")
            return False
    
    # Collect tokens
    total_tokens = 0
    doc_count = 0
    writer = None
    write_pos = 0
    
    print("Tokenizing documents...")
    pbar = tqdm(total=target_tokens, unit="tok", desc="Tokens") if HAS_TQDM else None
    
    try:
        # Streaming writers:
        # - npy: write directly into a memmap-backed .npy (no giant Python list)
        # - text: append chunks of space-separated ints
        if output_format == "npy":
            from numpy.lib.format import open_memmap  # type: ignore

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            writer = open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(int(target_tokens),))
            write_pos = 0
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            writer = open(output_path, "w", encoding="utf-8")

        for doc in dataset:
            # Get text content
            text = doc.get("text", "")
            if not text:
                continue
            
            # Tokenize
            tokens = enc.encode_ordinary(text)
            if not tokens:
                continue
            
            new_tokens = len(tokens)
            remaining = int(target_tokens) - int(total_tokens)
            take = int(new_tokens if new_tokens <= remaining else remaining)
            if take <= 0:
                break

            # Write
            if output_format == "npy":
                assert writer is not None
                assert hasattr(writer, "__setitem__")
                # Clip into uint16 range (GPT-2 vocab fits; this is defensive).
                arr = np.asarray(tokens[:take], dtype=np.int64)
                if arr.size > 0:
                    arr = np.clip(arr, 0, 65535).astype(np.uint16, copy=False)
                    writer[write_pos : write_pos + int(arr.size)] = arr
                    write_pos += int(arr.size)
            else:
                assert writer is not None
                # Write as space-separated integers (streaming)
                # Avoid huge join by chunking.
                chunk_size = 100_000
                for i in range(0, take, chunk_size):
                    chunk = tokens[i : i + chunk_size]
                    writer.write(" ".join(map(str, chunk)))
                    if i + chunk_size < take:
                        writer.write(" ")
                writer.write(" ")

            total_tokens += take
            doc_count += 1
            
            if pbar:
                pbar.update(take)
            
            # Check if we have enough
            if total_tokens >= target_tokens:
                break
            
            # Progress update every 1000 docs
            if doc_count % 1000 == 0 and not HAS_TQDM:
                print(f"  {doc_count:,} docs | {total_tokens:,} tokens ({100*total_tokens/target_tokens:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nInterrupted! Saving collected tokens...")
    
    # Explicit cleanup to avoid Bad file descriptor errors from background threads
    if 'dataset' in locals():
        del dataset
    gc.collect()

    if pbar:
        pbar.close()
    
    # Finalize writer
    if output_format == "npy":
        # If we didn't reach target_tokens (e.g., interruption), rewrite to exact length.
        # Note: shrinking a .npy in-place isn't supported; we create a truncated copy.
        if total_tokens < target_tokens:
            print(f"[warn] Collected {total_tokens:,} < target {target_tokens:,} tokens; writing truncated file.")
            tmp_path = output_path + ".partial.npy"
            from numpy.lib.format import open_memmap  # type: ignore
            mm_in = np.load(output_path, mmap_mode="r")
            mm_out = open_memmap(tmp_path, mode="w+", dtype=np.uint16, shape=(int(total_tokens),))
            mm_out[:] = mm_in[: int(total_tokens)]
            del mm_out
            del mm_in
            os.replace(tmp_path, output_path)
    else:
        if writer is not None:
            try:
                writer.flush()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass
    
    print(f"\nCollected {int(total_tokens):,} tokens from {doc_count:,} documents")
    print(f"Saved to {output_path}")

    # Verify
    file_size = os.path.getsize(output_path)
    print(f"Done! File size: {file_size / 1e6:.1f} MB")
    
    # Write metadata
    meta_path = output_path + ".meta"
    with open(meta_path, 'w') as f:
        f.write(f"tokens: {int(total_tokens)}\n")
        f.write(f"documents: {doc_count}\n")
        f.write(f"encoding: gpt2\n")
        f.write(f"vocab_size: {enc.n_vocab}\n")
        f.write(f"source: HuggingFaceFW/fineweb-edu\n")
        f.write(f"format: {output_format}\n")
    print(f"Metadata saved to {meta_path}")
    
    return True


def verify_existing(path: str) -> Optional[int]:
    """Check if file exists and return token count."""
    if not os.path.exists(path):
        return None
    
    meta_path = path + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if line.startswith("tokens:"):
                    return int(line.split(":")[1].strip())
    
    # Count tokens in file (slow)
    print(f"Counting tokens in {path}...")
    if path.endswith(".npy"):
        try:
            # Use mmap_mode='r' to peek without full load
            arr = np.load(path, mmap_mode='r')
            return len(arr)
        except Exception as e:
            print(f"Error reading npy: {e}")
            return None
    else:
        with open(path, 'r') as f:
            content = f.read()
            tokens = len(content.split())
        return tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    parser.add_argument("--tokens", type=str, default="100M",
                       help="Number of tokens to collect (e.g., 100M, 1B)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: fineweb_{tokens}.tokens)")
    parser.add_argument("--force", action="store_true",
                       help="Re-download even if file exists")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "text", "npy"],
                       help="Output format (text or npy). Auto infers from output filename.")
    
    args = parser.parse_args()
    
    target_tokens = parse_token_count(args.tokens)
    
    # Determine format and default path
    fmt = args.format
    
    if args.output:
        output_path = args.output
        if fmt == "auto":
            fmt = "npy" if output_path.endswith(".npy") else "text"
    else:
        # Format nicely: 100M, 1B, etc.
        if target_tokens >= 1_000_000_000:
            suffix = f"{target_tokens // 1_000_000_000}B"
        elif target_tokens >= 1_000_000:
            suffix = f"{target_tokens // 1_000_000}M"
        else:
            suffix = f"{target_tokens // 1000}K"
        
        if fmt == "auto":
            # Default to text for backward compat, unless user wants npy.
            # But wait, user query is specifically about npy.
            # I will let user specify via --output x.npy or --format npy
            fmt = "text"
            
        ext = "npy" if fmt == "npy" else "tokens"
        output_path = f"fineweb_{suffix.lower()}.{ext}"
    
    # Check if already exists
    if not args.force:
        existing = verify_existing(output_path)
        if existing is not None:
            print(f"File {output_path} already exists with {existing:,} tokens")
            if existing >= target_tokens:
                print("Sufficient tokens already available. Use --force to re-download.")
                return
            else:
                print(f"Need {target_tokens - existing:,} more tokens. Re-downloading...")
    
    # Download and tokenize
    success = download_and_tokenize(target_tokens, output_path, output_format=fmt)
    
    if success:
        print("\n" + "=" * 60)
        print(f"SUCCESS! Dataset ready at: {output_path}")
        print("=" * 60)
        print(f"\nTo train, run:")
        print(f"  python3.12 v28_transformer_decoupled_bottleneck_instrumented.py \\")
        print(f"      --data {output_path} \\")
        if fmt == "npy":
             print(f"      --data-format npy \\")
        print(f"      --out-dir runs/paper_experiment \\")
        print(f"      --tokenizer tiktoken \\")
        print(f"      --block 1024 \\")
        print(f"      --instrument medium")
    else:
        print("\nFailed to prepare dataset")
        return 1


if __name__ == "__main__":
    main()
