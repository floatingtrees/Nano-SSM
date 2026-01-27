"""
Stream, tokenize, and shard the FineWeb subset for SSM pretraining.

Saves fixed-length sequences of token ids as numpy .npy shards
and writes a metadata JSON describing the shards. Designed to run
offline on HPC: streaming dataset read, batched tokenization, and
sharded output to avoid large single files.

Example:
python data/process_fineweb.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --subset-name CC-MAIN-2024-10 \
  --split train \
  --streaming \
  --tokenizer gpt2 \
  --seq-len 16384 \
  --tokens-per-shard 5000000 \
  --outdir data/fineweb_tokens
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset

#finds the text field from a row in the dataset
#in the case of fine-web, the text column is "text"
def find_text_field(example: dict) -> str | None:
    if example is None:
        return None
    if isinstance(example, str):
        return example
    # common text-like keys
    for k in ("text", "content", "body", "html", "raw", "document", "page_text"):
        if k in example and example[k]:
            return example[k]
    # fallback: try to find largest string field
    best = None
    best_len = 0
    for v in example.values():
        if isinstance(v, str) and len(v) > best_len:
            best = v
            best_len = len(v)
    return best


def clean_html_tags(text: str) -> str:
    # simple HTML tag stripper to reduce noise; optional
    if not text:
        return text
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def save_shard(arrs: list[np.ndarray], outdir: Path, shard_id: int) -> str:
    # arrs: list of 1D uint32 arrays all of equal length (seq_len)
    if not arrs:
        raise ValueError("No sequences to save")
    # verify all arrays have same length
    lengths = [int(a.shape[0]) for a in arrs]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched sequence lengths in shard: {set(lengths)}")
    # verify all arrays have same dtype
    dtypes = [a.dtype for a in arrs]
    if len(set(dtypes)) != 1:
        raise TypeError(f"Mismatched dtypes in shard: {set(dtypes)}")

    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"tokens_shard_{shard_id:05d}.npy"
    if fname.exists():
        raise FileExistsError(f"Shard file already exists: {fname}")
    stack = np.stack(arrs, axis=0)
    np.save(fname, stack, allow_pickle=False)
    return str(fname)


def process_stream(
    dataset_iter: Iterable,
    tokenizer,
    seq_len: int,
    tokens_per_shard: int,
    outdir: Path,
    batch_size: int = 32,
    clean_html: bool = False,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
):
    tokens_buffer: list[int] = []
    seqs: list[np.ndarray] = []
    shards: list[str] = []
    shard_id = 0
    seqs_per_shard = max(1, int(tokens_per_shard // seq_len))

    batch_texts: list[str] = []

    def flush_batch():
        nonlocal batch_texts, tokens_buffer, seqs, shard_id
        if not batch_texts:
            return
        # try tokenizing the whole batch; on failure, tokenize per-example and log bad ones
        try:
            enc = tokenizer(batch_texts, add_special_tokens=False)
            ids_lists = enc["input_ids"]
        except Exception as e:
            # fallback: try per-example to isolate bad texts
            ids_lists = []
            for txt in batch_texts:
                try:
                    enc_single = tokenizer(txt, add_special_tokens=False)
                    ids_lists.append(enc_single["input_ids"])
                except Exception as e2:
                    # log bad example and skip
                    out_log = outdir / "bad_examples.log"
                    outdir.mkdir(parents=True, exist_ok=True)
                    with open(out_log, "a", encoding="utf-8") as lf:
                        preview = (txt[:500] + "...") if isinstance(txt, str) and len(txt) > 500 else str(txt)
                        lf.write(f"TOKENIZE_ERROR: {repr(preview)}\nException: {repr(e2)}\n\n")
                    continue
        for ids in ids_lists:
            tokens_buffer.extend(ids)
            while len(tokens_buffer) >= seq_len:
                seq = np.frombuffer(np.array(tokens_buffer[:seq_len], dtype=np.uint32), dtype=np.uint32).copy()
                seqs.append(seq)
                del tokens_buffer[:seq_len]
                if len(seqs) >= seqs_per_shard:
                    outpath = save_shard(seqs, outdir, shard_id)
                    shards.append(outpath)
                    shard_id_local = len(shards) - 1
                    print(f"Wrote shard {shard_id_local} -> {outpath} (seqs={len(seqs)})")
                    seqs = []
                    shard_id += 1
        batch_texts = []

    def passes_token_count_filter(example, min_tokens, max_tokens):
        count = example.get("token_count", None)
        if count is None:
            return False
        if not isinstance(count, int):
            try:
                count = int(count)
            except Exception:
                return False
        if min_tokens is not None and count < min_tokens:
            return False
        if max_tokens is not None and count > max_tokens:
            return False
        return True

    for example in dataset_iter:
        if not passes_token_count_filter(example, min_tokens, max_tokens):
            continue
        text = find_text_field(example)
        if not text:
            continue
        if not isinstance(text, str):
            # log non-string text and skip
            out_log = outdir / "bad_examples.log"
            outdir.mkdir(parents=True, exist_ok=True)
            with open(out_log, "a", encoding="utf-8") as lf:
                lf.write(f"NONSTRING_TEXT: example preview: {repr(str(text)[:500])}\n\n")
            continue
        if clean_html:
            text = clean_html_tags(text)
        batch_texts.append(text)
        if len(batch_texts) >= batch_size:
            flush_batch()

    # flush remaining
    flush_batch()

    # save any remaining sequences
    if seqs:
        outpath = save_shard(seqs, outdir, shard_id)
        shards.append(outpath)
        print(f"Wrote final shard {len(shards)-1} -> {outpath} (seqs={len(seqs)})")

    return shards, tokens_buffer


def main():
    p = argparse.ArgumentParser(description="Tokenize and shard FineWeb subset for SSM pretraining")
    p.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset-name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended for large datasets)")
    p.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name or path")
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--tokens-per-shard", type=int, default=5_000_000)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clean-html", action="store_true")
    p.add_argument("--min-tokens", type=int, default=None, help="Minimum token count for filtering")
    p.add_argument("--max-tokens", type=int, default=None, help="Maximum token count for filtering")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", args.tokenizer)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    except Exception:
        raise

    print("Loading dataset (streaming=%s)..." % args.streaming)
    ds_kwargs = {}
    if args.subset_name:
        name = args.subset_name
    else:
        name = None
    ds = load_dataset(args.dataset_name, name, split=args.split, streaming=args.streaming)

    # Pass min/max tokens to process_stream
    global min_tokens, max_tokens
    min_tokens = args.min_tokens
    max_tokens = args.max_tokens

    shards, leftover_tokens = process_stream(
        ds,
        tokenizer,
        seq_len=args.seq_len,
        tokens_per_shard=args.tokens_per_shard,
        outdir=outdir,
        batch_size=args.batch_size,
        clean_html=args.clean_html,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    meta = {
        "shards": shards,
        "seq_len": args.seq_len,
        "tokens_per_shard": args.tokens_per_shard,
        "tokenizer": args.tokenizer,
        "num_shards": len(shards),
        "leftover_tokens": len(leftover_tokens),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("Done. metadata written to", outdir / "metadata.json")


if __name__ == "__main__":
    main()
