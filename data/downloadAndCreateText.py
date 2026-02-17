"""
Stream a HuggingFace dataset, let the user select the text field (and optional token-count
field), filter by token count if requested, and write one cleaned text entry per line to
an output text file.

Usage examples:
python data/downloadAndCreateText.py --dataset-name wikitext --split train --out data/wikitext_lines.txt
python data/downloadAndCreateText.py --dataset-name HuggingFaceFW/fineweb-edu --subset-name CC-MAIN-2024-10 \
  --streaming --out data/fineweb_lines.txt

If `--text-field` is omitted the script will sample a few examples, print available
fields and prompt the user to choose which column contains the text.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

from datasets import load_dataset


def preview_val(v: Any, maxlen: int = 200) -> str:
    try:
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    except Exception:
        s = str(v)
    s = s.replace("\n", " ")
    if len(s) > maxlen:
        return s[: maxlen - 3] + "..."
    return s

#get an interator to stream dataset from Huggingface
def get_fresh_iter(dataset_name: str, subset_name: Optional[str], split: str, streaming: bool) -> Iterable:
    name = subset_name if subset_name else None
    ds = load_dataset(dataset_name, name, split=split, streaming=streaming)
    return ds

#Samples first n examples from the dataset iterator
def sample_examples(ds_iter: Iterable, n: int) -> List[Any]:
    return list(itertools.islice(ds_iter, n))

#Analyzes dataset samples in order to extract all fields
def discover_fields(samples: List[Any]) -> List[str]:
    fields = set()
    for ex in samples:
        if isinstance(ex, str):
            fields.add("__self__")
        elif isinstance(ex, dict):
            for k in ex.keys():
                fields.add(k)
    return sorted(fields)

#prompt user to enter in extra info (column name of text) if not provided as command line argument
def prompt_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    print(prompt)
    for i, c in enumerate(choices):
        print(f"  {i+1}) {c}")
    if default is not None:
        prompt_txt = f"Enter choice [1-{len(choices)}] (default={default}): "
    else:
        prompt_txt = f"Enter choice [1-{len(choices)}]: "
    while True:
        ans = input(prompt_txt).strip()
        if ans == "" and default is not None:
            if default in choices:
                return default
            try:
                idx = int(default) - 1
                return choices[idx]
            except Exception:
                return default
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception:
            if ans in choices:
                return ans
        print("Invalid choice, try again.")

#writes data to a textfile
def write_lines(
    ds_iter: Iterable,
    text_field: str,
    token_field: Optional[str],
    min_tokens: Optional[int],
    max_tokens: Optional[int],
    outpath: Path,
    progress_every: int = 1000,
):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(outpath, "w", encoding="utf-8") as fh:
        for i, ex in enumerate(ds_iter):
            if text_field == "__self__":
                text = ex if isinstance(ex, str) else None
            else:
                text = ex.get(text_field) if isinstance(ex, dict) else None
            if not text or not isinstance(text, str):
                skipped += 1
                continue

            # token filtering if requested
            if token_field and (min_tokens is not None or max_tokens is not None):
                count = None
                if isinstance(ex, dict) and token_field in ex:
                    try:
                        count = int(ex[token_field])
                    except Exception:
                        try:
                            count = int(float(ex[token_field]))
                        except Exception:
                            count = None
                if count is None:
                    skipped += 1
                    continue
                if min_tokens is not None and count < min_tokens:
                    skipped += 1
                    continue
                if max_tokens is not None and count > max_tokens:
                    skipped += 1
                    continue

            # one-line entry per dataset record
            line = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
            fh.write(line + "\n")
            written += 1
            if written % progress_every == 0:
                print(f"Written {written} lines (skipped {skipped})...")

    print(f"Done. Written {written} lines; skipped {skipped} records.")


def main():
    p = argparse.ArgumentParser(description="Stream a HuggingFace dataset and write raw text lines")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--subset-name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--out", required=True, help="Output text file path (one record per line)")
    p.add_argument("--sample-size", type=int, default=20, help="Number of examples to sample for field discovery")
    p.add_argument("--text-field", default=None, help="Column name to use for text (skip prompt)")
    p.add_argument("--token-field", default=None, help="Optional column name for approx token count per example")
    p.add_argument("--min-tokens", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    print("Loading a small sample to discover fields...")
    ds_iter = get_fresh_iter(args.dataset_name, args.subset_name, args.split, args.streaming)
    samples = sample_examples(ds_iter, args.sample_size)
    if not samples:
        print("No samples found from dataset - exiting.")
        return

    fields = discover_fields(samples)
    print("Discovered fields:")
    for f in fields:
        print(f"- {f}")
        # show a single preview value
        for ex in samples:
            if f == "__self__" and isinstance(ex, str):
                print(f"    example: {preview_val(ex)}")
                break
            if isinstance(ex, dict) and f in ex:
                print(f"    example: {preview_val(ex[f])}")
                break

    text_field = args.text_field
    if not text_field:
        text_field = prompt_choice("Choose the field that contains the text (or '__self__' if the item is a string):", fields)

    token_field = args.token_field
    if token_field is None:
        # allow user to pick a token count field or none
        choices = ["(none)"] + [f for f in fields if f != "__self__"]
        choice = prompt_choice("Choose a field that contains approximate token counts, or choose (none):", choices, default="(none)")
        if choice != "(none)":
            token_field = choice
        else:
            token_field = None

    print(f"Text field: {text_field}; token field: {token_field}")
    if (args.min_tokens is not None or args.max_tokens is not None) and token_field is None:
        print("Warning: min/max token filtering requested but no token field chosen; filters will be ignored.")

    print("Starting full pass; this may re-create the iterator to avoid consuming sampled items.")
    ds_iter_full = get_fresh_iter(args.dataset_name, args.subset_name, args.split, args.streaming)
    write_lines(ds_iter_full, text_field, token_field, args.min_tokens, args.max_tokens, Path(args.out))


if __name__ == "__main__":
    main()
