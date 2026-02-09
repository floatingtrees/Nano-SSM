"""Prototype Tokenizer + File-chunking dataloader class.

Provides `TokenizerDataLoader` which centralizes tokenizer config and
file-based chunked tokenization for training loops.

Notes
- This is a prototype focusing on correctness and clarity rather than
  absolute performance. For large corpora consider streaming + token-id
  buffering to avoid double-tokenization of lines.
"""
from typing import Callable, Generator, List, Optional, Union, Dict, Any
import os
import random
from pathlib import Path

import torch
import math

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - informative import error
    AutoTokenizer = None


class TokenizerDataLoader:
    """Class that holds tokenizer config and yields tokenized chunks from text files.

    Attributes:
        tokenizer_type: Currently only 'hf' supported (HuggingFace AutoTokenizer).
        tokenizer_name: Pretrained tokenizer name (e.g. 'gpt2').
        max_length: Default max token length to use for truncation/padding.
        data_input: Path to local .txt file to stream from (or None).
    """

    def __init__(self,
                 tokenizer_type: str = "hf",
                 tokenizer_name: str = "gpt2",
                 max_length: int = 1024,
                 data_input: Optional[str] = None,
                 tokenizer_args: Optional[dict] = None,
                 vocab_size: Optional[int] = None):
        if tokenizer_type != "hf":
            raise NotImplementedError("Only 'hf' tokenizer_type supported in prototype")

        if AutoTokenizer is None:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        tokenizer_args = tokenizer_args or {}
        self.tokenizer_type = tokenizer_type
        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size
        self.max_length = int(max_length)
        self.data_input = data_input
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_args)

        # allow overriding vocab size (useful when model vocab differs from HF tokenizer)
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        else:
            # try to infer from tokenizer
            self.vocab_size = getattr(self.tokenizer, "vocab_size", None)
            if self.vocab_size is None:
                try:
                    self.vocab_size = len(self.tokenizer.get_vocab())
                except Exception:
                    self.vocab_size = None


        #not used for now
        # Caching / sharding attributes
        self.cache_enabled: bool = False
        self.shard_dir: Optional[str] = None
        self.shard_size: int = 0  # number of sequences per shard
        self.shard_files: List[str] = []
        self.current_shard_idx: int = 0
        self.cached_shard: Optional[List[torch.Tensor]] = None
        self.cached_pos: int = 0
        self.cache_index: Optional[List[int]] = None  # randomized order within shard
        self.shuffle_within_shard: bool = True

    def tokenize_text(self,
                      text: Union[str, List[str]],
                      return_tensors: Optional[str] = None,
                      padding: Union[bool, str] = False,
                      truncation: bool = True,
                      max_length: Optional[int] = None) -> Dict[str, Any]:
        """Tokenize raw text (string or list of strings
        Returns the raw output dict from HF tokenizer.
        """
        ml = max_length or self.max_length
        return self.tokenizer(text,
                              padding="longest" if padding else False,
                              truncation=truncation,
                              max_length=ml,
                              return_tensors=return_tensors)

    def tokenize_file_chunks(self,
                             file_path: Optional[str] = None,   
                             chunk_size_tokens: int = 512,
                             overlap_tokens: int = 0,
                             file_mode: str = "lines",
                             skip_blank: bool = True,
                             return_tensors: Optional[str] = None,
                             truncation: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Yield tokenized chunks from a local text file, with no padding (contiguous token sequences).

        Buffer/Chunking Logic:
        - Reads the file line by line (if file_mode='lines').
        - Each line is tokenized to count its tokens (no special tokens).
        - Lines are accumulated in a buffer until the total token count >= chunk_size_tokens.
        - When the buffer is full, all lines in the buffer are joined with '\n', tokenized as a single sequence (no padding), and yielded as a chunk.
        - If a single line is longer than chunk_size_tokens, it is truncated and yielded as its own chunk.
        - If overlap_tokens > 0, the last N tokens of the previous chunk are kept as the start of the next chunk (to provide overlap between chunks).
        - At the end, any remaining lines in the buffer are yielded as a final chunk.

        Parameters:
            file_path: override `self.data_input` for this call.
            chunk_size_tokens: maximum token length per yielded chunk.
            overlap_tokens: keep this many tokens from end of previous chunk as start of next chunk.
            file_mode: 'lines' treats file as multiple short examples.
            skip_blank: whether to ignore empty lines when file_mode='lines'.
            return_tensors: pass to tokenizer (e.g. 'pt' or 'np').
            truncation: whether to truncate to chunk_size_tokens (should be True for SSM training).

        Returns:
            Generator of dicts (tokenizer output), each with a single contiguous sequence of tokens, no padding.
        """
        path = file_path or self.data_input
        if path is None:
            raise ValueError("No file path provided and `data_input` is None")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        buffer_lines: List[str] = []
        buffer_token_count = 0

        with open(path, "r", encoding="utf-8") as fh:
            if file_mode == "lines":
                for raw in fh:
                    line = raw.rstrip("\n")
                    if skip_blank and not line.strip():
                        continue

                    # Estimate tokens for the line (no special tokens added)
                    try:
                        ids = self.tokenizer.encode(line, add_special_tokens=False)
                        line_tokens = len(ids)
                    except Exception:
                        # Fallback: treat by words (very rough)
                        line_tokens = max(1, len(line.split()))

                    # If single line longer than chunk_size, yield truncated encoding (no padding)
                    if line_tokens >= chunk_size_tokens:
                        enc = self.tokenizer(line,
                                             truncation=True,
                                             max_length=chunk_size_tokens,
                                             padding=False,  # No padding
                                             return_tensors=return_tensors)
                        yield enc
                        # optionally keep overlap from this truncated chunk
                        if overlap_tokens > 0:
                            try:
                                tail_ids = enc["input_ids"][0][-overlap_tokens:]
                                tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
                                buffer_lines = [tail_text]
                                buffer_token_count = len(tail_ids)
                            except Exception:
                                buffer_lines = []
                                buffer_token_count = 0
                        continue

                    # Normal accumulation
                    buffer_lines.append(line)
                    buffer_token_count += line_tokens

                    if buffer_token_count >= chunk_size_tokens:
                        chunk_text = "\n".join(buffer_lines)
                        enc = self.tokenizer(chunk_text,
                                             truncation=True,
                                             max_length=chunk_size_tokens,
                                             padding=False,  # No padding
                                             return_tensors=return_tensors)
                        yield enc

                        # handle overlap: keep last `overlap_tokens` worth of text
                        if overlap_tokens > 0:
                            try:
                                last_ids = enc["input_ids"][0][-overlap_tokens:]
                                last_text = self.tokenizer.decode(last_ids, skip_special_tokens=True)
                                buffer_lines = [last_text]
                                buffer_token_count = len(last_ids)
                            except Exception:
                                buffer_lines = []
                                buffer_token_count = 0
                        else:
                            buffer_lines = []
                            buffer_token_count = 0

            else:
                # file_mode != 'lines': read entire file as one document
                text = fh.read()
                # naive chunking by tokenized sequence (may double-tokenize)
                try:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    for i in range(0, len(ids), chunk_size_tokens - overlap_tokens):
                        chunk_ids = ids[i:i + chunk_size_tokens]
                        # decode back to text and re-tokenize to get HF dict
                        chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                        enc = self.tokenizer(chunk_text,
                                             truncation=True,
                                             max_length=chunk_size_tokens,
                                             padding=False,  # No padding
                                             return_tensors=return_tensors)
                        yield enc
                except Exception:
                    # fallback: split by roughly chunk_size_tokens words
                    words = text.split()
                    approx = chunk_size_tokens
                    for i in range(0, len(words), approx - overlap_tokens):
                        chunk_text = " ".join(words[i:i + approx])
                        enc = self.tokenizer(chunk_text,
                                             truncation=True,
                                             max_length=chunk_size_tokens,
                                             padding=False,  # No padding
                                             return_tensors=return_tensors)
                        yield enc

        # yield leftover buffer if any
        if buffer_lines:
            chunk_text = "\n".join(buffer_lines)
            enc = self.tokenizer(chunk_text,
                                 truncation=True,
                                 max_length=chunk_size_tokens,
                                 padding=False,  # No padding
                                 return_tensors=return_tensors)
            yield enc

    def tokenize_batches(self,
                         file_path: Optional[str] = None,
                         seq_len: int = 512,
                         global_batch_size: int = 32,
                         overlap_tokens: int = 0,
                         file_mode: str = "lines",
                         skip_blank: bool = True,
                         return_tensors: Optional[str] = None,
                         drop_last: bool = True,
                         pin_memory: bool = False) -> Generator[torch.Tensor, None, None]:
        """
        Stream tokenized sequences and yield full batches of shape (global_batch_size, seq_len).

        This wraps `tokenize_file_chunks` (with `chunk_size_tokens=seq_len`) and accumulates
        individual sequences until `global_batch_size` sequences are available, at which point
        a single stacked tensor is yielded. Sequences that are not exactly `seq_len` are skipped
        (matching the behavior used in the training loop). If `drop_last` is False the final
        partial batch will be padded by repeating the last sequence to reach `global_batch_size`.
        """
        path = file_path or self.data_input
        if path is None:
            raise ValueError("No file path provided and `data_input` is None")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        buffer: List[torch.Tensor] = []

        gen = self.tokenize_file_chunks(file_path=path,
                                        chunk_size_tokens=seq_len,
                                        overlap_tokens=overlap_tokens,
                                        file_mode=file_mode,
                                        skip_blank=skip_blank,
                                        return_tensors=return_tensors)

        for enc in gen:
            ids = enc.get("input_ids")
            if ids is None:
                continue
            if isinstance(ids, list):
                seq = torch.tensor(ids[0], dtype=torch.long)
            else:
                seq = ids[0].detach().cpu().to(torch.long)

            # normalize token ids to be within [0, vocab_size-1] to avoid out-of-bounds
            if getattr(self, "vocab_size", None) is not None:
                seq = torch.clamp(seq, min=0, max=self.vocab_size - 1)

            # only accept sequences of exact desired length
            if seq.size(0) != seq_len:
                continue

            buffer.append(seq)

            if len(buffer) >= global_batch_size:
                batch = torch.stack(buffer[:global_batch_size], dim=0)
                if pin_memory:
                    batch = batch.pin_memory()
                yield batch
                buffer = buffer[global_batch_size:]

        # flush remaining buffer
        if buffer:
            if drop_last:
                return
            # pad by repeating last element
            while len(buffer) < global_batch_size:
                buffer.append(buffer[-1].clone())
            batch = torch.stack(buffer[:global_batch_size], dim=0)
            if pin_memory:
                batch = batch.pin_memory()
            yield batch

    # ---------------- Cache / Shard helpers ----------------
    def create_cache(self,
                     shard_dir: Union[str, Path],
                     shard_size: int = 256,
                     overwrite: bool = False,
                     chunk_size_tokens: int = 512,
                     overlap_tokens: int = 0) -> List[str]:
        """Tokenize the input file and write shards to `shard_dir`.

        Each shard contains up to `shard_size` token sequences saved via `torch.save`.
        Returns list of shard file paths.
        """
        if self.data_input is None:
            raise ValueError("data_input must be set on the TokenizerDataLoader to create cache")

        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        if overwrite:
            # remove existing shards
            for p in shard_dir.glob("shard_*.pt"):
                p.unlink()

        seq_buffer: List[torch.LongTensor] = []
        shard_files: List[str] = []
        shard_idx = 0

        for enc in self.tokenize_file_chunks(file_path or self.data_input,
                                            chunk_size_tokens=chunk_size_tokens,
                                            overlap_tokens=overlap_tokens):
            # extract input_ids as list or tensor
            ids = enc.get("input_ids")
            if ids is None:
                continue
            # tokenizer returns list-of-lists when return_tensors=None
            if isinstance(ids, list):
                seq = torch.tensor(ids[0], dtype=torch.long)
            else:
                # if tensors provided (e.g., return_tensors='pt')
                seq = ids[0].detach().cpu().to(torch.long)

            seq_buffer.append(seq)

            if len(seq_buffer) >= shard_size:
                shard_path = shard_dir / f"shard_{shard_idx:06d}.pt"
                torch.save(seq_buffer, shard_path)
                shard_files.append(str(shard_path))
                shard_idx += 1
                seq_buffer = []

        # save tail
        if seq_buffer:
            shard_path = shard_dir / f"shard_{shard_idx:06d}.pt"
            torch.save(seq_buffer, shard_path)
            shard_files.append(str(shard_path))

        # store cache metadata
        self.cache_enabled = True
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.shard_files = shard_files
        self.current_shard_idx = 0
        self.cached_shard = None
        self.cached_pos = 0

        return shard_files

    #idk why gpt generated two load_shard functions but since we don't need it just yet, I'll just keep one of them to avoid potential warnings/errors
    # def load_shard(self, idx: int, shuffle_within_shard: Optional[bool] = None):
    #     """Load shard file at index `idx` into memory and prepare index order."""
    #     if not self.shard_files:
    #         self._discover_shards()
    #     if not self.shard_files:
    #         raise RuntimeError("No shard files found; run create_cache first")
    #     idx = idx % len(self.shard_files)
    #     path = self.shard_files[idx]
    #     sequences = torch.load(path)
    #     # ensure list of tensors
    #     sequences = [s if torch.is_tensor(s) else torch.tensor(s, dtype=torch.long) for s in sequences]
    #     self.cached_shard = sequences
    #     self.cached_pos = 0
    #     self.current_shard_idx = idx
    #     if shuffle_within_shard is None:
    #         shuffle_within_shard = self.shuffle_within_shard
    #     if shuffle_within_shard:
    #         perm = torch.randperm(len(self.cached_shard)).tolist()
    #         self.cache_index = perm
    #     else:
    #         self.cache_index = list(range(len(self.cached_shard)))

    def load_shard(self, idx: int = 0, shuffle: bool = True) -> None:
        """Load shard `idx` into memory (cached_shard) as a list of LongTensors.

        If `shuffle` is True, the order of sequences within the shard is randomized.
        """
        if not self.cache_enabled or not self.shard_files:
            raise RuntimeError("Cache not created; call create_cache(...) first")

        if idx < 0 or idx >= len(self.shard_files):
            raise IndexError("shard index out of range")

        data = torch.load(self.shard_files[idx])
        # ensure tensors on CPU
        seqs = [s.detach().cpu().to(torch.long) for s in data]
        if shuffle:
            random.shuffle(seqs)

        self.cached_shard = seqs
        self.current_shard_idx = idx
        self.cached_pos = 0

    def iter_batches(self,
                     batch_size: int,
                     seq_len: int,
                     shuffle_shard: bool = True,
                     drop_last: bool = True) -> Generator[torch.LongTensor, None, None]:
        """Yield batches of shape (batch_size, seq_len) by loading shards on demand.

        For each loaded shard we split sequences into fixed-length windows of length
        `seq_len` using step = seq_len - overlap (must be > 0). Short sequences are skipped.
        Windows are shuffled (per-shard) when `shuffle_shard=True`.
        """
        if not self.cache_enabled or not self.shard_files:
            raise RuntimeError("Cache not created; call create_cache(...) first")

        num_shards = len(self.shard_files)
        shard_idx = 0

        while shard_idx < num_shards:
            self.load_shard(shard_idx, shuffle=shuffle_shard)
            if not self.cached_shard:
                shard_idx += 1
                continue

            # create windows from sequences in shard
            windows: List[torch.LongTensor] = []
            step = seq_len - getattr(self, "overlap_tokens", 0)
            if step <= 0:
                raise ValueError("seq_len must be larger than overlap_tokens")

            for seq in self.cached_shard:
                L = seq.size(0)
                if L < seq_len:
                    continue
                for i in range(0, L - seq_len + 1, step):
                    windows.append(seq[i:i + seq_len])

            if not windows:
                shard_idx += 1
                continue

            if shuffle_shard:
                random.shuffle(windows)

            # yield batches
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i:i + batch_size]
                if len(batch_windows) < batch_size and drop_last:
                    break
                yield torch.stack(batch_windows)

            shard_idx += 1

    # -------------------- Caching / sharding helpers --------------------
    def _discover_shards(self):
        """Populate `self.shard_files` from `self.shard_dir` if present."""
        if not self.shard_dir:
            self.shard_files = []
            return
        files = sorted(f for f in os.listdir(self.shard_dir) if f.endswith('.pt'))
        self.shard_files = [os.path.join(self.shard_dir, f) for f in files]

    def create_cache(self,
                     shard_dir: str,
                     shard_size: int,
                     seq_len: int,
                     overlap_tokens: int = 0,
                     chunk_size_tokens: int = 4096,
                     overwrite: bool = False):
        """Tokenize dataset and write shards to `shard_dir`.

        Produces fixed-length sequences of `seq_len` tokens (sliding windows)
        with `overlap_tokens` overlap between consecutive windows. Each shard
        contains `shard_size` sequences and is saved as a list of LongTensors
        using `torch.save`.
        """
        os.makedirs(shard_dir, exist_ok=True)
        if overwrite:
            # remove existing .pt files
            for f in os.listdir(shard_dir):
                if f.endswith('.pt'):
                    os.remove(os.path.join(shard_dir, f))

        seqs: List[torch.Tensor] = []
        shard_idx = 0
        buffer_ids: List[int] = []

        # Use tokenize_file_chunks to get large chunks of token ids, then window them
        for enc in self.tokenize_file_chunks(chunk_size_tokens=chunk_size_tokens,
                                             overlap_tokens=0,
                                             file_mode='document',
                                             return_tensors=None):
            # enc['input_ids'] may be a list-of-lists or tensor; normalize to list
            ids = enc.get('input_ids')
            if ids is None:
                continue
            if isinstance(ids, list):
                ids = ids[0]
            elif torch.is_tensor(ids):
                ids = ids[0].tolist()

            # append to buffer and create sliding windows
            buffer_ids.extend(ids)
            step = max(1, seq_len - overlap_tokens)
            start = 0
            # produce as many full windows as possible
            while start + seq_len <= len(buffer_ids):
                window = buffer_ids[start:start + seq_len]
                seqs.append(torch.tensor(window, dtype=torch.long))
                start += step

                # flush shard if full
                if len(seqs) >= shard_size:
                    shard_path = os.path.join(shard_dir, f"shard_{shard_idx:06d}.pt")
                    torch.save(seqs, shard_path)
                    shard_idx += 1
                    seqs = []

            # keep the leftover tail for next iteration
            buffer_ids = buffer_ids[start:]

        # flush remaining windows from buffer (if any)
        # create windows from remaining buffer if long enough
        step = max(1, seq_len - overlap_tokens)
        start = 0
        while start + seq_len <= len(buffer_ids):
            window = buffer_ids[start:start + seq_len]
            seqs.append(torch.tensor(window, dtype=torch.long))
            start += step

        if seqs:
            shard_path = os.path.join(shard_dir, f"shard_{shard_idx:06d}.pt")
            torch.save(seqs, shard_path)
            shard_idx += 1

        # finalize
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self._discover_shards()
        self.cache_enabled = True
        self.current_shard_idx = 0

    
    def get_next_sequence(self) -> Optional[torch.Tensor]:
        """Return next sequence tensor from cached shard, auto-loading next shard when needed.

        Returns None when no shards are available.
        """
        if not self.cache_enabled or not self.shard_files:
            return None
        if self.cached_shard is None or self.cached_pos >= len(self.cache_index or []):
            # load current shard (or next if exhausted)
            if self.cached_shard is None:
                self.load_shard(self.current_shard_idx)
            else:
                # move to next shard
                next_idx = (self.current_shard_idx + 1) % len(self.shard_files)
                self.load_shard(next_idx)

        if not self.cached_shard:
            return None

        idx = self.cache_index[self.cached_pos]
        seq = self.cached_shard[idx]
        self.cached_pos += 1
        return seq

    def iter_batches(self, batch_size: int, drop_last: bool = True) -> Generator[torch.Tensor, None, None]:
        """Yield batches of shape (batch_size, seq_len) from cached shards.

        This function relies on the cached shards containing fixed-length
        sequences (as created by `create_cache`). It will cycle through
        shards indefinitely; caller should break when desired.
        """
        if not self.cache_enabled:
            raise RuntimeError("Cache not enabled; call create_cache first")
        while True:
            batch: List[torch.Tensor] = []
            for _ in range(batch_size):
                seq = self.get_next_sequence()
                if seq is None:
                    break
                batch.append(seq)
            if not batch:
                break
            if len(batch) < batch_size:
                if drop_last:
                    break
                else:
                    # pad by repeating last element (rare)
                    while len(batch) < batch_size:
                        batch.append(batch[-1].clone())
            yield torch.stack(batch, dim=0)


if __name__ == "__main__":
    # Minimal usage example (update path to a small text file to try):
    sample = os.path.join(os.path.dirname(__file__), "../sample.txt")
    if not os.path.isfile(sample):
        sample = None

    dl = TokenizerDataLoader(tokenizer_name="gpt2", max_length=256, data_input=sample)

    # iterate a few chunks (no tensor return to keep small memory footprint)
    if dl.data_input is not None:
        for i, chunk in enumerate(dl.tokenize_file_chunks(chunk_size_tokens=128, overlap_tokens=8)):
            print(f"chunk {i}: input_ids_len=", len(chunk.get("input_ids", [])[0]))
            if i >= 3:
                break

