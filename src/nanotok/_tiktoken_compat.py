"""
Tiktoken parser to convert tiktoken encodings to HF tokenizer.json format
BPE Engine loads tokenizer.json directly, so we need to convert tiktoken encodings to that format
"""


from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.request import urlopen

_ENCODING_CONFIGS: dict[str, dict] = {
    "gpt2": {
        "format": "gpt2_bpe",
        "vocab_url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        "merges_url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "special_tokens": {"<|endoftext|>": 50256},
        "pt_mode": 3,
    },
    "r50k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "hash": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9f0571571",
        "special_tokens": {"<|endoftext|>": 50256},
        "pt_mode": 3,
    },
    "p50k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        "hash": "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        "special_tokens": {"<|endoftext|>": 50256},
        "pt_mode": 3,
    },
    "cl100k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "hash": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        "special_tokens": {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
        "pt_mode": 1,
    },
    "o200k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "hash": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0571571",
        "special_tokens": {
            "<|endoftext|>": 199999,
            "<|endofprompt|>": 200018,
        },
        "pt_mode": 2,
    },
}


def _cache_dir() -> Path:
    d = Path(os.environ.get("NANOTOK_CACHE", Path.home() / ".cache" / "nanotok"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_file(url: str) -> Path:
    filename = url.rsplit("/", 1)[-1]
    cache_path = _cache_dir() / filename

    if cache_path.exists():
        return cache_path

    with urlopen(url) as resp:
        cache_path.write_bytes(resp.read())
    return cache_path


def _load_tiktoken_bpe(path: Path) -> dict[bytes, int]:
    ranks: dict[bytes, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                ranks[base64.b64decode(parts[0])] = int(parts[1])
    return ranks


def _bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 byte-to-unicode mapping (standard across all OpenAI tokenizers)
    """
    bs: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _byte_encode(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token_bytes)


def _reconstruct_merges(
    mergeable_ranks: dict[bytes, int],
    byte_encoder: dict[int, str],
) -> list[str]:
    """
    reconstruct the BPE merge list by simulating BPE on each multi-byte token
    """
    merges: list[str] = []

    for token_bytes, _ in sorted(mergeable_ranks.items(), key=lambda x: x[1]):
        if len(token_bytes) == 1:
            continue

        # Repeatedly merge the lowest-rank adjacent pair until only 2 parts remain
        parts: list[bytes] = [bytes([b]) for b in token_bytes]
        while len(parts) > 2:
            min_idx, min_rank = -1, float("inf")
            for i in range(len(parts) - 1):
                pair_rank = mergeable_ranks.get(parts[i] + parts[i + 1])
                if pair_rank is not None and pair_rank < min_rank:
                    min_idx, min_rank = i, pair_rank
            if min_idx < 0:
                break
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]

        if len(parts) == 2:
            merges.append(f"{_byte_encode(parts[0], byte_encoder)} {_byte_encode(parts[1], byte_encoder)}")

    return merges


def _load_gpt2_bpe(vocab_path: Path, merges_path: Path) -> dict[bytes, int]:
    byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

    with open(vocab_path) as f:
        encoder: dict[str, int] = json.load(f)

    return {bytes(byte_decoder[c] for c in token_str): rank for token_str, rank in encoder.items()}


def tiktoken_to_hf_json(encoding_name: str) -> str:
    """
    convert a tiktoken encoding to HF tokenizer.json format string
    """

    cfg = _ENCODING_CONFIGS[encoding_name]
    special_tokens: dict[str, int] = cfg["special_tokens"]

    if cfg.get("format") == "gpt2_bpe":
        mergeable_ranks = _load_gpt2_bpe(
            _download_file(cfg["vocab_url"]),
            _download_file(cfg["merges_url"]),
        )
    else:
        mergeable_ranks = _load_tiktoken_bpe(_download_file(cfg["url"]))

    byte_encoder = _bytes_to_unicode()

    vocab = {_byte_encode(tok_bytes, byte_encoder): rank for tok_bytes, rank in mergeable_ranks.items()}
    vocab.update(special_tokens)

    merges = _reconstruct_merges(mergeable_ranks, byte_encoder)

    added_tokens = [
        {
            "id": tid, "content": tok, "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False, "special": True,
        }
        for tok, tid in special_tokens.items()
    ]

    return json.dumps({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "added_tokens": added_tokens,
    })


def get_tiktoken_pt_mode(encoding_name: str) -> int:
    cfg = _ENCODING_CONFIGS.get(encoding_name)
    if cfg is None:
        raise ValueError(f"Unknown tiktoken encoding: {encoding_name}")
    return cfg["pt_mode"]
