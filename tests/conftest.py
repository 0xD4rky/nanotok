import os
from pathlib import Path

import pytest


def _find_tokenizer_json() -> str | None:
    candidates = [
        os.environ.get("NANOTOK_TEST_TOKENIZER"),
        str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"),
    ]
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.is_file() and p.name == "tokenizer.json":
            return str(p)
        if p.is_dir():
            for tj in p.rglob("tokenizer.json"):
                return str(tj)
    return None


@pytest.fixture(scope="session")
def tokenizer_json_path() -> str:
    path = _find_tokenizer_json()
    if path is None:
        pytest.skip("No tokenizer.json found. Set NANOTOK_TEST_TOKENIZER env var.")
    return path


@pytest.fixture(scope="session")
def tokenizer(tokenizer_json_path):
    from nanotok import Tokenizer
    return Tokenizer.from_file(tokenizer_json_path)
