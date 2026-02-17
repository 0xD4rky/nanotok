from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def download_tokenizer(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for from_pretrained(). "
            "Install it with: pip install nanotok[hub]"
        ) from None

    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token

    tokenizer_path = Path(hf_hub_download(filename="tokenizer.json", **kwargs))

    config: dict[str, Any] = {}
    try:
        config_path = hf_hub_download(filename="tokenizer_config.json", **kwargs)
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        pass

    return tokenizer_path, config
