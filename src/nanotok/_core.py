from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanotok._nanotok_cpp import BPEEngine, PT_METASPACE, PT_SPLIT_SPC


class Tokenizer:

    def __init__(
        self,
        engine: BPEEngine,
        *,
        config: dict[str, Any] | None = None,
        prefix_token_ids: list[int] | None = None,
        suffix_token_ids: list[int] | None = None,
    ):
        self._engine = engine
        self._config = config or {}

        self._special_tokens: dict[str, int] = {}
        self._special_ids: set[int] = set()
        for tok in self._engine.get_added_tokens():
            if tok.special:
                self._special_tokens[tok.content] = tok.id
                self._special_ids.add(tok.id)

        self._eos_token = self._resolve_token("eos_token")
        self._bos_token = self._resolve_token("bos_token")
        self._pad_token = self._resolve_token("pad_token")
        self._unk_token = self._resolve_token("unk_token")
        self._chat_template = self._config.get("chat_template")
        self._padding_side = self._config.get("padding_side", "right")

        self._prefix_token_ids = prefix_token_ids or []
        self._suffix_token_ids = suffix_token_ids or []

        pt_mode = self._engine.get_pretokenizer_mode()
        self._strip_decode_prefix = (pt_mode == PT_METASPACE and
                                     self._engine.get_metaspace_add_prefix())

    def _resolve_token(self, field: str) -> str | None:
        val = self._config.get(field)
        if val is None:
            return None
        if isinstance(val, dict):
            return val.get("content")
        return str(val)

    @staticmethod
    def _parse_post_processor(tokenizer_json_path: Path) -> tuple[list[int], list[int]]:
        with open(tokenizer_json_path) as f:
            data = json.load(f)

        pp = data.get("post_processor")
        if not pp:
            return [], []

        if pp.get("type") == "Sequence":
            template = None
            for proc in pp.get("processors", []):
                if proc.get("type") == "TemplateProcessing":
                    template = proc
                    break
            if not template:
                return [], []
            pp = template

        if pp.get("type") != "TemplateProcessing":
            return [], []

        token_to_id = {t["content"]: t["id"] for t in data.get("added_tokens", [])}

        prefix, suffix = [], []
        seen_sequence = False
        for item in pp.get("single", []):
            if "Sequence" in item:
                seen_sequence = True
            elif "SpecialToken" in item:
                tid = token_to_id.get(item["SpecialToken"]["id"])
                if tid is not None:
                    (suffix if seen_sequence else prefix).append(tid)

        return prefix, suffix

    @classmethod
    def from_file(cls, path: str | Path) -> Tokenizer:
        path = Path(path)
        engine = BPEEngine(str(path))
        config = cls._load_config(path.parent)
        prefix, suffix = cls._parse_post_processor(path)
        return cls(engine, config=config, prefix_token_ids=prefix, suffix_token_ids=suffix)

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> Tokenizer:
        from nanotok._hub import download_tokenizer

        tokenizer_path, config = download_tokenizer(repo_id, **kwargs)
        engine = BPEEngine(str(tokenizer_path))
        prefix, suffix = cls._parse_post_processor(tokenizer_path)
        return cls(engine, config=config, prefix_token_ids=prefix, suffix_token_ids=suffix)

    @classmethod
    def from_tiktoken(cls, encoding_name: str) -> Tokenizer:
        from nanotok._tiktoken_compat import tiktoken_to_hf_json, get_tiktoken_pt_mode

        json_str = tiktoken_to_hf_json(encoding_name)
        engine = BPEEngine.from_json_string(json_str)
        engine.set_pretokenizer_mode(get_tiktoken_pt_mode(encoding_name))
        return cls(engine)

    @staticmethod
    def _load_config(directory: Path) -> dict[str, Any]:
        config_path = directory / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def encode(
        self,
        text: str,
        *,
        allowed_special: set[str] | None = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        if allowed_special is None:
            allowed_special = set(self._special_tokens) if add_special_tokens else set()
        ids = self._engine.encode(text, allowed_special)
        if add_special_tokens:
            ids = self._prefix_token_ids + ids + self._suffix_token_ids
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]
        text = self._engine.decode(ids)
        if self._strip_decode_prefix and text.startswith(" "):
            text = text[1:]
        return text

    def encode_batch(
        self,
        texts: list[str],
        *,
        allowed_special: set[str] | None = None,
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        if allowed_special is None:
            allowed_special = set(self._special_tokens) if add_special_tokens else set()
        batch = self._engine.batch_encode(texts, allowed_special)
        if add_special_tokens and (self._prefix_token_ids or self._suffix_token_ids):
            batch = [self._prefix_token_ids + ids + self._suffix_token_ids for ids in batch]
        return batch

    def clear_cache(self) -> None:
        self._engine.clear_cache()

    def set_cache_enabled(self, enabled: bool) -> None:
        self._engine.set_cache_enabled(enabled)

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        if skip_special_tokens:
            batch_ids = [[i for i in ids if i not in self._special_ids] for ids in batch_ids]
        results = self._engine.batch_decode(batch_ids)
        if self._strip_decode_prefix:
            results = [t[1:] if t.startswith(" ") else t for t in results]
        return results

    def token_to_id(self, token: str) -> int | None:
        result = self._engine.token_to_id(token)
        return None if result == -1 else result

    def id_to_token(self, id: int) -> str | None:
        result = self._engine.id_to_token(id)
        return None if result == "" else result

    @property
    def vocab_size(self) -> int:
        return self._engine.vocab_size()

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self._special_tokens)

    @property
    def eos_token(self) -> str | None:
        return self._eos_token

    @property
    def eos_token_id(self) -> int | None:
        return None if self._eos_token is None else self.token_to_id(self._eos_token)

    @property
    def bos_token(self) -> str | None:
        return self._bos_token

    @property
    def bos_token_id(self) -> int | None:
        return None if self._bos_token is None else self.token_to_id(self._bos_token)

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._pad_token = value

    @property
    def pad_token_id(self) -> int | None:
        return None if self._pad_token is None else self.token_to_id(self._pad_token)

    @property
    def unk_token(self) -> str | None:
        return self._unk_token

    @property
    def unk_token_id(self) -> int | None:
        return None if self._unk_token is None else self.token_to_id(self._unk_token)

    @property
    def padding_side(self) -> str:
        return self._padding_side

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        if value not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self._padding_side = value

    def __call__(
        self,
        text: str | list[str],
        *,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        all_ids = self.encode_batch(texts, add_special_tokens=add_special_tokens)

        if truncation and max_length is not None:
            all_ids = [ids[:max_length] for ids in all_ids]

        all_masks = [[1] * len(ids) for ids in all_ids]

        actual_max = max(len(ids) for ids in all_ids) if all_ids else 0
        if padding == "max_length" and max_length:
            target = max_length
        elif padding:
            target = actual_max
        else:
            target = 0

        if target > 0:
            pad_id = self.pad_token_id
            if pad_id is None:
                raise ValueError(
                    "Padding requested but the tokenizer has no pad token. "
                    "Set one with: tokenizer.pad_token = tokenizer.eos_token"
                )
            if self._padding_side == "left":
                all_ids = [[pad_id] * (target - len(ids)) + ids for ids in all_ids]
                all_masks = [[0] * (target - len(mask)) + mask for mask in all_masks]
            else:
                all_ids = [ids + [pad_id] * (target - len(ids)) for ids in all_ids]
                all_masks = [mask + [0] * (target - len(mask)) for mask in all_masks]

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_masks, dtype=torch.long),
            }

        if return_tensors == "np":
            import numpy as np
            return {
                "input_ids": np.array(all_ids, dtype=np.int64),
                "attention_mask": np.array(all_masks, dtype=np.int64),
            }

        return {
            "input_ids": all_ids[0] if is_single else all_ids,
            "attention_mask": all_masks[0] if is_single else all_masks,
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        from nanotok._chat_template import render_chat_template

        rendered = render_chat_template(
            template=self._chat_template,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self._bos_token or "",
            eos_token=self._eos_token or "",
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=True)
        return rendered

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"

    def __len__(self) -> int:
        return self.vocab_size
