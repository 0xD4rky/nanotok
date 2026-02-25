# nanotok

Fast BPE tokenizer. C++ core, Python interface.

Works with tiktoken encodings and any HuggingFace tokenizer.

## Install

```
uv pip install nanotok
```

With HuggingFace Hub and chat template support:

```
uv pip install "nanotok[all]"
```

## Usage

```python
from nanotok import Tokenizer

tok = Tokenizer.from_tiktoken("cl100k_base")

ids = tok.encode("hello world")
text = tok.decode(ids)
```

### Load from HuggingFace

```python
tok = Tokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

### Load from file

```python
tok = Tokenizer.from_file("path/to/tokenizer.json")
```

### Batch

```python
ids = tok.encode_batch(["hello", "world"])
texts = tok.decode_batch(ids)
```

### HuggingFace-style callable

```python
out = tok("hello world", padding=True, return_tensors="pt")
out["input_ids"]
out["attention_mask"]
```

### Chat templates

Requires `jinja2`.

```python
messages = [{"role": "user", "content": "hi"}]
tok.apply_chat_template(messages, tokenize=False)
```

### Special tokens

```python
ids = tok.encode("text", add_special_tokens=True)
text = tok.decode(ids, skip_special_tokens=True)

tok.eos_token_id
tok.bos_token_id
tok.vocab_size
```

## Supported tiktoken encodings

`gpt2`, `r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`

## Benchmarks

Averaged across Wikipedia, Code, News, Math, and Multilingual datasets.

| | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|
| nanotok | 29-33 | 63-69 |
| tiktoken | 15-16 | 53-59 |
| HF Tokenizers | 0.9-4 | 3-9 |

## License

MIT
