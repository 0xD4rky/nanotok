import pytest


def test_from_tiktoken_cl100k():
    tiktoken = pytest.importorskip("tiktoken")
    from nanotok import Tokenizer

    tok = Tokenizer.from_tiktoken("cl100k_base")
    enc = tiktoken.get_encoding("cl100k_base")

    texts = [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "def main():\n    print('hello')\n",
        "12345 + 67890 = 80235",
        "  spaces   and\ttabs\n",
    ]
    for text in texts:
        assert tok.encode(text) == enc.encode(text), f"Mismatch for {text!r}"


def test_from_tiktoken_roundtrip():
    pytest.importorskip("tiktoken")
    from nanotok import Tokenizer

    tok = Tokenizer.from_tiktoken("cl100k_base")
    text = "Hello, world! This is a test of the nanotok tokenizer."
    assert tok.decode(tok.encode(text)) == text


def test_from_tiktoken_o200k():
    tiktoken = pytest.importorskip("tiktoken")
    from nanotok import Tokenizer

    tok = Tokenizer.from_tiktoken("o200k_base")
    enc = tiktoken.get_encoding("o200k_base")
    assert tok.encode("hello world") == enc.encode("hello world")


def test_from_tiktoken_gpt2():
    tiktoken = pytest.importorskip("tiktoken")
    from nanotok import Tokenizer

    tok = Tokenizer.from_tiktoken("gpt2")
    enc = tiktoken.get_encoding("gpt2")
    assert tok.encode("hello world") == enc.encode("hello world")
