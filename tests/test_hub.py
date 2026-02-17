import pytest

pytestmark = pytest.mark.network


def test_from_pretrained_qwen():
    pytest.importorskip("huggingface_hub")
    from nanotok import Tokenizer

    tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    assert tok.vocab_size > 0

    ids = tok.encode("hello world")
    assert len(ids) > 0
    assert tok.decode(ids) == "hello world"
    assert tok.eos_token is not None


def test_from_pretrained_gpt2():
    pytest.importorskip("huggingface_hub")
    from nanotok import Tokenizer

    tok = Tokenizer.from_pretrained("openai-community/gpt2")

    ids = tok.encode("hello world")
    assert len(ids) > 0
    assert tok.decode(ids) == "hello world"
