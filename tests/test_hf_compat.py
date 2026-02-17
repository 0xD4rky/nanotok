import pytest


def test_call_returns_dict(tokenizer):
    result = tokenizer("hello world")
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "attention_mask" in result


def test_call_single_string(tokenizer):
    result = tokenizer("hello")
    assert isinstance(result["input_ids"], list)
    assert all(isinstance(i, int) for i in result["input_ids"])
    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert all(m == 1 for m in result["attention_mask"])


def test_call_batch(tokenizer):
    result = tokenizer(["hello", "world foo"])
    assert isinstance(result["input_ids"], list)
    assert len(result["input_ids"]) == 2


def test_call_truncation(tokenizer):
    result = tokenizer("hello world foo bar baz", truncation=True, max_length=3)
    assert len(result["input_ids"]) == 3


def test_call_padding(tokenizer):
    result = tokenizer(["hi", "hello world foo bar baz"], padding=True)
    assert len(result["input_ids"][0]) == len(result["input_ids"][1])


def test_call_padding_max_length(tokenizer):
    result = tokenizer("hi", padding="max_length", max_length=10)
    assert len(result["input_ids"]) == 10


def test_call_return_tensors_np(tokenizer):
    np = pytest.importorskip("numpy")
    result = tokenizer("hello world", return_tensors="np")
    assert hasattr(result["input_ids"], "shape")
    assert result["input_ids"].dtype == np.int64


def test_call_return_tensors_pt(tokenizer):
    torch = pytest.importorskip("torch")
    result = tokenizer("hello world", return_tensors="pt")
    assert torch.is_tensor(result["input_ids"])
    assert result["input_ids"].dtype == torch.long
