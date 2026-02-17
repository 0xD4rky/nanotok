def test_batch_encode_matches_individual(tokenizer):
    texts = ["Hello, world!", "The quick brown fox", "def main(): pass", "", "12345"]
    batch_result = tokenizer.encode_batch(texts)
    assert len(batch_result) == len(texts)

    for text, batch_ids in zip(texts, batch_result):
        assert batch_ids == tokenizer.encode(text), f"Mismatch for: {text!r}"


def test_decode_batch(tokenizer):
    texts = ["hello world", "foo bar baz", "testing 123"]
    decoded = tokenizer.decode_batch(tokenizer.encode_batch(texts))
    assert decoded == texts


def test_batch_encode_empty(tokenizer):
    assert tokenizer.encode_batch([]) == []


def test_batch_decode_empty(tokenizer):
    assert tokenizer.decode_batch([]) == []


def test_batch_encode_single(tokenizer):
    result = tokenizer.encode_batch(["hello"])
    assert len(result) == 1
    assert result[0] == tokenizer.encode("hello")
