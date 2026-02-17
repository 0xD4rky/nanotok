import pytest


def test_special_tokens_detected(tokenizer):
    specials = tokenizer.special_tokens
    assert isinstance(specials, dict)
    assert len(specials) > 0


def test_special_ids_are_ints(tokenizer):
    for token, token_id in tokenizer.special_tokens.items():
        assert isinstance(token, str)
        assert isinstance(token_id, int)


def test_encode_without_special(tokenizer):
    if not tokenizer.special_tokens:
        pytest.skip("No special tokens")

    special = next(iter(tokenizer.special_tokens))
    text = f"hello {special} world"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_encode_with_allowed_special(tokenizer):
    if not tokenizer.special_tokens:
        pytest.skip("No special tokens")

    special = next(iter(tokenizer.special_tokens))
    expected_id = tokenizer.special_tokens[special]
    ids = tokenizer.encode(f"hello{special}world", allowed_special={special})
    assert expected_id in ids


def test_decode_skip_special_tokens(tokenizer):
    if not tokenizer.special_tokens:
        pytest.skip("No special tokens")

    special = next(iter(tokenizer.special_tokens))
    special_id = tokenizer.special_tokens[special]

    normal_ids = tokenizer.encode("hello world")
    mixed_ids = normal_ids[:1] + [special_id] + normal_ids[1:]

    decoded_with = tokenizer.decode(mixed_ids, skip_special_tokens=False)
    decoded_without = tokenizer.decode(mixed_ids, skip_special_tokens=True)

    assert special in decoded_with
    assert special not in decoded_without


def test_eos_token_property(tokenizer):
    if tokenizer.eos_token is not None:
        assert isinstance(tokenizer.eos_token, str)
        assert tokenizer.eos_token_id is not None
