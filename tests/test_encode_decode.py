def test_roundtrip_hello(tokenizer):
    text = "hello world"
    ids = tokenizer.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == text


def test_roundtrip_empty(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_roundtrip_unicode(tokenizer):
    for text in ["Hello, world!", "Bonjour le monde", "Hola mundo"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_chinese(tokenizer):
    text = "你好世界"
    ids = tokenizer.encode(text)
    assert len(ids) > 0
    assert tokenizer.decode(ids) == text


def test_roundtrip_long(tokenizer):
    text = "The quick brown fox jumps over the lazy dog. " * 100
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_all_printable_ascii(tokenizer):
    text = "".join(chr(i) for i in range(32, 127))
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_whitespace(tokenizer):
    text = "  hello  \n  world  \t  "
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_newlines(tokenizer):
    text = "line1\nline2\nline3\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_returns_list_of_ints(tokenizer):
    ids = tokenizer.encode("test")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_roundtrip_code(tokenizer):
    text = 'def hello():\n    print("Hello, World!")\n    return 42\n'
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_numbers(tokenizer):
    text = "3.14159 2.71828 1234567890"
    assert tokenizer.decode(tokenizer.encode(text)) == text
