import pytest

MESSAGES = [{"role": "user", "content": "Hello!"}]


def test_chat_template_render(tokenizer):
    pytest.importorskip("jinja2")
    if tokenizer._chat_template is None:
        pytest.skip("No chat_template")

    result = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    assert isinstance(result, str)
    assert "Hello!" in result


def test_chat_template_tokenize(tokenizer):
    pytest.importorskip("jinja2")
    if tokenizer._chat_template is None:
        pytest.skip("No chat_template")

    result = tokenizer.apply_chat_template(MESSAGES, tokenize=True)
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)


def test_chat_template_generation_prompt(tokenizer):
    pytest.importorskip("jinja2")
    if tokenizer._chat_template is None:
        pytest.skip("No chat_template")

    without = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    assert len(with_prompt) >= len(without)
