from __future__ import annotations


def render_chat_template(
    *,
    template: str | None,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    bos_token: str = "",
    eos_token: str = "",
) -> str:
    if template is None:
        raise ValueError(
            "No chat_template found. Load a tokenizer that includes a "
            "chat_template in its tokenizer_config.json."
        )

    try:
        from jinja2 import BaseLoader, Environment
    except ImportError:
        raise ImportError(
            "jinja2 is required for apply_chat_template(). "
            "Install it with: pip install nanotok[chat]"
        ) from None

    def raise_exception(msg: str) -> str:
        raise ValueError(msg)

    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    env.globals["raise_exception"] = raise_exception

    return env.from_string(template).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
        eos_token=eos_token,
    )
