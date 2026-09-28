"""会话主题的模型提示与输出清理。"""

from __future__ import annotations

import json

from openai.types.chat import ChatCompletionMessageParam

TITLE_MAX_CHARS = 36
TITLE_PROMPT_MAX_BYTES = 960

TITLE_INSTRUCTIONS = (
    "Generate a concise, single-line task title of at most 36 characters and under five "
    "words where possible. Start with an imperative verb. Write in the user's language. "
    "Preserve ticket references, proper nouns, acronyms and code terms. "
    "Do not use quotes, markdown or trailing punctuation. Do not answer the request. "
    'Return only JSON with one field: {"title":"..."}.'
)


def title_messages(user_message: str) -> list[ChatCompletionMessageParam]:
    """限制首条任务的大小，不在 UTF-8 字符中间截断。"""
    remaining = TITLE_PROMPT_MAX_BYTES
    text = user_message.strip().encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
    return [
        {"role": "system", "content": TITLE_INSTRUCTIONS},
        {"role": "user", "content": text},
    ]


def parse_title(response: str | None) -> str | None:
    """只接受单行 JSON 标题；清理空白、引号和句末标点。"""
    if not response:
        return None
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"title"}
        or not isinstance(value["title"], str)
    ):
        return None
    title = " ".join(value["title"].split())
    title = title.strip("\"'`“”‘’ ").rstrip("。.!?！？ ")
    if not title or any(not char.isprintable() for char in title):
        return None
    return title[:TITLE_MAX_CHARS]
