"""对 OpenAI 兼容接口的最小封装。

任何提供 ``/v1/chat/completions`` 的服务（OpenAI、DeepSeek、通义、vLLM……）
都可以通过 ``base_url`` 接入。
"""

from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class LLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        **client_kwargs,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("缺少 API key：请设置环境变量 OPENAI_API_KEY，或传入 api_key 参数。")
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            **client_kwargs,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """发送一轮对话请求，返回 SDK 的 ChatCompletion 对象。"""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)
