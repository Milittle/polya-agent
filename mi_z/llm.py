"""对 OpenAI 兼容接口的最小封装。

任何提供 ``/v1/chat/completions`` 的服务（OpenAI、DeepSeek、通义、vLLM……）
都可以通过 ``base_url`` 接入。超时与重试交给 OpenAI SDK 原生机制：
429/5xx 自动指数退避重试（``max_retries`` 次），全部失败才抛给调用方。
"""

from __future__ import annotations

import logging
import os

from openai import APITimeoutError, OpenAI, RateLimitError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger("mi_z.llm")


class LLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
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
            timeout=timeout,
            max_retries=max_retries,
            **client_kwargs,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """发送一轮对话请求，返回 SDK 的 ChatCompletion 对象。

        429 会被 SDK 自动重试；重试把超时预算耗尽后表现为超时异常，这里补一句
        可操作的提示再抛出，避免误判成网络或模型能力问题。
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, RateLimitError) as exc:
            logger.warning(
                "请求失败（%s）。若持续超时：调大 LLM(timeout=...)；429 限流会被 SDK 自动重试，"
                "重试耗尽超时预算时也会表现为超时。",
                type(exc).__name__,
            )
            raise
