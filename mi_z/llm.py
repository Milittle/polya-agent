"""对 OpenAI 兼容接口的最小封装。

任何提供 ``/v1/chat/completions`` 的服务（OpenAI、DeepSeek、通义、vLLM……）
都可以通过 ``base_url`` 接入。超时与重试交给 OpenAI SDK 原生机制：
429/5xx 自动指数退避重试（``max_retries`` 次），全部失败才抛给调用方。

流式：``chat(..., on_delta=...)`` 走 ``stream=True``，片段边收边回调
``on_delta(kind, delta)``（kind ∈ "text" / "reasoning"），最终把 chunk 流
累积成与非流式**同形**的响应对象返回——调用方不需要感知两种形态的差异。
累积逻辑抽成纯函数 ``accumulate_stream``，可完全离线单测。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from types import SimpleNamespace

from openai import APITimeoutError, BadRequestError, OpenAI, RateLimitError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger("mi_z.llm")

DeltaCallback = Callable[[str, str], None]


def accumulate_stream(chunks: Iterable, on_delta: DeltaCallback | None = None) -> dict:
    """把 OpenAI 兼容的流式 chunk 序列累积成一条完整消息（纯函数，不发网络）。

    chunk 只按鸭子类型访问（SDK 的 ``ChatCompletionChunk`` 或测试假件均可）：
    ``delta.content`` / ``delta.reasoning_content``（DeepSeek 扩展字段，SDK 模型
    ``extra="allow"`` 可直接 getattr）各自拼接并实时回调；``delta.tool_calls[i]``
    按 index 键控合并——id/name 只在首片段携带（后续为 None），arguments 永远
    分片拼接，两个工具交错分片也能各自归位；尾部 usage-only chunk
    （``choices=[]``，由 ``stream_options={"include_usage": True}`` 产生）只承载
    usage。输出与非流式消息同形：纯工具轮的 content 是 None 而非空串。

    已知不处理的坏 provider 行为：每片段重复全量 arguments 会重复拼接；
    ``finish_reason`` 永不发则保持 None（Agent 不依赖它）。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    usage = None
    for chunk in chunks:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage  # 尾包可能只带 usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]  # 从不传 n>1
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
            if on_delta is not None:
                on_delta("reasoning", reasoning)
        if delta.content:  # 空串片段跳过，不发空回调
            content_parts.append(delta.content)
            if on_delta is not None:
                on_delta("text", delta.content)
        for tc in delta.tool_calls or []:
            index = tc.index if tc.index is not None else 0
            slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function is not None:  # 个别端点会发只带 index 的占位片段
                if tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
    return {
        "content": "".join(content_parts) or None,
        "reasoning_content": "".join(reasoning_parts) or None,
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _completion_from_stream(result: dict):
    """把 ``accumulate_stream`` 的结果包成与非流式响应同形的对象（鸭子类型）。

    不用 pydantic 真构造 ``ChatCompletion``——流式缺少 id 等字段会撞 SDK 校验；
    SimpleNamespace 足够覆盖 Agent / compact 的访问面
    （``choices[0].message.content / .tool_calls[i].id|.function.* /
    getattr(message, "reasoning_content") / .usage``）。
    """
    calls = [
        SimpleNamespace(
            # 个别端点流式不发 tool_call id：合成确定性 id，历史回传仍合法
            id=tc["id"] or f"call_{index}",
            function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
        )
        for index, tc in enumerate(result["tool_calls"])
    ]
    message = SimpleNamespace(content=result["content"], tool_calls=calls or None)
    if result["reasoning_content"]:
        message.reasoning_content = result["reasoning_content"]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=result["finish_reason"])],
        usage=result["usage"],
    )


class LLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
        **client_kwargs,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("缺少 API key：请设置环境变量 OPENAI_API_KEY，或传入 api_key 参数。")
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        # temperature=None 表示该模型不接受自定义温度（o 系列只允许默认 1），
        # 请求时不携带该参数
        self.temperature = temperature
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            **client_kwargs,
        )
        # 该端点是否接受 stream_options：个别 OpenAI 兼容端点会因它 400，
        # 首次失败即关闭并在本实例内记住（见 chat 的回退逻辑）
        self._stream_usage = True

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ):
        """发送一轮对话请求，返回与非流式同形的响应对象。

        ``on_delta(kind, delta)`` 给定时走流式：text / reasoning 片段边收边
        回调（工具参数分片不回调——UI 不渲染半截参数）；不给定时保持非流式
        路径不变（compact 摘要等内部调用、-p 模式都落在此分支）。

        429 会被 SDK 自动重试；重试把超时预算耗尽后表现为超时异常，这里补一句
        可操作的提示再抛出，避免误判成网络或模型能力问题。
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        try:
            if on_delta is None:
                return self.client.chat.completions.create(**kwargs)
            if self._stream_usage:
                kwargs["stream_options"] = {"include_usage": True}
            try:
                stream = self.client.chat.completions.create(stream=True, **kwargs)
            except BadRequestError:
                # 400 在 create 时即抛（流尚未开始迭代），回退安全：该端点不认
                # stream_options，去掉重试一次并记住，本实例之后不再携带。
                if "stream_options" not in kwargs:
                    raise
                self._stream_usage = False
                del kwargs["stream_options"]
                stream = self.client.chat.completions.create(stream=True, **kwargs)
            return _completion_from_stream(accumulate_stream(stream, on_delta))
        except (APITimeoutError, RateLimitError) as exc:
            logger.warning(
                "请求失败（%s）。若持续超时：调大 LLM(timeout=...)；429 限流会被 SDK 自动重试，"
                "重试耗尽超时预算时也会表现为超时。",
                type(exc).__name__,
            )
            raise
