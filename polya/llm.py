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

from .providers import profile_for, reasoning_params

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger("polya.llm")

DeltaCallback = Callable[[str, str], None]


class _StreamUnsupported(Exception):
    """端点拒绝流式请求（400 且信息指向 stream），调用方回退非流式。"""


# provider 超窗错误的特征片段（小写匹配）。OpenAI 系：context_length_exceeded /
# "This model's maximum context length is ..."；Anthropic："prompt is too long ..."。
# Agent 用它判定「压缩释放空间后重试一次」是否适用（pi 同款恢复语义）。
_OVERFLOW_MARKERS = (
    "context_length",
    "maximum context length",
    "prompt is too long",
    "prompt too long",
    "exceeds the context window",
    "exceeds context window",
    "input tokens exceed",
)


def is_context_overflow(exc: BaseException) -> bool:
    """判断异常是否为 provider 的上下文超窗错误（按错误文案特征匹配）。"""
    text = str(exc).lower()
    return any(marker in text for marker in _OVERFLOW_MARKERS)


def _new_stream_state() -> dict:
    return {
        "content_parts": [],
        "reasoning_parts": [],
        "tool_calls": {},
        "finish_reason": None,
        "usage": None,
    }


def fold_chunk(state: dict, chunk) -> list[tuple[str, str]]:
    """把一个流式 chunk 折叠进累积态 ``state``（就地修改），返回它产出的
    ``(kind, delta)`` 列表——回调式（accumulate_stream）与迭代器式
    （LLM.chat_iter）共享这一份折叠逻辑，行为不会分叉。

    chunk 只按鸭子类型访问（SDK 的 ``ChatCompletionChunk`` 或测试假件均可）：
    ``delta.content`` / ``delta.reasoning_content`` 各自拼接；``delta.tool_calls[i]``
    按 index 键控合并——id/name 只在首片段携带，arguments 永远分片拼接，
    两个工具交错分片也能各自归位；尾部 usage-only chunk 只承载 usage。
    """
    emitted: list[tuple[str, str]] = []
    chunk_usage = getattr(chunk, "usage", None)
    if chunk_usage is not None:
        state["usage"] = chunk_usage  # 尾包可能只带 usage
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return emitted
    choice = choices[0]  # 从不传 n>1
    if choice.finish_reason:
        state["finish_reason"] = choice.finish_reason
    delta = getattr(choice, "delta", None)
    if delta is None:
        return emitted
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        state["reasoning_parts"].append(reasoning)
        emitted.append(("reasoning", reasoning))
    if delta.content:  # 空串片段跳过
        state["content_parts"].append(delta.content)
        emitted.append(("text", delta.content))
    for tc in delta.tool_calls or []:
        index = tc.index if tc.index is not None else 0
        slot = state["tool_calls"].setdefault(index, {"id": "", "name": "", "arguments": ""})
        if tc.id:
            slot["id"] = tc.id
        if tc.function is not None:  # 个别端点会发只带 index 的占位片段
            if tc.function.name:
                slot["name"] = tc.function.name
            if tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    return emitted


def _finalize_stream(state: dict) -> dict:
    """累积态 → 与非流式消息同形的消息 dict（纯工具轮 content 为 None）。"""
    return {
        "content": "".join(state["content_parts"]) or None,
        "reasoning_content": "".join(state["reasoning_parts"]) or None,
        "tool_calls": [state["tool_calls"][index] for index in sorted(state["tool_calls"])],
        "finish_reason": state["finish_reason"],
        "usage": state["usage"],
    }


def accumulate_stream(chunks: Iterable, on_delta: DeltaCallback | None = None) -> dict:
    """把 OpenAI 兼容的流式 chunk 序列累积成一条完整消息（纯函数，不发网络）。

    片段经 :func:`fold_chunk` 折叠并实时回调 ``on_delta(kind, delta)``。
    已知不处理的坏 provider 行为：每片段重复全量 arguments 会重复拼接；
    ``finish_reason`` 永不发则保持 None（Agent 不依赖它）。
    """
    state = _new_stream_state()
    for chunk in chunks:
        for kind, delta in fold_chunk(state, chunk):
            if on_delta is not None:
                on_delta(kind, delta)
    return _finalize_stream(state)


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
        profile_name: str | None = None,
        **client_kwargs,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("缺少 API key：请设置环境变量 OPENAI_API_KEY，或传入 api_key 参数。")
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        # 来源标识：来自 ~/.polya/models.json 的哪个命名 profile（票 14）——状态行
        # 与 /models 选项器标记「当前」用；env/旗标直连时为 None
        self.profile_name = profile_name
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
        # 端点是否支持流式：不支持时回退非流式（首次 400 后记住）
        self._stream_supported = True
        # 流式瞬时错误的有界重试：只在尚未吐出任何片段时重开请求
        self._stream_retries = 1
        # 推理档位（/thinking，一家一策）：风格由模型能力档案决定，None=用厂商默认。
        self.reasoning_style = profile_for(self.model).reasoning_style
        self.thinking_level: str | None = None

    def _base_kwargs(self, messages: list[dict], tools: list[dict] | None) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        if self.thinking_level and self.reasoning_style != "none":
            kwargs.update(reasoning_params(self.reasoning_style, self.thinking_level))
        return kwargs

    def _open_stream(self, kwargs: dict):
        """打开流式请求；端点不认 stream_options 时去掉重试一次并记住，
        端点根本不支持流式时抛 :class:`_StreamUnsupported` 由调用方回退。"""
        if not self._stream_supported:
            raise _StreamUnsupported
        if self._stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return self.client.chat.completions.create(stream=True, **kwargs)
        except BadRequestError as exc:
            # 400 在 create 时即抛（流尚未开始迭代），回退安全。
            if "stream_options" in kwargs:
                self._stream_usage = False
                kwargs.pop("stream_options")
                try:
                    return self.client.chat.completions.create(stream=True, **kwargs)
                except BadRequestError as retry_exc:
                    if "stream" in str(retry_exc).lower():
                        self._stream_supported = False
                        raise _StreamUnsupported from retry_exc
                    raise
            if "stream" in str(exc).lower():
                self._stream_supported = False
                raise _StreamUnsupported from exc
            raise

    def chat_iter(self, messages: list[dict], tools: list[dict] | None = None):
        """流式的迭代器形态：``yield (kind, delta)``，结束的 ``StopIteration.value``
        是与非流式同形的完整响应。

        与 ``chat(on_delta=...)`` 共享 :func:`fold_chunk` 折叠逻辑——callback 是
        它的一个特例。生成器协议（ADR 0002）用这个接口逐段 ``yield`` 事件。

        健壮性：端点不支持流式时静默回退非流式（不吐任何片段，直接返回完整
        响应）；尚未吐出任何片段的瞬时错误（超时 / 限流）重开一次请求；一旦
        已有片段输出，错误直接上抛（重试会造成重复文本）。
        """
        kwargs = self._base_kwargs(messages, tools)
        attempts = 0
        while True:
            attempts += 1
            emitted = False
            try:
                stream = self._open_stream(kwargs)
            except _StreamUnsupported:
                return self.client.chat.completions.create(**kwargs)
            state = _new_stream_state()
            try:
                for chunk in stream:
                    for kind, delta in fold_chunk(state, chunk):
                        emitted = True
                        yield (kind, delta)
            except (APITimeoutError, RateLimitError) as exc:
                if not emitted and attempts <= self._stream_retries:
                    logger.warning(
                        "流式请求在首个片段前失败（%s），重试第 %d 次",
                        type(exc).__name__,
                        attempts,
                    )
                    continue
                logger.warning(
                    "请求失败（%s）。若持续超时：调大 LLM(timeout=...)；429 限流会被 SDK 自动重试，"
                    "重试耗尽超时预算时也会表现为超时。",
                    type(exc).__name__,
                )
                raise
            return _completion_from_stream(_finalize_stream(state))

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
        kwargs = self._base_kwargs(messages, tools)
        try:
            if on_delta is None:
                return self.client.chat.completions.create(**kwargs)
            try:
                return _completion_from_stream(
                    accumulate_stream(self._open_stream(kwargs), on_delta)
                )
            except _StreamUnsupported:
                # 端点不支持流式：静默回退非流式（无实时回调，但结果正确）
                return self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, RateLimitError) as exc:
            logger.warning(
                "请求失败（%s）。若持续超时：调大 LLM(timeout=...)；429 限流会被 SDK 自动重试，"
                "重试耗尽超时预算时也会表现为超时。",
                type(exc).__name__,
            )
            raise
