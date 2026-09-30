"""LLM 封装层的测试：不发真实请求，用假 client 验证参数透传与流式累积。"""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
from openai import BadRequestError

from polya.llm import LLM, _completion_from_stream, accumulate_stream


@pytest.fixture
def captured(monkeypatch):
    """把 OpenAI 构造函数换成假的，捕获收到的参数。"""
    box: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            box.update(kwargs)

    monkeypatch.setattr("polya.llm.OpenAI", FakeClient)
    return box


def test_timeout_and_retries_are_forwarded(captured):
    LLM(api_key="k", timeout=30, max_retries=5)
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 5


def test_defaults_are_sane(captured):
    LLM(api_key="k")
    assert captured["timeout"] == 120.0
    assert captured["max_retries"] == 2


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        LLM()


# ---------- accumulate_stream：纯函数累积（离线全覆盖） ----------


def make_chunk(
    content=None, reasoning=None, tool_calls=None, finish_reason=None, usage=None, with_choice=True
):
    delta = SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=tool_calls or []
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice] if with_choice else [], usage=usage)


def make_tc(index=0, id=None, name=None, arguments=None, with_function=True):
    function = SimpleNamespace(name=name, arguments=arguments) if with_function else None
    return SimpleNamespace(index=index, id=id, function=function)


def test_accumulate_joins_text_and_reasoning_with_interleaving():
    callbacks: list = []
    chunks = [
        make_chunk(reasoning="先想"),
        make_chunk(reasoning="清楚"),
        make_chunk(content="你好"),
        make_chunk(reasoning="再补一句"),
        make_chunk(content="世界", finish_reason="stop"),
    ]
    result = accumulate_stream(chunks, on_delta=lambda kind, delta: callbacks.append((kind, delta)))
    assert result["content"] == "你好世界"
    assert result["reasoning_content"] == "先想清楚再补一句"
    assert result["finish_reason"] == "stop"
    assert callbacks == [
        ("reasoning", "先想"),
        ("reasoning", "清楚"),
        ("text", "你好"),
        ("reasoning", "再补一句"),
        ("text", "世界"),
    ]


def test_accumulate_merges_tool_call_fragments_by_index():
    chunks = [
        make_chunk(tool_calls=[make_tc(0, id="c1", name="add", arguments='{"a"')]),
        # 两个工具交错分片
        make_chunk(tool_calls=[make_tc(1, id="c2", name="mul", arguments='{"x": ')]),
        make_chunk(tool_calls=[make_tc(0, arguments=": 2}")]),
        make_chunk(tool_calls=[make_tc(1, arguments="1}")]),
        make_chunk(tool_calls=[make_tc(1)], finish_reason="tool_calls"),  # 尾片无 id/name
    ]
    result = accumulate_stream(chunks)
    assert result["content"] is None  # 纯工具轮：与非流式同形
    assert result["tool_calls"] == [
        {"id": "c1", "name": "add", "arguments": '{"a": 2}'},
        {"id": "c2", "name": "mul", "arguments": '{"x": 1}'},
    ]


def test_accumulate_tolerates_missing_index_and_empty_function():
    chunks = [
        # 只带 index 的占位片段（function=None）直接跳过
        make_chunk(tool_calls=[SimpleNamespace(index=0, id=None, function=None)]),
        # index 缺失按 0（防御，OpenAI 必发）
        make_chunk(
            tool_calls=[
                SimpleNamespace(
                    index=None, id=None, function=SimpleNamespace(name="add", arguments='{"a": 1}')
                )
            ]
        ),
    ]
    result = accumulate_stream(chunks)
    assert result["tool_calls"] == [{"id": "", "name": "add", "arguments": '{"a": 1}'}]


def test_accumulate_captures_usage_only_tail_chunk():
    usage = SimpleNamespace(total_tokens=42)
    chunks = [
        make_chunk(content="答"),
        make_chunk(usage=usage, with_choice=False),  # stream_options 尾包：choices=[]
    ]
    result = accumulate_stream(chunks)
    assert result["content"] == "答"
    assert result["usage"] is usage


def test_accumulate_empty_stream_returns_all_none():
    result = accumulate_stream([], on_delta=lambda *a: pytest.fail("空流不应回调"))
    assert result == {
        "content": None,
        "reasoning_content": None,
        "tool_calls": [],
        "finish_reason": None,
        "usage": None,
    }


def test_accumulate_skips_empty_string_fragments():
    callbacks: list = []
    result = accumulate_stream(
        [make_chunk(content="", reasoning="")], on_delta=lambda kind, d: callbacks.append((kind, d))
    )
    assert result["content"] is None and result["reasoning_content"] is None
    assert callbacks == []


def test_completion_envelope_shape():
    usage = SimpleNamespace(total_tokens=1)
    result = {
        "content": "答",
        "reasoning_content": "想过",
        "tool_calls": [{"id": "", "name": "add", "arguments": "{}"}],
        "finish_reason": "tool_calls",
        "usage": usage,
    }
    completion = _completion_from_stream(result)
    message = completion.choices[0].message
    assert message.content == "答"
    assert message.reasoning_content == "想过"
    assert message.tool_calls[0].id == "call_0"  # 端点没发 id：合成确定性 id
    assert message.tool_calls[0].function.name == "add"
    assert completion.usage is usage
    assert completion.choices[0].finish_reason == "tool_calls"

    bare = _completion_from_stream(
        {
            "content": None,
            "reasoning_content": None,
            "tool_calls": [],
            "finish_reason": None,
            "usage": None,
        }
    )
    message = bare.choices[0].message
    assert message.content is None and message.tool_calls is None
    assert not hasattr(message, "reasoning_content")  # 属性仅在有内容时存在


# ---------- chat：流式路径与 stream_options 回退 ----------


class FakeCompletions:
    """记录 create 收到的 kwargs；流式返回预设 chunk 迭代器，非流式返回标记对象。

    reject_mode：None 正常；"stream_options" 只拒带 stream_options 的请求
    （模拟不支持该参数的端点）；"always" 一律 400（与 stream_options 无关）。
    """

    def __init__(self):
        self.kwargs_log: list[dict] = []
        self.chunks: list = []
        self.reject_mode: str | None = None
        self.non_stream_reply = SimpleNamespace(marker="non-stream")

    def create(self, **kwargs):
        self.kwargs_log.append(kwargs)
        should_reject = self.reject_mode == "always" or (
            self.reject_mode == "stream_options" and "stream_options" in kwargs
        )
        if should_reject:
            response = httpx2.Response(
                400, request=httpx2.Request("POST", "https://x.invalid/v1/chat/completions")
            )
            raise BadRequestError("bad request", response=response, body=None)
        if kwargs.get("stream"):
            return iter(self.chunks)
        return self.non_stream_reply


@pytest.fixture
def fake_llm(monkeypatch):
    completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr("polya.llm.OpenAI", FakeOpenAI)
    return LLM(api_key="k"), completions


def test_generate_title_records_usage_separately(monkeypatch):
    """标题请求用量单独计量，不进 total_usage（票 10）。"""
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"title":"会话标题"}'))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def with_options(self, **kwargs):
            captured["options"] = kwargs
            return self

    monkeypatch.setattr("polya.llm.OpenAI", FakeClient)
    llm = LLM(api_key="k")
    assert llm.title_usage == {}
    assert llm.generate_title("帮我修登录") == "会话标题"
    assert llm.title_usage == {"prompt_tokens": 11, "completion_tokens": 4}
    assert captured["options"] == {"timeout": 20.0, "max_retries": 0}


def test_chat_streams_when_on_delta_given(fake_llm):
    llm, completions = fake_llm
    completions.chunks = [
        make_chunk(content="你"),
        make_chunk(content="好"),
        make_chunk(usage=SimpleNamespace(total_tokens=7), with_choice=False),  # usage 尾包
    ]
    callbacks: list = []
    completion = llm.chat(
        [{"role": "user", "content": "hi"}], on_delta=lambda k, d: callbacks.append((k, d))
    )
    [kwargs] = completions.kwargs_log
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert callbacks == [("text", "你"), ("text", "好")]
    assert completion.choices[0].message.content == "你好"
    assert completion.usage.total_tokens == 7


def test_chat_without_delta_keeps_non_stream_path(fake_llm):
    llm, completions = fake_llm
    reply = llm.chat([{"role": "user", "content": "hi"}])
    [kwargs] = completions.kwargs_log
    assert "stream" not in kwargs and "stream_options" not in kwargs
    assert reply is completions.non_stream_reply


def test_chat_falls_back_when_stream_options_rejected(fake_llm):
    llm, completions = fake_llm
    completions.reject_mode = "stream_options"
    completions.chunks = [make_chunk(content="好", finish_reason="stop")]

    callbacks: list = []
    completion = llm.chat(
        [{"role": "user", "content": "hi"}], on_delta=lambda k, d: callbacks.append((k, d))
    )
    first, second = completions.kwargs_log
    assert "stream_options" in first and "stream_options" not in second
    assert llm._stream_usage is False  # 实例记住：该端点不支持
    assert completion.choices[0].message.content == "好"

    completions.reject_mode = None  # 第三次调用不再携带（记忆生效）
    completions.chunks = [make_chunk(content="呀")]
    llm.chat([{"role": "user", "content": "again"}], on_delta=lambda k, d: None)
    assert "stream_options" not in completions.kwargs_log[-1]


def test_chat_reraises_bad_request_unrelated_to_stream_options(fake_llm):
    llm, completions = fake_llm
    completions.reject_mode = "always"  # 400 与 stream_options 无关 → 不重试，直接上抛

    with pytest.raises(BadRequestError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert len(completions.kwargs_log) == 1
