"""加固阶段的测试：schema 表达力、提示词 section、token 预算、健壮性与归档。"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Literal

from polya import Agent, tool
from polya.compact import (
    COMPRESS_MARKER,
    MICROCLEAR_MARKER,
    effective_keep,
    extract_file_operations,
    microcompact,
)
from polya.llm import LLM
from polya.prompt import SystemPrompt, diff_sections, tool_guidelines, tool_snippets
from polya.tools import ToolRegistry, json_schema

# ---------- schema 表达力 ----------


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Point:
    x: int
    y: int = 0


def test_literal_and_enum_become_enum_schema():
    assert json_schema(Literal["a", "b"]) == {"type": "string", "enum": ["a", "b"]}
    assert json_schema(Color) == {"type": "string", "enum": ["red", "blue"]}


def test_annotated_description_lands_on_field():
    @tool
    def search(query: Annotated[str, "what to look for"], limit: int = 5) -> str:
        """搜索。"""
        return ""

    schema = ToolRegistry([search]).schemas()[0]
    params = schema["function"]["parameters"]
    assert params["properties"]["query"] == {"type": "string", "description": "what to look for"}
    assert params["properties"]["limit"] == {"type": "integer"}
    assert params["required"] == ["query"]  # limit 有默认值


def test_dict_and_nested_dataclass_schemas():
    assert json_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    nested = json_schema(Point)
    assert nested["type"] == "object"
    assert nested["properties"]["x"] == {"type": "integer"}
    assert nested["required"] == ["x"]


def test_unknown_type_gets_no_constraint_not_string():
    class Weird:
        pass

    assert json_schema(Weird) == {}  # 不再伪装成 string


def test_tool_snippet_and_guidelines_are_separate_channels():
    @tool(
        snippet="one line",
        guidelines=["rule A", "rule A", "rule B"],
    )
    def thing(path: str) -> str:
        """完整用法说明（进 API description）。"""
        return path

    item = ToolRegistry([thing]).get("thing")
    assert item.description == "完整用法说明（进 API description）。"
    assert item.snippet == "one line"
    assert tool_snippets(ToolRegistry([thing])) == "- thing: one line"
    assert tool_guidelines(ToolRegistry([thing])) == "- rule A\n- rule B"  # 去重保序


# ---------- 提示词 section ----------


def test_system_prompt_sections_are_tagged_and_ordered():
    prompt = SystemPrompt("BASE")
    prompt.set("tools", "- read: read files")
    prompt.set("cwd", "/work")
    prompt.set("empty", "")  # 空内容不产生 section
    rendered = prompt.render()
    assert rendered.startswith("BASE\n\n<tools>")
    assert "<cwd>\n/work\n</cwd>" in rendered
    assert "<empty>" not in rendered
    sections = prompt.sections()
    assert sections["preamble"] == "BASE"
    assert sections["tools"] == "<tools>\n- read: read files\n</tools>"


def test_diff_sections_reports_changed_and_removed():
    previous = {"preamble": "BASE", "skills": "<skills>x</skills>"}
    current = {"preamble": "BASE", "tools": "<tools>y</tools>"}
    patch = diff_sections(previous, current)
    assert patch == {"tools": "<tools>y</tools>", "skills": None}


# ---------- token 预算与文件追踪 ----------


def test_effective_keep_uses_token_budget():
    history = [
        {"role": "user", "content": "a" * 300},
        {"role": "tool", "content": "b" * 300},
        {"role": "tool", "content": "c" * 300},
    ]
    # 每条约 100+ token；预算只够最后一条
    assert effective_keep(history, keep=30, keep_tokens=150) == 1
    # 预算很大时保留全部
    assert effective_keep(history, keep=30, keep_tokens=10_000) == 3
    # 未给预算时退回消息条数语义
    assert effective_keep(history, keep=2, keep_tokens=None) == 2


def test_extract_file_operations_reads_and_writes():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"function": {"name": "edit_file", "arguments": '{"path": "b.py"}'}},
                {"function": {"name": "write_file", "arguments": '{"path": "c.py"}'}},
                {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                {"function": {"name": "read_file", "arguments": "not json"}},
            ],
        }
    ]
    read_files, modified_files = extract_file_operations(messages)
    assert read_files == {"a.py"}
    assert modified_files == {"b.py", "c.py"}


# ---------- 树入口只读回查（非破坏） ----------


def test_tree_read_preserves_original_after_override():
    from polya.tree import SessionTree

    tree = SessionTree()
    tree.reset_with_system("sys")
    entry = tree.append("tool", {"tool_call_id": "a", "content": "原始内容"})
    tree.override(entry.id, {**entry.payload, "content": "被改写的投影"})
    assert "原始内容" in tree.read(entry.id) and "被改写" not in tree.read(entry.id)
    assert tree.project()[1]["content"] == "被改写的投影"


# ---------- agent 健壮性 ----------


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_delta=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._replies.pop(0)


def response(message, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None
    )


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


@tool
def echo(text: str) -> str:
    """原样返回。"""
    return f"echo: {text}"


def test_bad_json_arguments_feed_error_back_to_model():
    llm = ScriptedLLM(
        [
            response(
                SimpleNamespace(
                    content=None,
                    tool_calls=[tool_call("c1", "echo", "{not valid json")],
                )
            ),
            response(SimpleNamespace(content="已修正", tool_calls=None)),
        ]
    )
    agent = Agent(llm=llm, tools=[echo])

    def handle(ev):
        from polya.agent import ToolCall

        if isinstance(ev, ToolCall):
            raise AssertionError("坏参数不应触发工具执行")
        return None

    from polya.agent import drive

    assert drive(agent.steps("干活"), handle) == "已修正"
    tool_message = next(m for m in agent.history if m.get("role") == "tool")
    assert "不是合法 JSON" in tool_message["content"]
    assert tool_message["tool_call_id"] == "c1"


def test_length_truncation_continues_then_answers():
    llm = ScriptedLLM(
        [
            response(SimpleNamespace(content="前半段", tool_calls=None), finish_reason="length"),
            response(SimpleNamespace(content="后半段完成", tool_calls=None), finish_reason="stop"),
        ]
    )
    agent = Agent(llm=llm, tools=[echo])
    assert agent.run("写长文") == "后半段完成"
    assert any(
        "truncated" in (m.get("content") or "") for m in agent.history if m.get("role") == "user"
    )


def test_compact_now_compacts_regardless_of_threshold():
    llm = ScriptedLLM([response(SimpleNamespace(content="#1: 摘要", tool_calls=None))])
    agent = Agent(llm=llm, tools=[echo], compress=True, keep_recent=0)
    agent.tree.replace_conversation(
        [
            {"role": "user", "content": "任务"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "function": {"name": "echo", "arguments": '{"text": "x"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "很长的结果" * 100},
        ]
    )
    message = agent.compact_now("聚焦任务")
    assert "Compacted" in message
    assert any(
        m.get("role") == "tool" and (m.get("content") or "").startswith(COMPRESS_MARKER)
        for m in agent.history
    )
    assert agent._last_summary is not None  # 迭代摘要已记下


# ---------- llm 健壮性 ----------


def make_llm() -> LLM:
    return LLM(api_key="test", model="test-model")


def test_stream_unsupported_falls_back_to_non_stream(monkeypatch):
    llm = make_llm()
    message = SimpleNamespace(content="answer", tool_calls=None)

    class Completions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is not True  # 回退路径不带 stream
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
            )

    monkeypatch.setattr(
        llm, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    )
    monkeypatch.setattr(llm, "_open_stream", lambda kwargs: (_ for _ in ()).throw(_unsupported()))

    chunks = llm.chat_iter([{"role": "user", "content": "hi"}])
    result = None
    try:
        while True:
            next(chunks)
    except StopIteration as stop:
        result = stop.value
    assert result.choices[0].message.content == "answer"


def _unsupported():
    from polya.llm import _StreamUnsupported

    return _StreamUnsupported()


def test_micro_compact_replaces_large_and_is_idempotent():
    from polya.tree import SessionTree

    tree = SessionTree()
    tree.reset_with_system("sys")
    tool_entry = tree.append("tool", {"tool_call_id": "a", "content": "z" * 2000})
    tree.append("tool", {"tool_call_id": "b", "content": "short"})
    result = microcompact(tree, keep=0, min_chars=2000)
    assert result is not None
    before, cleared = result
    assert cleared == 2000  # 返回清理的字符数
    messages = tree.conversation()
    assert messages[0]["content"].startswith(MICROCLEAR_MARKER)
    assert "history_read" in messages[0]["content"]
    assert messages[1]["content"] == "short"
    assert "z" * 2000 in tree.read(tool_entry.id)  # 原文留在树里
    again = microcompact(tree, keep=0, min_chars=2000)
    assert again is None  # 幂等：已清理的不再处理


def test_agent_micro_compresses_without_calling_llm():
    class NoCallLLM:
        def chat(self, *args, **kwargs):
            raise AssertionError("微压缩不应调用 LLM")

    agent = Agent(
        llm=NoCallLLM(),
        tools=[echo],
        compress=True,
        context_window=10000,
        keep_recent=1,
    )
    agent.tree.replace_conversation(
        [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "function": {"name": "echo", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "z" * 5000},
            {"role": "user", "content": "recent kept"},
        ]
    )
    result = agent._try_microcompress()
    assert result is not None
    _, cleared = result
    assert cleared == 5000
    assert any((m.get("content") or "").startswith(MICROCLEAR_MARKER) for m in agent.history)
    # 回查指针指向树里保留的原始输出（入口 id）
    pointer = next(
        m for m in agent.history if (m.get("content") or "").startswith(MICROCLEAR_MARKER)
    )
    assert "entry_id=4" in pointer["content"]
    raw = agent.tools.call("history_read", {"entry_id": 4})
    assert "z" * 100 in raw


def test_micro_disabled_for_thinking_bound_models():
    from polya.providers import ModelProfile

    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[echo],
        compress=True,
        context_window=10000,
        profile=ModelProfile(supports_inplace_tool_edit=False),
    )
    agent.tree.replace_conversation([{"role": "tool", "content": "z" * 5000}])
    assert agent._should_microcompress() is False


def test_cached_tokens_are_collected_and_accumulated():
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
        prompt_tokens_details=SimpleNamespace(cached_tokens=80),
    )
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=usage,
    )
    agent = Agent(llm=ScriptedLLM([reply]), tools=[echo])
    assert agent.run("hi") == "ok"
    assert agent.last_usage["cached_tokens"] == 80
    assert agent.total_usage["cached_tokens"] == 80


# ---------- i18n ----------


def test_language_selection_via_env(monkeypatch):
    """语言分层：POLYA_LANG 只影响界面文案；模型侧固定英文不受影响。"""
    from polya import i18n
    from polya.prompts import CODING_SYSTEM_PROMPT, msg

    monkeypatch.setenv("POLYA_LANG", "en")
    assert i18n.current_language() == "en"
    assert i18n.t("ui.render.plan_submitted") == "⏺ Plan submitted"
    monkeypatch.setenv("POLYA_LANG", "klingon")
    assert i18n.current_language() == "zh"  # 未知值回落
    assert i18n.t("ui.render.plan_submitted") == "⏺ 计划已提交"
    # 模型侧固定英文，且提示词带「按用户语言回复」防线（language-policy spec）。
    assert "user's language" in CODING_SYSTEM_PROMPT
    assert msg("agent.interrupted") == "Error: the user interrupted this task."
