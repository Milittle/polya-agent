"""用假的 LLM 验证 agent 的工具调用循环，不依赖网络。"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from types import SimpleNamespace

import pytest

from mi_z import Agent, ToolRegistry, tool
from mi_z.agent import DEFAULT_SYSTEM_PROMPT


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class ScriptedLLM:
    """按脚本依次返回预设回复，并记录每次收到的消息。

    usages 可选，与 replies 一一对应；缺省用 None 模拟不带 usage 的响应。
    """

    def __init__(self, replies, usages=None):
        self._replies = list(replies)
        self._usages = list(usages) if usages is not None else [None] * len(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None):
        # 做一次浅拷贝，避免后续对 messages 的修改影响断言
        self.calls.append({"messages": list(messages), "tools": tools})
        message = self._replies.pop(0)
        usage = self._usages.pop(0) if self._usages else None
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@tool
def add(a: int, b: int) -> str:
    """把两个整数相加。"""
    return str(a + b)


def test_schema_is_inferred_from_signature():
    [schema] = ToolRegistry([add]).schemas()
    assert schema["function"]["name"] == "add"
    assert schema["function"]["description"] == "把两个整数相加。"
    assert schema["function"]["parameters"] == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }


def test_optional_parameter_is_not_required():
    @tool
    def greet(name: str, greeting: str | None = None) -> str:
        """打招呼。"""
        return f"{greeting or '你好'}, {name}"

    schema = ToolRegistry([greet]).schemas()[0]
    assert schema["function"]["parameters"]["required"] == ["name"]
    assert schema["function"]["parameters"]["properties"]["greeting"]["type"] == "string"


def test_agent_runs_tool_then_answers():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 2, "b": 3}')]),
            make_message(content="答案是 5"),
        ]
    )
    agent = Agent(llm=llm, tools=[add])

    assert agent.run("2 + 3 等于几？") == "答案是 5"

    # 第二轮请求里应该带上工具执行结果
    second_messages = llm.calls[1]["messages"]
    tool_message = second_messages[-1]
    assert tool_message == {"role": "tool", "tool_call_id": "c1", "content": "5"}
    assert second_messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}


def test_unknown_tool_is_reported_back_to_model():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "nope", "{}")]),
            make_message(content="换个方式回答"),
        ]
    )
    agent = Agent(llm=llm, tools=[add])
    agent.run("随便问问")

    tool_message = llm.calls[1]["messages"][-1]
    assert "unknown tool" in tool_message["content"]


def test_history_is_preserved_across_turns():
    llm = ScriptedLLM([make_message(content="第一次答复"), make_message(content="第二次答复")])
    agent = Agent(llm=llm, tools=[add])

    agent.run("你好")
    agent.run("再见")

    roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]

    agent.reset()
    assert agent.history == []


def test_raises_when_max_steps_exceeded():
    looping = [make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 1, "b": 1}')])] * 3
    agent = Agent(llm=ScriptedLLM(looping), tools=[add], max_steps=3)

    with pytest.raises(RuntimeError):
        agent.run("死循环")


def test_approve_hook_can_deny_tool_call():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 1, "b": 2}')]),
            make_message(content="好的，那我不加了"),
        ]
    )
    seen = []

    def approve(tool, arguments):
        seen.append((tool.name, arguments))
        return False

    agent = Agent(llm=llm, tools=[add], approve=approve)
    agent.run("帮我加一下")

    assert seen == [("add", {"a": 1, "b": 2})]
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "拒绝" in tool_message["content"]


def test_approve_hook_can_allow_tool_call():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 2, "b": 3}')]),
            make_message(content="5"),
        ]
    )
    agent = Agent(llm=llm, tools=[add], approve=lambda tool, arguments: True)

    assert agent.run("加一下") == "5"
    assert llm.calls[1]["messages"][-1]["content"] == "5"


# ---------------------------------------------------------------- KV Cache 前缀稳定性
# 以下测试把《深入理解 AI Agent》2.3 节的三条铁律钉死：
# 前缀字节级不变（只追加、不改写）是 KV Cache / Prompt Cache 复用的前提。


def test_messages_only_grow_append_only():
    """每次请求的消息序列必须是上一次的严格扩展：旧消息一个字节都不能变。"""
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 2, "b": 3}')]),
            make_message(content="5"),
            make_message(content="第二次答复"),
        ]
    )
    agent = Agent(llm=llm, tools=[add])
    agent.run("2 + 3？")
    agent.run("再会")  # 第二轮 run 也要保持前缀稳定

    for earlier, later in pairwise(llm.calls):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


def test_tool_schemas_are_stable_across_calls():
    """工具 schema 的内容与顺序必须确定：工具定义位于上下文前部，顺序抖动会让缓存失效。"""
    registry = ToolRegistry([add])
    first = registry.schemas()

    @tool
    def echo(text: str) -> str:
        """原样返回。"""
        return text

    registry_before_run = ToolRegistry([add, echo])
    assert registry_before_run.schemas() == registry_before_run.schemas()
    assert first == ToolRegistry([add]).schemas()


def test_registry_freezes_after_first_run():
    """对话开始后增删工具必须报错：这是 2.3 节「工具定义定了就不改」的代码化。"""
    llm = ScriptedLLM([make_message(content="好的")])
    agent = Agent(llm=llm, tools=[add])
    assert agent.tools.frozen is False

    agent.run("你好")
    assert agent.tools.frozen is True

    @tool
    def echo(text: str) -> str:
        """原样返回。"""
        return text

    with pytest.raises(RuntimeError):
        agent.tools.add(echo)


def test_usage_is_accumulated_and_reset_clears_it():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    llm = ScriptedLLM(
        [make_message(content="好")],
        usages=[usage],
    )
    agent = Agent(llm=llm)

    agent.run("你好")

    assert agent.last_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert agent.total_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    agent.reset()
    assert agent.last_usage is None
    assert agent.total_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_missing_usage_is_tolerated():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 1, "b": 1}')]),
            make_message(content="2"),
        ]
    )
    agent = Agent(llm=llm, tools=[add])

    assert agent.run("1+1") == "2"
    assert agent.last_usage is None
    assert agent.total_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------- 状态栏（书 2.6）


def test_status_bar_appends_user_message_each_iteration():
    """状态栏以 user 消息出现在每次请求的末尾，含迭代号与工具计数；只追加不改写。"""
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 2, "b": 3}')]),
            make_message(content="5"),
        ]
    )
    agent = Agent(llm=llm, tools=[add], status_bar=True)
    agent.run("2+3")

    first_status = llm.calls[0]["messages"][-1]
    assert first_status["role"] == "user"
    assert "<agent_status>" in first_status["content"]
    assert "第 1/" in first_status["content"]  # 迭代号
    assert "尚未调用工具" in first_status["content"]  # 第一轮还没有调用

    second_status = llm.calls[1]["messages"][-1]
    assert "<agent_status>" in second_status["content"]
    assert "第 2/" in second_status["content"]
    assert "add: 1 次" in second_status["content"]  # 计数已累计

    # 持久追加：第二次请求的消息序列仍是第一次的严格扩展（KV Cache 纪律不被破坏）
    earlier, later = llm.calls
    assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


def test_tool_results_annotated_with_call_count_when_status_enabled():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 1, "b": 1}')]),
            make_message(content="2"),
        ]
    )
    agent = Agent(llm=llm, tools=[add], status_bar=True)
    agent.run("1+1")

    tool_message = llm.calls[1]["messages"][-2]  # 倒数第二是工具结果，最后是新一轮状态
    assert tool_message["content"].startswith("（add 第 1 次调用）")
    assert agent.tool_counts == {"add": 1}


def test_status_bar_off_by_default():
    llm = ScriptedLLM([make_message(content="好")])
    agent = Agent(llm=llm)
    agent.run("hi")
    assert all("<agent_status>" not in str(m.get("content")) for m in llm.calls[0]["messages"])


def test_custom_status_renderer_and_reset_clears_counts():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "add", '{"a": 1, "b": 1}')]),
            make_message(content="2"),
        ]
    )
    agent = Agent(llm=llm, tools=[add], status_bar=lambda snap: f"<x>第{snap.iteration}轮</x>")
    agent.run("1+1")

    assert llm.calls[0]["messages"][-1]["content"] == "<x>第1轮</x>"
    assert llm.calls[1]["messages"][-1]["content"] == "<x>第2轮</x>"

    agent.reset()
    assert agent.tool_counts == Counter()
