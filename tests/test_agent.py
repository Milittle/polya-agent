"""用假的 LLM 验证 agent 的工具调用循环，不依赖网络。"""

from __future__ import annotations

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
    """按脚本依次返回预设回复，并记录每次收到的消息。"""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None):
        # 做一次浅拷贝，避免后续对 messages 的修改影响断言
        self.calls.append({"messages": list(messages), "tools": tools})
        message = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


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
