"""todo_write 工具与 TodoStore 的测试：存储校验、状态栏渲染、端到端注入。"""

from __future__ import annotations

from types import SimpleNamespace

from polya import Agent, ToolRegistry
from polya.builtin import default_tools
from polya.status import StatusSnapshot, render_status
from polya.todos import TodoStore


def registry(todos=None) -> ToolRegistry:
    return ToolRegistry(default_tools(".", todos=todos))


def test_store_rewrite_validates_atomically():
    store = TodoStore()
    store.rewrite([{"content": "保留项", "status": "pending"}])

    try:
        store.rewrite(
            [
                {"content": "好项", "status": "in_progress"},
                {"content": "坏状态", "status": "done"},  # 非法 status
            ]
        )
        raised = False
    except ValueError as exc:
        raised = "invalid status" in str(exc)
    assert raised
    assert store.as_dicts() == [{"content": "保留项", "status": "pending"}]  # 原子：未变


def test_store_rejects_empty_content():
    store = TodoStore()
    try:
        store.rewrite([{"content": "  ", "status": "pending"}])
        raised = False
    except ValueError as exc:
        raised = "content" in str(exc)
    assert raised


def test_store_empty_rewrite_clears():
    store = TodoStore()
    store.rewrite([{"content": "a", "status": "pending"}])
    assert store.rewrite([]) == 0
    assert len(store) == 0


def test_todo_write_tool_only_when_store_passed():
    assert "todo_write" not in [t.name for t in default_tools(".")]
    assert "todo_write" in [t.name for t in default_tools(".", todos=TodoStore())]


def test_todo_write_tool_updates_store_and_echoes():
    store = TodoStore()
    tools = registry(todos=store)

    result = tools.call(
        "todo_write",
        {
            "items": [
                {"content": "读代码", "status": "completed"},
                {"content": "修 bug", "status": "in_progress"},
            ]
        },
    )

    assert "(2 items)" in result
    assert "[1] [completed] 读代码" in result
    assert store.as_dicts()[1] == {"content": "修 bug", "status": "in_progress"}

    assert "TODO list cleared" in tools.call("todo_write", {"items": []})
    assert len(store) == 0


def test_todo_write_bad_input_returns_error_to_model():
    tools = registry(todos=TodoStore())
    result = tools.call("todo_write", {"items": [{"content": "x", "status": "wat"}]})
    assert "Error" in result and "invalid status" in result


def test_status_bar_renders_todos_with_raw_labels():
    snapshot = StatusSnapshot(
        iteration=1,
        max_steps=10,
        todos=[
            {"content": "读代码", "status": "completed"},
            {"content": "修 bug", "status": "in_progress"},
        ],
    )
    text = render_status(snapshot)
    assert "- TODO list:" in text
    assert "[1] [completed] 读代码" in text
    assert "[2] [in_progress] 修 bug" in text


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_delta=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        message = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_todo_write_flows_into_status_bar_end_to_end():
    """todo_write 写入的清单，下一轮状态栏（上下文末尾）必须能看到。"""
    store = TodoStore()
    llm = ScriptedLLM(
        [
            _message(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "todo_write",
                        '{"items": [{"content": "定位问题", "status": "in_progress"},'
                        ' {"content": "修复", "status": "pending"}]}',
                    )
                ]
            ),
            _message(content="清单已建"),
        ]
    )
    agent = Agent(llm=llm, tools=default_tools(".", todos=store), status_bar=True, todos=store)
    agent.run("修一下")

    second_status = llm.calls[1]["messages"][-1]["content"]
    assert "- TODO list:" in second_status
    assert "[1] [in_progress] 定位问题" in second_status
    assert "[2] [pending] 修复" in second_status

    agent.reset()
    assert len(store) == 0  # 清单随会话重置
