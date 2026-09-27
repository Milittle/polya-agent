"""两阶段模式（规划 → 执行）的测试：只读约束、计划提交结束本轮、缓存纪律。"""

from __future__ import annotations

from types import SimpleNamespace

from polya import Agent, tool


@tool
def add(a: int, b: int) -> str:
    """加法（测试用只读工具）。"""
    return str(a + b)


@tool(name="write_thing", kind="write")
def write_thing(content: str) -> str:
    """写点东西（测试用危险工具）。"""
    return f"已写入: {content}"


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_delta=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        message = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_exit_plan_mode_registered_only_in_plan_mode():
    plain = Agent(llm=ScriptedLLM([]), tools=[add])
    assert "exit_plan_mode" not in [t.name for t in plain.tools]

    planner = Agent(llm=ScriptedLLM([]), tools=[add], plan_mode=True)
    assert "exit_plan_mode" in [t.name for t in planner.tools]


def test_builtin_run_submits_plan_ends_turn_and_releases_mode():
    """内置驱动：提交计划即结束本轮并释放 plan 模式，返回计划文本。"""
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "write_thing", '{"content": "x"}')]),  # 被拒
            _message(tool_calls=[_call("c2", "exit_plan_mode", '{"plan": "三步走"}')]),  # 提交
            _message(content="这轮不该被请求"),  # run 在计划提交后结束，不该消费
        ]
    )
    agent = Agent(llm=llm, tools=[add, write_thing], plan_mode=True)

    assert agent.run("做事") == "三步走"
    assert agent.plan_mode is False  # 展示即释放执行
    denial = llm.calls[1]["messages"][-1]["content"]
    assert "规划模式" in denial and "exit_plan_mode" in denial
    # 工具数组全程不变（缓存纪律）：每次请求的 tools 完全一致
    assert llm.calls[0]["tools"] == llm.calls[1]["tools"]


def test_plan_mode_denies_write_but_allows_read():
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "write_thing", '{"content": "x"}')]),
            _message(tool_calls=[_call("c2", "add", '{"a": 1, "b": 2}')]),
            _message(content="完成"),
        ]
    )
    agent = Agent(llm=llm, tools=[add, write_thing], plan_mode=True)
    assert agent.run("做事") == "完成"
    assert "规划模式" in llm.calls[1]["messages"][-1]["content"]  # 写被拒
    assert llm.calls[2]["messages"][-1]["content"] == "3"  # 只读放行
    assert agent.plan_mode is True  # 未提交计划，保持规划模式


def test_leave_plan_mode_releases_writes():
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "write_thing", '{"content": "y"}')]),
            _message(content="完成"),
        ]
    )
    agent = Agent(llm=llm, tools=[write_thing], plan_mode=True)
    agent.leave_plan_mode()
    assert agent.run("做事") == "完成"
    assert llm.calls[1]["messages"][-1]["content"] == "已写入: y"


def test_plan_go_command_releases_plan_mode():
    from polya.commands import handle_command

    agent = Agent(llm=ScriptedLLM([]), tools=[add], plan_mode=True, plan_capable=True)
    assert "已批准" in handle_command("/plan go", agent)
    assert agent.plan_mode is False


def test_status_bar_shows_plan_mode():
    llm = ScriptedLLM([_message(content="好")])
    agent = Agent(llm=llm, tools=[add], plan_mode=True, status_bar=True)
    agent.run("hi")
    assert "规划" in llm.calls[0]["messages"][-1]["content"]

    llm2 = ScriptedLLM([_message(content="好")])
    agent2 = Agent(llm=llm2, tools=[add], status_bar=True)
    agent2.run("hi")
    assert "规划" not in llm2.calls[0]["messages"][-1]["content"]
