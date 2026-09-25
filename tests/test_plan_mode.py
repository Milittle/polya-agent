"""两阶段模式（规划 → 执行）的测试：只读约束、计划审批、缓存纪律。"""

from __future__ import annotations

from types import SimpleNamespace

from mi_z import Agent, tool


@tool
def add(a: int, b: int) -> str:
    """加法（测试用只读工具）。"""
    return str(a + b)


@tool(name="write_thing", dangerous=True)
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

    def chat(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        message = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_exit_plan_mode_registered_only_in_plan_mode():
    plain = Agent(llm=ScriptedLLM([]), tools=[add])
    assert "exit_plan_mode" not in [t.name for t in plain.tools]

    planner = Agent(llm=ScriptedLLM([]), tools=[add], plan_mode=True)
    assert "exit_plan_mode" in [t.name for t in planner.tools]


def test_full_cycle_deny_submit_approve_execute():
    """规划模式拒写 → 提交计划（默认自动批准）→ 写操作放行。"""
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "write_thing", '{"content": "x"}')]),  # 被拒
            _message(tool_calls=[_call("c2", "exit_plan_mode", '{"plan": "三步走"}')]),  # 批准
            _message(tool_calls=[_call("c3", "write_thing", '{"content": "y"}')]),  # 放行
            _message(content="完成"),
        ]
    )
    agent = Agent(llm=llm, tools=[add, write_thing], plan_mode=True)

    assert agent.run("做事") == "完成"

    denial = llm.calls[1]["messages"][-1]["content"]  # 第一轮的 tool 回填
    assert "规划模式" in denial and "exit_plan_mode" in denial

    approval = llm.calls[2]["messages"][-1]["content"]  # 第二轮的 tool 回填
    assert "已批准" in approval and "执行模式" in approval

    executed = llm.calls[3]["messages"][-1]["content"]  # 第三轮的 tool 回填
    assert executed == "已写入: y"
    assert agent.plan_mode is False

    # 工具数组全程不变（缓存纪律）：每次请求的 tools 完全一致
    assert llm.calls[0]["tools"] == llm.calls[1]["tools"] == llm.calls[2]["tools"]


def test_plan_rejection_keeps_readonly():
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "exit_plan_mode", '{"plan": "烂计划"}')]),
            _message(tool_calls=[_call("c2", "write_thing", '{"content": "x"}')]),  # 仍被拒
            _message(content="好吧"),
        ]
    )
    agent = Agent(llm=llm, tools=[write_thing], plan_mode=True, approve_plan=lambda plan: False)
    agent.run("做事")

    rejection = llm.calls[1]["messages"][-1]["content"]
    assert "被拒绝" in rejection
    assert agent.plan_mode is True  # 仍在规划模式
    still_denied = llm.calls[2]["messages"][-1]["content"]
    assert "规划模式" in still_denied


def test_readonly_tools_work_in_plan_mode():
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "add", '{"a": 1, "b": 2}')]),
            _message(content="3"),
        ]
    )
    agent = Agent(llm=llm, tools=[add, write_thing], plan_mode=True)
    assert agent.run("算数") == "3"
    assert llm.calls[1]["messages"][-1]["content"] == "3"  # 只读工具未被拦截


def test_status_bar_shows_plan_mode():
    llm = ScriptedLLM([_message(content="好")])
    agent = Agent(llm=llm, tools=[add], plan_mode=True, status_bar=True)
    agent.run("hi")
    assert "规划" in llm.calls[0]["messages"][-1]["content"]

    llm2 = ScriptedLLM([_message(content="好")])
    agent2 = Agent(llm=llm2, tools=[add], status_bar=True)
    agent2.run("hi")
    assert "规划" not in llm2.calls[0]["messages"][-1]["content"]


def test_exit_plan_mode_after_execution_is_noop():
    llm = ScriptedLLM(
        [
            _message(tool_calls=[_call("c1", "exit_plan_mode", '{"plan": "p"}')]),
            _message(tool_calls=[_call("c2", "exit_plan_mode", '{"plan": "又交"}')]),
            _message(content="ok"),
        ]
    )
    agent = Agent(llm=llm, tools=[add], plan_mode=True)
    agent.run("做事")
    noop = llm.calls[2]["messages"][-1]["content"]
    assert "已处于执行模式" in noop
