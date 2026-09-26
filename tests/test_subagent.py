"""子代理（context-engineering 票 03 / 07）：隔离、深度 1、共享审批、用量并入。"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from polya import Agent
from polya.approval import ApprovalGate
from polya.loop import run_tool_call
from polya.permissions import Rule
from polya.subagent import SubagentRunner
from polya.tools import ToolRegistry


def resp(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


def tc(cid, name, args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))


def usage(p=0, c=0, cached=0):
    details = SimpleNamespace(cached_tokens=cached) if cached else None
    return SimpleNamespace(
        prompt_tokens=p, completion_tokens=c, total_tokens=p + c, prompt_tokens_details=details
    )


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_delta=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._replies.pop(0)


def _runner(tmp_path, llm, **kwargs) -> tuple[Agent, SubagentRunner]:
    parent = Agent(llm=llm, tools=[])
    runner = SubagentRunner(root=str(tmp_path), **kwargs)
    runner.attach(parent)
    return parent, runner


def test_child_writes_file_and_report_returns(tmp_path):
    llm = ScriptedLLM(
        [
            resp(tool_calls=[tc("c1", "write_file", '{"path": "out.txt", "content": "hi"}')]),
            resp(content="报告：已写 out.txt"),
        ]
    )
    parent, runner = _runner(tmp_path, llm, max_steps=5)
    report = runner.run("写文件", "写 out.txt")
    assert "报告：已写 out.txt" in report
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi"
    assert parent.subagent is runner


def test_child_write_denied_in_plan_mode_but_finishes(tmp_path):
    llm = ScriptedLLM(
        [
            resp(tool_calls=[tc("c1", "write_file", '{"path": "out.txt", "content": "hi"}')]),
            resp(content="报告：写操作被拒绝"),
        ]
    )
    parent, runner = _runner(tmp_path, llm, max_steps=5)
    parent.plan_mode = True
    report = runner.run("写文件", "写 out.txt")
    assert "报告" in report  # 子不崩，以报告收尾
    assert not (tmp_path / "out.txt").exists()


def test_child_tools_have_history_read_but_no_task(tmp_path):
    llm = ScriptedLLM([resp(content="完成")])
    _, runner = _runner(tmp_path, llm)
    runner.run("探查", "随便看看")
    child = runner.last_child
    assert child is not None
    assert child.tools.get("task") is None  # 深度 1
    assert child.tools.get("history_read") is not None  # 子压缩启用


def test_child_usage_merged_into_parent(tmp_path):
    llm = ScriptedLLM([resp(content="完成", usage=usage(p=100, c=10, cached=40))])
    parent, runner = _runner(tmp_path, llm)
    runner.run("探查", "做点事")
    assert parent.total_usage["prompt_tokens"] == 100
    assert parent.total_usage["completion_tokens"] == 10
    assert parent.total_usage["cached_tokens"] == 40
    assert parent.last_usage is None  # 父 last_usage 不被污染（压缩触发不受影响）


def test_stop_interrupts_child_and_keeps_tool_pairing(tmp_path):
    stop = threading.Event()

    class StopLLM:
        def __init__(self):
            self.n = 0

        def chat(self, messages, tools=None, on_delta=None):
            self.n += 1
            if self.n == 1:
                stop.set()
                return resp(tool_calls=[tc("c1", "list_dir", '{"path": "."}')])
            return resp(content="done")

    parent, runner = _runner(tmp_path, StopLLM())
    runner.stop = stop
    report = runner.run("探查", "列目录")
    assert "中断" in report
    child = runner.last_child
    declared = {
        call["id"]
        for message in child.history
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    answered = {
        message["tool_call_id"] for message in child.history if message.get("role") == "tool"
    }
    assert declared <= answered  # 中断后 tool_call_id 仍全部回填


def test_child_compacts_midrun(tmp_path):
    """票 07 变体 1：子中途真实压缩，父历史不受扰，tool 配对完整。"""

    class CompactLLM:
        def __init__(self):
            self.normal = 0

        def chat(self, messages, tools=None, on_delta=None):
            if tools is None:  # 压缩摘要调用
                return resp(content="#2: 已完成目录探查（原始输出已折叠）")
            self.normal += 1
            if self.normal == 1:
                return resp(
                    tool_calls=[tc("c1", "list_dir", '{"path": "."}')],
                    usage=usage(p=8100),
                )
            return resp(content="报告：完成", usage=usage(p=50))

    parent = Agent(llm=CompactLLM(), tools=[], context_window=10000, keep_recent=0)
    runner = SubagentRunner(root=str(tmp_path))
    runner.attach(parent)
    report = runner.run("探查", "列目录并总结")
    assert "报告：完成" in report
    child = runner.last_child
    assert any("[COMPRESSED]" in (m.get("content") or "") for m in child.history)
    declared = {
        call["id"]
        for message in child.history
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    answered = {
        message["tool_call_id"] for message in child.history if message.get("role") == "tool"
    }
    assert declared <= answered
    assert parent.history == [] or all(m.get("role") != "tool" for m in parent.history)
    assert parent.total_usage["prompt_tokens"] >= 8100  # 子用量已并入


def test_result_truncated_to_limit(tmp_path):
    llm = ScriptedLLM([resp(content="x" * 500)])
    _, runner = _runner(tmp_path, llm, result_limit=50)
    report = runner.run("汇报", "写长报告")
    assert "已截断" in report


def test_max_steps_exceeded_returns_error(tmp_path):
    llm = ScriptedLLM([resp(tool_calls=[tc("c1", "list_dir", '{"path": "."}')]) for _ in range(6)])
    _, runner = _runner(tmp_path, llm, max_steps=2)
    report = runner.run("探查", "一直列目录")
    assert "Error" in report


# ---------- 票 07：集成场景（父调 task，子真实干活，报告回父历史） ----------


def test_integration_parent_delegates_and_child_works(tmp_path):
    class RoutingLLM:
        """父子共用同一实例：按请求 tools 是否含 task 分派剧本（锁「同实例串行」）。"""

        def __init__(self):
            self.parent = [
                resp(
                    tool_calls=[
                        tc("t1", "task", '{"description": "写文件", "prompt": "写 out.txt"}')
                    ]
                ),
                resp(content="父层已收到子报告"),
            ]
            self.child = [
                resp(tool_calls=[tc("c1", "write_file", '{"path": "out.txt", "content": "hi"}')]),
                resp(content="报告：已写 out.txt"),
            ]

        def chat(self, messages, tools=None, on_delta=None):
            names = {t["function"]["name"] for t in (tools or [])}
            return (self.parent if "task" in names else self.child).pop(0)

    llm = RoutingLLM()
    parent = Agent(llm=llm, tools=[])
    runner = SubagentRunner(root=str(tmp_path))
    runner.attach(parent)
    # 父级注册 task 工具（真实 registry）
    parent.tools.add(runner.task_tool())
    gate = ApprovalGate(interactive=False)
    gate.allow_all = True
    runner.bind(gate, None, False)
    runner.dispatch = lambda child, ev: run_tool_call(
        child, None, ev, False, gate, unrestricted=True, origin="子任务", progress=None
    )

    # 迷你父驱动：ToolCall 经 run_tool_call 执行（task 工具从此进）
    gen = parent.steps("让子代理写文件")
    to_send = None
    answer = None
    while True:
        try:
            ev = gen.send(to_send)
        except StopIteration as stop:
            answer = stop.value
            break
        if ev.event == "tool_call":
            to_send = run_tool_call(
                parent, None, ev, False, gate, unrestricted=True, origin=None, progress=None
            )
        else:
            to_send = None

    assert answer == "父层已收到子报告"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi"
    tool_messages = [m for m in parent.history if m.get("role") == "tool"]
    assert any("报告：已写 out.txt" in m["content"] for m in tool_messages)


def test_session_rule_applies_to_child_without_prompt(tmp_path):
    llm = ScriptedLLM(
        [
            resp(tool_calls=[tc("c1", "bash", '{"command": "cat x"}')]),
            resp(content="完成"),
        ]
    )
    parent, runner = _runner(tmp_path, llm)
    gate = ApprovalGate(interactive=False)
    gate.rules.append(Rule("bash", "cat"))  # 会话规则对子生效
    runner.bind(gate, None, False)

    prompted = []
    gate.screen = lambda *a, **k: prompted.append(True)  # type: ignore[method-assign]
    runner.dispatch = lambda child, ev: run_tool_call(
        child, None, ev, False, gate, unrestricted=False, origin="子任务", progress=None
    )
    (tmp_path / "x").write_text("content", encoding="utf-8")
    runner.run("读文件", "cat x")
    assert prompted == []  # 命中规则，未弹审批
    assert gate.rejected_reason is None


def test_task_tool_without_attach_reports_error():
    runner = SubagentRunner(root=".")
    registry = ToolRegistry([runner.task_tool()])
    result = registry.call("task", {"description": "d", "prompt": "p"})
    assert result.startswith("Error:") and "attach" in result
