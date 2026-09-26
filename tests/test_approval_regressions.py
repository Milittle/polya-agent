"""用户反馈：拒绝即停、会话全授权和默认单行输入。"""

import asyncio
from io import StringIO

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from test_interactive import FakeLLM, call, reply, until

from polya import Agent, tool
from polya.input import InputBox
from polya.loop import ApprovalGate, ApprovalOutcome, run_task
from polya.render import TerminalRenderer


def test_reject_stops_remaining_tools_and_model_requests():
    executed = []

    @tool(kind="exec")
    def command() -> str:
        """Command awaiting approval."""
        executed.append("command")
        return "done"

    @tool
    def later() -> str:
        """A later read should also stop after rejection."""
        executed.append("later")
        return "done"

    llm = FakeLLM([reply(calls=[call("command", "a"), call("later", "b")]), reply("continued")])
    agent = Agent(llm=llm, tools=[command, later], approve=lambda *_: False, status_bar=False)
    gate = ApprovalGate(True)
    gate.screen = lambda *a, **kw: ApprovalOutcome(False, reason="stop")
    with pytest.raises(InterruptedError):
        run_task(agent, TerminalRenderer(Console(file=StringIO())), "go", True, gate)
    assert executed == []
    assert len(llm.requests) == 1
    results = [m for m in agent.history if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["a", "b"]
    assert "stop" in results[0]["content"]


def test_approval_offers_session_wide_allow(monkeypatch):
    @tool(kind="exec")
    def bash(command: str) -> str:
        """Run a command."""
        return ""

    def select(options, **kwargs):
        labels = [label for label, _ in options]
        assert "Allow all for session" in labels
        return labels.index("Allow all for session")

    monkeypatch.setattr("polya.approval._select_option", select)
    gate = ApprovalGate(True)
    outcome = gate.screen(bash, {"command": "cd src && pytest"})
    assert outcome.approved
    assert outcome.allow_all


def test_empty_input_renders_one_row(tmp_path):
    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=DummyOutput())

        async def scenario():
            task = asyncio.create_task(box.ask_async({}))
            window = box._session.layout.current_window
            await until(lambda: window.render_info is not None)
            try:
                assert window.render_info.window_height == 1
            finally:
                pipe.send_text("\x04")
                with pytest.raises(EOFError):
                    await task

        asyncio.run(scenario())


def test_all_authorization_covers_different_tools_but_not_high_risk():
    from polya.agent import ToolCall
    from polya.loop import _run_tool

    executed = []

    @tool(kind="exec")
    def bash(command: str) -> str:
        """Fake shell."""
        executed.append(command)
        return "ok"

    @tool(kind="write")
    def write(path: str) -> str:
        """Fake write."""
        executed.append(path)
        return "ok"

    agent = Agent(llm=FakeLLM([]), tools=[bash, write], approve=lambda *_: False)
    gate = ApprovalGate(True)
    prompts = []

    def screen(tool, arguments, high_risk=False):
        prompts.append((tool.name, high_risk))
        return ApprovalOutcome(not high_risk, allow_all=not high_risk)

    gate.screen = screen
    renderer = TerminalRenderer(Console(file=StringIO()))
    for i, (name, arguments) in enumerate(
        [
            ("bash", {"command": "pytest tests"}),
            ("bash", {"command": "cd src && ls"}),
            ("write", {"path": "app.py"}),
            ("bash", {"command": "sudo something"}),
        ]
    ):
        _run_tool(
            agent, renderer, ToolCall(name=name, call_id=str(i), arguments=arguments), True, gate
        )
    assert prompts == [("bash", False), ("bash", True)]
    assert executed == ["pytest tests", "cd src && ls", "app.py"]


def test_ask_without_terminal_never_executes():
    from polya.agent import ToolCall
    from polya.loop import _run_tool

    @tool(kind="exec")
    def command() -> str:
        """Must not run without authorization."""
        raise AssertionError("executed without permission")

    agent = Agent(llm=FakeLLM([]), tools=[command], approve=lambda *_: False)
    result = _run_tool(
        agent,
        TerminalRenderer(Console(file=StringIO())),
        ToolCall(name="command", call_id="a", arguments={}),
        False,
        ApprovalGate(False),
    )
    assert "默认拒绝" in result


def test_cancel_command_edit_does_not_restore_original_command(monkeypatch):
    from polya.approval import _ask_line

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("prompt_toolkit.prompt", cancel)
    assert _ask_line("edit", default="original command") == ""


def test_input_grows_with_content_then_shrinks(tmp_path):
    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=DummyOutput())

        async def scenario():
            task = asyncio.create_task(box.ask_async({}))
            window = box._session.layout.current_window
            await until(lambda: window.render_info is not None)
            buffer = box._session.default_buffer
            for text, rows in [("a\nb\nc", 3), ("\n".join("x" for _ in range(9)), 6), ("", 1)]:
                buffer.text = text
                box._session.app.invalidate()
                await until(lambda rows=rows: window.render_info.window_height == rows)
            pipe.send_text("\x04")
            with pytest.raises(EOFError):
                await task

        asyncio.run(scenario())


def test_rejection_pauses_queue_but_allows_new_instruction(tmp_path):
    from test_interactive import session_for

    @tool(kind="exec")
    def command() -> str:
        """Should be denied."""
        raise AssertionError("denied command executed")

    llm = FakeLLM([reply(calls=[call("command", "a")]), reply("new task done")])
    with session_for(tmp_path, llm, [command], approve=lambda *_: False) as (session, _, _):

        def deny(*args, **kwargs):
            session.enqueue("old queued task")
            return ApprovalOutcome(False)

        session.gate.screen = deny
        session._work("first task")
        assert session.queue_paused
        assert list(session.pending) == ["old queued task"]
        session._work("new instruction")
        assert len(llm.requests) == 2
        assert llm.requests[-1][-1]["content"] == "new instruction"
        assert list(session.pending) == ["old queued task"]
        session._local("/permissions all")
        assert session.gate.allow_all and session.state["mode"] == "yolo"
        session._local("/permissions ask")
        assert not session.gate.allow_all and session.state["mode"] == "normal"
