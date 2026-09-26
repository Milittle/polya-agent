"""常驻输入与后台任务的集成边界，不访问模型服务。"""

import asyncio
import threading
from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from polya import Agent, tool
from polya.input import InputBox, InputSuspended
from polya.loop import ApprovalOutcome, InteractiveSession, run_task
from polya.render import TerminalRenderer


def reply(content=None, calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=None,
    )


def call(name, identifier):
    return SimpleNamespace(id=identifier, function=SimpleNamespace(name=name, arguments="{}"))


class FakeLLM:
    model = "test-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, **kwargs):
        self.requests.append(list(messages))
        return next(self.responses)


async def until(condition):
    async def poll():
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), 3)


@contextmanager
def session_for(tmp_path, llm, tools=(), approve=None):
    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=DummyOutput())
        output = StringIO()
        renderer = TerminalRenderer(Console(file=output))
        renderer.use_scrollback(Console(file=output))
        agent = Agent(llm=llm, tools=tools, approve=approve, status_bar=False, prefix_check=True)
        session = InteractiveSession(agent, str(tmp_path), renderer, box)
        yield session, pipe, output


def test_slash_picker_uses_live_input_without_calling_model(tmp_path):
    llm = FakeLLM([])
    with session_for(tmp_path, llm) as (session, pipe, output):

        async def scenario():
            task = asyncio.create_task(session.run())
            await until(lambda: session.box._session.app.is_running)
            pipe.send_text("/plan\r")
            buffer = session.box._session.default_buffer
            await until(lambda: buffer.text == "/plan " and buffer.complete_state is not None)
            assert not session.agent.plan_mode
            pipe.send_text("\t")
            await until(lambda: buffer.text == "/plan on" and buffer.complete_state is None)
            assert not session.agent.plan_mode
            pipe.send_text("\r")
            await until(lambda: session.agent.plan_mode and not session.state["busy"])
            pipe.send_text("\x04")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        assert not llm.requests
        assert "已进入规划模式" in output.getvalue()


@pytest.mark.parametrize("command", ["/reset", "/clear", "/new", "/exit", "/quit"])
def test_task_end_command_holds_following_queue_entries(tmp_path, command):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.enqueue(command)
        session.enqueue("after command")
        assert session._pop(boundary=True) is None
        assert list(session.pending) == [command, "after command"]
        assert "当前任务结束后执行" in output.getvalue()
        assert session._pop() == command


def test_queued_permissions_apply_at_boundary_and_resume_reports_state(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.enqueue("/permissions all")
        assert not session.gate.allow_all
        session._boundary()
        assert session.gate.allow_all
        assert "下一轮请求前执行" in output.getvalue()
        session.queue_paused = True
        session.state["queue_paused"] = True
        session._local("/resume extra")
        assert session.queue_paused
        session._local("/resume")
        assert not session.queue_paused and not session.state["queue_paused"]
        assert "已恢复排队任务" in output.getvalue()


def test_resume_during_input_handoff_does_not_get_stuck_in_paused_queue(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.queue_paused = True
        session.state["queue_paused"] = True
        session.enqueue("/resume")
        assert not session.queue_paused and not session.pending


def test_invalid_reset_preserves_context_indicator(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.renderer._ctx_used = 100
        session.state["context"] = "ctx 10%"
        session._local("/clear extra")
        assert session.renderer._ctx_used == 100
        assert "context" in session.state


def test_clear_keeps_session_identity_and_queue(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.topic = "old-topic"
        session.gate.rules.append("rule")
        session.enqueue("later message")
        session._local("/clear")
        assert session.topic == "old-topic" and list(session.gate.rules) == ["rule"]
        assert list(session.pending) == ["later message"]
        assert not session.agent.history


def test_new_resets_topic_rules_queue_and_reprints_banner(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        titles = []
        session.renderer._console.set_window_title = titles.append
        session.agent.history.append({"role": "user", "content": "stale"})
        session.topic = "old-topic"
        session.gate.rules.append("rule")
        session.enqueue("later message")
        session._local("/new")
        assert not session.agent.history and session.topic is None
        assert not session.gate.rules and not session.pending
        assert session.state["queued"] == 0
        assert titles and titles[-1] == "polya"
        assert "已丢弃 1 条排队消息" in output.getvalue()
        assert "polya · v" in output.getvalue()  # 启动区重印


def test_busy_input_enters_next_request_after_entire_tool_batch(tmp_path):
    entered, release = threading.Event(), threading.Event()

    @tool
    def slow() -> str:
        """Hold the first tool while the user types."""
        entered.set()
        assert release.wait(3)
        return "first result"

    @tool
    def second() -> str:
        """Complete the same batch."""
        return "second result"

    llm = FakeLLM([reply(calls=[call("slow", "a"), call("second", "b")]), reply("done")])
    with session_for(tmp_path, llm, [slow, second]) as (session, pipe, output):

        async def scenario():
            task = asyncio.create_task(session.run())
            pipe.send_text("start\r")
            await until(entered.is_set)
            pipe.send_text("also check callers\r")
            await until(lambda: session.state.get("queued") == 1)
            release.set()
            await until(lambda: len(llm.requests) == 2 and not session.state["busy"])
            pipe.send_text("\x04")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        messages = llm.requests[1]
        assert [m["role"] for m in messages[-3:]] == ["tool", "tool", "user"]
        assert messages[-1]["content"] == "also check callers"
        assert "已排队" in output.getvalue()


def test_approval_releases_input_and_restores_folded_paste_and_cursor(tmp_path):
    @tool(kind="write")
    def change() -> str:
        """An operation requiring approval."""
        return "changed"

    llm = FakeLLM([reply(calls=[call("change", "a")]), reply("done")])
    with session_for(tmp_path, llm, [change], approve=lambda *_: False) as (session, pipe, _):
        original = "\n".join(f"line-{i}" for i in range(20))
        seen = []

        def approval(*args, **kwargs):
            seen.append(session.box._session.app.is_running)
            assert session.box._draft.text.endswith("[Pasted #1 +20 lines]")
            assert session.box._draft.cursor_position == 2
            return ApprovalOutcome(False)

        session._screen = approval

        async def scenario():
            task = asyncio.create_task(session.run())
            await until(lambda: session.box._session.app.is_running)
            pipe.send_text("draft \x1b[200~" + original + "\x1b[201~")
            await until(lambda: bool(session.box._tokens))
            session.box._session.default_buffer.cursor_position = 2
            # Start work while retaining the draft in the active prompt.
            session.state["busy"] = True
            work = asyncio.create_task(asyncio.to_thread(session._work, "start"))
            await asyncio.wait_for(work, 3)
            await until(lambda: session.box._session.app.is_running)
            buffer = session.box._session.default_buffer
            assert buffer.cursor_position == 2
            assert session.box._expand_pastes(buffer.text) == "draft " + original
            pipe.send_text("\x03\x03\x03")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        assert seen == [False]


def test_escape_preserves_draft_and_completed_tool_result(tmp_path):
    entered, release = threading.Event(), threading.Event()

    @tool
    def slow() -> str:
        """Wait for interrupt."""
        entered.set()
        assert release.wait(3)
        return "completed before stopping"

    llm = FakeLLM([reply(calls=[call("slow", "a")])])
    with session_for(tmp_path, llm, [slow]) as (session, pipe, _):

        async def scenario():
            task = asyncio.create_task(session.run())
            pipe.send_text("start\r")
            await until(entered.is_set)
            pipe.send_text("keep this draft\x1b")
            await until(session.stop.is_set)
            assert session.state["stopping"]
            release.set()
            await until(lambda: not session.state["busy"])
            assert session.box._session.default_buffer.text == "keep this draft"
            pipe.send_text("\x03\x03\x03")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        assert len(llm.requests) == 1
        assert session.agent.history[-1]["content"] == "completed before stopping"


def test_suspend_resumes_real_async_prompt(tmp_path):
    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=DummyOutput())

        async def scenario():
            prompt = asyncio.create_task(box.ask_async({}))
            pipe.send_text("hello")
            await until(lambda: box._session.default_buffer.text == "hello")
            box.suspend()
            with pytest.raises(InputSuspended):
                await prompt
            prompt = asyncio.create_task(box.ask_async({}))
            pipe.send_text(" world\r")
            assert await asyncio.wait_for(prompt, 3) == "hello world"

        asyncio.run(scenario())


def test_live_previews_commit_to_scrollback_once():
    output = StringIO()
    renderer = TerminalRenderer()
    renderer.use_scrollback(Console(file=output))
    with renderer:
        assert renderer._live is None
        renderer.update("text_delta", {"delta": "unique first\nlast fragment"})
        assert output.getvalue() == ""
        assert "last fragment" in renderer.preview(80)
        assert renderer.has_preview
        renderer.update("assistant_message", {"content": "unique first\nlast fragment"})
        assert not renderer.has_preview
        renderer.update("tool_call", {"name": "bash", "arguments": {"command": "echo hello"}})
        renderer.update("tool_output_delta", {"line": "unique tool output"})
        assert "unique tool output" in renderer.preview(80)
        assert "unique tool output" not in output.getvalue()
        renderer.update("tool_result", {"name": "bash", "result": "unique tool output"})
    assert output.getvalue().count("unique first") == 1
    assert output.getvalue().count("last fragment") == 1
    assert output.getvalue().count("unique tool output") == 1
    assert "unique tool output" in renderer.expand_blocks()
    assert "Running Bash" not in output.getvalue()
    assert "Ran Bash  $ echo hello" in output.getvalue()
    assert not renderer.has_preview


def test_interrupt_backfills_unexecuted_tools():
    stop = threading.Event()

    @tool
    def first() -> str:
        """Request stop during execution."""
        stop.set()
        return "actual result"

    @tool
    def second() -> str:
        """Must not run."""
        raise AssertionError("second tool ran after interrupt")

    agent = Agent(
        llm=FakeLLM([reply(calls=[call("first", "a"), call("second", "b")])]),
        tools=[first, second],
        status_bar=False,
    )
    with pytest.raises(InterruptedError):
        run_task(agent, TerminalRenderer(Console(file=StringIO())), "go", True, stop=stop)
    results = [m for m in agent.history if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["a", "b"]
    assert results[0]["content"] == "actual result"
    assert "Error" in results[1]["content"]


def test_stop_after_tool_declaration_backfills_before_first_execution():
    stop = threading.Event()
    agent = Agent(
        llm=FakeLLM([reply(calls=[call("missing", "a")])]),
        tools=[],
        status_bar=False,
    )
    renderer = TerminalRenderer(Console(file=StringIO()))
    update = renderer.update

    def stop_on_declaration(event, payload):
        update(event, payload)
        if event == "assistant_message":
            stop.set()

    renderer.update = stop_on_declaration
    with pytest.raises(InterruptedError):
        run_task(agent, renderer, "go", True, stop=stop)
    assert agent.history[-1]["role"] == "tool"
    assert agent.history[-1]["tool_call_id"] == "a"


def test_narrow_status_preserves_mode_and_interrupt(tmp_path):
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.utils import get_cwidth

    class NarrowOutput(DummyOutput):
        def get_size(self):
            return Size(rows=12, columns=40)

    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=NarrowOutput())
        box._state = {"busy": True, "mode": "plan", "model": "very-long-model-name"}
        text = "".join(fragment for _, fragment in box._bottom_bar())
        assert get_cwidth(text) <= 40
        assert "esc to interrupt" in "".join(t for _, t in box._working_bar())
        assert "plan" in text
        assert "very-long-model-name" not in text


def test_reset_waits_for_task_end_and_preserves_queue_order(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.agent.history.append({"role": "user", "content": "original"})
        for text in ("first", "/reset", "after reset"):
            session.enqueue(text)
        session._boundary()
        assert session.agent.history[-1]["content"] == "first"
        assert list(session.pending) == ["/reset", "after reset"]
        assert session._local(session._pop())
        assert session.agent.history == []
        assert session._pop() == "after reset"


def test_streamed_shell_failure_shows_exit_code():
    output = StringIO()
    renderer = TerminalRenderer()
    renderer.use_scrollback(Console(file=output))
    renderer.update("tool_call", {"name": "bash"})
    renderer.update("tool_output_delta", {"line": "failed check"})
    renderer.update("tool_result", {"name": "bash", "result": "failed check\n退出码 2"})
    assert "Failed Bash" in output.getvalue()
    assert "exit 2" in output.getvalue()
    assert "Ran Bash" not in output.getvalue()


def test_task_timer_survives_tool_events_and_resets_for_new_task(tmp_path, monkeypatch):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        monkeypatch.setattr("polya.loop.time.monotonic", lambda: 100.0)
        session._start_task()
        assert session.state["started_at"] == 100.0
        monkeypatch.setattr("polya.loop.time.monotonic", lambda: 120.0)
        session.renderer.update("tool_call", {"name": "bash"})
        session.renderer.update("tool_result", {"name": "bash", "result": "done"})
        assert session.state["started_at"] == 100.0
        session._start_task()
        assert session.state["started_at"] == 120.0


def test_session_identity_rename_clear_and_new(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        assert session.state["project"] == str(tmp_path)
        session._local("/rename 修复输入框")
        assert session.topic == session.state["topic"] == "修复输入框"
        session._local("/clear")
        assert session.state["topic"] == "修复输入框"
        session._local("/rename")
        assert session.topic == "修复输入框"
        session._local("/rename bad\x1btitle")
        assert session.topic == "修复输入框"
        session._local("/new")
        assert session.topic is None and session.state["topic"] is None
        assert session.state["project"] == str(tmp_path)
        assert "已更新会话主题" in output.getvalue()


def test_interruption_pauses_queue_until_explicit_resume(tmp_path):
    entered, release = threading.Event(), threading.Event()

    @tool
    def slow() -> str:
        """Hold a running operation while the user requests interruption."""
        entered.set()
        assert release.wait(3)
        return "done"

    llm = FakeLLM([reply(calls=[call("slow", "a")]), reply("resumed")])
    with session_for(tmp_path, llm, [slow]) as (session, pipe, output):

        async def scenario():
            task = asyncio.create_task(session.run())
            pipe.send_text("start\r")
            await until(entered.is_set)
            session.enqueue("later")
            session.interrupt()
            assert session.queue_paused and session.state["stopping"]
            assert "正在停止" in session._resume_queue()
            assert session.queue_paused
            release.set()
            await until(lambda: not session.state["busy"])
            assert list(session.pending) == ["later"]
            assert len(llm.requests) == 1
            pipe.send_text("/resume\r")
            await until(lambda: len(llm.requests) == 2 and not session.state["busy"])
            pipe.send_text("\x04")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        assert "本轮已中断" in output.getvalue()
        assert "本轮结束" in output.getvalue()
        assert "任务成功" not in output.getvalue()


def test_task_failure_pauses_pending_messages(tmp_path):
    class BrokenLLM(FakeLLM):
        def chat(self, messages, **kwargs):
            session.enqueue("later")
            raise ConnectionError("connection lost")

    with session_for(tmp_path, BrokenLLM([])) as (session, _, output):
        session._start_task()
        session._work("fail")
        assert session.queue_paused and list(session.pending) == ["later"]
        assert "本轮失败" in output.getvalue()


def test_queued_supplement_receives_delivery_receipt(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.enqueue("keep the interface")
        assert "下一轮请求前交给模型" in output.getvalue()
        session._boundary()
        assert session.agent.history[-1]["content"] == "keep the interface"
        assert "补充已交给模型" in output.getvalue()
