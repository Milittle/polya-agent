"""常驻输入与后台任务的集成边界，不访问模型服务。"""

import asyncio
import threading
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from polya import Agent, tool
from polya.input import InputBox, InputSuspended
from polya.loop import InteractiveSession, run_task
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
def session_for(tmp_path, llm, tools=(), **agent_kwargs):
    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=DummyOutput())
        output = StringIO()
        renderer = TerminalRenderer(Console(file=output))
        renderer.use_scrollback(Console(file=output))
        agent = Agent(llm=llm, tools=tools, status_bar=False, prefix_check=True, **agent_kwargs)
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


def test_tick_repaints_only_while_busy(tmp_path):
    """空窗期（思考/长工具无事件）计时靠周期重绘，空闲时不重绘。"""
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.tick_interval = 0.01
        calls: list[int] = []
        session.box._session.app.invalidate = lambda: calls.append(1)

        async def scenario():
            session.state["busy"] = False
            ticker = asyncio.create_task(session._tick())
            await asyncio.sleep(0.04)
            assert calls == []  # 空闲不重绘
            session.state["busy"] = True
            await until(lambda: len(calls) >= 3)
            session.closing = True
            await asyncio.wait_for(ticker, 1)

        asyncio.run(scenario())


@pytest.mark.parametrize("command", ["/reset", "/clear", "/new", "/exit", "/quit"])
def test_idle_command_refused_while_busy(tmp_path, command):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.state["busy"] = True
        session._submit_input(command)
        assert "先按 Esc 中断" in output.getvalue()
        assert not session.closing  # /exit 未被误执行


def test_queued_message_applies_at_boundary(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.state["busy"] = True
        session._submit_input("补充一句")
        assert "下一轮请求前交给模型" in output.getvalue()
        session._boundary()
        assert session.agent.history[-1]["content"] == "补充一句"
        assert "补充已交给模型" in output.getvalue()


def test_idle_command_borrows_terminal_without_freezing_the_loop(tmp_path, monkeypatch):
    """空闲命令必须在 worker 线程执行。

    若命令在主循环线程执行，`_borrow_terminal` 的 `call_soon_threadsafe` +
    `ready.wait()` 会与主循环互相死等，整个 TUI 冻结（/login 卡死回归）。
    """
    from polya import commands as commands_mod
    from polya.commands import Command

    handler_thread: list[int | None] = [None]
    borrowed = threading.Event()

    def handler(ctx, arg):
        handler_thread[0] = threading.get_ident()

        def work():
            borrowed.set()
            return "borrowed ok"

        return ctx.in_terminal(work)

    monkeypatch.setitem(commands_mod.BY_NAME, "/testterm", Command("/testterm", "借道", handler))

    with session_for(tmp_path, FakeLLM([])) as (session, pipe, output):
        loop_thread = None

        async def scenario():
            nonlocal loop_thread
            loop_thread = threading.get_ident()
            task = asyncio.create_task(session.run())
            await until(lambda: session.box._session.app.is_running)
            pipe.send_text("/testterm\r")
            await until(lambda: borrowed.is_set())
            await until(lambda: "borrowed ok" in output.getvalue())
            assert handler_thread[0] is not None
            assert handler_thread[0] != loop_thread
            pipe.send_text("\x04")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())


def test_follow_up_queues_separately_from_steering(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.state["busy"] = True
        session.box.last_kind = "follow-up"
        session._submit_input("随后再处理")
        assert list(session.follow_up) == ["随后再处理"] and not session.steering
        assert "本轮结束后交给模型" in output.getvalue()


def test_dequeue_moves_queue_back_to_editor(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.state["busy"] = True
        session._submit_input("第一条")
        session.box.last_kind = "follow-up"
        session._submit_input("第二条")
        text = session._dequeue_to_editor()
        assert text == "第一条\n第二条"
        assert not session.steering and not session.follow_up
        assert session.state["queued"] == 0


def test_invalid_reset_preserves_context_indicator(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.renderer._ctx_used = 100
        session.state["context_pct"] = 10
        session._local("/clear extra")
        assert session.renderer._ctx_used == 100
        assert "context_pct" in session.state


def test_status_computes_uncached_input_and_cost(tmp_path):
    llm = FakeLLM([])
    llm.model = "deepseek-flash"
    with session_for(tmp_path, llm) as (session, _, _):
        session.renderer.total_usage = {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 500_000,
            "cached_tokens": 800_000,
        }
        session._status(session.renderer)
        # pi 语义：↑input 扣掉缓存读取
        assert session.state["input_tokens"] == 200_000
        assert session.state["cached_tokens"] == 800_000
        # 0.2*0.3 + 0.5*1.2 + 0.8*0.006 = 0.06 + 0.6 + 0.0048
        assert session.state["cost"] == pytest.approx(0.6648)
        assert session.state["subscribed"] is False


def test_clear_alias_starts_fresh_session_and_drops_queue(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.agent.history.append({"role": "user", "content": "stale"})
        session.topic = "old-topic"
        session.state["busy"] = True
        session._submit_input("later message")
        session.state["busy"] = False
        session._local("/clear")
        assert not session.agent.history and session.topic is None
        assert not session.steering and not session.follow_up
        assert session.state["queued"] == 0
        assert "已开始新会话" in output.getvalue()
        assert "已丢弃 1 条排队消息" in output.getvalue()


def test_new_resets_topic_queue_and_reprints_banner(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        titles = []
        session.renderer._console.set_window_title = titles.append
        session.agent.history.append({"role": "user", "content": "stale"})
        session.topic = "old-topic"
        session.state["busy"] = True
        session._submit_input("later message")
        session.state["busy"] = False
        session._local("/new")
        assert not session.agent.history and session.topic is None
        assert not session.steering and not session.follow_up
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
        run_task(agent, TerminalRenderer(Console(file=StringIO())), "go", stop=stop)
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
        run_task(agent, renderer, "go", stop=stop)
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


def test_boundary_injects_steering_in_order(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        session.agent.history.append({"role": "user", "content": "original"})
        session.state["busy"] = True
        for text in ("first", "second"):
            session._submit_input(text)
        session._boundary()
        assert [m["content"] for m in session.agent.history[-2:]] == ["first", "second"]
        assert not session.steering


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


def test_session_identity_rename_and_new_session(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        assert session.state["project"] == str(tmp_path)
        session._local("/rename 修复输入框")
        assert session.topic == session.state["topic"] == "修复输入框"
        session._local("/rename")
        assert session.topic == "修复输入框"
        session._local("/rename bad\x1btitle")
        assert session.topic == "修复输入框"
        session._local("/clear")  # /clear 现同为开新会话：主题重置
        assert session.topic is None and session.state["topic"] is None
        assert session.state["project"] == str(tmp_path)
        assert "已更新会话主题" in output.getvalue()


def test_first_task_topic_is_persisted_as_session_title(tmp_path, monkeypatch):
    """自动主题写进会话元数据：autosave 落盘后 /resume 看得到主题，而非只剩时间戳名。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with session_for(tmp_path, FakeLLM([reply(content="done")])) as (session, _, _):
        session._work("修复登录页面")
        assert session.topic == session.agent.session_title == "修复登录页面"
        saved = tmp_path / ".polya" / "sessions" / f"{session.agent.session_name}.jsonl"
        assert '"title": "修复登录页面"' in saved.read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["/new", "/clear", "/reset"])
def test_session_commands_wipe_screen_before_welcome(tmp_path, command):
    """/new /clear /reset 清屏后重印启动区：终端像刚启动的新会话。"""
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        # 清屏走 rich Control.clear()，只在终端上输出 ANSI：测试用强制终端捕获。
        session.renderer._console = Console(file=output, force_terminal=True)
        output.write("旧输出\n")
        session._local(command)
        wiped = output.getvalue()
        assert "\x1b[2J" in wiped  # rich Control.clear()：擦全屏 + 光标归位
        assert wiped.index("\x1b[2J") < wiped.index("polya · v")  # 先清屏，再印 banner
        assert "和你一起理解" in wiped
        assert "旧输出" not in wiped[wiped.index("\x1b[2J") :]  # 旧内容只留在清屏之前


def test_interrupt_returns_queued_messages_to_editor(tmp_path):
    entered, release = threading.Event(), threading.Event()

    @tool
    def slow() -> str:
        """Hold a running operation while the user requests interruption."""
        entered.set()
        assert release.wait(3)
        return "done"

    llm = FakeLLM([reply(calls=[call("slow", "a")])])
    with session_for(tmp_path, llm, [slow]) as (session, pipe, output):

        async def scenario():
            task = asyncio.create_task(session.run())
            pipe.send_text("start\r")
            await until(entered.is_set)
            pipe.send_text("later\r")  # 忙时 → steering 队列
            await until(lambda: session.state.get("queued") == 1)
            session.interrupt()
            assert session.state["stopping"]
            release.set()
            await until(lambda: not session.state["busy"])
            # Esc 中断后，排队消息回到编辑器，不静默丢弃
            await until(lambda: "later" in session.box._session.default_buffer.text)
            assert not session.steering
            assert len(llm.requests) == 1
            pipe.send_text("\x03\x03\x03")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        assert "本轮已中断" in output.getvalue()


def test_task_failure_keeps_queue(tmp_path):
    class BrokenLLM(FakeLLM):
        def chat(self, messages, **kwargs):
            session.state["busy"] = True
            session._submit_input("later")
            raise ConnectionError("connection lost")

    with session_for(tmp_path, BrokenLLM([])) as (session, _, output):
        session._start_task()
        session._work("fail")
        assert list(session.steering) == ["later"]
        assert "本轮失败" in output.getvalue()


def test_queued_supplement_receives_delivery_receipt(tmp_path):
    with session_for(tmp_path, FakeLLM([])) as (session, _, output):
        session.state["busy"] = True
        session._submit_input("keep the interface")
        assert "下一轮请求前交给模型" in output.getvalue()
        session._boundary()
        assert session.agent.history[-1]["content"] == "keep the interface"
        assert "补充已交给模型" in output.getvalue()


def test_ctrl_s_callback_sets_default_model(tmp_path, monkeypatch):
    from polya.models import ModelEntry, ModelsConfig, ProviderEntry

    path = tmp_path / "models.json"
    config = ModelsConfig()
    config.add(
        "p",
        ProviderEntry(
            "https://x.example/v1",
            "sk-x-12345678",
            "m-a",
            [ModelEntry("m-a"), ModelEntry("m-b")],
        ),
    )
    config.save(path)
    monkeypatch.setattr("polya.models.default_path", lambda: path)

    with session_for(tmp_path, FakeLLM([])) as (session, _, _):
        assert session.box.on_set_default.__self__ is session  # 驱动已接线
        message = session.box.on_set_default("p/m-b")
        assert "已设为默认启动模型" in message
        assert ModelsConfig.load().active == "p/m-b"


# ---------- 票 02：预算检查点续跑 ----------


@tool
def _noop() -> str:
    """No-op tool for budget tests."""
    return "ok"


def test_run_task_budget_checkpoint_continues_to_final():
    llm = FakeLLM(
        [
            reply(calls=[call("_noop", "c1")]),
            reply(calls=[call("_noop", "c2")]),
            reply(content="完成"),
        ]
    )
    agent = Agent(llm=llm, tools=[_noop], status_bar=False, max_steps=2, max_continuations=1)
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output))
    assert run_task(agent, renderer, "做两次") == "final"
    text = output.getvalue()
    assert "检查点" in text  # dim 提示，不是失败
    assert "完成" in text


def test_run_task_budget_exhausted_outcome():
    llm = FakeLLM([reply(calls=[call("_noop", f"c{i}")]) for i in range(2)])
    agent = Agent(llm=llm, tools=[_noop], status_bar=False, max_steps=2, max_continuations=0)
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output))
    assert run_task(agent, renderer, "一直做") == "budget"
    assert "续跑上限" in output.getvalue()


def test_budget_exhausted_is_not_a_failure(tmp_path, monkeypatch):
    """预算收尾走正常路径：给可继续提示，不出现 [任务失败]。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    llm = FakeLLM([reply(calls=[call("_noop", f"c{i}")]) for i in range(2)])
    with session_for(
        tmp_path, llm, tools=[_noop], max_steps=2, max_continuations=0
    ) as (session, pipe, output):

        async def scenario():
            task = asyncio.create_task(session.run())
            await until(lambda: session.box._session.app.is_running)
            pipe.send_text("一直做\r")
            await until(lambda: "达检查点收尾" in output.getvalue())
            pipe.send_text("\x04")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        text = output.getvalue()
        assert "已达单轮预算" in text  # 可继续提示
        assert "[任务失败]" not in text


# ---------- 票 07：无进展熔断接线 ----------


def test_run_task_loop_guard_stops_with_outcome():
    llm = FakeLLM([reply(calls=[call("_noop", f"c{i}")]) for i in range(2)])
    agent = Agent(
        llm=llm, tools=[_noop], status_bar=False, max_steps=0,
        loop_guard=True, loop_repeat_limit=1,
    )
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output))
    assert run_task(agent, renderer, "循环") == "no_progress"
    assert "已停止" in output.getvalue()


def test_loop_guard_stop_is_not_a_failure(tmp_path, monkeypatch):
    """熔断收尾走正常路径：可继续提示，不出现 [任务失败]。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    llm = FakeLLM([reply(calls=[call("_noop", f"c{i}")]) for i in range(2)])
    with session_for(
        tmp_path, llm, tools=[_noop], max_steps=0, loop_guard=True, loop_repeat_limit=1
    ) as (session, pipe, output):

        async def scenario():
            task = asyncio.create_task(session.run())
            await until(lambda: session.box._session.app.is_running)
            pipe.send_text("循环\r")
            await until(lambda: "检测到重复调用，已停止" in output.getvalue())
            pipe.send_text("\x04")
            await asyncio.wait_for(task, 3)

        asyncio.run(scenario())
        text = output.getvalue()
        assert "已停止" in text
        assert "[任务失败]" not in text
