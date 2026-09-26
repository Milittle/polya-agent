"""交互驱动（ADR 0002 的消费方）：消费 agent 生成器 → 渲染 / 权限判定 / 审批 /
执行 → ``send`` 回结果。

- 事件适配：``renderer.update(ev.event, event_payload(ev))``——事件对象到渲染器
  词表的同名同键翻译，渲染器内部接口不动。
- 工具执行权在本层（executor 共用）；bash 实时输出经 ``default_tools(
  on_shell_output=)`` 的 tap 直喂渲染器（装配在 cli.build_agent）——执行期流是
  驱动层事务，不是引擎旁路。
- 常驻交互：Esc 在事件边界关闭生成器并回填未决工具；Ctrl+C 专职输入框。
  审批通过握手借用输入，完成后恢复草稿；非交互仍走内置驱动。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .agent import Agent, Iteration, PlanSubmitted, ToolCall, event_payload
from .approval import (
    ApprovalGate,
    ApprovalOutcome,  # noqa: F401 - compatibility import
    ApprovalRejected,
    terminal_approve,  # noqa: F401 - compatibility import
    terminal_approve_plan,  # noqa: F401 - compatibility import
)
from .commands import (
    BUSY_HINTS,
    HELP_TEXT,  # noqa: F401 - compatibility import
    CommandContext,
    command_error,
    dispatch_command,
    handle_command,  # noqa: F401 - compatibility import
    parse_command,
)
from .executor import execute
from .filefind import ProjectFiles
from .input import InputBox, InputSuspended
from .render import TerminalRenderer, console

# ---------- 生成器驱动 ----------


def _result_summary(result: str, duration: float, limit: int = 80) -> str:
    first = result.strip().splitlines()[0] if result.strip() else ""
    if len(first) > limit:
        first = first[: limit - 1] + "…"
    return f"{first} · {round(duration, 2)}s"


def run_tool_call(
    agent: Agent,
    renderer: TerminalRenderer | None,
    ev: ToolCall,
    interactive: bool,
    gate: ApprovalGate,
    *,
    unrestricted: bool | None = None,
    origin: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """一次工具调用的驱动侧处理：渲染 → decide → 审批 → 执行 → 渲染结果。

    ``progress`` 为 None（父会话）：三个渲染事件全发，与旧行为逐字节相同。非 None
    （子代理路径）：不发渲染事件（不抢单槽、不进滚动区），改用 ``progress`` 输出
    进度行与结果摘要；审批仍走同一 gate，``unrestricted`` 由调用方显式传父会话取值。
    """
    if unrestricted is None:
        unrestricted = agent.approve is None
    if progress is None and renderer is not None:
        renderer.update("tool_review", event_payload(ev))
    item = agent.tools.get(ev.name)
    approved = False
    if item is None:
        result, duration = f"Error: unknown tool '{ev.name}'", 0.0
    else:
        outcome = gate.authorize(
            item,
            ev.arguments,
            plan=agent.plan_mode,
            unrestricted=unrestricted,
            interactive=interactive,
            origin=origin,
        )
        approved = outcome.approved
        if outcome.approved:
            if progress is None and renderer is not None:
                renderer.update("tool_call", {**event_payload(ev), "arguments": outcome.arguments})
            result, duration = execute(item, outcome.arguments or {})
        else:
            result, duration = outcome.reason or "Error: Tool call denied", 0.0
    if progress is not None:
        progress(f"← {ev.name} {_result_summary(result, duration)}")
    elif renderer is not None:
        renderer.update(
            "tool_result",
            {
                "name": ev.name,
                "call_id": ev.call_id,
                "result": result,
                "duration_s": round(duration, 3),
                "error": result.startswith("Error"),
                "denied": item is not None and not approved,
            },
        )
    return result


def _run_tool(
    agent: Agent, renderer: TerminalRenderer, ev: ToolCall, interactive: bool, gate: ApprovalGate
) -> str:
    """父会话薄包装：进度渲染走原有三事件路径（测试直接 import 本函数）。"""
    return run_tool_call(agent, renderer, ev, interactive, gate)


def run_task(
    agent: Agent,
    renderer: TerminalRenderer,
    text: str,
    interactive: bool,
    gate: ApprovalGate | None = None,
    *,
    stop: threading.Event | None = None,
    on_boundary=None,
) -> None:
    """消费一次 ``steps()``：事件转发渲染器，ToolCall/PlanSubmitted 就地处理。

    Ctrl+C 落在驱动侧（审批 / 执行）时 ``gen.close()`` 触发 agent 的
    GeneratorExit 回填，历史保持合法后中断向上传播。
    """
    if gate is None:
        gate = ApprovalGate(interactive)
    gate.rejected_reason = None
    gen = agent.steps(text)
    to_send = None
    try:
        with renderer:
            while True:
                if stop is not None and stop.is_set() and to_send is None:
                    raise InterruptedError("用户请求中断")
                try:
                    ev = gen.send(to_send)
                except StopIteration:
                    if gate.rejected_reason is not None:
                        raise ApprovalRejected(gate.rejected_reason) from None
                    return
                except RuntimeError:
                    if gate.rejected_reason is not None:
                        raise ApprovalRejected(gate.rejected_reason) from None
                    raise
                to_send = None
                if gate.rejected_reason is not None:
                    raise ApprovalRejected(gate.rejected_reason)
                if stop is not None and stop.is_set():
                    raise InterruptedError("用户请求中断")
                if isinstance(ev, Iteration) and on_boundary is not None:
                    renderer.update(ev.event, event_payload(ev))
                    to_send = on_boundary()
                elif isinstance(ev, ToolCall):
                    to_send = _run_tool(agent, renderer, ev, interactive, gate)
                elif isinstance(ev, PlanSubmitted):
                    renderer.update("plan_approval", {})
                    to_send = agent._handle_plan(ev.plan)
                    renderer.update("plan_result", {"approved": not agent.plan_mode})
                    if agent.plan_mode:
                        gate.rejected_reason = "用户拒绝了计划"
                else:
                    renderer.update(ev.event, event_payload(ev))
    finally:
        gen.close()


def _topic_from(first_input: str) -> str:
    """A readable local provisional topic; never make an extra model request."""
    text = " ".join(first_input.split())
    return "".join(char for char in text if char.isprintable())[:48] or "新会话"


def _run_shell_bang(agent: Agent, root: str, command: str, say) -> None:
    """``!`` 前缀：本地跑 shell，输出进上下文（8000 字符截断，与工具结果同规）。"""
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=root, timeout=60
        )
        output = (completed.stdout + completed.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - shell 失败只报本次，REPL 存活
        output = f"Error: {type(exc).__name__}: {exc}"
    say(f"$ {command}", "yellow")
    if output:
        say(output, "none")
    truncated = output if len(output) <= 8000 else output[:8000] + "\n…（已截断）"
    # 直接注入历史（不触发 LLM 轮）：作为后续对话的上下文证据
    agent.history.append({"role": "user", "content": f"[shell] $ {command}\n{truncated}"})


def _append_project_memory(root: str, text: str, say) -> None:
    """``#`` 前缀：把一行记忆追加到项目根 AGENTS.md（会话启动时注入系统提示词）。"""
    path = Path(root) / "AGENTS.md"
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if header_needed:
            fh.write("# 项目记忆（polya 会话启动时自动载入；# 前缀追加）\n\n")
        fh.write(f"{text}\n")
    say(f"已记入 {path}", "dim")


def print_welcome(root: str, model: str, output: Console) -> None:
    path = str(Path(root).resolve())
    home = str(Path.home())
    if path == home or path.startswith(home + os.sep):
        path = "~" + path[len(home) :]
    output.set_window_title("polya")
    output.print(Text(f"  polya · v{version('polya')}", style="bold cyan"))
    output.print(Text("  和你一起理解问题、制定计划、完成验证", style="dim"))
    output.print()
    output.print(Text(f"  {path} · {model}", style="dim"), overflow="ellipsis", no_wrap=True)
    output.print(Text("  /help 查看命令\n", style="dim"))


class InteractiveSession:
    """输入在主事件循环运行；一个 worker 独占 agent 和命令执行。

    审批通过握手借用终端，主输入退出后才允许审批读 stdin。
    普通消息在下一模型调用前注入；会重置或结束会话的命令等当前任务收尾。
    """

    def __init__(self, agent: Agent, root: str, renderer: TerminalRenderer, box: InputBox):
        self.agent, self.root, self.renderer, self.box = agent, root, renderer, box
        self.pending: deque[str] = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.closing = False
        self.topic: str | None = None
        self.queue_paused = False
        self.state = {
            "model": agent.llm.model,
            "project": str(Path(root).resolve()),
            "topic": None,
            "mode": "normal",
            "busy": False,
        }
        self.requests: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.box.on_interrupt = self.interrupt
        self.gate = ApprovalGate(True, renderer, root)
        self._screen = self.gate.screen
        self.gate.screen = lambda *args, **kwargs: self._borrow_terminal(  # type: ignore[method-assign]
            lambda: self._screen(*args, **kwargs)
        )
        self._approve_plan: Callable[[str], bool] | None = agent.approve_plan
        if self._approve_plan is not None:
            agent.approve_plan = lambda plan: self._borrow_terminal(
                lambda: bool(self._approve_plan and self._approve_plan(plan))
            )
        self.renderer.on_status = self._status
        self.box.preview = self.renderer.preview
        self.renderer.session_footer = self.box.session_footer
        self._status(renderer)
        # 子代理（票 04）：重绑到会话 gate（授权规则 / allow_all 共享），dispatch
        # 走 run_tool_call（origin 标「子任务」、progress 进 task 尾窗），unrestricted
        # 显式取父会话审批模式，避免子 approve=None 被当成 yolo。
        runner = getattr(self.agent, "subagent", None)
        if runner is not None:
            runner.bind(self.gate, self.stop, True)
            runner.progress = lambda line: self.renderer.update(
                "task_progress", {"name": "task", "line": line}
            )
            runner.dispatch = lambda child, ev: run_tool_call(
                child,
                self.renderer,
                ev,
                True,
                self.gate,
                unrestricted=self.agent.approve is None,
                origin="子任务",
                progress=runner.progress,
            )

    def say(self, text: str, style: str = "none") -> None:
        self.renderer._console.print(text, style=style, markup=False)

    def _status(self, renderer: TerminalRenderer) -> None:
        # /models 切换后状态行跟随（llm 实例整个换掉，model/profile_name 都变）
        self.state["model"] = self.agent.llm.model
        self.state["topic"] = self.topic
        self.state["profile"] = getattr(self.agent.llm, "profile_name", None)
        self.state["permissions"] = (
            "all" if self.gate.allow_all or self.agent.approve is None else "ask"
        )
        self.state["mode"] = (
            "plan"
            if self.agent.plan_mode
            else "yolo"
            if (self.gate.allow_all or self.agent.approve is None)
            else "normal"
        )
        self.state["status"] = renderer._status_label()
        self.state["preview_active"] = renderer.has_preview
        if renderer.context_window and renderer._ctx_used is not None:
            self.state["context"] = f"ctx {renderer._ctx_used * 100 // renderer.context_window}%"
        self.box._session.app.invalidate()

    def _start_task(self) -> None:
        self.stop.clear()
        self.state.update(
            busy=True,
            stopping=False,
            status="Waiting for model",
            started_at=time.monotonic(),
        )

    def interrupt(self) -> None:
        if self.state["busy"]:
            self.stop.set()
            self.state["stopping"] = True
            self._pause_queue()
            self.box._session.app.invalidate()

    def _pause_queue(self) -> None:
        self.queue_paused = True
        self.state["queue_paused"] = True

    def enqueue(self, text: str) -> None:
        command, _ = parse_command(text)
        if command is not None and command.busy == "control":
            self.say(dispatch_command(text, self._command_context()) or "")
            return
        with self.lock:
            self.pending.append(text)
            self.state["queued"] = len(self.pending)
        hint = f"（{BUSY_HINTS[command.busy]}）" if command else "（下一轮请求前交给模型）"
        self.say("＋ 已排队" + hint + "：" + text.replace("\n", " ")[:60], "dim")

    def _pop(self, *, boundary: bool = False) -> str | None:
        with self.lock:
            if not self.pending:
                return None
            # 重置和退出不能在仍存活的生成器内部执行。
            command, _ = parse_command(self.pending[0])
            if boundary and command is not None and command.busy == "task_end":
                return None
            text = self.pending.popleft()
            self.state["queued"] = len(self.pending)
            return text

    def _resume_queue(self) -> str:
        if self.state.get("stopping"):
            return "正在停止当前操作；停止后用 /resume 恢复队列。"
        if not self.queue_paused:
            return "排队任务未暂停。"
        self.queue_paused = False
        self.state["queue_paused"] = False
        return "已恢复排队任务。"

    def _command_context(self) -> CommandContext:
        return CommandContext(
            self.agent,
            self.renderer,
            self.gate,
            self._resume_queue,
            self._restart,
            in_terminal=self._borrow_terminal,  # /models add 向导借道审批的让位机制
            rename=self._rename,
        )

    def _rename(self, topic: str) -> str:
        self.topic = topic
        self.state["topic"] = topic
        self.renderer._console.set_window_title(f"polya · {topic}")
        self.box._session.app.invalidate()
        return f"已更新会话主题：{topic}"

    def _restart(self) -> str:
        """/new 的会话级重置：主题、窗口标题、授权规则与排队消息；返回丢弃附注。"""
        self.topic = None
        self.state["topic"] = None
        self.queue_paused = False
        self.state["queue_paused"] = False
        self.renderer._console.set_window_title("polya")
        self.gate.rules.clear()
        with self.lock:
            dropped = len(self.pending)
            self.pending.clear()
            self.state["queued"] = 0
        print_welcome(self.root, self.agent.llm.model, self.renderer._console)
        return f"（已丢弃 {dropped} 条排队消息）" if dropped else ""

    def _local(self, text: str) -> bool:
        if text.startswith("/"):
            command, _ = parse_command(text)
            output = dispatch_command(text, self._command_context())
            if output is None:
                self.closing = True
            else:
                if (
                    command is not None
                    and command.name in ("/clear", "/new")
                    and not command_error(text)
                ):
                    self.renderer._ctx_used = None
                    self.state.pop("context", None)
                self.say(output)
            self._status(self.renderer)
            return True
        if text.startswith("!") and len(text) > 1:
            self.state["status"] = "Running Bash"
            _run_shell_bang(self.agent, self.root, text[1:].strip(), self.say)
            return True
        if text.startswith("#") and len(text) > 1:
            _append_project_memory(self.root, text[1:].strip(), self.say)
            return True
        return False

    def _boundary(self) -> None:
        while not self.stop.is_set() and not self.queue_paused:
            text = self._pop(boundary=True)
            if text is None:
                break
            if not self._local(text):
                self.agent.history.append({"role": "user", "content": text})
                self.say("❯ " + text, "cyan")
                self.say("＋ 补充已交给模型，将用于下一轮请求。", "dim")

    def _borrow_terminal(self, callback):
        if self.stop.is_set() or self.closing:
            raise InterruptedError("审批已取消")
        if self.loop is None:
            # 事件循环尚未就绪（理论上不会走到）：直接跑回调，避免死等。
            return callback()
        ready, done = threading.Event(), threading.Event()
        self.loop.call_soon_threadsafe(self.requests.put_nowait, (ready, done))
        ready.wait()
        try:
            if self.closing or self.stop.is_set():
                raise InterruptedError("审批已取消")
            return callback()
        except KeyboardInterrupt as exc:
            raise InterruptedError("审批已取消") from exc
        finally:
            done.set()

    def _work(self, text: str) -> None:
        task = not text.startswith(("/", "!", "#"))
        started = time.monotonic()
        outcome = "本轮结束"
        try:
            if not self._local(text):
                if self.topic is None:
                    self.topic = _topic_from(text)
                    self.state["topic"] = self.topic
                    self.renderer._console.set_window_title(f"polya · {self.topic}")
                self.say("❯ " + text, "cyan")
                run_task(
                    self.agent,
                    self.renderer,
                    text,
                    True,
                    self.gate,
                    stop=self.stop,
                    on_boundary=self._boundary,
                )
        except ApprovalRejected:
            outcome = "本轮已拒绝"
            self._pause_queue()
            self.say("已拒绝并停止当前任务，等待新指令。排队任务已暂停，/resume 恢复。", "yellow")
        except InterruptedError:
            outcome = "本轮已中断"
            self._pause_queue()
            self.say(
                "已中断本次任务；已完成步骤保留。排队任务已暂停，/resume 恢复；也可输入新任务。",
                "yellow",
            )
        except Exception as exc:  # noqa: BLE001 - 网络和工具错误不结束会话
            outcome = "本轮失败"
            self._pause_queue()
            self.say(f"[任务失败] {type(exc).__name__}: {exc}", "red")
            self.say("排队任务已暂停，/resume 恢复；也可输入新任务。", "yellow")
        finally:
            if task:
                elapsed = max(0, int(time.monotonic() - started))
                self.say(f"── {outcome} · {elapsed}s", "dim")

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        prompt = None
        worker = None
        request = asyncio.create_task(self.requests.get())
        try:
            while not self.closing:
                if worker is None and not self.queue_paused:
                    text = self._pop()
                    if text is not None:
                        self._start_task()
                        worker = asyncio.create_task(asyncio.to_thread(self._work, text))
                if prompt is None:
                    prompt = asyncio.create_task(self.box.ask_async(self.state))
                tasks = [prompt, request] + ([worker] if worker is not None else [])
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if worker is not None and worker in done:
                    await worker
                    worker = None
                    self.state.update(busy=False, stopping=False)
                    self.state.pop("started_at", None)
                    self.box.refresh_file_index()  # agent 可能刚写过文件
                    self.box._session.app.invalidate()
                if prompt in done:
                    try:
                        text = prompt.result().strip()
                    except (EOFError, KeyboardInterrupt):
                        self.closing = True
                    else:
                        command, _ = parse_command(text)
                        if command is not None and command.busy == "control":
                            self.say(dispatch_command(text, self._command_context()) or "")
                        elif text:
                            if self.queue_paused and worker is None:
                                self._start_task()
                                worker = asyncio.create_task(asyncio.to_thread(self._work, text))
                            elif worker is None:
                                with self.lock:
                                    self.pending.append(text)
                            else:
                                self.enqueue(text)
                    prompt = None
                if request in done:
                    ready, finished = request.result()
                    if prompt is not None:
                        self.box.suspend()
                        try:
                            submitted = await prompt
                        except InputSuspended:
                            pass
                        except (EOFError, KeyboardInterrupt):
                            self.closing = True
                        else:
                            if submitted.strip():
                                self.enqueue(submitted.strip())
                        prompt = None
                    ready.set()
                    await asyncio.to_thread(finished.wait)
                    request = asyncio.create_task(self.requests.get())
        finally:
            self.closing = True
            self.stop.set()
            # 退出也完成借用握手，避免 worker 等待一个已经消失的输入框。
            if prompt is not None and not prompt.done():
                self.box.suspend()
                try:
                    await prompt
                except (InputSuspended, EOFError, KeyboardInterrupt):
                    pass
            while worker is not None and not worker.done():
                done, _ = await asyncio.wait([worker, request], return_when=asyncio.FIRST_COMPLETED)
                if request in done:
                    ready, finished = request.result()
                    ready.set()
                    await asyncio.to_thread(finished.wait)
                    request = asyncio.create_task(self.requests.get())
            request.cancel()
            try:
                await request
            except asyncio.CancelledError:
                pass
            if worker is not None:
                await worker
            self.agent.approve_plan = self._approve_plan
            self.renderer.on_status = lambda renderer: None
            self.box.preview = lambda width, max_lines: ""
            self.renderer.session_footer = None


def run_repl(agent: Agent, root: str, renderer: TerminalRenderer) -> None:
    if sys.stdin.isatty():
        # stdout/stderr 都经同一个代理排在输入区上方；不使用全屏终端。
        with patch_stdout(raw=True):
            output = Console()
            renderer.use_scrollback(output)
            print_welcome(root, agent.llm.model, output)
            box = InputBox(files=ProjectFiles(Path(root)))
            asyncio.run(InteractiveSession(agent, root, renderer, box).run())
        return

    def say(message: str, style: str = "none") -> None:
        console.print(message, style=style, markup=False)

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not text:
            continue
        if text.startswith("/"):
            result = handle_command(text, agent, renderer)
            if result is None:
                return
            say(result)
        elif text.startswith("!") and len(text) > 1:
            _run_shell_bang(agent, root, text[1:].strip(), say)
        elif text.startswith("#") and len(text) > 1:
            _append_project_memory(root, text[1:].strip(), say)
        else:
            try:
                console.print(Markdown(agent.run(text)))
                say(f"[用量] {agent.total_usage}", "dim")
            except KeyboardInterrupt:
                say("已中断本次任务", "yellow")
            except Exception as exc:  # noqa: BLE001
                say(f"[任务失败] {type(exc).__name__}: {exc}", "red")
