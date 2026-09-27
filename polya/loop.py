"""交互驱动（ADR 0002 的消费方）：消费 agent 生成器 → 渲染 / 审查 / 执行 /
steering 排队 → ``send`` 回结果。

- 事件适配：``renderer.update(ev.event, event_payload(ev))``——事件对象到渲染器
  词表的同名同键翻译，渲染器内部接口不动。
- 工具执行权在本层（executor 共用）；bash 实时输出经 ``default_tools(
  on_shell_output=)`` 的 tap 直喂渲染器（装配在 cli.build_agent）——执行期流是
  驱动层事务，不是引擎旁路。
- 常驻交互：Esc 在事件边界关闭生成器并回填未决工具；Ctrl+C 专职输入框。
  忙时输入分 steering / follow-up 两类；终端借用仅服务 /login 向导。
"""

from __future__ import annotations

import asyncio
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

from .agent import (
    Agent,
    BudgetCheckpoint,
    BudgetExhausted,
    Iteration,
    PlanSubmitted,
    ToolCall,
    event_payload,
)
from .commands import (
    HELP_TEXT,  # noqa: F401 - compatibility import
    CommandContext,
    command_error,
    dispatch_command,
    handle_command,  # noqa: F401 - compatibility import
    parse_command,
)
from .executor import execute
from .filefind import ProjectFiles
from .gitinfo import current_branch
from .input import InputBox, InputSuspended
from .models import ModelsConfig, format_context_window
from .providers import SUBSCRIPTION_PROVIDERS, estimate_cost
from .render import TerminalRenderer, console
from .review import Reviewer, is_plan_approval

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
    reviewer: Reviewer | None = None,
    *,
    origin: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """一次工具调用的驱动侧处理：渲染 → 审查 → 执行 → 渲染结果。

    审查器（review.py）默认放行；deny 只作为工具结果回传模型，不弹窗、不让位。
    ``progress`` 为 None（父会话）：三个渲染事件全发；非 None（子代理路径）：
    不发渲染事件，改用 ``progress`` 输出进度行与结果摘要。
    """
    if reviewer is None:
        reviewer = agent.reviewer
    if progress is None and renderer is not None:
        renderer.update("tool_review", event_payload(ev))
    item = agent.tools.get(ev.name)
    approved = False
    if item is None:
        result, duration = f"Error: unknown tool '{ev.name}'", 0.0
    else:
        outcome = reviewer.review(item, ev.arguments, plan=agent.plan_mode, origin=origin)
        approved = outcome.verdict == "allow"
        if approved:
            if progress is None and renderer is not None:
                renderer.update("tool_call", event_payload(ev))
            result, duration = execute(item, ev.arguments)
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


def run_task(
    agent: Agent,
    renderer: TerminalRenderer,
    text: str,
    reviewer: Reviewer | None = None,
    *,
    stop: threading.Event | None = None,
    on_boundary=None,
    on_plan: Callable[[str], None] | None = None,
) -> str:
    """消费一次 ``steps()``：事件转发渲染器，ToolCall/PlanSubmitted 就地处理。

    返回本轮收尾原因：``"plan"``（提交计划，驱动层据此记 plan_pending）、
    ``"budget"``（预算连跳上限收尾）或 ``"final"``（给出最终答案）。中断时
    ``gen.close()`` 触发 agent 的 GeneratorExit 回填，历史保持合法。
    """
    if reviewer is None:
        reviewer = agent.reviewer
    gen = agent.steps(text)
    to_send = None
    outcome = "final"
    try:
        with renderer:
            while True:
                if stop is not None and stop.is_set() and to_send is None:
                    raise InterruptedError("用户请求中断")
                try:
                    ev = gen.send(to_send)
                except StopIteration:
                    return outcome
                to_send = None
                if stop is not None and stop.is_set():
                    raise InterruptedError("用户请求中断")
                if isinstance(ev, BudgetCheckpoint):
                    # 软检查点：dim 提示后原地续跑，回合不结束（票 02）。
                    renderer.update(ev.event, event_payload(ev))
                    to_send = None
                elif isinstance(ev, BudgetExhausted):
                    renderer.update(ev.event, event_payload(ev))
                    outcome = "budget"
                    to_send = None
                elif isinstance(ev, Iteration) and on_boundary is not None:
                    renderer.update(ev.event, event_payload(ev))
                    to_send = on_boundary()
                elif isinstance(ev, ToolCall):
                    to_send = run_tool_call(agent, renderer, ev, reviewer)
                elif isinstance(ev, PlanSubmitted):
                    renderer.update("plan_submitted", {"plan": ev.plan})
                    if on_plan is not None:
                        on_plan(ev.plan)
                    # 回填计划结果并结束本轮；plan_mode 由驱动层在批准时翻转。
                    try:
                        gen.send("计划已展示；本轮结束，等待用户指示。")
                    except StopIteration:
                        pass
                    return "plan"
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
    agent.append_user_message(f"[shell] $ {command}\n{truncated}")


def _append_project_memory(root: str, text: str, say) -> None:
    """``#`` 前缀：把一行记忆追加到项目根 AGENTS.md（会话启动时注入系统提示词）。"""
    path = Path(root) / "AGENTS.md"
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if header_needed:
            fh.write("# 项目记忆（polya 会话启动时自动载入；# 前缀追加）\n\n")
        fh.write(f"{text}\n")
    say(f"已记入 {path}", "dim")


def print_welcome(
    output: Console,
    *,
    project_memory: str | None = None,
    trusted: bool = True,
) -> None:
    """日常启动 banner（票 01）：只报身份，不重复底栏的模型/目录/快捷键。

    底栏（`input.py` 的 env/bottom bar）常驻显示模型、窗口、目录、主题、模式与
    `/help` 提示；banner 一次性滚走，只留版本、标语与底栏没有的项目级信息。
    """
    output.set_window_title("polya")
    output.print(Text(f"  polya · v{version('polya')}", style="bold cyan"))
    output.print(Text("  和你一起理解问题、制定计划、完成验证", style="dim"))
    if not trusted:
        output.print(
            Text(
                "  ⚠ 未信任此目录：AGENTS.md / 项目 skills 未加载（/trust 查看）",
                style="yellow",
            )
        )
    elif project_memory:
        output.print(Text("  已加载项目记忆 AGENTS.md", style="dim"))
    output.print()


class InteractiveSession:
    """输入在主事件循环运行；一个 worker 独占 agent 和命令执行。

    /login 向导通过握手借用终端，主输入退出后才允许它读 stdin。
    普通消息在下一模型调用前注入；会重置或结束会话的命令需先让任务空闲。
    """

    tick_interval: float = 1.0  # 忙碌条重绘周期（测试可缩小）

    def __init__(self, agent: Agent, root: str, renderer: TerminalRenderer, box: InputBox):
        self.agent, self.root, self.renderer, self.box = agent, root, renderer, box
        # 忙时输入分两类：steering=下一模型请求前注入；follow_up=本轮结束后作为
        # 下一任务。（pi 语义：Enter=steering，Alt+Enter=follow-up，Alt+Up=取回。）
        self.steering: deque[str] = deque()
        self.follow_up: deque[str] = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.closing = False
        self.topic: str | None = None
        self.state = {
            "model": agent.llm.model,
            "project": str(Path(root).resolve()),
            "topic": None,
            "mode": "normal",
            "busy": False,
        }
        self.requests: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None  # 事件循环线程 id（_borrow_terminal 防死锁）
        self.box.on_interrupt = self.interrupt
        self.box.on_dequeue = self._dequeue_to_editor
        self.box.on_set_default = self._set_default_model
        self.renderer.on_status = self._status
        self.box.preview = self.renderer.preview
        self.plan_pending = False
        self._status(renderer)
        # 子代理（票 04）：共享父会话审查器；dispatch 走 run_tool_call（origin 标
        # 「子任务」、progress 进 task 尾窗）。子代理继承父 plan_mode。
        runner = getattr(self.agent, "subagent", None)
        if runner is not None:
            runner.bind(self.stop, True)
            runner.progress = lambda line: self.renderer.update(
                "task_progress", {"name": "task", "line": line}
            )
            runner.dispatch = lambda child, ev: run_tool_call(
                child,
                self.renderer,
                ev,
                self.agent.reviewer,
                origin="子任务",
                progress=runner.progress,
            )

    def say(self, text: str, style: str = "none") -> None:
        self.renderer._console.print(text, style=style, markup=False)

    def _status(self, renderer: TerminalRenderer) -> None:
        # /model 切换后状态行跟随（llm 实例整个换掉，model/profile_name 都变）
        llm = self.agent.llm
        self.state["model"] = llm.model
        self.state["topic"] = self.topic
        self.state["profile"] = getattr(llm, "profile_name", None)
        self.state["thinking"] = getattr(llm, "thinking_level", None) or ""
        self.state["mode"] = "plan" if self.agent.plan_mode else "normal"
        self.state["status"] = renderer._status_label()
        self.state["preview_active"] = renderer.has_preview
        self.state["branch"] = current_branch(self.root)
        # 用量与费用：累计 token 取渲染器存的 total（跨压缩保留）；命中率取最近一次
        # 请求的 cached/prompt。费用按 PRICES 估算，无价表或订阅制 provider 不显示金额。
        totals = renderer.total_usage or {}
        prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
        completion_tokens = int(totals.get("completion_tokens", 0) or 0)
        cached_tokens = int(totals.get("cached_tokens", 0) or 0)
        # pi 语义：↑input 只算未命中缓存的输入，缓存读取单列为 CR（否则与 CR 重叠）。
        input_tokens = max(0, prompt_tokens - cached_tokens)
        self.state["input_tokens"] = input_tokens
        self.state["output_tokens"] = completion_tokens
        self.state["cached_tokens"] = cached_tokens
        self.state["cache_hit"] = (
            renderer._ctx_cached / renderer._ctx_used * 100 if renderer._ctx_used else None
        )
        self.state["subscribed"] = getattr(llm, "profile_name", None) in SUBSCRIPTION_PROVIDERS
        self.state["cost"] = estimate_cost(
            llm.model, input_tokens, completion_tokens, cached_tokens
        )
        self.state["auto"] = bool(getattr(self.agent, "compress", False))
        # 窗口大小（票 01）：第二零刻即可显示容量，首次请求后才追加占用百分比。
        # 只赋值、不 pop：占用在 /clear /new 时由 _run_command 显式清零（见下）。
        if renderer.context_window:
            self.state["window"] = format_context_window(renderer.context_window)
        if renderer.context_window and renderer._ctx_used is not None:
            self.state["context_pct"] = renderer._ctx_used * 100 // renderer.context_window
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
        """Esc：请求中断当前任务；停止后队列里的消息回到编辑器（pi 语义）。"""
        if self.state["busy"]:
            self.stop.set()
            self.state["stopping"] = True
            self.box._session.app.invalidate()

    def _sync_queue_state(self) -> None:
        with self.lock:
            steering, follow_up = len(self.steering), len(self.follow_up)
        self.state["steering"] = steering
        self.state["follow_up"] = follow_up
        self.state["queued"] = steering + follow_up
        self.box._session.app.invalidate()

    def _queue_input(self, text: str, kind: str) -> None:
        """忙时输入：steering 在下一模型请求前注入，follow-up 在本轮结束后。"""
        with self.lock:
            (self.steering if kind == "steering" else self.follow_up).append(text)
        self._sync_queue_state()
        hint = "（下一轮请求前交给模型）" if kind == "steering" else "（本轮结束后交给模型）"
        self.say("＋ 已排队" + hint + "：" + text.replace("\n", " ")[:60], "dim")

    def _pop_next_task(self) -> str | None:
        """下一个任务：steering 优先，其次 follow-up。"""
        with self.lock:
            if self.steering:
                text = self.steering.popleft()
            elif self.follow_up:
                text = self.follow_up.popleft()
            else:
                return None
        self._sync_queue_state()
        return text

    def _set_default_model(self, ref: str) -> str:
        """Ctrl+S（/model 选项器内）：把 ref 写为默认启动模型，返回 flash 提示。"""
        config = ModelsConfig.load()
        try:
            provider_id, model = config.use(ref)
            config.save()
        except ValueError as exc:
            return f"[错误] {exc}"
        return f"已设为默认启动模型：{provider_id}/{model}"

    def _dequeue_to_editor(self) -> str:
        """Alt+Up / Esc：把排队消息取回编辑器。"""
        with self.lock:
            items = [*self.steering, *self.follow_up]
            self.steering.clear()
            self.follow_up.clear()
        self._sync_queue_state()
        return "\n".join(items)

    def _command_context(self) -> CommandContext:
        return CommandContext(
            self.agent,
            self.renderer,
            restart=self._restart,
            in_terminal=self._borrow_terminal,  # /login 向导借道终端让位
            rename=self._rename,
        )

    def _rename(self, topic: str) -> str:
        self.topic = topic
        self.state["topic"] = topic
        self.renderer._console.set_window_title(f"polya · {topic}")
        self.box._session.app.invalidate()
        return f"已更新会话主题：{topic}"

    def _restart(self) -> str:
        """/new 的会话级重置：清屏重印启动区、主题、标题、计划与排队消息。"""
        self.topic = None
        self.state["topic"] = None
        self.plan_pending = False
        self.renderer._console.set_window_title("polya")
        with self.lock:
            dropped = len(self.steering) + len(self.follow_up)
            self.steering.clear()
            self.follow_up.clear()
        self._sync_queue_state()
        # 清屏再重印启动区：/new /clear /reset 后终端像刚启动的新会话，旧输出
        # 不留在滚动区。经 patched stdout 写出会先擦除输入区、写完再重绘，
        # 输入框与状态栏回到屏幕底部（见 issues/10-fixed-bottom-tui.md 的全屏方案）。
        self.renderer._console.clear()
        print_welcome(
            self.renderer._console,
            project_memory=self.agent.project_memory,
            trusted=self.agent.trusted,
        )
        return f"（已丢弃 {dropped} 条排队消息）" if dropped else ""

    def _local(self, text: str) -> bool:
        if text.startswith("/"):
            command, _ = parse_command(text)
            if command is not None:
                self._run_command(text, command)
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
        """工具批次全部回填后、下一模型请求前，注入 steering 消息。"""
        while not self.stop.is_set():
            with self.lock:
                if not self.steering:
                    break
                text = self.steering.popleft()
            self._sync_queue_state()
            if not self._local(text):
                self.agent.append_user_message(text)
                self.say("❯ " + text, "cyan")
                self.say("＋ 补充已交给模型，将用于下一轮请求。", "dim")

    def _borrow_terminal(self, callback):
        if self.stop.is_set() or self.closing:
            raise InterruptedError("终端借用已取消")
        if self.loop is None or threading.get_ident() == self._loop_thread:
            # 事件循环未就绪，或调用方就在主循环线程上：握手会自我死等
            # （call_soon_threadsafe 排的回调永远排不到）。直接跑回调，宁可短暂
            # 阻塞也不冻结会话。命令现由 run() 起独立 worker 任务，正常不会走到这里。
            return callback()
        ready, done = threading.Event(), threading.Event()
        self.loop.call_soon_threadsafe(self.requests.put_nowait, (ready, done))
        ready.wait()
        try:
            if self.closing or self.stop.is_set():
                raise InterruptedError("终端借用已取消")
            return callback()
        except KeyboardInterrupt as exc:
            raise InterruptedError("终端借用已取消") from exc
        finally:
            done.set()

    def _on_plan(self, plan: str) -> None:
        self.plan_pending = True

    def _work(self, text: str) -> None:
        task = not text.startswith(("/", "!", "#"))
        started = time.monotonic()
        outcome = "本轮结束"
        try:
            if self.plan_pending and task:
                if is_plan_approval(text):
                    self.agent.leave_plan_mode()
                    self.plan_pending = False
                    self.say("已批准计划，进入执行。", "green")
                else:
                    self.say("计划修改意见已交给模型；仍在计划模式（只读）。", "dim")
            if not self._local(text):
                if self.topic is None:
                    self.topic = _topic_from(text)
                    self.state["topic"] = self.topic
                    self.renderer._console.set_window_title(f"polya · {self.topic}")
                self.say("❯ " + text, "cyan")
                task_outcome = run_task(
                    self.agent,
                    self.renderer,
                    text,
                    self.agent.reviewer,
                    stop=self.stop,
                    on_boundary=self._boundary,
                    on_plan=self._on_plan,
                )
                if task_outcome == "plan":
                    outcome = "等待计划确认"
                elif task_outcome == "budget":
                    # 软检查点收尾：不是失败，历史完整，下一条消息即可继续。
                    outcome = "达检查点收尾"
                    self.say(
                        "已达单轮预算，历史已保留；继续请直接发送下一条消息。", "yellow"
                    )
        except InterruptedError:
            outcome = "本轮已中断"
            self.say("已中断本次任务；已完成步骤保留，排队消息回到输入框。", "yellow")
        except Exception as exc:  # noqa: BLE001 - 网络和工具错误不结束会话
            outcome = "本轮失败"
            self.say(f"[任务失败] {type(exc).__name__}: {exc}", "red")
        finally:
            if task:
                elapsed = max(0, int(time.monotonic() - started))
                self.say(f"── {outcome} · {elapsed}s", "dim")
                # 自动落盘（session-lifecycle 票 02）：每个任务收尾静默保存，
                # /resume 因此列出真实用过的会话。
                try:
                    self.agent.autosave()
                except Exception as exc:  # noqa: BLE001 - 落盘失败不结束会话
                    self.say(f"[自动保存失败] {type(exc).__name__}: {exc}", "yellow")

    def _run_command(self, text: str, command) -> None:
        """命令在主循环（输入线程）执行；`idle` 命令遇到运行中的任务则拒绝。"""
        if command.idle and self.state.get("busy"):
            self.say(f"当前任务运行中；先按 Esc 中断再执行 {command.name}。", "yellow")
            return
        output = dispatch_command(text, self._command_context())
        if output is None:
            self.closing = True
            return
        if command.name in (
            "/new",
            "/resume",
            "/fork",
            "/clone",
            "/load",
        ) and not command_error(text):
            self.renderer._ctx_used = None
            self.renderer.total_usage = {}
            self.state.pop("context_pct", None)
            self.state.pop("cache_hit", None)
        # 切/换会话后把驱动层主题与窗口标题同步到会话元数据（/new 走 restart 自清）。
        if command.name in ("/resume", "/fork", "/clone", "/load"):
            self.topic = self.agent.session_title
            self.state["topic"] = self.topic
            title = f"polya · {self.topic}" if self.topic else "polya"
            self.renderer._console.set_window_title(title)
        if command.name == "/plan":
            self.plan_pending = False
        self.say(output)
        self._status(self.renderer)

    def _submit_input(self, text: str) -> None:
        """分流一次提交。

        忙时：命令就地执行（`idle` 命令自会拒绝），消息排队（Enter=steering，
        Alt=follow-up）。空闲时一律进队列，由 `run()` 起独立 worker/命令任务——
        命令必须在主循环之外执行，否则终端让位握手（`_borrow_terminal`）会死锁。
        """
        if not text:
            return
        command, _ = parse_command(text)
        if self.state.get("busy"):
            if command is not None:
                self._run_command(text, command)
            else:
                self._queue_input(text, self.box.last_kind)
            return
        with self.lock:
            self.steering.append(text)
        self._sync_queue_state()

    async def _tick(self) -> None:
        """每秒重绘忙碌条：思考/长工具的空窗期没有事件，计时否则会冻在某一秒。"""
        while not self.closing:
            await asyncio.sleep(self.tick_interval)
            if self.state.get("busy"):
                self.box._session.app.invalidate()

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        prompt = None
        worker = None
        command = None
        ticker = asyncio.create_task(self._tick())
        request = asyncio.create_task(self.requests.get())
        try:
            while not self.closing:
                if worker is None and command is None:
                    text = self._pop_next_task()
                    if text is not None:
                        entry, _ = parse_command(text)
                        if entry is not None:
                            # 命令也走独立任务：终端让位握手要求主循环保持可调度，
                            # 网络/交互输入也不冻 UI（_borrow_terminal）。
                            command = asyncio.create_task(
                                asyncio.to_thread(self._run_command, text, entry)
                            )
                        else:
                            self._start_task()
                            worker = asyncio.create_task(asyncio.to_thread(self._work, text))
                if prompt is None and command is None:
                    prompt = asyncio.create_task(self.box.ask_async(self.state))
                tasks = [t for t in (prompt, request, worker, command) if t is not None]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if worker is not None and worker in done:
                    await worker
                    worker = None
                    interrupted = self.stop.is_set()
                    self.state.update(busy=False, stopping=False)
                    self.state.pop("started_at", None)
                    if interrupted:
                        # Esc 中断：队列里的消息回到编辑器，不静默丢弃。
                        pending = self._dequeue_to_editor()
                        if pending:
                            self.box.insert_pending(pending)
                    self.box.refresh_file_index()  # agent 可能刚写过文件
                    self.box._session.app.invalidate()
                if command is not None and command in done:
                    await command
                    command = None
                    self.box.refresh_file_index()  # 命令可能刚写过文件
                    self.box._session.app.invalidate()
                if prompt is not None and prompt in done:
                    try:
                        text = prompt.result().strip()
                    except (EOFError, KeyboardInterrupt):
                        self.closing = True
                    else:
                        self._submit_input(text)
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
                            self._submit_input(submitted.strip())
                        prompt = None
                    ready.set()
                    await asyncio.to_thread(finished.wait)
                    request = asyncio.create_task(self.requests.get())
        finally:
            self.closing = True
            self.stop.set()
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass
            # 退出也完成借用握手，避免 worker/命令等待一个已经消失的输入框。
            if prompt is not None and not prompt.done():
                self.box.suspend()
                try:
                    await prompt
                except (InputSuspended, EOFError, KeyboardInterrupt):
                    pass
            while any(t is not None and not t.done() for t in (worker, command)):
                running = [t for t in (worker, command) if t is not None and not t.done()]
                done, _ = await asyncio.wait(
                    [*running, request], return_when=asyncio.FIRST_COMPLETED
                )
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
            if command is not None:
                await command
            self.renderer.on_status = lambda renderer: None
            self.box.preview = lambda width, max_lines: ""


def run_repl(agent: Agent, root: str, renderer: TerminalRenderer) -> None:
    if sys.stdin.isatty():
        # stdout/stderr 都经同一个代理排在输入区上方；不使用全屏终端。
        with patch_stdout(raw=True):
            output = Console()
            renderer.use_scrollback(output)
            print_welcome(
                output,
                project_memory=agent.project_memory,
                trusted=agent.trusted,
            )
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
