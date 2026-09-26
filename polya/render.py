"""Agent 事件的终端渲染器。

常驻交互由 prompt_toolkit 展示 live 区；完成的正文与工具结果一次写入滚动区。
思考和工具结果保留有界归档，供 /expand 使用；流式 bash 结果不重复打印。
独立使用时保留 Rich Live 渲染，非终端不创建 Live。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from io import StringIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

console = Console()  # stdout：只承载答案与命令输出（-p 可安全重定向/管道）
ui = Console(stderr=True)  # stderr：状态条 / 日志 / 审批面板等“界面”输出

# 阶段标签：thinking=等待首个片段 / streaming=正文流入 / tool=工具执行中
PHASE_THINKING = "thinking"
PHASE_STREAMING = "streaming"
PHASE_TOOL = "tool"


def display_tool_name(name: str) -> str:
    """Display labels are independent of model-facing tool identifiers."""
    return name.replace("_", " ").title()


def _collapse(text: str, max_lines: int = 8, max_chars: int = 600) -> tuple[str, int]:
    """折叠长文本：(展示文本, 隐藏行数)。

    先截前 ``max_lines`` 行，再按 ``max_chars`` 截断（允许行中截断、尾加 …，
    隐藏行数仍按剩余整行计）。空文本返回 ("", 0)。
    """
    if not text:
        return "", 0
    lines = text.splitlines()
    shown = lines[:max_lines]
    hidden = len(lines) - len(shown)
    partial = "\n".join(shown)
    if len(partial) > max_chars:
        partial = partial[:max_chars] + "…"
    return partial, hidden


def _args_preview(arguments: dict, limit: int = 60) -> str:
    """工具头行的参数预览：紧凑 JSON，超长截断。"""
    preview = json.dumps(arguments, ensure_ascii=False, separators=(", ", ": "))
    if len(preview) > limit:
        preview = preview[:limit] + "…"
    return preview


def _header_arg(name: str, arguments: dict) -> str:
    """工具头行的人话参数：按工具特化（命令/路径/URL 直接可读），兜底紧凑 JSON。"""
    if name == "task" and arguments.get("description") is not None:
        desc = str(arguments["description"]).replace("\n", " ")
        return desc[:60] + ("…" if len(desc) > 60 else "")
    if name == "bash" and arguments.get("command") is not None:
        return "$ " + str(arguments["command"]).replace("\n", " ⏎ ")
    if (
        name in ("read_file", "write_file", "edit_file", "multi_edit", "list_dir")
        and arguments.get("path") is not None
    ):
        arg = str(arguments["path"])
        if name == "read_file" and arguments.get("start_line") is not None:
            end = arguments.get("end_line")
            arg += f":{arguments['start_line']}" + (f"-{end}" if end is not None else "-")
        if name == "multi_edit" and isinstance(arguments.get("edits"), list):
            arg += f" ({len(arguments['edits'])} edits)"
        return arg
    if name in ("grep", "glob") and arguments.get("pattern") is not None:
        arg = str(arguments["pattern"])
        if name == "grep" and arguments.get("glob"):
            arg += f"  ·  glob {arguments['glob']}"
        return arg
    if name == "web_fetch" and arguments.get("url") is not None:
        return str(arguments["url"])
    if name == "todo_write" and isinstance(arguments.get("items"), list):
        return f"{len(arguments['items'])} items"
    if name == "exit_plan_mode":
        return f"Submit plan ({len(str(arguments.get('plan') or ''))} chars)"
    return _args_preview(arguments)


def _fmt_tokens(count: int) -> str:
    """token 数的人话格式：1234 → 1.2k、128000 → 128k。"""
    if count < 1000:
        return str(count)
    value = count / 1000
    return f"{value:.0f}k" if value >= 100 else f"{value:.1f}k"


def _thinking_summary(reasoning: str) -> Text:
    """思考折叠行（Claude Code 的 ✻ 语汇）：字数 + 首行摘要，dim italic 单行。

    不展示秒数——交错 thinking（DeepSeek interleave）下计时含糊。
    """
    stripped = reasoning.strip()
    summary = Text(f"✻ 思考 {len(reasoning)} 字", style="dim italic")
    if stripped:
        first = stripped.splitlines()[0]
        summary.append(f"：{first[:40]}", style="dim italic")
        if len(first) > 40:
            summary.append("…", style="dim italic")
    return summary


class TerminalRenderer:
    """Agent 事件 → 终端渲染。驱动将事件翻译后调用 ``update``。

    ``render`` 只读组装（Live 刷新线程会并发调用，沿用无锁模式：共享量只在
    ``update`` 里原子重绑）。实例本身即上下文管理器：``with renderer:`` 期间
    live 区刷新、退出即消隐；``pause``/``resume`` 供阻塞输入（审批）前后
    暂停，避免状态行盖住 ``input`` 正在等待的那一行。
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        min_render_interval: float = 0.12,
        text_tail_chars: int = 1200,
        reasoning_tail_lines: int = 3,
        max_result_lines: int = 3,
        max_result_chars: int = 400,
        refresh_per_second: int = 10,
        context_window: int = 0,
        tool_output_tail_lines: int = 8,
    ) -> None:
        self.step = 0
        self.max_steps = 0
        self.phase = PHASE_THINKING
        self.phase_t0 = time.monotonic()
        self.current_tool: str | None = None
        self._console = console if console is not None else Console(stderr=True)
        self._min_render_interval = min_render_interval
        self._text_tail_chars = text_tail_chars
        self._reasoning_tail_lines = reasoning_tail_lines
        self._max_result_lines = max_result_lines
        self._max_result_chars = max_result_chars
        self._refresh_per_second = refresh_per_second
        # 上下文占用展示：context_window 为 0（未知）时不显示；占用取最近一次
        # 请求的 prompt_tokens（usage 事件），压缩后回落可见
        self.context_window = context_window
        self._ctx_used: int | None = None
        self._ctx_cached = 0
        self._tool_output_tail_lines = max(1, tool_output_tail_lines)
        self._tool_out: list[str] = []  # 运行中工具的实时输出尾窗（bash tap 喂入）
        self._tool_dropped = 0  # 尾窗装不下而丢弃的行数（渲染 … 标记用）
        self._reasoning_buf: list[str] = []
        self._text_buf: list[str] = []
        self._live_text: Markdown | None = None  # 节流缓存的正文尾窗
        self._dirty = False
        self._last_rebuild = 0.0
        self._live: Live | None = None
        self._blocks: list[dict] = []  # 滚动区已提交块的留档（/expand 用）
        self._max_blocks = 20
        self._next_block_id = 1
        self.scrollback = False
        self._preview_key: tuple[object, ...] | None = None
        self._preview_ansi = ""
        self._preview_at = 0.0
        self._current_arguments: dict = {}
        self.on_status = lambda renderer: None
        self.session_footer: Callable[[], list] | None = None

    def use_scrollback(self, console: Console) -> None:
        """常驻输入模式：仅追加输出，终端刷新完全交给 prompt_toolkit。"""
        self.scrollback = True
        self._console = console

    @property
    def has_preview(self) -> bool:
        return bool(self._text_buf or self.current_tool)

    def preview(self, width: int, max_lines: int = 8) -> str:
        """Render a bounded live tail for prompt_toolkit, without writing to the terminal.

        Render the whole Markdown block before cropping physical lines so lists,
        tables and fenced code keep their structure. Only the UI thread owns this cache.
        """
        text = "".join(self._text_buf)
        tool = self.current_tool
        if not text and not tool:
            return ""
        width, max_lines = max(1, width), max(1, max_lines)
        output = tuple(self._tool_out)
        arguments = self._current_arguments.copy()
        phase = self.phase
        key = (text, tool, output, str(arguments), phase, width, max_lines)
        now = time.monotonic()
        cached_key = self._preview_key
        if key == cached_key:
            return self._preview_ansi
        if (
            cached_key is not None
            and key[1] == cached_key[1]
            and key[3:] == cached_key[3:]
            and now - self._preview_at < self._min_render_interval
        ):
            return self._preview_ansi
        buffer = StringIO()
        console = Console(file=buffer, width=width, force_terminal=True, color_system="standard")
        if text:
            console.print(Markdown(text))
        elif output:
            console.print(Text("\n".join(output), style="dim"))
        lines = buffer.getvalue().splitlines()
        folded = len(lines) > max_lines or (not text and self._tool_dropped > 0)
        header = (
            "⏺" if text else f"{self._status_label()} {_header_arg(tool or '', arguments or {})}"
        )
        if folded:
            header += " · …"
        buffer = StringIO()
        console = Console(file=buffer, width=width, force_terminal=True, color_system="standard")
        console.print(Text(header, style="cyan"), overflow="ellipsis", no_wrap=True)
        self._preview_ansi = buffer.getvalue() + "\n".join(lines[-max_lines:])
        self._preview_ansi = self._preview_ansi.rstrip("\n")
        self._preview_key, self._preview_at = key, now
        return self._preview_ansi

    # ---------- 事件入口 ----------

    def update(self, event: str, payload: dict) -> None:
        if event == "iteration":
            self.step = payload.get("step", self.step)
            self.max_steps = payload.get("max_steps", self.max_steps)
            self.phase = PHASE_THINKING
            self.phase_t0 = time.monotonic()
            self._reasoning_buf.clear()
            self._text_buf.clear()
            self._live_text = None
            self._dirty = False
            self._tool_out.clear()
            self._tool_dropped = 0
        elif event == "reasoning_delta":
            self._reasoning_buf.append(payload.get("delta", ""))
            self._dirty = True
        elif event == "text_delta":
            self._text_buf.append(payload.get("delta", ""))
            self.phase = PHASE_STREAMING
            self._dirty = True
            if not self.scrollback:
                self._maybe_rebuild()
        elif event == "assistant_message":
            self._commit(payload)
        elif event in ("tool_review", "tool_approval"):
            self.phase = "reviewing" if event == "tool_review" else "approval"
            self.current_tool = payload.get("name")
            self._current_arguments = dict(payload.get("arguments") or {})
        elif event == "tool_call":
            self.phase = PHASE_TOOL
            self.phase_t0 = time.monotonic()
            self.current_tool = payload.get("name")
            self._current_arguments = dict(payload.get("arguments") or {})
            self._tool_out.clear()
            self._tool_dropped = 0
            if not self.scrollback:
                self._print_tool_header(payload.get("name") or "?", payload.get("arguments") or {})
        elif event == "tool_output_delta":
            # 运行中工具的实时输出（bash tap 由 CLI 侧直接喂入，不经
            # agent 事件——引擎不感知 UI）。运行中只刷新尾窗，结果完成后归档。
            line = payload.get("line")
            if line:
                self._tool_out.append(str(line))
                overflow = len(self._tool_out) - self._tool_output_tail_lines
                if overflow > 0:
                    del self._tool_out[:overflow]
                    self._tool_dropped += overflow
        elif event == "task_progress":
            # 子代理进度（票 04）：同款尾窗，并重申 task 相位——闸门内的
            # tool_approval 事件会改写单槽，须归位到父 task。
            line = payload.get("line")
            if line:
                self._tool_out.append(str(line))
                overflow = len(self._tool_out) - self._tool_output_tail_lines
                if overflow > 0:
                    del self._tool_out[:overflow]
                    self._tool_dropped += overflow
            self.current_tool = "task"
            self.phase = PHASE_TOOL
        elif event == "plan_approval":
            self.phase = "approval"
        elif event == "plan_result":
            self.phase = PHASE_THINKING
            self._console.print(
                Text(
                    "✔ 计划已批准，继续执行。" if payload.get("approved") else "计划已拒绝。",
                    style="green" if payload.get("approved") else "yellow",
                )
            )
        elif event == "approval_granted":
            self._print_approval(payload)
        elif event == "tool_result":
            self._print_tool_result(payload)
            self.current_tool = None
            self.phase = PHASE_THINKING
            self._tool_out.clear()
            self._tool_dropped = 0
        elif event == "usage":
            # 上下文占用 = 最近一次请求的 prompt_tokens（当前真正在窗内的量）；
            # cached_tokens 是其中命中提示缓存的部分，用于展示缓存收益。
            last = payload.get("last") or {}
            self._ctx_used = last.get("prompt_tokens")
            self._ctx_cached = last.get("cached_tokens") or 0
        elif event == "compaction":
            self._print_compaction(payload)

        self.on_status(self)

    # ---------- 滚动区输出 ----------

    def _record_block(self, block: dict) -> int:
        """滚动区每提交一块就留档（环上限 20），供 /expand 展开全文。"""
        block["id"] = self._next_block_id
        self._next_block_id += 1
        self._blocks.append(block)
        del self._blocks[: -self._max_blocks]
        return block["id"]

    def _commit(self, payload: dict) -> None:
        """一段 assistant 消息完成：思考折叠行 + 完整 Markdown 进滚动区，清缓冲。"""
        # Clear the transient preview before publishing the completed block.
        self._text_buf.clear()
        self._preview_key = None
        reasoning = payload.get("reasoning")
        if reasoning:
            self._console.print(_thinking_summary(reasoning))
            self._record_block({"kind": "thinking", "content": reasoning})
        content = payload.get("content") or ""
        if content.strip():
            # 工具块之后接正文：空行分组，避免与 ⎿ 块连成一片
            if self._blocks and self._blocks[-1]["kind"] == "tool":
                self._console.print()
            if self.scrollback:
                self._console.print(Text("⏺", style="cyan"))
            self._console.print(Markdown(content))
        self._reasoning_buf.clear()
        self._text_buf.clear()
        self._live_text = None
        self._dirty = False

    def _print_approval(self, payload: dict) -> None:
        name = payload["name"]
        arguments = dict(payload["arguments"])
        block_id = self._record_block(
            {
                "kind": "approval",
                "name": name,
                "arguments": arguments,
                "scope": payload.get("scope", "This call only"),
            }
        )
        command = arguments.get("command")
        summary = (
            str(command)
            if command is not None
            else (display_tool_name(name) + " " + _header_arg(name, arguments))
        )
        summary = " ".join(summary.splitlines())
        if len(summary) > 100:
            summary = summary[:100] + "…"
        receipt = Text("✔ ", style="green")
        receipt.append("You approved polya to run ")
        receipt.append(summary, style="dim")
        self._console.print(receipt)
        self._console.print(Text(f"  + Show details: /details {block_id}", style="dim"))

    def _print_tool_header(self, name: str, arguments: dict) -> None:
        self._console.print()  # 块间空行分组
        arg = _header_arg(name, arguments) if arguments else ""
        if len(arg) > 100:
            arg = arg[:100] + "…"
        header = Text()
        header.append("Running ", style="dim")
        header.append(display_tool_name(name), style="bold cyan")
        if arg:
            header.append(f"  {arg}", style="bold" if name == "bash" else "dim")
        self._console.print(header)

    def _print_compaction(self, payload: dict) -> None:
        """压缩事件：全量（LLM 摘要）与微压缩（无 LLM 指针折叠）各一句。"""
        before = payload.get("before", 0)
        after = payload.get("after", 0)
        if payload.get("mode") == "micro":
            cleared = payload.get("cleared", 0)
            message = f"⌁ 上下文微压缩：清理旧工具结果约 {cleared} 字符（可 history_read 回查）"
        else:
            message = f"⌁ 上下文已压缩：{before} → {after} 条消息"
        self._console.print(Text(message, style="dim"))

    def _print_tool_result(self, payload: dict) -> None:
        """Compact completion summary, with a bounded preview and archived details."""
        result = payload.get("result") or ""
        duration = payload.get("duration_s", 0.0)
        body, hidden = _collapse(result, self._max_result_lines, self._max_result_chars)
        exit_match = re.search(r"(?:退出码|Exit code) (-?\d+|-)\s*$", result)
        exit_code = exit_match[1] if exit_match else None
        error = payload.get("error") or exit_code not in (None, "0")
        name = payload.get("name") or self.current_tool or "?"
        status = "Denied" if payload.get("denied") else "Failed" if error else "Ran"
        if (
            name in ("bash", "bash_output")
            and result.rsplit("\n", 1)[-1].startswith("仍在运行")
            and not error
        ):
            status = "Running"
        block_id = self._record_block(
            {
                "kind": "tool",
                "name": name,
                "arguments": self._current_arguments.copy(),
                "result": result,
                "duration_s": duration,
                "status": status,
            }
        )

        header = Text(status + " ", style="red" if error else "dim")
        header.append(display_tool_name(name), style="bold cyan")
        if self.scrollback:
            arg = _header_arg(name, self._current_arguments)
            if arg:
                header.append(f"  {arg[:100]}", style="dim")
        header.append(f" · {duration}s", style="dim")
        if exit_code is not None:
            header.append(f" · exit {exit_code}", style="red" if error else "dim")
        header.append(f" · + Show details: /details {block_id}", style="dim")
        self._console.print(header)
        for line in body.splitlines():
            self._console.print(Text("  " + line, style="red" if error else "dim"))
        if hidden or body != result:
            self._console.print(Text("  … More output in details", style="dim"))
        if not result:
            self._console.print(Text("  No output", style="dim"))

    def expand_blocks(self, count: int = 5) -> str:
        """展开最近 count 块的全文（/expand 命令的输出，纯文本走命令通道）。"""
        if not self._blocks:
            return "（暂无可展开的块——先跑一个任务，或非终端会话不记录）"
        return "\n\n".join(self._block_details(block) for block in self._blocks[-count:])

    def show_details(self, block_id: int) -> str:
        """Stable IDs never silently point at a newer call when the archive rolls over."""
        for block in self._blocks:
            if block["id"] == block_id:
                return self._block_details(block)
        return f"Details #{block_id} unavailable (not found or no longer retained)."

    @staticmethod
    def _block_details(block: dict) -> str:
        if block["kind"] == "thinking":
            content = block["content"]
            return f"✻ 思考全文（{len(content)} 字）：\n{content}"
        name = display_tool_name(block["name"])
        arguments = json.dumps(block["arguments"], ensure_ascii=False, indent=2)
        if block["kind"] == "approval":
            return f"Approved {name}\nScope: {block['scope']}\n{arguments}"
        return f"{block['status']} {name} · {block['duration_s']}s\n{arguments}\n{block['result']}"

    # ---------- live 区 ----------

    def _maybe_rebuild(self) -> None:
        """节流重建正文尾窗：脏且距上次重建超过间隔才动（首帧恒立即）。"""
        if not self._dirty:
            return
        if time.monotonic() - self._last_rebuild < self._min_render_interval:
            return
        text = "".join(self._text_buf)[-self._text_tail_chars :]
        self._live_text = Markdown(text) if text else None
        self._dirty = False
        self._last_rebuild = time.monotonic()

    def _status_label(self) -> str:
        if self.phase == PHASE_TOOL:
            return f"Running {display_tool_name(self.current_tool or '')}"
        if self.phase == "reviewing":
            return "Reviewing"
        if self.phase == "approval":
            return "Awaiting approval"
        if self.phase == PHASE_STREAMING:
            return "Responding"
        if self._reasoning_buf:
            return "Thinking"
        return "Waiting for model"

    def render(self) -> RenderableType:
        """live 区内容：思考尾窗 / 正文尾窗 / 工具实时输出尾窗 / 状态行（纯读）。"""
        parts: list[RenderableType] = []
        if self.phase == PHASE_THINKING and self._reasoning_buf:
            lines = "".join(self._reasoning_buf).splitlines()
            tail = lines[-self._reasoning_tail_lines :]
            if len(tail) < len(lines):
                tail = ["…", *tail]
            parts.append(Text("\n".join(tail), style="dim italic"))
        if self._live_text is not None:
            parts.append(self._live_text)
        if self.phase == PHASE_TOOL and self._tool_out:
            # 长命令运行中不再只有 spinner 干转：输出尾窗实时滚动（全量仍由
            # tool_result 的 ⎿ 块承载，这里只解盲等）
            tail = ["…", *self._tool_out] if self._tool_dropped else list(self._tool_out)
            parts.append(Text("\n".join(tail), style="dim"))
        elapsed = int(time.monotonic() - self.phase_t0)
        label = f" {self._status_label()} · step {self.step}/{self.max_steps} · {elapsed}s"
        if self.context_window and self._ctx_used:
            percent = self._ctx_used * 100 // self.context_window
            label += (
                f" · {_fmt_tokens(self._ctx_used)}/{_fmt_tokens(self.context_window)}（{percent}%）"
            )
            if self._ctx_cached:
                cache_percent = self._ctx_cached * 100 // self._ctx_used
                label += f" · cache {cache_percent}%"
        parts.append(Spinner("dots", text=Text(label, style="cyan")))
        return Group(*parts)

    # ---------- 生命周期 ----------

    def __enter__(self) -> TerminalRenderer:
        if self._console.is_terminal and not self.scrollback:
            self._live = Live(
                get_renderable=self.render,
                console=self._console,
                transient=True,
                refresh_per_second=self._refresh_per_second,
            )
            self._live.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.scrollback:
            partial = "".join(self._text_buf)
            self._text_buf.clear()
            self.current_tool = None
            self._tool_out.clear()
            self._preview_key = None
            if partial.strip():
                self._console.print(Text("⏺ Partial response", style="dim"))
                self._console.print(Markdown(partial))
            self.on_status(self)
        if self._live is not None:
            self._live.stop()
            self._live = None

    def pause(self) -> None:
        """暂停刷新（如审批 ``input`` 前），避免状态行盖住输入行。非终端 no-op。"""
        if self._live is not None:
            self._live.stop()

    def resume(self) -> None:
        if self._live is not None:
            self._live.start()
