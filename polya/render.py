"""Agent 事件的终端渲染器。

正文按段落边界实时提交进滚动区（原生 scrollback），未完成的尾部留在编辑器
上方的短尾窗（prompt_toolkit 通过 ``preview()`` 自取）；工具结果一次写入。
思考与工具结果保留有界归档，供 /details 使用；流式 bash 结果不重复打印。
"""

from __future__ import annotations

import json
import re
import time
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

console = Console()  # stdout：只承载答案与命令输出（-p 可安全重定向/管道）
ui = Console(stderr=True)  # stderr：状态条 / 日志等“界面”输出

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


def _stream_cut(text: str, in_fence: bool, budget: int) -> tuple[int, bool]:
    """正文流式提交点：返回 (可提交字符数, 切点之后的围栏态)，0 表示暂不提交。

    优先切在「围栏外的空行」——段落边界，保住表格/列表/代码块的完整；没有空行
    且完整行数超过 ``budget`` 时，退到围栏外的最后一个换行。切点始终落在围栏外，
    代码块不会被劈成多段。
    """
    lines = text.split("\n")
    fence = in_fence
    para_cut = line_cut = 0
    para_fence = line_fence = in_fence
    pos = 0
    for index, line in enumerate(lines):
        last = index == len(lines) - 1
        end = pos + len(line) + (0 if last else 1)
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if not last and not fence:
            line_cut, line_fence = end, fence
            if not line.strip():
                para_cut, para_fence = end, fence
        pos = end
    if para_cut:
        return para_cut, para_fence
    if line_cut and len(lines) - 1 > budget:
        return line_cut, line_fence
    return 0, in_fence


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


def _thinking_summary(reasoning: str) -> Text:
    """思考标题行（Claude Code 的 ✻ 语汇）：字数，dim italic 单行。

    不展示秒数——交错 thinking（DeepSeek interleave）下计时含糊。尾窗正文由
    ``_reasoning_tail`` 逐行落在标题下方，标题只报字数。
    """
    return Text(f"✻ 思考 {len(reasoning)} 字", style="dim italic")


def _reasoning_tail(reasoning: str, limit: int) -> list[str]:
    """思考尾部：末 ``limit`` 行，被截断时前置一行省略号。

    流式尾窗与定稿共用同一切法：思考结束时落进滚动区的那几行，和过程中滚动的
    尾窗一样大。
    """
    lines = reasoning.splitlines()
    tail = lines[-limit:]
    if len(tail) < len(lines):
        tail = ["…", *tail]
    return tail


class TerminalRenderer:
    """Agent 事件 → 终端渲染。驱动将事件翻译后调用 ``update``。

    正文按段落边界实时提交进滚动区，未完成的尾部留在编辑器上方的尾窗
    （prompt_toolkit 通过 ``preview()`` 自取）。实例即上下文管理器：
    ``with renderer:`` 退出时把未提交正文标记为 Partial response 留在滚动区。
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        min_render_interval: float = 0.12,
        max_result_lines: int = 3,
        max_result_chars: int = 400,
        context_window: int = 0,
        tool_output_tail_lines: int = 8,
        reasoning_tail_lines: int = 3,
        text_flush_lines: int = 8,
    ) -> None:
        self.step = 0
        self.max_steps = 0
        self.phase = PHASE_THINKING
        self.phase_t0 = time.monotonic()
        self.current_tool: str | None = None
        self._console = console if console is not None else Console(stderr=True)
        self._min_render_interval = min_render_interval
        self._max_result_lines = max_result_lines
        self._max_result_chars = max_result_chars
        # 上下文占用展示：context_window 为 0（未知）时不显示；占用取最近一次
        # 请求的 prompt_tokens（usage 事件），压缩后回落可见
        self.context_window = context_window
        self._ctx_used: int | None = None
        self._ctx_cached = 0
        self.total_usage: dict = {}  # 会话累计用量（usage 事件的 total）
        self._tool_output_tail_lines = max(1, tool_output_tail_lines)
        self._reasoning_tail_lines = max(1, reasoning_tail_lines)  # 思考尾窗行数
        self._text_flush_lines = max(1, text_flush_lines)  # 无空行时的正文提交行预算
        self._tool_out: list[str] = []  # 运行中工具的实时输出尾窗（bash tap 喂入）
        self._tool_dropped = 0  # 尾窗装不下而丢弃的行数（渲染 … 标记用）
        self._reasoning_buf: list[str] = []
        self._text_buf: list[str] = []
        self._answer_started = False  # 本段正文的 ⏺ 标题是否已打（流式多块只打一次）
        self._in_fence = False  # 正文是否停在代码围栏内（提交切点须避开围栏）
        self._blocks: list[dict] = []  # 滚动区已提交块的留档（/details 用）
        self._max_blocks = 20
        self._next_block_id = 1
        self.scrollback = False
        self._preview_key: tuple[object, ...] | None = None
        self._preview_ansi = ""
        self._preview_at = 0.0
        self._current_arguments: dict = {}
        self.on_status = lambda renderer: None

    def use_scrollback(self, console: Console) -> None:
        """常驻输入模式：仅追加输出，终端刷新完全交给 prompt_toolkit。"""
        self.scrollback = True
        self._console = console

    @property
    def has_preview(self) -> bool:
        return bool(self._text_buf or self.current_tool or self._reasoning_buf)

    def preview(self, width: int, max_lines: int = 8) -> str:
        """Render a bounded live tail for prompt_toolkit, without writing to the terminal.

        Render the whole Markdown block before cropping physical lines so lists,
        tables and fenced code keep their structure. Only the UI thread owns this cache.
        """
        text = "".join(self._text_buf)
        tool = self.current_tool
        reasoning = "".join(self._reasoning_buf)
        if not text and not tool and not reasoning:
            return ""
        width, max_lines = max(1, width), max(1, max_lines)
        output = tuple(self._tool_out)
        arguments = self._current_arguments.copy()
        phase = self.phase
        # reasoning 放在「增长字段」区（与 text/output 同侧）：节流只比较结构字段
        # （key[1] tool 与 key[4:] arguments/phase/宽高），思考逐片增长不触发重渲。
        key = (text, tool, output, reasoning, str(arguments), phase, width, max_lines)
        now = time.monotonic()
        cached_key = self._preview_key
        if key == cached_key:
            return self._preview_ansi
        if (
            cached_key is not None
            and key[1] == cached_key[1]
            and key[4:] == cached_key[4:]
            and now - self._preview_at < self._min_render_interval
        ):
            return self._preview_ansi
        buffer = StringIO()
        console = Console(file=buffer, width=width, force_terminal=True, color_system="standard")
        if text:
            console.print(Markdown(text))
        elif output:
            console.print(Text("\n".join(output), style="dim"))
        elif reasoning and not tool:
            # 思考尾窗（dim italic）：流式思考的存活信号。全文只在 commit 后按尾窗
            # 大小落滚动区、经 /details 展开——不实时落思考，避免 append-only 刷屏。
            tail = _reasoning_tail(reasoning, self._reasoning_tail_lines)
            console.print(Text("\n".join(tail), style="dim italic"))
        lines = buffer.getvalue().splitlines()
        folded = len(lines) > max_lines or (not text and self._tool_dropped > 0)
        if text:
            header = "⏺"
        elif reasoning and not tool:
            header = f"✻ 思考中 · {len(reasoning)} 字"
        else:
            header = f"{self._status_label()} {_header_arg(tool or '', arguments or {})}"
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
            self._tool_out.clear()
            self._tool_dropped = 0
            self._answer_started = False
            self._in_fence = False
        elif event == "reasoning_delta":
            self._reasoning_buf.append(payload.get("delta", ""))
        elif event == "text_delta":
            self._text_buf.append(payload.get("delta", ""))
            self.phase = PHASE_STREAMING
            self._stream_text()
        elif event == "assistant_message":
            self._commit(payload)
        elif event == "tool_review":
            self.phase = "reviewing"
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
            # 闸门内的事件会改写单槽，须归位到父 task。
            line = payload.get("line")
            if line:
                self._tool_out.append(str(line))
                overflow = len(self._tool_out) - self._tool_output_tail_lines
                if overflow > 0:
                    del self._tool_out[:overflow]
                    self._tool_dropped += overflow
            self.current_tool = "task"
            self.phase = PHASE_TOOL
        elif event == "plan_submitted":
            self.phase = PHASE_THINKING
            self._print_plan(payload.get("plan") or "")
        elif event == "tool_result":
            self._print_tool_result(payload)
            self.current_tool = None
            self.phase = PHASE_THINKING
            # 该工具轮的思考已随 assistant_message 折叠进滚动区，清掉以免尾窗回映。
            self._reasoning_buf.clear()
            self._tool_out.clear()
            self._tool_dropped = 0
        elif event == "usage":
            # 上下文占用 = 最近一次请求的 prompt_tokens（当前真正在窗内的量）；
            # cached_tokens 是其中命中提示缓存的部分，用于展示缓存收益。
            # total 为会话累计（跨压缩保留），状态栏的 ↑/↓/R 取它。
            last = payload.get("last") or {}
            self._ctx_used = last.get("prompt_tokens")
            self._ctx_cached = last.get("cached_tokens") or 0
            self.total_usage = payload.get("total") or {}
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
        """一段 assistant 消息完成：补交正文尾部 + 思考尾窗落滚动区，清缓冲。"""
        # Clear the transient preview before publishing the completed block.
        self._preview_key = None
        reasoning = payload.get("reasoning")
        if reasoning:
            self._print_thinking(reasoning)
            self._record_block({"kind": "thinking", "content": reasoning})
        content = payload.get("content") or ""
        # 流式期间完整段落已落滚动区，这里只补未提交的尾部；无 delta 的非流式
        # 回复（text_buf 空且 ⏺ 未打）直接落 content。
        text = "".join(self._text_buf)
        if text or not self._answer_started:
            self._print_answer(text or content)
        self._reasoning_buf.clear()
        self._text_buf.clear()
        self._answer_started = False
        self._in_fence = False

    def _stream_text(self) -> None:
        """正文流式提交（路线 B）：以段落边界为切点，完整块即时渲染进滚动区。

        未打完的尾部留在小尾窗；无空行且行数超预算时退到围栏外的最后一个换行，
        保证长块也能持续走、尾窗不无限增长。
        """
        text = "".join(self._text_buf)
        cut, in_fence = _stream_cut(text, self._in_fence, self._text_flush_lines)
        if not cut:
            return
        self._print_answer(text[:cut])
        self._in_fence = in_fence
        self._text_buf = [text[cut:]] if text[cut:] else []

    def _print_answer(self, chunk: str) -> None:
        """把正文的一块渲染进滚动区；「⏺」标题与工具块后的空行只打一次。"""
        if not chunk.strip():
            return
        if not self._answer_started:
            if self._blocks and self._blocks[-1]["kind"] == "tool":
                self._console.print()  # 与 ⎿ 块空行分组
            if self.scrollback:
                self._console.print(Text("⏺", style="cyan"))
            self._answer_started = True
        self._console.print(Markdown(chunk))

    def _print_thinking(self, reasoning: str) -> None:
        """思考定稿：字数标题 + 尾窗大小的灰斜体行（与流式尾窗对得上）。"""
        self._console.print(_thinking_summary(reasoning))
        for line in _reasoning_tail(reasoning, self._reasoning_tail_lines):
            self._console.print(Text(line, style="dim italic"))

    def _print_plan(self, plan: str) -> None:
        """计划提交：全文进滚动区（不再弹阻断式选择器）。"""
        if not plan.strip():
            return
        if self._blocks and self._blocks[-1]["kind"] == "tool":
            self._console.print()
        self._console.print(Text("⏺ 计划已提交", style="cyan"))
        self._console.print(Markdown(plan))
        self._console.print(
            Text("  /plan go 开始执行 · 或直接输入修改意见（仍在计划模式）", style="dim")
        )

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
        return f"{block['status']} {name} · {block['duration_s']}s\n{arguments}\n{block['result']}"

    # ---------- 状态标签（preview 头行用） ----------

    def _status_label(self) -> str:
        if self.phase == PHASE_TOOL:
            return f"Running {display_tool_name(self.current_tool or '')}"
        if self.phase == "reviewing":
            return "Reviewing"
        if self.phase == PHASE_STREAMING:
            return "Responding"
        if self._reasoning_buf:
            return "Thinking"
        return "Waiting for model"

    # ---------- 生命周期 ----------

    def __enter__(self) -> TerminalRenderer:
        return self

    def __exit__(self, *exc) -> None:
        # 中断时把尚未提交的正文标记为 Partial response 留在滚动区，清理 tail。
        partial = "".join(self._text_buf)
        self._text_buf.clear()
        # 中断于思考阶段时半截思考不能留在尾窗（has_preview 含 reasoning，须显式清）。
        self._reasoning_buf.clear()
        self.current_tool = None
        self._tool_out.clear()
        self._preview_key = None
        self._answer_started = False
        self._in_fence = False
        if partial.strip():
            self._console.print(Text("⏺ Partial response", style="dim"))
            self._console.print(Markdown(partial))
        self.on_status(self)
