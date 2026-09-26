"""pi 风格终端渲染器：滚动区永久追加 + 底部小型 live 区。

Agent 只发事件（``on_event``），本模块把事件流渲染成终端输出：

- **滚动区**（终端原生 scrollback）：完成即打印、永不重绘——思考折叠行、
  工具头行、折叠后的工具结果、完整 Markdown 段落。rich 只在**同一 Console
  实例**内协调 ``print`` 与 ``Live``（print 自动排在 live 区上方），因此滚动
  区与 live 区共用同一个控制台。
- **live 区**（``transient=True``，随 Live 消隐）：spinner 状态行 + 正在流式
  输出的尾窗（思考末几行 / 正文尾窗 Markdown）。增量只置脏、按
  ``min_render_interval`` 节流重建缓存对象，避免增长的 Markdown 被 Live 的
  刷新频率反复重解析。段落完成（assistant_message / 工具块）即提交进滚动区，
  live 区缩回状态行。

非终端环境：不建 Live（``Console.is_terminal`` 判断），滚动区照常打印——
headless 也能看到完整轨迹；``pause``/``resume`` 全部 no-op。
"""

from __future__ import annotations

import json
import time

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
            arg += f"（{len(arguments['edits'])} 处）"
        return arg
    if name in ("grep", "glob") and arguments.get("pattern") is not None:
        arg = str(arguments["pattern"])
        if name == "grep" and arguments.get("glob"):
            arg += f"  ·  glob {arguments['glob']}"
        return arg
    if name == "web_fetch" and arguments.get("url") is not None:
        return str(arguments["url"])
    if name == "todo_write" and isinstance(arguments.get("items"), list):
        return f"{len(arguments['items'])} 项"
    if name == "exit_plan_mode":
        return f"提交计划（{len(str(arguments.get('plan') or ''))} 字）"
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
    """Agent 事件 → 终端渲染。``update`` 即传给 ``Agent.on_event`` 的回调。

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
        max_result_lines: int = 8,
        max_result_chars: int = 600,
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
            self._maybe_rebuild()
        elif event == "assistant_message":
            self._commit(payload)
        elif event == "tool_call":
            self.phase = PHASE_TOOL
            self.phase_t0 = time.monotonic()
            self.current_tool = payload.get("name")
            self._tool_out.clear()
            self._tool_dropped = 0
            self._print_tool_header(payload.get("name") or "?", payload.get("arguments") or {})
        elif event == "tool_output_delta":
            # 运行中工具的实时输出（bash tap 由 CLI 侧直接喂入，不经
            # agent.on_event——引擎不感知 UI）。只进 live 区尾窗，不落滚动区：
            # 全量输出由随后的 tool_result 事件以 ⎿ 块正式提交。
            line = payload.get("line")
            if line:
                self._tool_out.append(str(line))
                overflow = len(self._tool_out) - self._tool_output_tail_lines
                if overflow > 0:
                    del self._tool_out[:overflow]
                    self._tool_dropped += overflow
        elif event == "tool_result":
            self._print_tool_result(payload)
            self.current_tool = None
            self._tool_out.clear()
            self._tool_dropped = 0
        elif event == "usage":
            # 上下文占用 = 最近一次请求的 prompt_tokens（当前真正在窗内的量）
            self._ctx_used = (payload.get("last") or {}).get("prompt_tokens")

    # ---------- 滚动区输出 ----------

    def _record_block(self, block: dict) -> None:
        """滚动区每提交一块就留档（环上限 20），供 /expand 展开全文。"""
        self._blocks.append(block)
        del self._blocks[: -self._max_blocks]

    def _commit(self, payload: dict) -> None:
        """一段 assistant 消息完成：思考折叠行 + 完整 Markdown 进滚动区，清缓冲。"""
        reasoning = payload.get("reasoning")
        if reasoning:
            self._console.print(_thinking_summary(reasoning))
            self._record_block({"kind": "thinking", "content": reasoning})
        content = payload.get("content") or ""
        if content.strip():
            # 工具块之后接正文：空行分组，避免与 ⎿ 块连成一片
            if self._blocks and self._blocks[-1]["kind"] == "tool":
                self._console.print()
            self._console.print(Markdown(content))
        self._reasoning_buf.clear()
        self._text_buf.clear()
        self._live_text = None
        self._dirty = False

    def _print_tool_header(self, name: str, arguments: dict) -> None:
        self._console.print()  # 块间空行分组
        arg = _header_arg(name, arguments)
        if len(arg) > 100:
            arg = arg[:100] + "…"
        header = Text()
        header.append("⏺ ", style="bold cyan")
        header.append(name, style="cyan")
        if arg:
            header.append(f"  {arg}", style="dim")
        self._console.print(header)

    def _print_tool_result(self, payload: dict) -> None:
        """结果用 Claude Code 的树形连接符：首行 ``  ⎿ ``、续行 4 空格对齐。"""
        result = payload.get("result") or ""
        duration = payload.get("duration_s", 0.0)
        body, hidden = _collapse(result, self._max_result_lines, self._max_result_chars)
        style = "red" if payload.get("error") else None
        if not result:
            self._console.print(Text(f"  ⎿ （无输出） · {duration}s", style="dim"))
        else:
            lines = body.splitlines()
            for index, line in enumerate(lines):
                prefix = "  ⎿ " if index == 0 else "    "
                text = Text(prefix + line, style=style)
                if index == len(lines) - 1 and not hidden:
                    text.append(f" · {duration}s", style="dim")
                self._console.print(text)
            if hidden:
                tail = f"    … 还有 {hidden} 行未显示（/expand 查看全文） · {duration}s"
                self._console.print(Text(tail, style="dim"))
        block = {
            "kind": "tool",
            "name": payload.get("name") or "?",
            "result": result,
            "duration_s": duration,
        }
        self._record_block(block)

    def expand_blocks(self, count: int = 5) -> str:
        """展开最近 count 块的全文（/expand 命令的输出，纯文本走命令通道）。"""
        if not self._blocks:
            return "（暂无可展开的块——先跑一个任务，或非终端会话不记录）"
        parts = []
        for block in self._blocks[-count:]:
            if block["kind"] == "thinking":
                content = block["content"]
                parts.append(f"✻ 思考全文（{len(content)} 字）：\n{content}")
            else:
                name, result = block["name"], block["result"]
                duration = block["duration_s"]
                parts.append(f"⏺ {name} 结果全文 · {duration}s：\n{result}")
        return "\n\n".join(parts)

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
            return f"运行 {self.current_tool or ''}"
        if self.phase == PHASE_STREAMING:
            return "回复中"
        return "思考中"

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
        label = f" {self._status_label()} · 第 {self.step}/{self.max_steps} 轮 · {elapsed}s"
        if self.context_window and self._ctx_used:
            percent = self._ctx_used * 100 // self.context_window
            label += (
                f" · {_fmt_tokens(self._ctx_used)}/{_fmt_tokens(self.context_window)}（{percent}%）"
            )
        parts.append(Spinner("dots", text=Text(label, style="cyan")))
        return Group(*parts)

    # ---------- 生命周期 ----------

    def __enter__(self) -> TerminalRenderer:
        if self._console.is_terminal:
            self._live = Live(
                get_renderable=self.render,
                console=self._console,
                transient=True,
                refresh_per_second=self._refresh_per_second,
            )
            self._live.start()
        return self

    def __exit__(self, *exc) -> None:
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
