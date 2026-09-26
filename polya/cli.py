"""polya 命令行入口：交互式 REPL 与单任务模式。

用法::

    uv run polya                     # 交互式 REPL（当前目录为工作目录）
    uv run polya -p "修复测试" --plan  # 单任务模式：执行一次即退出

交互设计对标 pi（badlogic/pi-mono）的极简风格（详见 polya/ui.py）：滚动区永久
追加 + 底部小型 live 区。中断分级——输入处 EOF / Ctrl+C 统一退出；``run()``
执行中 Ctrl+C 仅终止本轮回提示符（历史保留，缺失的 tool 结果由 Agent 补齐后
序列仍合法；流式被打断时半截内容不落历史，序列天然合法）。

**tty 与管道分流**：stdin 是终端时，``TerminalRenderer`` 接管全部用户可见输出
（流式正文、工具块、状态行统一走 ``ui``/stderr 控制台——rich 只在同一 Console
内协调 Live 与滚动区打印），答案由 ``assistant_message`` 事件实时渲染；非终端
（-p、管道喂 stdin）完全不接渲染器，答案照旧只走 stdout，可安全重定向/管道。

审批：危险工具执行前展示**变更预览**（写类工具与磁盘现状比对的红绿 diff、bash
的完整命令），随后弹出 ``❯`` 选择列表（允许 / 本会话内总是允许 / 拒绝，
↑/↓+Enter 或数字键，光标默认停在拒绝），面板用 Panel 框出；阻塞输入前
``renderer.pause()`` 暂停 live 区刷新。状态行随 usage 事件展示上下文占用；
bash 运行中的输出经 tap 实时进 live 区尾窗。命令输出（/todos、/status 等）
含 ``[1]`` 这类方括号，走 rich 时一律 ``markup=False``。
"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich_argparse import RichHelpFormatter

from .agent import Agent
from .builtin import CODING_SYSTEM_PROMPT, default_tools
from .llm import LLM
from .providers import profile_for
from .todos import _STATUS_LABELS, TodoStore
from .tools import Tool
from .ui import TerminalRenderer

console = Console()  # stdout：只承载答案与命令输出（-p 可安全重定向/管道）
ui = Console(stderr=True)  # stderr：状态条 / 日志 / 审批面板等“界面”输出

SLASH_COMMANDS = ["/help", "/todos", "/status", "/plan", "/expand", "/reset", "/exit", "/quit"]

HELP_TEXT = """\
命令：
  /help            显示本帮助
  /todos           显示当前 TODO 清单
  /status          显示会话状态（模式 / 历史 / 工具计数 / token 用量）
  /plan on|off     开启/关闭规划模式（只读约束 + 计划审批）
  /expand [N]      展开最近 N 块（默认 5）的工具结果 / 思考全文
  /reset           清空对话历史、TODO 与统计
  /exit, /quit     退出（输入处 Ctrl+D / Ctrl+C 同效）"""


def handle_command(cmd: str, agent: Agent, renderer: TerminalRenderer | None = None) -> str | None:
    """处理一条斜杠命令，返回要打印的文本；返回 ``None`` 表示退出 REPL。"""
    name, _, arg = cmd.partition(" ")
    arg = arg.strip()
    if name in ("/exit", "/quit"):
        return None
    if name == "/help":
        return HELP_TEXT
    if name == "/todos":
        items = agent.todos.as_dicts()
        if not items:
            return "（TODO 清单为空）"
        # 状态标签与状态栏（status.py）同源中文，两处展示不打架
        return "\n".join(
            f"[{index}] [{_STATUS_LABELS[item['status']]}] {item['content']}"
            for index, item in enumerate(items, 1)
        )
    if name == "/status":
        mode = "规划中（只读）" if agent.plan_mode else "执行"
        calls = dict(agent.tool_counts)
        return "\n".join(
            [
                f"模式: {mode}",
                f"历史消息: {len(agent.history)} 条",
                f"工具调用: {calls if calls else '（无）'}",
                f"token 用量: {agent.total_usage}",
            ]
        )
    if name == "/plan":
        if arg == "on":
            agent.plan_mode = True
            if agent.tools.get("exit_plan_mode") is None:
                return "已进入规划模式（注意：未注册 exit_plan_mode 工具，计划无法提交批准）。"
            return "已进入规划模式：只读探查，模型完成计划后会调用 exit_plan_mode 提交。"
        if arg == "off":
            agent.plan_mode = False
            return "已退出规划模式。"
        return "用法: /plan on|off"
    if name == "/expand":
        try:
            count = int(arg) if arg else 5
        except ValueError:
            return "用法: /expand [N]（N 为块数，默认 5）"
        if renderer is None:
            return "（非终端会话不记录块，无法展开）"
        return renderer.expand_blocks(count)
    if name == "/reset":
        agent.reset()
        return "已清空对话历史、TODO 与统计。"
    return f"未知命令 {name}，/help 查看可用命令。"


def _select_option(
    options: list[tuple[str, str]], *, cancel_index: int, initial: int | None = None
) -> int:
    """渲染一个 ``❯`` 单选列表：↑/↓ 移动、Enter 确认、数字键直达、Esc 取消。

    与主输入框同为 prompt_toolkit，同一套 ``❯`` 视觉语言。每个选项是
    ``(标签, 说明)``；``cancel_index`` 是 Esc/EOF 的落点。Ctrl+C 照旧抛
    KeyboardInterrupt（沿 run 循环的中断分级）。
    """
    state = {"index": cancel_index if initial is None else initial}

    def fragments():
        rows = []
        for index, (label, desc) in enumerate(options):
            selected = index == state["index"]
            marker = "❯ " if selected else "  "
            rows.append(
                ("class:selected" if selected else "class:option", f"{marker}{index + 1}. {label}")
            )
            if desc:
                rows.append(("class:desc", f" — {desc}"))
            rows.append(("", "\n"))
        rows.append(("class:desc", "↑/↓ 选择 · Enter 确认 · 数字直达 · Esc 取消"))
        return rows

    bindings = KeyBindings()

    def choose(index: int):
        def handler(event):
            state["index"] = index
            event.app.exit(result=index)

        return handler

    for index in range(len(options)):
        bindings.add(str(index + 1))(choose(index))

    @bindings.add("up")
    def _up(event):
        state["index"] = (state["index"] - 1) % len(options)

    @bindings.add("down")
    def _down(event):
        state["index"] = (state["index"] + 1) % len(options)

    @bindings.add("enter")
    def _enter(event):
        event.app.exit(result=state["index"])

    @bindings.add("escape")
    def _escape(event):
        state["index"] = cancel_index
        event.app.exit(result=cancel_index)

    app = Application(
        layout=Layout(Window(FormattedTextControl(fragments, show_cursor=False))),
        key_bindings=bindings,
        style=Style.from_dict(
            {"selected": "bold cyan", "option": "", "desc": "fg:ansibrightblack"}
        ),
        full_screen=False,
    )
    try:
        return app.run()
    except EOFError:
        return cancel_index


def _apply_edits_draft(text: str, edits: list[dict]) -> tuple[str, str | None]:
    """在内存里模拟 multi_edit（语义与 builtin 一致）：返回 (草稿, 失败原因)。

    失败原因非 None 时真实执行也会失败——预览照给（diff 仍展示已可确定的部分），
    但把失败处明确标出，不制造「批准了却没执行」的错觉。
    """
    draft = text
    for index, edit in enumerate(edits, 1):
        old = str(edit.get("old_string", ""))
        new = str(edit.get("new_string", ""))
        if not old:
            return draft, f"第 {index} 处编辑缺少 old_string"
        count = draft.count(old)
        if count == 0:
            return draft, f"第 {index} 处编辑未找到 old_string（执行将失败）"
        if count > 1 and not edit.get("replace_all"):
            return draft, f"第 {index} 处 old_string 出现 {count} 次，不唯一（执行将失败）"
        draft = draft.replace(old, new) if edit.get("replace_all") else draft.replace(old, new, 1)
    return draft, None


def _diff_lines(old: str, new: str, path: str, max_lines: int = 40) -> list[str]:
    """行级 unified diff（上下文 2 行）：红删绿增的原料，超长截断并标注。"""
    diff = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return ["（内容无变化）"]
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"… 还有 {len(diff) - max_lines} 行未显示（批准后执行完整变更）"]
    return diff


def _approval_body(name: str, arguments: dict, root: Path | None) -> list[Text]:
    """审批面板正文：写类工具给 diff、bash 给完整命令、其余兜底参数预览。"""
    if name == "bash" and arguments.get("command") is not None:
        lines = [f"$ {part}" for part in str(arguments["command"]).splitlines() or [""]]
        if len(lines) > 8:
            lines = lines[:8] + [f"… 还有 {len(lines) - 8} 行"]
        return [Text(line, style="yellow") for line in lines]
    if name in ("write_file", "edit_file", "multi_edit") and root is not None:
        path = str(arguments.get("path", ""))
        body: list[Text] = []
        try:
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                return [Text(f"路径越界，将被拒绝：{path}", style="red")]
            old = target.read_text(encoding="utf-8") if target.is_file() else ""
        except (OSError, UnicodeDecodeError):
            return [Text("（无法读取原文件，diff 预览不可用）", style="dim")]
        error = None
        if name == "write_file":
            new = str(arguments.get("content", ""))
        elif name == "edit_file":
            new, error = _apply_edits_draft(
                old,
                [
                    {
                        "old_string": arguments.get("old_string", ""),
                        "new_string": arguments.get("new_string", ""),
                        "replace_all": arguments.get("replace_all", False),
                    }
                ],
            )
        else:
            edits = arguments.get("edits") if isinstance(arguments.get("edits"), list) else []
            new, error = _apply_edits_draft(old, edits)
        if error:
            body.append(Text(error, style="red"))
        diff = _diff_lines(old, new, path)
        body.extend(
            Text(
                line,
                style="green"
                if line.startswith("+")
                else "red"
                if line.startswith("-")
                else "cyan"
                if line.startswith("@")
                else None,
            )
            for line in diff
        )
        return body
    preview = str(arguments)
    if len(preview) > 120:
        preview = preview[:120] + "…"
    return [Text(f"{name}({preview})")]


def terminal_approve(
    interactive: bool,
    renderer: TerminalRenderer | None = None,
    root: str | os.PathLike[str] | None = None,
):
    """构建终端审批回调：危险工具执行前展示**变更预览**并弹出选择列表。

    选项：允许本次 / 本会话内总是允许（该工具不再询问）/ 拒绝。光标默认停在
    「拒绝」——Enter 单按绝不放行，放行须 ↑ 或数字键主动选择。写类工具先算
    diff（与磁盘上的现状比对）再问——绝不盲批；bash 展示完整命令。非交互
    环境（管道/CI）默认拒绝——宁可打断任务，不静默执行写操作。

    询问前 ``renderer.pause()``、结束后 ``resume()``：阻塞输入期间若刷新
    线程仍在重绘，状态行会盖住用户正在交互的那一行。
    """
    allowed: set[str] = set()
    root_path = Path(root).resolve() if root is not None else None

    def approve(tool: Tool, arguments: dict) -> bool:
        if not tool.dangerous:
            return True
        if tool.name in allowed:
            return True
        if not interactive:
            ui.print(
                Panel(
                    Group(*_approval_body(tool.name, arguments, root_path)),
                    title="非交互环境，默认拒绝",
                    border_style="red",
                )
            )
            return False
        if renderer is not None:
            renderer.pause()
        try:
            ui.print(
                Panel(
                    Group(*_approval_body(tool.name, arguments, root_path)),
                    title="危险工具执行审批",
                    border_style="yellow",
                )
            )
            choice = _select_option(
                [
                    ("允许", "执行本次调用"),
                    ("总是允许", f"本会话内 {tool.name} 不再询问"),
                    ("拒绝", "不执行，让模型调整方案"),
                ],
                cancel_index=2,
            )
        finally:
            if renderer is not None:
                renderer.resume()
        if choice == 0:
            return True
        if choice == 1:
            allowed.add(tool.name)
            ui.print(f"本会话内 {tool.name} 不再询问", style="dim")
            return True
        return False

    return approve


def terminal_approve_plan(interactive: bool, renderer: TerminalRenderer | None = None):
    """构建计划审批回调：打印计划全文后弹出选择列表。非交互环境默认拒绝（--yes 可全自动）。"""

    def approve_plan(plan: str) -> bool:
        if not interactive:
            ui.print("非交互环境，默认拒绝计划；用 --yes 自动批准", style="red")
            return False
        if renderer is not None:
            renderer.pause()
        try:
            ui.print(Panel(Markdown(plan), title="执行计划", border_style="cyan"))
            choice = _select_option(
                [
                    ("批准", "按计划进入执行模式（写操作仍受审批）"),
                    ("拒绝", "继续规划，修改后重新提交"),
                ],
                cancel_index=1,
            )
        finally:
            if renderer is not None:
                renderer.resume()
        return choice == 0

    return approve_plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="polya",
        description="本地编码代理：读文件、改代码、跑命令，干活前先问一句。",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("--root", default=".", help="工作目录，所有文件操作被限制在内（默认 .）")
    parser.add_argument(
        "-p",
        "--prompt",
        help="单任务模式：执行一次任务后打印答案并退出（不做交互确认，需配合 --yes）",
    )
    parser.add_argument(
        "--plan", action="store_true", help="启动时进入规划模式（先探查提计划，批准后才动写操作）"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="自动批准一切审批（危险工具与计划）；仅在信任任务时使用",
    )
    parser.add_argument(
        "--max-steps", type=int, default=25, help="单次任务的最大迭代轮数（默认 25）"
    )
    parser.add_argument("--model", help="模型名（默认取 OPENAI_MODEL 环境变量）")
    parser.add_argument("--base-url", help="API 地址（默认取 OPENAI_BASE_URL 环境变量）")
    parser.add_argument("--api-key", help="API key（默认取 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--no-status", action="store_true", help="关闭 Agent 状态栏（调试用）")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出，仍显示分块进度（不支持流式的端点用）",
    )
    parser.add_argument("--no-compress", action="store_true", help="关闭上下文压缩（默认开启）")
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="上下文窗口（token），超过 80%% 触发压缩（默认取模型档案，如 128000）",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=30,
        help="压缩保留区：最近 N 条消息内的工具结果不压缩、状态栏不删除（默认 30）",
    )
    parser.add_argument(
        "--prefix-check",
        action="store_true",
        help="开启前缀不变量断言：每次请求必须是上一次的严格扩展（调试用，有比较开销）",
    )
    return parser.parse_args(argv)


def build_agent(
    args: argparse.Namespace, llm=None, renderer: TerminalRenderer | None = None
) -> Agent:
    """按 CLI 参数构建 Agent。``llm`` 参数供测试注入假实现。"""
    todos = TodoStore()
    if llm is None:
        # 模型档案决定温度等默认参数（o 系列不接受自定义温度），Agent 侧再用
        # 同一份档案决定压缩策略与前缀纪律
        profile = profile_for(args.model or os.getenv("OPENAI_MODEL"))
        llm = LLM(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=profile.temperature,
        )
    else:
        profile = profile_for(getattr(llm, "model", None))
    interactive = sys.stdin.isatty()
    return Agent(
        llm=llm,
        tools=default_tools(
            root=args.root,
            todos=todos,
            # bash 运行中的实时输出直接喂渲染器（agent 线程内同步回调）。
            # 引擎不感知 UI；headless 下 renderer.update 只积累不打印，无副作用。
            on_shell_output=(
                (lambda line: renderer.update("tool_output_delta", {"name": "bash", "line": line}))
                if renderer is not None
                else None
            ),
        ),
        system_prompt=CODING_SYSTEM_PROMPT,
        approve=None if args.yes else terminal_approve(interactive, renderer, root=args.root),
        approve_plan=None if args.yes else terminal_approve_plan(interactive, renderer),
        status_bar=not args.no_status,
        todos=todos,
        plan_mode=args.plan,
        plan_capable=True,  # exit_plan_mode 构造时注册，/plan 随时切换而不动工具数组
        max_steps=args.max_steps,
        compress=not args.no_compress,
        context_window=args.context_window,
        keep_recent=args.keep_recent,
        profile=profile,
        prefix_check=args.prefix_check,
        stream=not args.no_stream,
    )


class SlashCommandCompleter(Completer):
    """只补全开头的斜杠命令：整行不是命令（普通任务）时不给任何建议。

    一旦出现空格（如 ``/plan on`` 的参数部分）即停止——命令名补全到此为止。
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command in SLASH_COMMANDS:
            if command.startswith(text):
                yield Completion(command, start_position=-len(text))


def _display_width(text: str) -> int:
    """粗略显示宽度：CJK 记 2 列（横线填充用，不追求精确 wcwidth）。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def _rule(label: str, width: int) -> str:
    """一条 ─ 横线，label 嵌在开头（Claude Code 输入框的上下框线）。"""
    label = f" {label} " if label else ""
    fill = max(0, width - _display_width(label) - 2)
    return "──" + label + "─" * fill


def _prompt_message(state: dict) -> list:
    """输入框顶线 + ``❯`` 提示符；主题已知时嵌在顶线（``── ✳ topic ──``）。"""
    width = shutil.get_terminal_size((100, 24)).columns
    topic = state.get("topic")
    rule = _rule(f"✳ {topic}" if topic else "", width)
    return [("", "\n"), ("class:rule", rule), ("class:rule", "\n"), ("class:prompt", "❯ ")]


def make_session() -> PromptSession:
    """Claude Code 风格的输入框：上下两条横线 + ``❯`` 提示符 + 多行编辑。

    顶线（message，随会话主题更新）与底线（bottom_toolbar，按键提示）围出输入
    区；按键约定：**Enter 提交**，Alt+Enter（Escape,Enter）插入换行，行尾反斜杠
    + Enter 是换行的备用入口（部分终端会吃掉 Alt 键）。历史、斜杠补全、灰色
    历史建议照旧；多行提交整体入历史。提示符每轮由 ``repl`` 传入（主题可变）。
    """
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def _submit(event):
        buffer = event.current_buffer
        document = buffer.document
        if not (document.is_cursor_at_the_end and document.on_last_line):
            buffer.insert_text("\n")  # 光标在行中间/非末行：Enter 当换行用
            return
        if document.current_line_before_cursor.endswith("\\"):
            buffer.delete_before_cursor(1)  # 经典续行：去掉反斜杠换行
            buffer.insert_text("\n")
            return
        buffer.validate_and_handle()

    def bottom_rule() -> list:
        width = shutil.get_terminal_size((100, 24)).columns
        return [("class:rule", _rule("Enter 发送 · Alt+Enter 换行 · /help 命令", width))]

    state_dir = Path.home() / ".polya"  # 状态目录：输入历史现居于此，配置将放这里
    state_dir.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: [("class:continuation", "… ")],
        bottom_toolbar=bottom_rule,
        placeholder=[("class:placeholder", "输入任务（Alt+Enter 换行），/help 查看命令")],
        style=Style.from_dict(
            {
                "prompt": "bold cyan",
                "rule": "fg:ansibrightblack",
                "bottom-toolbar": "bg:default fg:ansibrightblack",
                "continuation": "dim",
                "placeholder": "dim",
            }
        ),
        history=FileHistory(str(state_dir / "history")),
        completer=SlashCommandCompleter(),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
    )


def _slugify(text: str, max_len: int = 48) -> str:
    """压成 kebab-case slug：小写、只留 [a-z0-9-]、空白/标点归并为分隔符。"""
    text = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    return text[:max_len].rstrip("-")


def _topic_from(first_input: str) -> str:
    """从首个任务**本地**推断会话主题：slug 化用户输入，纯中文退化为截断原文。

    不为此调 LLM——发布级产品不在首任务里藏一次隐性额外请求（延迟与费用
    都不可见）。主题只是装饰，绝不影响会话。
    """
    return _slugify(first_input) or first_input.strip().replace("\n", " ")[:24] or "new-session"


def repl(agent: Agent, root: str, renderer: TerminalRenderer) -> None:
    """交互主循环：斜杠命令本地处理，其余输入交给 agent。

    **tty 分流**（见模块 docstring）：终端下渲染器接 ``agent.on_event``，答案由
    ``assistant_message`` 事件实时渲染（流式），滚动区与 live 区统一走 ``ui``
    控制台；非终端下不接渲染器，答案照旧走 stdout（``console``），可安全管道。
    异常兜底 ``Exception``：API/网络错误只报本次任务，REPL 必须存活。
    """
    interactive = sys.stdin.isatty()
    session = make_session() if interactive else None
    state: dict = {"topic": None}  # 会话主题（首任务后推断），嵌入输入框顶线与终端标题
    if interactive:
        agent.on_event = renderer.update
        if ui.is_terminal:
            ui.set_window_title("polya")

    def say(message: str, style: str) -> None:
        if interactive:
            ui.print(message, style=style, markup=False)
        else:
            console.print(message, style=style, markup=False)

    say(f"polya（模型: {agent.llm.model}，工作目录: {os.path.abspath(root)}）", "none")
    say("输入任务开始；/help 查看命令；Ctrl+D 退出。规划模式可用 /plan on 开启。\n", "none")
    while True:
        try:
            if session is not None:
                user_input = session.prompt(lambda: _prompt_message(state)).strip()
            else:
                user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input.startswith("/"):
            output = handle_command(user_input, agent, renderer)
            if output is None:
                return
            say(output, "none")
            continue
        try:
            if interactive:
                # 答案已在 assistant_message 事件里实时渲染，这里只补用量行
                with renderer:
                    agent.run(user_input)
                ui.print(f"[用量] {agent.total_usage}", style="dim", markup=False)
            else:
                console.print(Markdown(agent.run(user_input)))
                console.print(f"[用量] {agent.total_usage}", style="dim", markup=False)
        except KeyboardInterrupt:
            say(
                "\n[已中断本次任务；已完成步骤保留在历史中，可继续对话或 /reset 重来]",
                "yellow",
            )
        except Exception as exc:  # noqa: BLE001 - REPL 必须存活：API/网络错误只报本次任务
            say(f"[任务失败] {type(exc).__name__}: {exc}", "red")
        if interactive and state["topic"] is None:
            # Claude Code 同款：首个任务后定会话主题，嵌入输入框顶线并写终端标签页
            state["topic"] = _topic_from(user_input)
            if ui.is_terminal:
                ui.set_window_title(f"✳ {state['topic']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()  # 与 demo.py 一致：从项目 .env 读取 OPENAI_* 配置
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
        handlers=[
            RichHandler(
                # 与渲染器的 Live 共用 ui(stderr) 控制台：rich 只在同一 Console 实例内
                # 协调 Live 与其它输出，日志才会排在 live 区上方而非与 spinner 撞行；
                # stderr 也保证 stdout 只承载答案（-p 可安全重定向/管道）。
                console=ui,
                show_time=False,
                show_level=False,
                show_path=False,
                markup=False,
                highlighter=NullHighlighter(),
            )
        ],
    )
    # SDK 的 HTTP 明细日志（httpx2）会混进 live 区刷屏，抬到 WARNING 只留异常
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    renderer = TerminalRenderer(ui)
    agent = build_agent(args, renderer=renderer)
    renderer.context_window = agent.context_window  # 状态行的上下文占用展示用
    if args.prompt is not None:
        try:
            console.print(Markdown(agent.run(args.prompt)))
        except KeyboardInterrupt:
            print("\n[已中断]", file=sys.stderr)
            return 130
        except RuntimeError as exc:
            print(f"[任务失败] {exc}", file=sys.stderr)
            return 1
        console.print(f"[用量] {agent.total_usage}", style="dim", markup=False)
        return 0
    repl(agent, args.root, renderer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
