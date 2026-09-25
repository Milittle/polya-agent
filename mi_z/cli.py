"""mi-z 命令行入口：交互式 REPL 与单任务模式。

用法::

    uv run mi-z                     # 交互式 REPL（当前目录为工作目录）
    uv run mi-z -p "修复测试" --plan  # 单任务模式：执行一次即退出

交互设计对标《深入理解 AI Agent》chapter5 coding-agent 的 main.py：输入处的
EOF / Ctrl+C 统一翻译成 ``/quit`` 复用命令分发（避免三处退出逻辑）；中断分级——
输入处中断=退出，``run()`` 执行中中断=仅终止本轮回提示符（历史保留，缺失的
tool 结果由 Agent 补齐后序列仍合法）。与它不同的是我们有终端审批：危险工具
执行前 ``y/N/a`` 交互确认，这是 mi-z 的 ``approve`` 钩子接到终端的实现。

显示层用 rich：模型回答按 Markdown 渲染、审批与计划用 Panel 框出。命令输出
（/todos、/status 等）仍走裸 ``print``——它们含 ``[1]``、``[in_progress]`` 这类
方括号，交给 rich 会被当标记解析。凡带用户/模型内容的 rich 输出一律 ``markup=False``
或以 ``Text``/``Markdown`` 包裹，避免内容里的方括号触发标记错误。

输入层用 prompt_toolkit（**仅当 stdin 是终端**）：历史持久化 + 斜杠命令补全。
管道/CI 下退回裸 ``input``，不在非 tty 环境里驱动全屏行编辑器。

任务执行期间用 rich ``Live`` 在底部显示实时状态（第 N 轮 / 当前工具），事件由
``Agent.on_event`` 钩子推送；日志经 ``RichHandler`` 排在状态区上方，避免打断。
Live 同样仅在终端启用；本项是唯一的核心改动（Agent 多了一个可选回调）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich_argparse import RichHelpFormatter

from .agent import Agent
from .builtin import CODING_SYSTEM_PROMPT, default_tools
from .llm import LLM
from .providers import profile_for
from .todos import TodoStore
from .tools import Tool

console = Console()

SLASH_COMMANDS = ["/help", "/todos", "/status", "/plan", "/reset", "/exit", "/quit"]

HELP_TEXT = """\
命令：
  /help            显示本帮助
  /todos           显示当前 TODO 清单
  /status          显示会话状态（模式 / 历史 / 工具计数 / token 用量）
  /plan on|off     开启/关闭规划模式（只读约束 + 计划审批）
  /reset           清空对话历史、TODO 与统计
  /exit, /quit     退出（输入处 Ctrl+D / Ctrl+C 同效）"""


def handle_command(cmd: str, agent: Agent) -> str | None:
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
        return "\n".join(
            f"[{index}] [{item['status']}] {item['content']}" for index, item in enumerate(items, 1)
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
    if name == "/reset":
        agent.reset()
        return "已清空对话历史、TODO 与统计。"
    return f"未知命令 {name}，/help 查看可用命令。"


def terminal_approve(interactive: bool):
    """构建终端审批回调：危险工具执行前询问。

    ``y`` 本次允许，``n`` 拒绝，``a`` 本会话内该工具自动放行（免重复确认）。
    非交互环境（管道/CI）默认拒绝——宁可打断任务，不静默执行写操作。
    """
    allowed: set[str] = set()

    def approve(tool: Tool, arguments: dict) -> bool:
        if not tool.dangerous:
            return True
        if tool.name in allowed:
            return True
        preview = str(arguments)
        if len(preview) > 120:
            preview = preview[:120] + "…"
        if not interactive:
            console.print(
                Panel(
                    Text(f"{tool.name}({preview})"),
                    title="非交互环境，默认拒绝",
                    border_style="red",
                )
            )
            return False
        console.print(
            Panel(
                Text(f"{tool.name}({preview})"),
                title="危险工具执行审批",
                subtitle="[green]y[/] 允许  [red]n[/] 拒绝  [yellow]a[/] 本会话内自动放行",
                border_style="yellow",
            )
        )
        try:
            answer = input("确认 [y/N/a] ").strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes"):
            return True
        if answer == "a":
            allowed.add(tool.name)
            console.print(f"本会话内 {tool.name} 不再询问", style="dim")
            return True
        return False

    return approve


def terminal_approve_plan(interactive: bool):
    """构建计划审批回调：打印计划全文后询问。非交互环境默认拒绝（--yes 可全自动）。"""

    def approve_plan(plan: str) -> bool:
        if not interactive:
            console.print("非交互环境，默认拒绝计划；用 --yes 自动批准", style="red")
            return False
        console.print(Panel(Markdown(plan), title="执行计划", border_style="cyan"))
        try:
            answer = input("批准该计划并进入执行模式? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")

    return approve_plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mi-z",
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


def build_agent(args: argparse.Namespace, llm=None) -> Agent:
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
        tools=default_tools(root=args.root, todos=todos),
        system_prompt=CODING_SYSTEM_PROMPT,
        approve=None if args.yes else terminal_approve(interactive),
        approve_plan=None if args.yes else terminal_approve_plan(interactive),
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


def make_session() -> PromptSession:
    """构建带历史与斜杠补全的输入会话；历史持久化到 ``~/.mi_z_history``。"""
    return PromptSession(
        history=FileHistory(str(Path.home() / ".mi_z_history")),
        completer=SlashCommandCompleter(),
        auto_suggest=AutoSuggestFromHistory(),
    )


class LiveStatusBar:
    """把 Agent 的进度事件渲染成 rich Spinner，供 Live 实时刷新。

    ``update`` 即传给 ``Agent.on_event`` 的回调；``render`` 读当前状态生成可
    渲染对象（每帧调用一次，故状态须在实例上而非闭包外）。
    """

    def __init__(self) -> None:
        self.step = 0
        self.max_steps = 0
        self.tool: str | None = None

    def update(self, event: str, payload: dict) -> None:
        if event == "iteration":
            self.step = payload.get("step", self.step)
            self.max_steps = payload.get("max_steps", self.max_steps)
            self.tool = None  # 新一轮开始：上一轮的工具名作废
        elif event == "tool_call":
            self.tool = payload.get("name")

    def render(self):
        label = f"第 {self.step}/{self.max_steps} 轮"
        if self.tool:
            label += f" · 调用 {self.tool}"
        return Spinner("dots", text=Text(f" {label}…", style="cyan"))


def status_live(status: LiveStatusBar):
    """任务执行期间在终端底部显示实时状态；非终端（管道/CI）下为 no-op。"""
    if not console.is_terminal:
        return nullcontext()
    return Live(
        get_renderable=status.render,
        console=console,
        transient=True,
        refresh_per_second=10,
    )


def repl(agent: Agent, root: str) -> None:
    """交互主循环：斜杠命令本地处理，其余输入交给 agent。

    stdin 是终端时用 prompt_toolkit（历史 + 补全），否则退回 ``input``；任务执行
    期间用 Live 显示实时状态（仅终端）。``session.prompt`` 与 ``Live`` 都独占终端，
    但二者不同时活跃（Live 只在 ``run()`` 期间开，之后随即关闭），故不冲突。
    """
    print(f"mi-z（模型: {agent.llm.model}，工作目录: {os.path.abspath(root)}）")
    print("输入任务开始；/help 查看命令；Ctrl+D 退出。规划模式可用 /plan on 开启。\n")
    session = make_session() if sys.stdin.isatty() else None
    status = LiveStatusBar()
    agent.on_event = status.update
    while True:
        try:
            if session is not None:
                user_input = session.prompt("> ").strip()
            else:
                user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input.startswith("/"):
            output = handle_command(user_input, agent)
            if output is None:
                return
            print(output)
            continue
        try:
            with status_live(status):
                answer = agent.run(user_input)
            console.print(Markdown(answer))
            console.print(f"[用量] {agent.total_usage}", style="dim", markup=False)
        except KeyboardInterrupt:
            console.print(
                "\n[已中断本次任务；已完成步骤保留在历史中，可继续对话或 /reset 重来]",
                style="yellow",
                markup=False,
            )
        except RuntimeError as exc:
            console.print(f"[任务失败] {exc}", style="red", markup=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()  # 与 demo.py 一致：从项目 .env 读取 OPENAI_* 配置
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
        handlers=[
            RichHandler(
                # 日志走 stderr：stdout 只承载答案（-p 模式可安全重定向/管道）。与
                # stdout 上的 Live 经 stderr 重定向协同——日志自动排在状态区上方。
                console=Console(stderr=True),
                show_time=False,
                show_level=False,
                show_path=False,
                markup=False,
                highlighter=NullHighlighter(),
            )
        ],
    )
    agent = build_agent(args)
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
    repl(agent, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
