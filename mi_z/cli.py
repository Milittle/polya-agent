"""mi-z 命令行入口：交互式 REPL 与单任务模式。

用法::

    uv run mi-z                     # 交互式 REPL（当前目录为工作目录）
    uv run mi-z -p "修复测试" --plan  # 单任务模式：执行一次即退出

交互设计对标《深入理解 AI Agent》chapter5 coding-agent 的 main.py：输入处的
EOF / Ctrl+C 统一翻译成 ``/quit`` 复用命令分发（避免三处退出逻辑）；中断分级——
输入处中断=退出，``run()`` 执行中中断=仅终止本轮回提示符（历史保留，缺失的
tool 结果由 Agent 补齐后序列仍合法）。与它不同的是我们有终端审批：危险工具
执行前 ``y/N/a`` 交互确认，这是 mi-z 的 ``approve`` 钩子接到终端的实现。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from .agent import Agent
from .builtin import CODING_SYSTEM_PROMPT, default_tools
from .llm import LLM
from .providers import profile_for
from .todos import TodoStore
from .tools import Tool

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
        if not interactive:
            print(f"[非交互环境，默认拒绝] {tool.name}({arguments})")
            return False
        preview = str(arguments)
        if len(preview) > 120:
            preview = preview[:120] + "…"
        try:
            answer = input(f"允许执行 {tool.name}({preview})? [y/N/a] ").strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes"):
            return True
        if answer == "a":
            allowed.add(tool.name)
            print(f"[本会话内 {tool.name} 不再询问]")
            return True
        return False

    return approve


def terminal_approve_plan(interactive: bool):
    """构建计划审批回调：打印计划全文后询问。非交互环境默认拒绝（--yes 可全自动）。"""

    def approve_plan(plan: str) -> bool:
        if not interactive:
            print("[非交互环境，默认拒绝计划；用 --yes 自动批准]")
            return False
        print("\n" + "=" * 60 + f"\n{plan}\n" + "=" * 60)
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


def repl(agent: Agent, root: str) -> None:
    """交互主循环：斜杠命令本地处理，其余输入交给 agent。"""
    print(f"mi-z（模型: {agent.llm.model}，工作目录: {os.path.abspath(root)}）")
    print("输入任务开始；/help 查看命令；Ctrl+D 退出。规划模式可用 /plan on 开启。\n")
    while True:
        try:
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
            print(agent.run(user_input))
            print(f"\n[用量] {agent.total_usage}")
        except KeyboardInterrupt:
            print("\n[已中断本次任务；已完成步骤保留在历史中，可继续对话或 /reset 重来]")
        except RuntimeError as exc:
            print(f"[任务失败] {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()  # 与 demo.py 一致：从项目 .env 读取 OPENAI_* 配置
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    agent = build_agent(args)
    if args.prompt is not None:
        try:
            print(agent.run(args.prompt))
        except KeyboardInterrupt:
            print("\n[已中断]", file=sys.stderr)
            return 130
        except RuntimeError as exc:
            print(f"[任务失败] {exc}", file=sys.stderr)
            return 1
        print(f"\n[用量] {agent.total_usage}")
        return 0
    repl(agent, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
