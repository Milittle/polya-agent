"""polya 命令行入口：参数解析、装配与单任务模式。

用法::

    uv run polya                     # 交互式 REPL（当前目录为工作目录）
    uv run polya -p "修复测试" --plan  # 单任务模式：执行一次即退出

交互驱动在 polya/loop.py（生成器协议的消费者），渲染在 polya/render.py，
输入在 polya/input.py——本模块只做 argparse、Agent 装配与 tty 分流：

- 终端：``TerminalRenderer(ui)`` 消费事件流，bash 实时输出经
  ``on_shell_output`` tap 直喂渲染器（执行期流是驱动层事务，见 loop.py）。
- 非终端（-p、管道）：不接渲染器，走内置驱动 ``run()``，答案只走 stdout，
  可安全重定向/管道。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from .agent import Agent
from .builtin import CODING_SYSTEM_PROMPT, default_tools
from .llm import LLM
from .loop import run_repl, terminal_approve, terminal_approve_plan
from .providers import profile_for
from .render import TerminalRenderer, console, ui
from .todos import TodoStore


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


def _project_memory(root: str) -> str | None:
    """读项目根 AGENTS.md（项目记忆，spec「项目记忆」）：存在才读，启动一次。"""
    path = Path(root) / "AGENTS.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return content or None


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
    # 项目记忆（AGENTS.md）启动读一次、拼进系统提示词尾部——会话内不变，不违
    # 「系统提示词静态」铁律的精神（铁律防的是逐轮变更破缓存；# 前缀写入后
    # 下次会话生效）。缺失即跳过。
    system_prompt = CODING_SYSTEM_PROMPT
    memory = _project_memory(args.root)
    if memory:
        system_prompt = f"{CODING_SYSTEM_PROMPT}\n\n# 项目记忆（AGENTS.md，启动时载入）\n\n{memory}"
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
        system_prompt=system_prompt,
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
    run_repl(agent, args.root, renderer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
