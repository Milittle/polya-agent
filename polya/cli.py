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
from .loop import run_repl, run_tool_call
from .models import ModelsConfig, resolve_connection, resolve_context_window
from .providers import profile_for
from .render import TerminalRenderer, console, ui
from .skills import SkillCatalog
from .subagent import SubagentRunner
from .todos import TodoStore
from .trust import has_trust_requiring_resources, is_trusted, set_decision
from .trust import trust as trust_path


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
        help="单任务模式：执行一次任务后打印答案并退出（默认放行，以进程权限运行）",
    )
    parser.add_argument(
        "--plan", action="store_true", help="启动时进入规划模式（先探查提计划，批准后才动写操作）"
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
        "--no-microcompact",
        action="store_true",
        help="关闭微压缩（无 LLM 的旧工具结果指针清理，默认开启，阈值 60%%）",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="上下文窗口（token），触发压缩的阈值由此推算（默认取模型档案，如 128000）",
    )
    parser.add_argument(
        "--reserve-tokens",
        type=int,
        default=16384,
        help="全量压缩的绝对预留（token）：小窗口下先于 80%% 触发，保证留出响应空间（默认 16384）",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=30,
        help="压缩保留区：最近 N 条消息内的工具结果不压缩、状态栏不删除（默认 30）",
    )
    parser.add_argument(
        "--keep-recent-tokens",
        type=int,
        default=None,
        help="压缩保留区的 token 预算（优先于 --keep-recent；按消息从末向前累积）",
    )
    parser.add_argument(
        "--prefix-check",
        action="store_true",
        help="开启前缀不变量断言：每次请求必须是上一次的严格扩展（调试用，有比较开销）",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="信任当前目录：加载项目 AGENTS.md / 项目 skills（非交互场景必需）",
    )
    parser.add_argument(
        "--no-trust",
        action="store_true",
        help="不信任当前目录：落 false 决定并按未信任启动（覆盖父目录继承）",
    )
    return parser.parse_args(argv)


def _confirm_trust(root: str) -> bool:
    """交互式信任门：一行 y/N（普通 input，不抢终端）。默认不信任。"""
    path = str(Path(root).resolve())
    ui.print(f"首次在此目录使用 polya：{path}")
    ui.print("信任后才能加载该目录的 AGENTS.md 与项目 skills（工具照常以进程权限运行）。")
    try:
        answer = input("信任此目录并继续？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes", "是")


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
        # 启动解析（票 14/03）：旗标 > active provider（~/.polya/models.json）> 环境变量。
        # 模型档案决定温度等默认参数（o 系列不接受自定义温度）；窗口走解析链
        # （条目发现值 → 静态表 → 默认，票 03），显式 --context-window 优先。
        config = ModelsConfig.load()
        model, base_url, api_key, profile_name = resolve_connection(
            args.model, args.base_url, args.api_key, config
        )
        profile = profile_for(model)
        active = config.active_entry()
        context_window = (
            args.context_window
            if args.context_window is not None
            else resolve_context_window(model, active[1] if active else None)
        )
        llm = LLM(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=profile.temperature,
            profile_name=profile_name,
        )
    else:
        profile = profile_for(getattr(llm, "model", None))
        context_window = args.context_window
    interactive = sys.stdin.isatty()
    # 项目信任门（trust.py，pi 模型）：未信任不加载项目资源（AGENTS.md / 项目
    # skills），防陌生仓库的指令注入；工具仍以进程权限在 root 内运行。
    # --trust / --no-trust 覆盖；否则查 trust.json（含父目录继承）。
    if args.trust:
        trusted = True
    elif args.no_trust:
        trusted = False
    else:
        trusted = is_trusted(args.root)
    memory = _project_memory(args.root) if trusted else None
    runner = SubagentRunner(root=args.root, memory=memory, renderer=renderer)
    tools = [
        *default_tools(
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
        runner.task_tool(),  # 子代理（票 03/04）：kind=delegate，构建期注册先于 freeze
    ]
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt=CODING_SYSTEM_PROMPT,
        project_memory=memory,
        cwd=args.root,
        # 默认放行：审查器缝在 Agent（review.py，AllowAllReviewer）；plan 只读
        # 约束由它实现。未来模型审查器（Jev 类）实现同一协议接入。
        status_bar=not args.no_status,
        todos=todos,
        plan_mode=args.plan,
        plan_capable=True,  # exit_plan_mode 构造时注册，/plan 随时切换而不动工具数组
        max_steps=args.max_steps,
        compress=not args.no_compress,
        context_window=context_window,
        micro_threshold=None if args.no_microcompact else 0.6,
        keep_recent=args.keep_recent,
        keep_recent_tokens=args.keep_recent_tokens,
        reserve_tokens=args.reserve_tokens,
        profile=profile,
        prefix_check=args.prefix_check,
        stream=not args.no_stream,
        skills=SkillCatalog.discover(args.root, trusted=trusted),
    )
    runner.attach(agent)
    agent.trusted = trusted  # /trust 状态查询用（会话启动时的实际信任态）
    # 默认 runner（-p / 管道）：走子 Agent 自带审查器（继承父 reviewer）。交互
    # REPL 会在 InteractiveSession 里重绑 dispatch 到父渲染器与共享审查器。
    runner.bind(None, interactive)

    def _child_dispatch(child, ev):
        return run_tool_call(child, renderer, ev, child.reviewer, origin="子任务")

    runner.dispatch = _child_dispatch
    return agent


def main(argv: list[str] | None = None) -> int:
    # profile 的录入与管理全部在会话内 /models（add 交互向导），无外置子命令
    args = parse_args(argv)
    load_dotenv()  # 与 demo.py 一致：从项目 .env 读取 OPENAI_* 配置
    # 信任门：--trust / --no-trust 直接落盘；交互首次进「有可保护资源」的陌生
    # 目录问一次；非交互不弹门（未信任即不加载项目资源，可用 --trust 显式声明）。
    if args.trust:
        trust_path(args.root)
    elif args.no_trust:
        set_decision(args.root, False)
    elif (
        args.prompt is None
        and sys.stdin.isatty()
        and has_trust_requiring_resources(args.root)
        and not is_trusted(args.root)
    ):
        if not _confirm_trust(args.root):
            return 0
        trust_path(args.root)
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
    try:
        agent = build_agent(args, renderer=renderer)
    except ValueError as exc:  # models.json 坏配置：指到文件，不甩 traceback
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
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
