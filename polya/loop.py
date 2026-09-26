"""交互驱动（ADR 0002 的消费方）：消费 agent 生成器 → 渲染 / 权限判定 / 审批 /
执行 → ``send`` 回结果。

- 事件适配：``renderer.update(ev.event, event_payload(ev))``——事件对象到渲染器
  词表的同名同键翻译，渲染器内部接口不动。
- 工具执行权在本层（executor 共用）；bash 实时输出经 ``default_tools(
  on_shell_output=)`` 的 tap 直喂渲染器（装配在 cli.build_agent）——执行期流是
  驱动层事务，不是引擎旁路。
- 中断分级：任务执行中 Ctrl+C 捕获后 ``gen.close()``，agent 在 GeneratorExit
  路径回填未决 ToolCall（历史完整可续）；输入处 Ctrl+C 双击 / Ctrl+D 退出。
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .agent import Agent, PlanSubmitted, ToolCall, event_payload
from .executor import execute
from .input import InputBox
from .permissions import Context, Rule, decide, rule_for
from .render import TerminalRenderer, console, ui
from .todos import _STATUS_LABELS
from .tools import Tool

HELP_TEXT = """\
命令：
  /help            显示本帮助
  /todos           显示当前 TODO 清单
  /status          显示会话状态（模式 / 历史 / 工具计数 / token 用量）
  /plan on|off     开启/关闭规划模式（只读约束 + 计划审批）
  /expand [N]      展开最近 N 块（默认 5）的工具结果 / 思考全文
  /reset           清空对话历史、TODO 与统计
  /exit, /quit     退出（输入处 Ctrl+D / Ctrl+C 双击同效）"""


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


def _ask_line(label: str, default: str | None = None) -> str:
    """审批辅助输入行（拒绝理由 / 修改命令）。独立函数便于测试替身。"""
    from prompt_toolkit import prompt

    try:
        return prompt(label, default=default or "")
    except (EOFError, KeyboardInterrupt):
        return default or ""


class ApprovalOutcome:
    """一次审批的结果：放行与否 + 可能的规则 / 改写命令 / 拒绝理由。"""

    def __init__(
        self,
        approved: bool,
        rule: Rule | None = None,
        command: str | None = None,
        reason: str | None = None,
    ):
        self.approved = approved
        self.rule = rule
        self.command = command
        self.reason = reason


class ApprovalGate:
    """会话审批闸门（spec「审批交互」）：四选项、默认拒绝、会话级授权规则累积。

    光标默认停在「拒绝」——Enter 单按绝不放行；高危不给前缀授权出口（Q9）；
    修改后执行仅 bash（Q13）；复合命令 / 无法取前缀的工具不给选项 2（Q14/Q18）。
    """

    def __init__(
        self,
        interactive: bool,
        renderer: TerminalRenderer | None = None,
        root: str | os.PathLike[str] | None = None,
    ):
        self.interactive = interactive
        self.renderer = renderer
        self.root_path = Path(root).resolve() if root is not None else None
        self.rules: list[Rule] = []

    def screen(self, tool: Tool, arguments: dict, high_risk: bool = False) -> ApprovalOutcome:
        """渲染变更预览并弹四选项；非交互环境预览后直接拒绝。"""
        if not self.interactive:
            ui.print(
                Panel(
                    Group(*_approval_body(tool.name, arguments, self.root_path)),
                    title="非交互环境，默认拒绝",
                    border_style="red",
                )
            )
            return ApprovalOutcome(approved=False)

        rule = None if high_risk else rule_for(tool, arguments)
        can_modify = tool.kind == "exec" and isinstance(arguments.get("command"), str)
        # 动态选项表：高危不给授权出口（Q9）；取不出前缀不给（Q14/Q18）
        options: list[tuple[str, str]] = [("允许", "执行本次调用")]
        if rule is not None:
            options.append(("本会话前缀授权", f"{rule} 起不再询问"))
        if can_modify:
            options.append(("修改后执行", "预填原命令，改完执行（仅 bash）"))
        deny_index = len(options)
        options.append(("拒绝", "不执行，让模型调整方案（可附理由）"))

        if self.renderer is not None:
            self.renderer.pause()
        try:
            ui.print(
                Panel(
                    Group(*_approval_body(tool.name, arguments, self.root_path)),
                    title="危险工具执行审批",
                    border_style="yellow",
                )
            )
            choice = _select_option(options, cancel_index=deny_index, initial=deny_index)
        finally:
            if self.renderer is not None:
                self.renderer.resume()

        if choice == 0:
            return ApprovalOutcome(approved=True)
        if rule is not None and choice == 1:
            return ApprovalOutcome(approved=True, rule=rule)
        if can_modify and choice == (2 if rule is not None else 1):
            modified = _ask_line("修改命令: ", default=str(arguments["command"]))
            return ApprovalOutcome(approved=bool(modified.strip()), command=modified.strip())
        reason = _ask_line("拒绝理由（回车跳过）: ").strip() or None
        return ApprovalOutcome(approved=False, reason=reason)

    def as_approve(self):
        """兼容 approve(tool, arguments) -> bool 钩子（内置驱动 run() 用）：
        非交互即预览+拒绝，交互路径正常四选项（规则入会话集）。"""

        def approve(tool: Tool, arguments: dict) -> bool:
            if not tool.dangerous:
                return True
            if any(r.matches(tool, arguments) for r in self.rules):
                return True
            outcome = self.screen(tool, arguments)
            if outcome.rule is not None:
                self.rules.append(outcome.rule)
                ui.print(f"本会话内 {outcome.rule} 起不再询问", style="dim")
            if not outcome.approved and outcome.reason:
                ui.print(f"理由：{outcome.reason}", style="dim")
            return outcome.approved

        return approve


def terminal_approve(
    interactive: bool,
    renderer: TerminalRenderer | None = None,
    root: str | os.PathLike[str] | None = None,
):
    """构建终端审批回调（旧签名兼容，内置驱动 run() / -p 模式用）。

    询问前 ``renderer.pause()``、结束后 ``resume()``：阻塞输入期间若刷新
    线程仍在重绘，状态行会盖住用户正在交互的那一行。
    """
    return ApprovalGate(interactive, renderer, root).as_approve()


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


# ---------- 生成器驱动 ----------


def _run_tool(
    agent: Agent, renderer: TerminalRenderer, ev: ToolCall, interactive: bool, gate: ApprovalGate
) -> str:
    """一次工具调用的驱动侧处理：渲染 → decide → 审批 → 执行 → 渲染结果。"""
    renderer.update("tool_call", event_payload(ev))
    item = agent.tools.get(ev.name)
    if item is None:
        result, duration = f"Error: unknown tool '{ev.name}'", 0.0
    else:
        decision = decide(
            item, ev.arguments, Context(plan=agent.plan_mode, rules=tuple(gate.rules))
        )
        if decision.verdict == "deny":
            result, duration = decision.reason, 0.0
        elif decision.verdict == "ask" and interactive and item.dangerous:
            outcome = gate.screen(item, ev.arguments, high_risk=(decision.reason == "high-risk"))
            if outcome.rule is not None:
                gate.rules.append(outcome.rule)
                ui.print(f"本会话内 {outcome.rule} 起不再询问", style="dim")
            if not outcome.approved:
                suffix = f"：{outcome.reason}" if outcome.reason else ""
                result, duration = f"Error: 用户拒绝了工具调用 {ev.name}{suffix}", 0.0
            else:
                arguments = dict(ev.arguments)
                if outcome.command is not None:  # 修改后执行（Q13：仅 bash）
                    arguments["command"] = outcome.command
                result, duration = execute(item, arguments)
        else:
            result, duration = execute(item, ev.arguments)
    renderer.update(
        "tool_result",
        {
            "name": ev.name,
            "call_id": ev.call_id,
            "result": result,
            "duration_s": round(duration, 3),
            "error": result.startswith("Error"),
        },
    )
    return result


def run_task(
    agent: Agent,
    renderer: TerminalRenderer,
    text: str,
    interactive: bool,
    gate: ApprovalGate | None = None,
) -> None:
    """消费一次 ``steps()``：事件转发渲染器，ToolCall/PlanSubmitted 就地处理。

    Ctrl+C 落在驱动侧（审批 / 执行）时 ``gen.close()`` 触发 agent 的
    GeneratorExit 回填，历史保持合法后中断向上传播。
    """
    if gate is None:
        gate = ApprovalGate(interactive)
    gen = agent.steps(text)
    to_send = None
    try:
        with renderer:
            while True:
                try:
                    ev = gen.send(to_send)
                except StopIteration:
                    return
                to_send = None
                if isinstance(ev, ToolCall):
                    to_send = _run_tool(agent, renderer, ev, interactive, gate)
                elif isinstance(ev, PlanSubmitted):
                    to_send = agent._handle_plan(ev.plan)
                else:
                    renderer.update(ev.event, event_payload(ev))
    finally:
        gen.close()


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


def run_repl(agent: Agent, root: str, renderer: TerminalRenderer) -> None:
    """交互主循环：斜杠命令 / ``!`` shell / ``#`` 记忆本地处理，其余交给 agent。

    **tty 分流**：终端下渲染器消费 ``steps()`` 事件流实时渲染，滚动区与
    live 区统一走 ``ui`` 控制台；非终端下走内置驱动 ``run()``，答案只走
    stdout（``console``），可安全管道。异常兜底 ``Exception``：API/网络错误
    只报本次任务，REPL 必须存活。
    """
    interactive = sys.stdin.isatty()
    box = InputBox() if interactive else None
    gate = ApprovalGate(interactive, renderer, root)
    state: dict = {"topic": None, "model": agent.llm.model}  # 输入框状态栏数据
    if interactive:
        if ui.is_terminal:
            ui.set_window_title("polya")

    def say(message: str, style: str) -> None:
        if interactive:
            ui.print(message, style=style, markup=False)
        else:
            console.print(message, style=style, markup=False)

    say(f"polya（模型: {agent.llm.model}，工作目录: {os.path.abspath(root)}）", "none")
    say(
        "输入任务开始；/help 命令 · ! 跑 shell · # 记项目记忆；Ctrl+C 双击或 Ctrl+D 退出。\n",
        "none",
    )
    while True:
        state["mode"] = "规划" if agent.plan_mode else "执行"
        state["rules"] = len(gate.rules)
        try:
            if box is not None:
                user_input = box.ask(state).strip()
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
        if user_input.startswith("!") and len(user_input) > 1:
            _run_shell_bang(agent, root, user_input[1:].strip(), say)
            continue
        if user_input.startswith("#") and len(user_input) > 1:
            _append_project_memory(root, user_input[1:].strip(), say)
            continue
        try:
            if interactive:
                run_task(agent, renderer, user_input, interactive, gate)
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
