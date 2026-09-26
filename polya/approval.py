"""Terminal approval and session authorization, shared by both agent drivers."""

from __future__ import annotations

import difflib
import os
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .permissions import Context, Rule, assess, decide, rule_for
from .render import TerminalRenderer, display_tool_name, ui
from .tools import Tool


def _select_option(
    options: list[tuple[str, str]],
    *,
    cancel_index: int,
    initial: int | None = None,
    footer: Callable[[], list] | None = None,
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

    def rule():
        return [("class:desc", "─" * get_app().output.get_size().columns)]

    app: Application[int] = Application(
        layout=Layout(
            HSplit(
                [
                    Window(FormattedTextControl(rule), height=1),
                    Window(
                        FormattedTextControl(fragments, show_cursor=False), dont_extend_height=True
                    ),
                    Window(FormattedTextControl(rule), height=1),
                    Window(
                        FormattedTextControl(
                            [
                                (
                                    "class:desc",
                                    "↑/↓ select · Enter confirm · number select · Esc cancel",
                                )
                            ]
                        ),
                        height=1,
                    ),
                    *([Window(FormattedTextControl(footer), height=2)] if footer else []),
                ]
            )
        ),
        erase_when_done=True,
        key_bindings=bindings,
        style=Style.from_dict(
            {
                "selected": "bold cyan",
                "option": "",
                "desc": "fg:ansibrightblack",
                "rule": "fg:ansibrightblack",
            }
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
            raw_edits = arguments.get("edits")
            edits = (
                [item for item in raw_edits if isinstance(item, dict)]
                if isinstance(raw_edits, list)
                else []
            )
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
                else "none",
            )
            for line in diff
        )
        return body
    preview = str(arguments)
    if len(preview) > 120:
        preview = preview[:120] + "…"
    return [Text(f"{display_tool_name(name)}({preview})")]


def _ask_line(label: str, default: str | None = None) -> str:
    """审批辅助输入行（拒绝理由 / 修改命令）。独立函数便于测试替身。"""
    from prompt_toolkit import prompt

    try:
        return prompt(label, default=default or "")
    except (EOFError, KeyboardInterrupt):
        return ""


class ApprovalRejected(InterruptedError):
    """用户明确拒绝后停止整轮，而非让模型自动换一种方式继续。"""


class ApprovalOutcome:
    """一次审批的结果：放行与否 + 可能的规则 / 改写命令 / 拒绝理由。"""

    def __init__(
        self,
        approved: bool,
        rule: Rule | None = None,
        command: str | None = None,
        reason: str | None = None,
        allow_all: bool = False,
        arguments: dict | None = None,
    ):
        self.approved = approved
        self.rule = rule
        self.command = command
        self.reason = reason
        self.allow_all = allow_all
        self.arguments = arguments


class ApprovalGate:
    """会话审批闸门（spec「审批交互」）：会话授权选项、默认拒绝、会话级授权规则累积。

    普通操作默认 Allow once，高危/未知默认 Deny；高危不给前缀授权出口（Q9）；
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
        self.allow_all = False
        self.rejected_reason: str | None = None

    def screen(
        self, tool: Tool, arguments: dict, high_risk: bool = False, origin: str | None = None
    ) -> ApprovalOutcome:
        """渲染变更预览并弹审批选项；非交互环境预览后直接拒绝。"""
        if not self.interactive:
            ui.print(
                Panel(
                    Group(*_approval_body(tool.name, arguments, self.root_path)),
                    title="非交互环境，默认拒绝",
                    border_style="red",
                )
            )
            return ApprovalOutcome(approved=False)

        assessment = assess(tool, arguments)
        high_risk = high_risk or assessment.risk == "high"
        rule = None if high_risk else rule_for(tool, arguments)
        can_modify = tool.kind == "exec" and isinstance(arguments.get("command"), str)
        # 动态选项表：高危不给授权出口（Q9）；取不出前缀不给（Q14/Q18）
        options: list[tuple[str, str]] = [("Allow once", "Run this call")]
        if rule is not None:
            options.append(("Allow prefix for session", f"{rule} will not prompt again"))
        if can_modify:
            options.append(("Edit command", "Edit, then review again"))
        all_index = None
        if not high_risk:
            all_index = len(options)
            options.append(
                ("Allow all for session", "Allow subsequent calls; high-risk calls still prompt")
            )
        deny_index = len(options)
        options.append(("Deny", "Stop this task; optionally give a reason"))

        if self.renderer is not None:
            self.renderer.pause()
        try:
            ui.print(
                Panel(
                    Group(
                        Text(assessment.reason, style="dim"),
                        *_approval_body(tool.name, arguments, self.root_path),
                    ),
                    title=f"Approve {display_tool_name(tool.name)} · {assessment.risk} risk"
                    + (f" · {origin}" if origin else ""),
                    border_style="yellow",
                )
            )
            choice = _select_option(
                options,
                cancel_index=deny_index,
                initial=0 if assessment.default_allow and not high_risk else deny_index,
                **(
                    {"footer": self.renderer.session_footer}
                    if self.renderer and self.renderer.session_footer
                    else {}
                ),
            )
        finally:
            if self.renderer is not None:
                self.renderer.resume()

        if all_index is not None and choice == all_index:
            return ApprovalOutcome(approved=True, allow_all=True)
        if choice == 0:
            return ApprovalOutcome(approved=True)
        if rule is not None and choice == 1:
            return ApprovalOutcome(approved=True, rule=rule)
        if can_modify and choice == (2 if rule is not None else 1):
            modified = _ask_line("Edit command: ", default=str(arguments["command"]))
            return ApprovalOutcome(approved=bool(modified.strip()), command=modified.strip())
        reason = _ask_line("Reason (Enter to skip): ").strip() or None
        return ApprovalOutcome(approved=False, reason=reason)

    def authorize(
        self,
        tool: Tool,
        arguments: dict,
        *,
        plan: bool = False,
        unrestricted: bool = False,
        interactive: bool | None = None,
        origin: str | None = None,
    ) -> ApprovalOutcome:
        """Evaluate and approve the exact call; edited commands go through a fresh review."""
        current = dict(arguments)
        interactive = self.interactive if interactive is None else interactive
        while True:
            if tool.kind == "write" and self.root_path and current.get("path"):
                target = (self.root_path / str(current["path"])).resolve()
                if not target.is_relative_to(self.root_path):
                    return ApprovalOutcome(False, reason="Error: Path is outside the workspace")
            decision = decide(
                tool,
                current,
                Context(
                    plan=plan,
                    rules=tuple(self.rules),
                    yolo=self.allow_all or unrestricted,
                ),
            )
            if decision.verdict == "deny":
                return ApprovalOutcome(False, reason=decision.reason)
            if decision.verdict == "allow":
                return ApprovalOutcome(True, arguments=current)
            if not interactive:
                return ApprovalOutcome(
                    False, reason="Error: 非交互环境默认拒绝 (approval required)"
                )
            if self.renderer:
                self.renderer.update("tool_approval", {"name": tool.name, "arguments": current})
            if origin is None:
                outcome = self.screen(tool, current, high_risk=decision.reason == "high-risk")
            else:
                # 仅在子任务路径传 origin，保持既有 screen 调用方（测试 monkeypatch）兼容
                outcome = self.screen(
                    tool, current, high_risk=decision.reason == "high-risk", origin=origin
                )
            if not outcome.approved:
                reason = f"Error: 用户拒绝了工具调用 {tool.name}"
                if outcome.reason:
                    reason += f": {outcome.reason}"
                self.rejected_reason = reason
                return ApprovalOutcome(False, reason=reason)
            if outcome.command is not None:
                current["command"] = outcome.command
                continue
            if outcome.rule is not None:
                self.rules.append(outcome.rule)
                ui.print(f"Allowed {outcome.rule}; will not prompt again this session", style="dim")
            if outcome.allow_all:
                self.allow_all = True
                ui.print(
                    "Session calls allowed; high-risk calls still require approval", style="dim"
                )
            outcome.arguments = current
            if self.renderer:
                scope = (
                    "All session calls; high-risk calls still require approval"
                    if outcome.allow_all
                    else str(outcome.rule)
                    if outcome.rule
                    else "This call only"
                )
                self.renderer.update(
                    "approval_granted",
                    {
                        "name": tool.name,
                        "arguments": current,
                        "scope": scope,
                    },
                )
            else:
                receipt = Text("✔ You approved polya to run ")
                receipt.append(
                    str(current.get("command") or display_tool_name(tool.name)), style="dim"
                )
                ui.print(receipt)
            return outcome

    def as_approve(self):
        """Legacy bool hook: copy the approved, possibly edited arguments back to the driver."""

        def approve(tool: Tool, arguments: dict) -> bool:
            outcome = self.authorize(tool, arguments)
            if outcome.approved:
                arguments.clear()
                arguments.update(outcome.arguments or {})
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
            footer = renderer.session_footer if (renderer and renderer.session_footer) else None
            choice = _select_option(
                [
                    ("批准", "按计划进入执行模式（写操作仍受审批）"),
                    ("Deny", "停止本轮；可输入新要求修改计划"),
                ],
                cancel_index=1,
                footer=footer,
            )
        finally:
            if renderer is not None:
                renderer.resume()
        return choice == 0

    return approve_plan
