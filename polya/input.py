"""Claude Code 风格输入框：多行编辑、补全、粘贴折叠、双击 Ctrl+C 退出。

对外只暴露 :class:`InputBox` 的 ``ask(state)``：渲染一轮输入并返回提交文本
（粘贴占位符已展开）。退出走异常通道——空框 2 秒内双击 Ctrl+C 抛
``KeyboardInterrupt``、Ctrl+D 抛 ``EOFError``，与 REPL 现行退出路径一致。

状态栏数据（模式 / 模型 / 上下文占比 / 规则数）由驱动层经 ``state`` 注入，
本模块不反取 agent 状态（依赖单向，spec「模块划分」）。
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (
    Completer,
    Completion,
    PathCompleter,
    merge_completers,
)
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

# 斜杠命令表：(命令, 说明)——补全菜单的说明列与 cli 的 /help 同源
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "显示本帮助"),
    ("/todos", "显示当前 TODO 清单"),
    ("/status", "显示会话状态（模式 / 用量 / 工具计数）"),
    ("/plan", "开启/关闭规划模式（on|off）"),
    ("/expand", "展开最近 N 块工具结果 / 思考全文"),
    ("/reset", "清空对话历史、TODO 与统计"),
    ("/exit", "退出"),
    ("/quit", "退出"),
]

PASTE_FOLD_THRESHOLD = 10  # 粘贴超过此行数即折叠为占位符
QUIT_WINDOW_S = 2.0  # 空框双击 Ctrl+C 的判定窗口（秒）
KEY_HINTS = "Enter 发送 · Alt+Enter/Ctrl+J 换行 · /help 命令"


def _display_width(text: str) -> int:
    """粗略显示宽度：CJK 记 2 列（横线填充用，不追求精确 wcwidth）。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def _rule(label: str, width: int) -> str:
    """一条 ─ 横线，label 嵌在开头（Claude Code 输入框的上下框线）。"""
    label = f" {label} " if label else ""
    fill = max(0, width - _display_width(label) - 2)
    return "──" + label + "─" * fill


def prompt_message(state: dict) -> list:
    """输入框顶线 + ``❯`` 提示符；主题已知时嵌在顶线（``── ✳ topic ──``）。"""
    width = shutil.get_terminal_size((100, 24)).columns
    topic = state.get("topic")
    rule = _rule(f"✳ {topic}" if topic else "", width)
    return [("", "\n"), ("class:rule", rule), ("class:rule", "\n"), ("class:prompt", "❯ ")]


class SlashCompleter(Completer):
    """补全开头的斜杠命令（带说明列）；整行不是命令时不给任何建议。

    一旦出现空格（如 ``/plan on`` 的参数部分）即停止——命令名补全到此为止。
    """

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command, description in SLASH_COMMANDS:
            if command.startswith(text):
                yield Completion(command, start_position=-len(text), display_meta=description)


class AtPathCompleter(Completer):
    """``@`` 触发的文件路径补全：对光标前 ``@`` 开头的词做相对路径补全。"""

    def __init__(self) -> None:
        self._paths = PathCompleter()

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        token = text.rsplit(None, 1)[-1] if text.split() else ""
        if not token.startswith("@") or len(token) < 2:
            return
        fragment = token[1:]
        probe = Document(fragment, len(fragment))
        for completion in self._paths.get_completions(probe, complete_event):
            # PathCompleter 的语义是「后缀插入」（start_position 通常为 0），
            # 原样透传，勿改成替换——否则拼出的路径会丢前半段。
            yield Completion(
                completion.text,
                start_position=completion.start_position,
                display=completion.display,
                display_meta=completion.display_meta,
            )


class InputBox:
    """多行输入框。``ask(state)`` 渲染一轮输入并返回提交文本（粘贴已展开）。"""

    def __init__(self, history_path: Path | None = None, *, input=None, output=None):
        self._state: dict = {}
        self._pastes: list[str] = []  # 折叠登记：原文按序号存取
        self._tokens: list[str] = []
        self._last_cancel = 0.0
        self._hint = ""
        self._hint_until = 0.0
        self._session = self._build(history_path, input, output)

    # ---- 对外接口 ----

    def ask(self, state: dict) -> str:
        """渲染一轮输入。退出信号以异常上行（KeyboardInterrupt / EOFError）。"""
        self._state = state
        text = self._session.prompt(self._message)
        expanded = self._expand_pastes(text)
        self._pastes.clear()
        self._tokens.clear()
        return expanded

    # ---- 组装 ----

    def _build(self, history_path: Path | None, input, output) -> PromptSession:
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("c-j")
        def _newline_ctrl_j(event):
            # Ctrl+J：部分终端吃掉 Alt 键时的换行备用入口
            event.current_buffer.insert_text("\n")

        @bindings.add("enter")
        def _enter(event):
            self._submit(event.current_buffer)

        @bindings.add("c-c")
        def _cancel(event):
            try:
                self._on_cancel(event.current_buffer)
            finally:
                event.app.invalidate()

        @bindings.add(Keys.BracketedPaste)
        def _paste(event):
            self._on_paste(event.data or "", event.current_buffer)

        if history_path is None:
            state_dir = Path.home() / ".polya"  # 状态目录：输入历史现居于此
            state_dir.mkdir(parents=True, exist_ok=True)
            history_path = state_dir / "history"

        return PromptSession(
            input=input,
            output=output,
            multiline=True,
            prompt_continuation=lambda width, number, soft: [("class:continuation", "… ")],
            bottom_toolbar=self._bottom_bar,
            placeholder=[("class:placeholder", "输入任务（Alt+Enter 换行），/help 命令，@ 补路径")],
            style=Style.from_dict(
                {
                    "prompt": "bold cyan",
                    "rule": "fg:ansibrightblack",
                    "bottom-toolbar": "bg:default fg:ansibrightblack",
                    "continuation": "dim",
                    "placeholder": "dim",
                }
            ),
            history=FileHistory(str(history_path)),
            completer=merge_completers(SlashCompleter(), AtPathCompleter()),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=bindings,
        )

    # ---- 渲染 ----

    def _message(self) -> list:
        return prompt_message(self._state)

    def _bottom_bar(self) -> list:
        """底线状态栏：左会话状态（state 注入），右快捷键提示 / 临时提示。"""
        state = self._state
        left = []
        for key in ("mode", "model", "context"):
            if state.get(key):
                left.append(str(state[key]))
        if state.get("rules") is not None:
            left.append(f"规则 {state['rules']}")
        hint = self._hint if time.monotonic() < self._hint_until else KEY_HINTS
        label = " · ".join([*left, hint])
        width = shutil.get_terminal_size((100, 24)).columns
        return [("class:rule", _rule(label, width))]

    def _flash_hint(self, message: str) -> None:
        """临时提示占据状态栏右侧一小段时间。"""
        self._hint = message
        self._hint_until = time.monotonic() + 3.0

    # ---- 按键语义（抽出为方法，便于单测直接驱动 Buffer）----

    def _submit(self, buffer) -> None:
        """Enter：补全菜单打开时选中补全项；否则末行提交、行中/续行换行。"""
        state = buffer.complete_state
        if state is not None and state.completions:
            buffer.apply_completion(state.current_completion or state.completions[0])
            return
        document = buffer.document
        if not (document.is_cursor_at_the_end and document.on_last_line):
            buffer.insert_text("\n")  # 光标在行中间/非末行：Enter 当换行用
            return
        if document.current_line_before_cursor.endswith("\\"):
            buffer.delete_before_cursor(1)  # 经典续行：去掉反斜杠换行
            buffer.insert_text("\n")
            return
        buffer.validate_and_handle()

    def _on_cancel(self, buffer) -> None:
        """Ctrl+C：有文本先清空；空框 2 秒内双击退出（KeyboardInterrupt 上行）。"""
        if buffer.text:
            buffer.reset()
            self._flash_hint("已清空（空框双击 Ctrl+C 退出）")
            return
        now = time.monotonic()
        if now - self._last_cancel <= QUIT_WINDOW_S:
            raise KeyboardInterrupt
        self._last_cancel = now
        self._flash_hint("再按一次 Ctrl+C 退出")

    def _on_paste(self, data: str, buffer) -> None:
        """大段粘贴折叠为 ``[Pasted #N +M lines]``，提交时展开（见 ask）。"""
        lines = len(data.splitlines())
        if lines <= PASTE_FOLD_THRESHOLD:
            buffer.insert_text(data)
            return
        self._pastes.append(data)
        token = f"[Pasted #{len(self._pastes)} +{lines} lines]"
        self._tokens.append(token)
        buffer.insert_text(token)

    def _expand_pastes(self, text: str) -> str:
        for token, content in zip(self._tokens, self._pastes, strict=True):
            text = text.replace(token, content)
        return text
