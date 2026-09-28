"""常驻多行输入：编辑、补全、粘贴折叠和自适应状态栏。

同步 ask() 用于独立输入，ask_async()/suspend() 支持后台任务与 /login 向导的终端
让位。草稿以 Document 保留光标，粘贴登记直到提交才清空。状态由驱动注入。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, to_filter
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu, CompletionsMenuControl
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from .commands import COMMANDS, command_error, parse_command
from .filefind import ProjectFiles
from .i18n import t
from .models import format_tokens

PASTE_FOLD_THRESHOLD = 10  # 粘贴超过此行数即折叠为占位符
QUIT_WINDOW_S = 2.0  # 空框双击 Ctrl+C 的判定窗口（秒）
KEY_HINTS = t("ui.input.key_hints")

# 补全菜单样式（2026-09 样式原型裁决，变体 B「极简暗色」）：无底色，未选中
# 暗灰、选中亮青加粗。每个类都显式 bg:default——PT 默认样式是浅灰块 +
# 选中白底反白，漏写任一类都会渗透回默认色块（见 issues/01-menu-style.md）。
MENU_STYLE: dict[str, str] = {
    "completion-menu": "bg:default fg:#808080 noreverse",
    "completion-menu.completion": "bg:default",
    "completion-menu.completion.current": "bg:default fg:ansibrightcyan bold noreverse",
    "completion-menu.meta.completion": "bg:default fg:#5f5f5f",
    "completion-menu.meta.completion.current": "bg:default fg:#d7d7d7",
}


def _display_width(text: str) -> int:
    """使用终端列宽计算，涵盖中文与组合字符。"""
    return get_cwidth(text)


def _rule(label: str, width: int) -> str:
    text = _fit(f"── {label} " if label else "", width)
    return text + "─" * max(0, width - get_cwidth(text))


def _fit(text: str, width: int) -> str:
    if get_cwidth(text) <= width:
        return text
    result = ""
    for char in text:
        if get_cwidth(result + char) > max(0, width - 1):
            break
        result += char
    return result + ("…" if width else "")


def _tail(text: str, width: int) -> str:
    """保留尾部（项目基名/后缀），前面用 … 占位。"""
    if width <= 0:
        return ""
    if get_cwidth(text) <= width:
        return text
    tail = ""
    for char in reversed(text):
        if get_cwidth(char + tail) > width - 1:
            break
        tail = char + tail
    return "…" + tail


def _align(left: str, right: str, width: int) -> str:
    """一行的左右两段：右段优先保真，左段先截，至少留 2 空格间隔。"""
    if not right:
        return _fit(left, width)
    right = _fit(right, width)
    left = _fit(left, max(0, width - get_cwidth(right) - 2))
    gap = max(2, width - get_cwidth(left) - get_cwidth(right))
    return _fit(left + " " * gap + right, width)


def prompt_message(state: dict) -> list:
    """主题留在终端标题中，输入区只有提示符。"""
    return [("class:prompt", "❯ ")]


class InputSuspended(Exception):
    """/models add 向导临时取得终端输入权；草稿和粘贴登记继续保留。"""


class SlashCompleter(Completer):
    """Complete a leading command or its enumerated argument, never prompt prose.

    Codex 式过滤：大小写不敏感，精确命中排前缀命中前；别名不单列——命中即补主名。
    """

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in document.text or document.text_after_cursor:
            return
        if not any(char.isspace() for char in text):
            query = text.lower()

            def bucket(command) -> int:
                names = [command.name.lower(), *(alias.lower() for alias in command.aliases)]
                if query in names:
                    return 0  # 精确命中（含别名）
                if any(name.startswith(query) for name in names):
                    return 1  # 前缀命中
                return 2

            matched = [c for c in COMMANDS if bucket(c) < 2]
            matched.sort(key=bucket)  # 稳定排序：桶内保持声明序
            for entry in matched:
                yield Completion(
                    entry.name,
                    start_position=-len(text),
                    display=entry.name,
                    display_meta=_command_meta(entry),
                )
            return
        command, argument = parse_command(text)
        if command is None or len(text.split()) > 2 or (argument and text[-1].isspace()):
            return
        for value, label in command.effective_choices():
            if value.startswith(argument):
                yield Completion(value, start_position=-len(argument), display_meta=label)


def _command_meta(command) -> str:
    parts = [p for p in (command.argument_hint, command.description) if p]
    if command.aliases:
        parts.append(t("ui.input.aliases", aliases=", ".join(command.aliases)))
    return " · ".join(parts)


class AtPathCompleter(Completer):
    """``@`` 触发的全项目模糊文件补全：光标前 ``@`` 开头的词整词替换插入。"""

    def __init__(self, files: ProjectFiles) -> None:
        self._files = files

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        token = text.rsplit(None, 1)[-1] if text.split() else ""
        if not token.startswith("@"):
            return
        fragment = token[1:]
        # 整词替换（start_position=-len(fragment)）接管 @ 后的全部已输入，
        # 与旧 PathCompleter 的后缀插入不同：模糊命中的是完整相对路径。
        for path in self._files.search(fragment):
            yield Completion(path, start_position=-len(fragment))


class InputBox:
    """多行输入框。``ask(state)`` 渲染一轮输入并返回提交文本（粘贴已展开）。"""

    def __init__(
        self,
        history_path: Path | None = None,
        *,
        input=None,
        output=None,
        files: ProjectFiles | None = None,
    ):
        self._state: dict = {}
        self.on_interrupt = lambda: None
        self.on_dequeue = lambda: ""  # Alt+Up：由驱动提供排队消息文本
        self.on_set_default = lambda ref: ""  # Ctrl+S：由驱动写默认模型并返回提示
        self.last_kind = "steering"  # 上次提交语义：steering（Enter）/ follow-up（Alt+Enter）
        self.preview = lambda width, max_lines: ""
        self._draft = Document("")
        self._pastes: list[str] = []  # 折叠登记：原文按序号存取
        self._tokens: list[str] = []
        self._last_cancel = 0.0
        self._hint = ""
        self._hint_until = 0.0
        self._menu_complete = False  # Tab 发起的 menu-complete 运行标记（见 _tab）
        # rg 索引懒构建：不触发 @ 补全的会话不会运行子进程（测试与非交互路径零成本）。
        self._files = files or ProjectFiles(Path.cwd())
        self._session = self._build(history_path, input, output)

    # ---- 对外接口 ----

    def ask(self, state: dict) -> str:
        """渲染一轮输入。退出信号以异常上行（KeyboardInterrupt / EOFError）。"""
        self._state = state
        text = self._session.prompt(self._message)
        return self._submitted(text)

    async def ask_async(self, state: dict) -> str:
        self._state = state
        try:
            text = await self._session.prompt_async(self._message, default=self._draft)
        except KeyboardInterrupt as exc:
            raise EOFError from exc
        return self._submitted(text)

    def refresh_file_index(self) -> None:
        """任务结束后由驱动调用：agent 可能刚写过文件，后台重建 @ 索引。"""
        self._files.refresh_soon()

    def suspend(self) -> None:
        if self._session.app.is_running and not self._session.app.is_done:
            self._draft = self._session.default_buffer.document
            self._session.app.exit(exception=InputSuspended())

    def insert_pending(self, text: str) -> None:
        """把排队消息送回编辑器（Esc 中断 / Alt+Up 取回）；prompt 未在跑时存为草稿。"""
        if not text:
            return
        if self._session.app.is_running and not self._session.app.is_done:
            buffer = self._session.default_buffer
            if buffer.text and not buffer.text.endswith("\n"):
                buffer.insert_text("\n")
            buffer.insert_text(text)
            self._session.app.invalidate()
        else:
            prefix = self._draft.text
            self._draft = Document((prefix + "\n" + text) if prefix else text)

    def _submitted(self, text: str) -> str:
        self._draft = Document("")
        expanded = self._expand_pastes(text)
        self._pastes.clear()
        self._tokens.clear()
        return expanded

    # ---- 组装 ----

    def _build(self, history_path: Path | None, input, output) -> PromptSession:
        bindings = KeyBindings()

        @bindings.add("c-j")
        def _newline(event):
            # Ctrl+J：换行（Enter=发送，Alt+Enter=追加）
            event.current_buffer.insert_text("\n")

        @bindings.add("c-s")
        def _save_default(event):
            # Ctrl+S：在 /model 选项器里把当前高亮设为默认启动模型（票 05）；
            # 返回值由驱动提供（写 models.json.active），空串表示未处理。
            buffer = event.current_buffer
            state = buffer.complete_state
            command, _ = parse_command(buffer.text)
            if command is None or command.name != "/model" or state is None:
                return
            completion = state.current_completion
            if completion is None:
                return
            message = self.on_set_default(completion.text)
            if message:
                self._flash_hint(message)
            event.app.invalidate()

        @bindings.add("enter")
        def _enter(event):
            self._submit(event.current_buffer, "steering")

        @bindings.add("escape", "enter")
        def _follow_up(event):
            # Alt+Enter：本轮结束后再交给模型（pi follow-up）
            self._submit(event.current_buffer, "follow-up")

        @bindings.add("escape", "up")
        def _dequeue(event):
            # Alt+Up：把排队消息取回编辑器（pi 语义）
            text = self.on_dequeue()
            if text:
                buffer = event.current_buffer
                if buffer.text and not buffer.text.endswith("\n"):
                    buffer.insert_text("\n")
                buffer.insert_text(text)
            event.app.invalidate()

        @bindings.add("c-i")
        def _tab(event):
            buffer = event.current_buffer
            state = buffer.complete_state
            if state is not None and state.completions:
                buffer.apply_completion(state.current_completion or state.completions[0])
            else:
                # select_first=True：Tab 打开菜单即插入首项（menu-complete，对齐
                # CC/Codex）。_menu_complete 标记让 _preselect_first 钩子给这条
                # 显式路径让路（钩子先改 index 会废掉库的插入与无增量重置）。
                self._menu_complete = True
                buffer.start_completion(select_first=True)

        @bindings.add("c-c")
        def _cancel(event):
            try:
                self._on_cancel(event.current_buffer)
            except KeyboardInterrupt:
                event.app.exit(exception=KeyboardInterrupt())
            finally:
                event.app.invalidate()

        @bindings.add("escape")
        def _escape(event):
            if event.current_buffer.complete_state:
                event.current_buffer.cancel_completion()
            else:
                self.on_interrupt()

        @bindings.add(Keys.BracketedPaste)
        def _paste(event):
            self._on_paste(event.data or "", event.current_buffer)

        if history_path is None:
            state_dir = Path.home() / ".polya"  # 状态目录：输入历史现居于此
            state_dir.mkdir(parents=True, exist_ok=True)
            history_path = state_dir / "history"

        session: PromptSession = PromptSession(
            input=input,
            output=output,
            multiline=True,
            prompt_continuation=lambda width, number, soft: [("class:continuation", "  ")],
            erase_when_done=True,
            refresh_interval=0.5,
            # 自动补全统一由下方 _auto_complete 驱动（插入与删除都触发）；
            # 库自带的 insert 驱动只覆盖插入且重复起任务，关掉保持单一来源。
            complete_while_typing=False,
            placeholder=[("class:placeholder", t("ui.input.placeholder"))],
            style=Style.from_dict(
                {
                    "prompt": "bold cyan",
                    "rule": "fg:ansibrightblack",
                    "bottom-toolbar": "bg:default fg:ansibrightblack",
                    "continuation": "dim",
                    "placeholder": "dim",
                    **MENU_STYLE,
                }
            ),
            history=FileHistory(str(history_path)),
            completer=merge_completers([SlashCompleter(), AtPathCompleter(self._files)]),
            key_bindings=bindings,
        )

        # 复用 PromptSession 的编辑、历史与搜索控件；边线和状态同属一个布局。
        window = session.layout.current_window
        window.height = Dimension(min=1, max=6)
        window.dont_extend_height = to_filter(True)
        container = session.layout.container

        # 3.0.53 的补全菜单是输入窗下方的浮层；输入窗限高（1–6 行、不占满）后
        # 下方没有可画的行，菜单整体消失。改为把菜单做成输入区上方的实体行、
        # 向上生长（CC/Codex 同款形态），原浮层永不渲染。
        main = container.children[0].alternative_content  # type: ignore[attr-defined]
        main.floats[0].content = CompletionsMenu(extra_filter=to_filter(False))
        buffer = session.default_buffer

        # PT 只在插入时重启自动补全：打错回删到匹配前缀后菜单不会回来。
        # 改由文本变化驱动（插入与删除都触发），触发条件与两个 completer
        # 的入口条件对齐：行首单行命令，或光标前 @ 开头的词。
        def _auto_complete(buffer) -> None:
            self._menu_complete = False  # 任何文本变化都终结 Tab 发起的补全运行
            document = buffer.document
            if document.text.startswith("/"):
                wants = "\n" not in document.text and not document.text_after_cursor
            else:
                before = document.text_before_cursor
                token = before.rsplit(None, 1)[-1] if before.split() else ""
                wants = token.startswith("@")
            if wants:
                buffer.start_completion()

        # 自动弹出即预选首项（CC/Codex 同款）。不能用 start_completion(
        # select_first=True)：那是 menu-complete 语义，会把首项增量写进输入框；
        # 也不能无守卫地在加载期改 complete_index——会废掉库对「唯一无增量
        # 补全」（打全命令名）的重置，留下僵尸菜单。这里同步复刻同一条
        # 判定：命中即让路，由库收起菜单；随后 _preselect_first 的空态分支
        # 以重建态重开（见 _reopen_complete_command，票 04）。
        def _preselect_first(buffer) -> None:
            state = buffer.complete_state
            if state is None:
                self._menu_complete = False  # 空结果重置：Tab 运行已结束，撤销标记
                self._reopen_complete_command(buffer)
                return
            if self._menu_complete:
                return  # Tab 的 select_first 路径由库收尾（含插入与无增量重置）
            if state.complete_index is not None or not state.completions:
                return
            document = buffer.document
            first = state.completions[0]
            replaced = document.text_before_cursor[
                len(document.text_before_cursor) + first.start_position :
            ]
            if len(state.completions) == 1 and replaced == first.text:
                return  # 唯一无增量：让库重置状态并收起菜单
            state.complete_index = 0

        buffer.on_text_changed.add_handler(_auto_complete)
        buffer.on_completions_changed.add_handler(_preselect_first)

        def menu_visible() -> bool:
            state = buffer.complete_state
            return state is not None and bool(state.completions)

        menu = ConditionalContainer(
            Window(
                content=CompletionsMenuControl(),
                height=Dimension(min=1, max=6),
                style="class:completion-menu",
                dont_extend_height=to_filter(True),
            ),
            filter=Condition(menu_visible),
        )

        def rule():
            return [("class:rule", "─" * session.output.get_size().columns)]

        session.app.layout = Layout(
            HSplit(
                [
                    ConditionalContainer(
                        Window(
                            FormattedTextControl(self._live_preview),
                            dont_extend_height=True,
                            height=Dimension(min=1, max=9),
                        ),
                        filter=Condition(lambda: bool(self._state.get("preview_active"))),
                    ),
                    ConditionalContainer(
                        Window(FormattedTextControl(self._working_bar), height=1),
                        filter=Condition(lambda: bool(self._state.get("busy"))),
                    ),
                    menu,
                    Window(FormattedTextControl(rule), height=1),
                    container,
                    Window(FormattedTextControl(rule), height=1),
                    Window(FormattedTextControl(self._environment_bar), height=1),
                    Window(FormattedTextControl(self._usage_bar), height=1),
                    Window(FormattedTextControl(self._bottom_bar), height=1),
                ]
            ),
            focused_element=window,
        )
        session.app.timeoutlen = 0.5
        return session

    def _reopen_complete_command(self, buffer) -> None:
        """命令名打全后重开菜单（票 04）：库的「唯一无增量」重置会把它收起。

        CC/Codex 形态：输入完整的 `/help` 时菜单仍列出该命令（选中态 + 说明），
        Enter 照常执行、Esc 关闭。与 _submit 的选项器同款手法：公开
        CompletionState 构造 + 手动 fire；下一次文本变化由 _text_changed 清空
        重启，不会僵尸化。仅在库收起（state 为 None）的事件里调用——Esc 走
        cancel_completion，不触发本事件，关闭后不会回弹。
        """
        document = buffer.document
        text = document.text
        if (
            not text.startswith("/")
            or "\n" in text
            or document.text_after_cursor
            or any(char.isspace() for char in text)
        ):
            return
        query = text.lower()
        command = next(
            (
                c
                for c in COMMANDS
                if query in (c.name.lower(), *(alias.lower() for alias in c.aliases))
            ),
            None,
        )
        if command is None:
            return
        buffer.complete_state = CompletionState(
            buffer.document,
            [
                Completion(
                    command.name,
                    start_position=-len(text),
                    display=command.name,
                    display_meta=_command_meta(command),
                )
            ],
            0,
        )
        buffer.on_completions_changed.fire()

    # ---- 渲染 ----

    def _message(self) -> list:
        return prompt_message(self._state)

    def _live_preview(self):
        size = self._session.output.get_size()
        # Leave room for the editor, its borders, status and completion menu.
        max_lines = max(1, min(8, (size.rows - 11) // 2))
        return ANSI(self.preview(size.columns, max_lines))

    def _working_bar(self) -> list:
        """Task-wide status remains above the editor, independent of individual tools."""
        started_at = self._state.get("started_at")
        elapsed = max(0, int(time.monotonic() - started_at)) if started_at is not None else 0
        minutes, seconds = divmod(elapsed, 60)
        duration = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
        if self._state.get("stopping"):
            label = "Stopping"
            hint = "waiting: " + self._state.get("status", "current operation")
        else:
            label = self._state.get("status", "Waiting for model")
            hint = (
                "esc to close completions"
                if self._session.default_buffer.complete_state
                else "esc to interrupt"
            )
        width = self._session.output.get_size().columns
        text = f"  {label} · {duration} ({hint})"
        # Keep the action legible on narrow terminals before adding a timer.
        if get_cwidth(text) > width:
            text = f"  {label} ({hint})"
        return [("class:rule", _fit(text, width))]

    def _environment_bar(self) -> list:
        """项目身份行：项目（~ 缩写）(分支) • 主题，右对齐 provider · 模型 · 思考档。

        窗口与用量在下一行 `_usage_bar`；模式与提示在 `_bottom_bar`。宽度不足时
        依次舍弃 provider → 思考档 → 分支 → 主题，最后保项目基名与模型。
        """
        width = max(0, self._session.output.get_size().columns - 3)
        state = self._state
        project = state.get("project") or "—"
        home = str(Path.home())
        if project == home or project.startswith(home + "/"):
            project = "~" + project[len(home) :]
        branch = state.get("branch")
        topic = state.get("topic")
        model = state.get("model") or "—"
        thinking = state.get("thinking")
        profile = state.get("profile")

        def compose(
            with_provider: bool, with_thinking: bool, with_branch: bool, with_topic: bool
        ) -> tuple[str, str]:
            left = f"{project} ({branch})" if with_branch and branch else project
            if with_topic and topic:
                left = f"{left} • {topic}"
            right = f"{model} • {thinking}" if with_thinking and thinking else model
            if with_provider and profile:
                right = f"({profile}) {right}"
            return left, right

        # 窄屏舍弃顺序：provider → 思考档 → 分支 → 主题；最后保项目基名与模型。
        left, right = compose(True, True, True, True)
        for flags in (
            (False, True, True, True),
            (False, False, True, True),
            (False, False, False, True),
            (False, False, False, False),
        ):
            if get_cwidth(left) + get_cwidth(right) + 2 <= width:
                break
            left, right = compose(*flags)
        basename = Path(project).name or project
        reserve = min(get_cwidth(basename), max(1, width // 3))
        if get_cwidth(right) > max(0, width - reserve - 2):
            right = _fit(right, max(0, width - reserve - 2))
        budget = max(0, width - get_cwidth(right) - 2)
        if get_cwidth(left) > budget:
            left = _tail(left, budget)  # 保项目基名/后缀
        return [("class:rule", "  " + _align(left, right, width))]

    def _usage_bar(self) -> list:
        """累计用量 + 上下文：↑输入 ↓输出 R缓存 CH命中% $费用 ctx 占比/窗口 (auto)。"""
        width = max(0, self._session.output.get_size().columns - 3)
        state = self._state
        parts: list[str] = []
        input_tokens = state.get("input_tokens") or 0
        output_tokens = state.get("output_tokens") or 0
        cached_tokens = state.get("cached_tokens") or 0
        if input_tokens:
            parts.append(f"↑{format_tokens(input_tokens)}")
        if output_tokens:
            parts.append(f"↓{format_tokens(output_tokens)}")
        if cached_tokens:
            parts.append(f"CR{format_tokens(cached_tokens)}")
        hit = state.get("cache_hit")
        if cached_tokens and hit is not None:
            parts.append(f"CH{hit:.1f}%")
        cost = state.get("cost")
        subscribed = state.get("subscribed")
        if cost is not None:
            # 订阅制也照列价估算，但标 (sub)：套餐内不实际计费（pi 同款）。
            parts.append(f"${cost:.3f}" + (" (sub)" if subscribed else ""))
        elif subscribed and (input_tokens or output_tokens or cached_tokens):
            parts.append("(sub)")
        context = self._context_segment()
        if context:
            parts.append(context)
        return [("class:rule", "  " + _fit(" ".join(parts), width))]

    def _context_segment(self) -> str:
        """上下文片段：`ctx 23%/128k (auto)`；窗口未知时不显示，auto 单独保留。"""
        state = self._state
        window = state.get("window")
        percent = state.get("context_pct")
        if window and percent is not None:
            segment = f"ctx {percent}%/{window}"
        elif window:
            segment = f"ctx —/{window}"
        else:
            segment = ""
        if state.get("auto"):
            segment = f"{segment} (auto)" if segment else "(auto)"
        return segment

    def _bottom_bar(self) -> list:
        """模式与队列在左，随上下文变化的操作提示在右。"""
        state = self._state
        width = max(0, self._session.output.get_size().columns - 3)
        busy = state.get("busy", False)
        mode = state.get("mode", "normal")
        if state.get("queued"):
            mode += " · " + t("ui.input.queued_count", count=state['queued'])
        hint = t("ui.input.hint_steer") if busy else KEY_HINTS
        buffer = self._session.default_buffer
        if buffer.complete_state:
            hint = t("ui.input.hint_select")
        elif buffer.text.lstrip().startswith("/"):
            command, _ = parse_command(buffer.text)
            if command is not None:
                # 菜单收起时（如 Esc 关闭、参数阶段）描述不能随之消失：
                # 底栏接过参数提示与说明；菜单打开时说明在菜单里，不重复。
                hint = " · ".join(p for p in (command.argument_hint, command.description) if p)
        elif buffer.text and not busy:
            hint = t("ui.input.hint_multiline")
        flashed = time.monotonic() < self._hint_until
        if flashed:
            hint = self._hint
        # 窄屏：提示退到最短，但模式（plan/normal）一定保留。
        if not flashed and get_cwidth(mode) + get_cwidth(hint) + 4 > width:
            hint = (
            t("ui.input.hint_esc_close")
            if buffer.complete_state
            else t("ui.input.hint_enter_steer")
            if busy
            else "/help"
        )
        return [("class:rule", "  " + _align(mode, hint, width))]

    def _flash_hint(self, message: str) -> None:
        """临时提示占据状态栏右侧一小段时间。"""
        self._hint = message
        self._hint_until = time.monotonic() + 3.0

    # ---- 按键语义（抽出为方法，便于单测直接驱动 Buffer）----

    def _submit(self, buffer, kind: str = "steering") -> None:
        """Enter：命令高亮补全直接执行；文件补全只插入；否则末行提交、行中换行。"""
        self.last_kind = kind
        state = buffer.complete_state
        if state is not None and state.completions:
            before = state.original_document.text
            buffer.apply_completion(state.current_completion or state.completions[0])
            # 对齐 Claude Code：命令补全落到下方直接执行；文件等补全只插入。
            if buffer.text != before and not before.startswith("/"):
                return
        document = buffer.document
        if not (document.is_cursor_at_the_end and document.on_last_line):
            buffer.insert_text("\n")  # 光标在行中间/非末行：Enter 当换行用
            return
        if document.current_line_before_cursor.endswith("\\"):
            buffer.delete_before_cursor(1)  # 经典续行：去掉反斜杠换行
            buffer.insert_text("\n")
            return
        text = self._expand_pastes(buffer.text).strip()
        if text.startswith("/"):
            error = command_error(text, allow_picker=True)
            if error:
                self._flash_hint(error)
                return
            command, argument = parse_command(text)
            choices = command.effective_choices() if command is not None else ()
            if command is not None and choices and not argument:
                # A picker inside the existing editor: no terminal handoff or
                # session mutation until the user submits a complete command.
                buffer.document = Document(command.name + " ")
                completions = [
                    Completion(
                        value,
                        display_meta=label
                        + (
                            t("ui.input.current")
                            if value == self._current_choice(command.name)
                            else ""
                        ),
                    )
                    for value, label in choices
                ]
                # CompletionState 公开构造替掉 buffer._set_completions 私有 API
                # （prompt_toolkit 3.0.53 验证）；complete_index=0 预选首项。
                buffer.complete_state = CompletionState(buffer.document, completions, 0)
                buffer.on_completions_changed.fire()
                hint = t("ui.input.hint_selector_run")
                if command.name == "/model":
                    hint = t("ui.input.hint_selector_model")
                self._flash_hint(hint)
                return
        buffer.validate_and_handle()

    def _current_choice(self, name: str) -> str:
        if name == "/plan":
            return "on" if self._state.get("mode") == "plan" else "off"
        if name == "/model":
            profile, model = self._state.get("profile"), self._state.get("model")
            return f"{profile}/{model}" if profile and model else (model or "")
        if name == "/thinking":
            return self._state.get("thinking") or ""
        return ""

    def _on_cancel(self, buffer) -> None:
        """Ctrl+C：有文本先清空；空框 2 秒内双击退出（KeyboardInterrupt 上行）。"""
        if buffer.text:
            buffer.reset()
            self._pastes.clear()
            self._tokens.clear()
            self._last_cancel = 0.0
            self._flash_hint(t("ui.input.cleared"))
            return
        now = time.monotonic()
        if now - self._last_cancel <= QUIT_WINDOW_S:
            raise KeyboardInterrupt
        self._last_cancel = now
        self._flash_hint(t("ui.input.quit_confirm"))

    def _on_paste(self, data: str, buffer) -> None:
        """大段粘贴折叠为 ``[Pasted #N +M lines]``，提交时展开（见 ask）。"""
        data = data.replace("\r\n", "\n").replace("\r", "\n")
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
