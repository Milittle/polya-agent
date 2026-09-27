"""input.py 测试：粘贴折叠、Ctrl+C 语义、补全、菜单 Enter 选中、状态栏。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from shutil import which
from subprocess import run

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from polya.filefind import ProjectFiles
from polya.input import (
    KEY_HINTS,
    PASTE_FOLD_THRESHOLD,
    AtPathCompleter,
    InputBox,
    SlashCompleter,
)

requires_rg = pytest.mark.skipif(which("rg") is None, reason="rg 不在 PATH（工具层同依赖）")


@contextmanager
def make_box(tmp_path):
    with create_pipe_input() as pipe:
        yield InputBox(history_path=tmp_path / "history", input=pipe, output=DummyOutput()), pipe


async def until(condition):
    async def poll():
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), 3)


def big_paste(lines: int = PASTE_FOLD_THRESHOLD + 5) -> str:
    return "\n".join(f"line-{i}" for i in range(lines))


# ---------- ask() 端到端（管道驱动） ----------


def test_ask_returns_submitted_text(tmp_path):
    with make_box(tmp_path) as (box, pipe):
        pipe.send_text("hello polya\r")
        assert box.ask({}) == "hello polya"


def test_bracketed_paste_folds_in_live_session(tmp_path):
    with make_box(tmp_path) as (box, pipe):
        pipe.send_text("\x1b[200~" + big_paste() + "\x1b[201~")
        pipe.send_text("\r")
        # 输入框里是折叠占位符，提交拿到的是展开原文
        assert box.ask({}) == big_paste()


def test_double_ctrl_c_quits(tmp_path):
    with make_box(tmp_path) as (box, pipe):
        pipe.send_text("\x03")  # 空框第一次：提示
        pipe.send_text("\x03")  # 2 秒内第二次：退出
        with pytest.raises(KeyboardInterrupt):
            box.ask({})


# ---------- 按键语义（直接驱动 Buffer） ----------


def test_ctrl_c_clears_text_first(tmp_path):
    with make_box(tmp_path) as (box, _):
        buffer = Buffer()
        buffer.insert_text("abc")
        box._on_cancel(buffer)
        assert buffer.text == ""  # 有文本先清空
        box._on_cancel(buffer)  # 空框第一次：只提示，不退出
        with pytest.raises(KeyboardInterrupt):
            box._on_cancel(buffer)  # 2 秒内第二次：退出


def test_paste_fold_registers_token_and_expands(tmp_path):
    with make_box(tmp_path) as (box, _):
        buffer = Buffer()

        box._on_paste("one\ntwo\nthree", buffer)  # 阈值内：原样插入
        assert buffer.text == "one\ntwo\nthree"

        original = big_paste()
        box._on_paste(original, buffer)  # 超阈值：折叠
        assert "[Pasted #1 +15 lines]" in buffer.text
        assert box._expand_pastes(buffer.text).endswith(original)


def test_enter_runs_highlighted_command(tmp_path):
    with make_box(tmp_path) as (box, _):
        accepted = []
        buffer = Buffer(
            completer=SlashCompleter(), accept_handler=lambda b: accepted.append(b.text) or False
        )
        buffer.insert_text("/he")
        # start_completion 需要事件循环；测试直接置入补全态（私有 API，仅此处使用）
        from prompt_toolkit.completion import Completion

        buffer._set_completions(completions=[Completion("/help", start_position=-3)])
        box._submit(buffer)
        assert accepted == ["/help"]  # 对齐 CC：高亮命令补全后 Enter 直接执行


def test_enter_opens_picker_for_choice_commands(tmp_path):
    with make_box(tmp_path) as (box, _):
        buffer = Buffer(completer=SlashCompleter())
        buffer.insert_text("/pl")
        from prompt_toolkit.completion import Completion

        buffer._set_completions(completions=[Completion("/plan", start_position=-3)])
        box._submit(buffer)
        assert buffer.text == "/plan "  # 补出主名并进入选项器，而非直接执行
        state = buffer.complete_state
        assert state is not None, "选项器菜单应已打开"
        assert [c.text for c in state.completions] == ["on", "go", "off"]
        assert state.complete_index == 0  # 公开 CompletionState 构造预选首项


def test_menu_reopens_after_backspace(tmp_path):
    """打错的命令删回匹配前缀后，菜单要重新弹出。

    PT 默认只在插入时触发自动补全；backspace 只清空状态不重启，菜单会一直
    藏到下一次插入。"""
    with make_box(tmp_path) as (box, pipe):

        async def scenario():
            task = asyncio.ensure_future(box.ask_async({}))
            await until(lambda: box._session.app.is_running)
            pipe.send_text("/heq")  # 无匹配：菜单收起
            buffer = box._session.default_buffer
            await until(lambda: buffer.complete_state is None)
            await asyncio.sleep(0.1)  # 等补全任务彻底收敛（真实人手节奏），避免任务延迟掩盖 bug
            assert buffer.complete_state is None
            pipe.send_text("\x7f")  # backspace → "/he"
            await until(
                lambda: (
                    buffer.complete_state is not None
                    and [c.text for c in buffer.complete_state.completions] == ["/help"]
                )
            )
            pipe.send_text("\x03")
            await until(lambda: "已清空" in box._hint)
            pipe.send_text("\x03")
            pipe.send_text("\x03")
            await until(lambda: task.done())

        asyncio.run(scenario())


def test_auto_popup_preselects_first_completion(tmp_path):
    """输入过滤命中多个候选时，弹出即预选首项（CC/Codex 同款），而非空选。"""
    with make_box(tmp_path) as (box, pipe):

        async def scenario():
            task = asyncio.ensure_future(box.ask_async({}))
            await until(lambda: box._session.app.is_running)
            pipe.send_text("/re")  # 命中 /reload /resume /rewind /rename
            buffer = box._session.default_buffer

            def picked():
                state = buffer.complete_state
                return (
                    state is not None and len(state.completions) > 1 and state.complete_index == 0
                )

            await until(picked)
            assert buffer.complete_state.completions[0].text == "/reload"
            pipe.send_text("\x03")
            await until(lambda: "已清空" in box._hint)
            pipe.send_text("\x03")
            pipe.send_text("\x03")
            await until(lambda: task.done())

        asyncio.run(scenario())


def test_full_command_closes_menu_without_zombie(tmp_path):
    """打全命令名（唯一无增量补全）菜单应关闭，且后续回删还能重开（僵尸回归）。"""
    with make_box(tmp_path) as (box, pipe):

        async def scenario():
            task = asyncio.ensure_future(box.ask_async({}))
            await until(lambda: box._session.app.is_running)
            pipe.send_text("/help")  # 唯一补全无增量 → 菜单收起
            buffer = box._session.default_buffer
            await until(lambda: buffer.complete_state is None)
            await asyncio.sleep(0.1)
            pipe.send_text("\x7f")  # 回删 → "/hel"：若状态僵尸化，重启会被拦截
            await until(
                lambda: (
                    buffer.complete_state is not None
                    and [c.text for c in buffer.complete_state.completions] == ["/help"]
                )
            )
            pipe.send_text("\x03")
            await until(lambda: "已清空" in box._hint)
            pipe.send_text("\x03")
            pipe.send_text("\x03")
            await until(lambda: task.done())

        asyncio.run(scenario())


def test_escape_closes_menu_and_it_stays_closed(tmp_path):
    """Esc 关菜单后不能自动重开：cancel 会恢复文本，若文本变化驱动不区分
    cancel 与人工编辑，菜单会立刻弹回，永远关不掉。"""
    with make_box(tmp_path) as (box, pipe):

        async def scenario():
            task = asyncio.ensure_future(box.ask_async({}))
            await until(lambda: box._session.app.is_running)
            pipe.send_text("/re")  # 菜单弹出（多候选，预选首项）
            buffer = box._session.default_buffer
            await until(
                lambda: buffer.complete_state is not None and buffer.complete_state.completions
            )
            pipe.send_text("\x1b")  # Esc 关闭
            await until(lambda: buffer.complete_state is None)
            await asyncio.sleep(0.2)
            assert buffer.complete_state is None  # 不回弹
            assert buffer.text == "/re"  # 关闭不改动输入
            pipe.send_text("\x03")
            await until(lambda: "已清空" in box._hint)
            pipe.send_text("\x03")
            pipe.send_text("\x03")
            await until(lambda: task.done())

        asyncio.run(scenario())


def test_tab_selects_first_completion_and_enter_runs_it(tmp_path):
    with make_box(tmp_path) as (box, pipe):

        async def scenario():
            task = asyncio.ensure_future(box.ask_async({}))
            await until(lambda: box._session.app.is_running)
            pipe.send_text("/exi")
            await until(lambda: box._session.default_buffer.complete_state is not None)
            pipe.send_text("\x1b")  # Esc 关掉自动弹出，再用 Tab 主动打开
            await until(lambda: box._session.default_buffer.complete_state is None)
            pipe.send_text("\t")

            def picked():
                state = box._session.default_buffer.complete_state
                return state is not None and state.complete_index == 0

            await until(picked)
            assert box._session.default_buffer.text == "/exit"  # Tab=插入首项（CC 同义）
            pipe.send_text("\r")
            await until(lambda: task.done())
            assert task.result() == "/exit"  # 预选中的命令 Enter 直接提交执行

        asyncio.run(scenario())


# ---------- 补全器 ----------


def test_slash_completer_carries_descriptions():
    completions = list(SlashCompleter().get_completions(Document("/pl"), None))
    by_text = {c.text: c for c in completions}
    assert "/plan" in by_text
    assert "规划" in "".join(str(by_text["/plan"].display_meta))


def test_slash_completer_matches_case_insensitive():
    assert [c.text for c in SlashCompleter().get_completions(Document("/HE"), None)] == ["/help"]
    texts = [c.text for c in SlashCompleter().get_completions(Document("/EX"), None)]
    assert texts == ["/export", "/exit"]  # 前缀桶内保持声明序


def test_slash_completer_alias_yields_canonical_name():
    completions = list(SlashCompleter().get_completions(Document("/qu"), None))
    assert [c.text for c in completions] == ["/exit"]  # 别名不单列菜单行
    assert "/quit" in "".join(str(completions[0].display_meta))  # 别名标注在 meta


def _make_tree(tmp_path):
    run(["git", "init"], cwd=tmp_path, capture_output=True)  # rg 仅在 git 仓库内尊重 .gitignore
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x", encoding="utf-8")
    (tmp_path / "src" / "api.py").write_text("x", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "app-notes.md").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "secret.py").write_text("x", encoding="utf-8")
    return ProjectFiles(tmp_path)


@requires_rg
def test_at_completer_fuzzy_searches_whole_project(tmp_path):
    completer = AtPathCompleter(_make_tree(tmp_path))
    completions = list(completer.get_completions(Document("读一下 @ap"), None))
    texts = [c.text for c in completions]
    assert "src/app.py" in texts and "docs/app-notes.md" in texts
    # basename 命中优先于仅路径命中由打分器单测锁定；此处锁插入语义
    top = next(c for c in completions if c.text == "src/app.py")
    assert top.start_position == -2  # 整词替换 @ 后的已输入片段


@requires_rg
def test_at_completer_respects_gitignore_and_lists_hidden(tmp_path):
    files = _make_tree(tmp_path)
    completer = AtPathCompleter(files)
    texts = [c.text for c in completer.get_completions(Document("@sec"), None)]
    assert "ignored/secret.py" not in texts
    texts = [c.text for c in completer.get_completions(Document("@"), None)]
    assert ".hidden" in texts  # 未入 ignore 的隐藏文件照常出现


def test_at_completer_ignores_plain_text(tmp_path):
    completer = AtPathCompleter(_make_tree(tmp_path))
    assert list(completer.get_completions(Document("读一下 src/ap"), None)) == []


# ---------- 补全菜单样式 ----------


def test_completion_menu_style_is_background_free():
    """菜单样式裁决（原型变体 B）：无底色、无反白；选中行亮青加粗。

    PT 默认会给菜单浅灰块、选中行白底反白；漏写任一类都会渗透回默认色块。"""
    from prompt_toolkit.styles import Style, merge_styles
    from prompt_toolkit.styles.defaults import default_ui_style

    from polya.input import MENU_STYLE

    merged = merge_styles([default_ui_style(), Style.from_dict(MENU_STYLE)])
    current = merged.get_attrs_for_style_str(
        "class:completion-menu class:completion-menu.completion.current"
    )
    meta = merged.get_attrs_for_style_str(
        "class:completion-menu class:completion-menu.meta.completion"
    )
    for attrs in (current, meta):
        assert attrs.bgcolor in (None, "default"), attrs
        assert not attrs.reverse, attrs
    assert current.bold and current.color == "ansibrightcyan"


# ---------- 状态栏 ----------


def _bar_text(box) -> str:
    return "".join(fragment for _, fragment in box._bottom_bar())


def test_identity_usage_and_bottom_bars_split_state(tmp_path):
    with make_box(tmp_path) as (box, _):
        box._state = {
            "mode": "plan",
            "model": "deepseek-flash",
            "profile": "deepseek",
            "thinking": "high",
            "project": "/home/u/polya",
            "branch": "main",
            "topic": "修复输入框",
            "window": "128k",
            "context_pct": 12,
            "auto": True,
            "input_tokens": 12345,
            "output_tokens": 678,
            "cached_tokens": 12000,
            "cache_hit": 80.0,
        }
        identity = "".join(t for _, t in box._environment_bar())
        usage = "".join(t for _, t in box._usage_bar())
        bottom = _bar_text(box)
        # 身份行：模型带 provider/思考档，项目带分支与主题
        assert "(deepseek) deepseek-flash • high" in identity
        assert "main" in identity and "修复输入框" in identity
        # 用量行：累计 token、缓存命中、上下文占用与 auto
        for part in ("↑12.3k", "↓678", "CR12.0k", "CH80.0%", "ctx 12%/128k (auto)"):
            assert part in usage, part
        # 操作行：模式与快捷键
        assert "plan" in bottom and KEY_HINTS in bottom

        box._flash_hint("再按一次 Ctrl+C 退出")
        assert "再按一次 Ctrl+C 退出" in _bar_text(box)
        assert KEY_HINTS not in _bar_text(box)  # 临时提示顶替快捷键区


def test_usage_bar_shows_subscription_and_unknown_window(tmp_path):
    with make_box(tmp_path) as (box, _):
        box._state = {
            "input_tokens": 1000,
            "output_tokens": 200,
            "subscribed": True,
            "auto": True,
        }
        usage = "".join(t for _, t in box._usage_bar())
        assert "(sub)" in usage and "(auto)" in usage
        assert "ctx" not in usage  # 窗口未知时不显示占用
        # 已知价表则显示金额而非 (sub)
        box._state.update(subscribed=True, cost=0.1234)
        assert "$0.123" in "".join(t for _, t in box._usage_bar())


def test_usage_bar_omits_empty_state(tmp_path):
    with make_box(tmp_path) as (box, _):
        box._state = {}
        assert "".join(t for _, t in box._usage_bar()).strip() == ""


def test_bottom_bar_without_state_shows_hints_only(tmp_path):
    with make_box(tmp_path) as (box, _):
        assert KEY_HINTS in _bar_text(box)


def test_bottom_bar_keeps_command_description_when_menu_closed(tmp_path):
    """命令打全后补全菜单收起（无增量重置），底栏须接过说明而不是只回显命令名。"""
    with make_box(tmp_path) as (box, _):
        buffer = Buffer()  # 不挂补全钩子，隔离底栏渲染
        buffer.document = Document("/new")
        box._session.default_buffer = buffer
        text = _bar_text(box)
        assert "旧会话保留" in text

        buffer.document = Document("/rename")
        assert "重命名当前会话主题" in _bar_text(box)
        assert "<主题>" in _bar_text(box)  # 仍需参数时保留用法


def test_working_bar_explains_escape_and_stopping(tmp_path):
    from prompt_toolkit.completion import Completion

    with make_box(tmp_path) as (box, _):
        box._state = {"busy": True}

        def text():
            return "".join(fragment for _, fragment in box._working_bar())

        assert "Waiting for model · 0s (esc to interrupt)" in text()
        box._session.default_buffer._set_completions(completions=[Completion("test")])
        assert "esc to close completions" in text()
        box._state["stopping"] = True
        assert "Stopping" in text() and "waiting" in text()
        assert "esc to interrupt" not in text()


def test_working_timer_formats_elapsed_time(monkeypatch, tmp_path):
    with make_box(tmp_path) as (box, _):
        box._state = {"busy": True, "started_at": 100.0}
        monkeypatch.setattr("polya.input.time.monotonic", lambda: 172.0)
        text = "".join(fragment for _, fragment in box._working_bar())
        assert "Waiting for model · 1m 12s (esc to interrupt)" in text
        box._state["started_at"] = 170.0
        text = "".join(fragment for _, fragment in box._working_bar())
        assert "Waiting for model · 2s" in text


@pytest.mark.parametrize("width", [24, 40, 80, 120])
def test_footer_preserves_model_project_and_topic_with_unicode(tmp_path, width):
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.utils import get_cwidth

    class Output(DummyOutput):
        def get_size(self):
            return Size(rows=24, columns=width)

    with create_pipe_input() as pipe:
        box = InputBox(tmp_path / "history", input=pipe, output=Output())
        box._state = {
            "model": "test-model",
            "profile": "acme",
            "thinking": "high",
            "window": "200k",
            "project": "/very/long/parent/项目",
            "branch": "feature/x",
            "topic": "修复输入框和工具反馈",
            "context_pct": 23,
            "auto": True,
            "mode": "normal",
            "input_tokens": 12345,
            "output_tokens": 678,
            "cached_tokens": 12000,
            "cache_hit": 80.0,
        }
        identity = "".join(t for _, t in box._environment_bar())
        usage = "".join(t for _, t in box._usage_bar())
        bottom = _bar_text(box)
        for line in (identity, usage, bottom):
            assert get_cwidth(line) <= width
        assert "test-model" in identity and "项目" in identity
        assert "normal" in bottom
        if width >= 80:
            assert "200k" in usage
            assert "ctx 23%/200k" in usage and "auto" in usage
            assert "修复输入框和工具反馈" in identity
        if width >= 120:
            assert "acme" in identity


def test_working_bar_uses_actual_phase(tmp_path):
    with make_box(tmp_path) as (box, _):
        for phase in ("Thinking", "Responding", "Running Bash", "Reviewing"):
            box._state = {"busy": True, "status": phase}
            assert phase in "".join(t for _, t in box._working_bar())
        box._state = {"busy": True, "status": "Reviewing", "stopping": True}
        text = "".join(t for _, t in box._working_bar())
        assert "Stopping" in text and "Reviewing" in text
