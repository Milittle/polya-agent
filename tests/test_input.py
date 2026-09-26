"""input.py 测试：粘贴折叠、Ctrl+C 语义、补全、菜单 Enter 选中、状态栏。"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from polya.input import (
    KEY_HINTS,
    PASTE_FOLD_THRESHOLD,
    AtPathCompleter,
    InputBox,
    SlashCompleter,
)


@contextmanager
def make_box(tmp_path):
    with create_pipe_input() as pipe:
        yield InputBox(history_path=tmp_path / "history", input=pipe, output=DummyOutput()), pipe


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


def test_enter_applies_completion_when_menu_open(tmp_path):
    with make_box(tmp_path) as (box, _):
        buffer = Buffer(completer=SlashCompleter())
        buffer.insert_text("/pl")
        # start_completion 需要事件循环；测试直接置入补全态（私有 API，仅此处使用）
        from prompt_toolkit.completion import Completion

        buffer._set_completions(completions=[Completion("/plan", start_position=-3)])
        state = buffer.complete_state
        assert state is not None and state.completions, "补全菜单应已打开"
        box._submit(buffer)
        assert buffer.text == "/plan"  # 选中补全项而非提交半个命令


# ---------- 补全器 ----------


def test_slash_completer_carries_descriptions():
    completions = list(SlashCompleter().get_completions(Document("/pl"), None))
    by_text = {c.text: c for c in completions}
    assert "/plan" in by_text
    assert "规划" in "".join(str(by_text["/plan"].display_meta))


def test_at_path_completer_completes_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x", encoding="utf-8")
    completions = list(AtPathCompleter().get_completions(Document("读一下 @src/ap"), None))
    assert completions, "@src/ap 应给出路径补全"
    # PathCompleter 的 text 是待插入后缀，完整路径看 display
    assert any("app" in str(c.display) for c in completions)


def test_at_path_completer_ignores_plain_text():
    assert list(AtPathCompleter().get_completions(Document("读一下 src/ap"), None)) == []
    assert list(AtPathCompleter().get_completions(Document("@"), None)) == []


# ---------- 状态栏 ----------


def _bar_text(box) -> str:
    return "".join(fragment for _, fragment in box._bottom_bar())


def test_bottom_bar_shows_state_and_hints(tmp_path):
    with make_box(tmp_path) as (box, _):
        box._state = {"mode": "规划", "model": "deepseek-chat", "context": "12.8k/128k", "rules": 2}
        text = _bar_text(box)
        for part in ("规划", "deepseek-chat", "12.8k/128k", "规则 2", KEY_HINTS):
            assert part in text, part

        box._flash_hint("再按一次 Ctrl+C 退出")
        assert "再按一次 Ctrl+C 退出" in _bar_text(box)
        assert KEY_HINTS not in _bar_text(box)  # 临时提示顶替快捷键区


def test_bottom_bar_without_state_shows_hints_only(tmp_path):
    with make_box(tmp_path) as (box, _):
        assert KEY_HINTS in _bar_text(box)
