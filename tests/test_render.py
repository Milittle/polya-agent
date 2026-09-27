"""TerminalRenderer 测试：注入捕获输出的 Console，滚动区可捕获。"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from polya.render import TerminalRenderer, _collapse


def make_renderer(**kwargs) -> tuple[TerminalRenderer, StringIO]:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    return TerminalRenderer(console, **kwargs), buf


def feed_tool_call(renderer, name="bash", arguments=None, call_id="c1"):
    renderer.update("tool_call", {"name": name, "call_id": call_id, "arguments": arguments or {}})


def feed_tool_result(renderer, result, duration_s=0.4, error=False, call_id="c1", name="bash"):
    renderer.update(
        "tool_result",
        {
            "name": name,
            "call_id": call_id,
            "result": result,
            "duration_s": duration_s,
            "error": error,
        },
    )


# ---------- _collapse 纯函数 ----------


def test_collapse_limits_lines_and_chars():
    text = "\n".join(f"line{i}" for i in range(20))
    shown, hidden = _collapse(text, max_lines=8, max_chars=600)
    assert shown == "\n".join(f"line{i}" for i in range(8))
    assert hidden == 12

    shown, hidden = _collapse("x" * 1000, max_lines=8, max_chars=600)
    assert shown == "x" * 600 + "…" and hidden == 0

    assert _collapse("", 8, 600) == ("", 0)


# ---------- 滚动区：工具块（Claude Code 树形：⏺ 头 + ⎿ 结果） ----------


def test_tool_block_header_and_collapsed_result():
    renderer, buf = make_renderer()
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    feed_tool_call(renderer, arguments={"command": "wc -w README.md"})
    feed_tool_result(renderer, "\n".join(f"行{i}" for i in range(20)))

    out = buf.getvalue()
    assert "Running Bash" in out and "$ wc -w README.md" in out
    assert "  行0" in out and "  行2" in out  # 首行 ⎿ 连接符、续行 4 空格
    assert "行3" not in out  # 折叠到前 8 行
    assert "+ Show details: /details 1" in out


def test_short_result_shows_all_lines_with_duration():
    renderer, buf = make_renderer()
    feed_tool_call(renderer, name="add", arguments={"a": 2, "b": 3})
    feed_tool_result(renderer, "5", duration_s=0.01)

    out = buf.getvalue()
    assert "  5" in out and "0.01s" in out and "还有" not in out


def test_empty_result_shows_placeholder_tail():
    renderer, buf = make_renderer()
    feed_tool_call(renderer)
    feed_tool_result(renderer, "", duration_s=0.2)
    assert "No output" in buf.getvalue() and "0.2s" in buf.getvalue()


def test_error_result_is_printed_in_red():
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, no_color=False, width=120
    )  # 终端模式才输出 ANSI
    renderer = TerminalRenderer(console)
    feed_tool_call(renderer)
    feed_tool_result(renderer, "Error: 用户拒绝了工具调用 bash", error=True)
    assert "\x1b[31m" in buf.getvalue()  # red


# ---------- 滚动区：assistant 提交 ----------


def test_assistant_message_collapses_thinking_and_prints_content_once():
    renderer, buf = make_renderer()
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    reasoning = "长" * 80
    renderer.update("reasoning_delta", {"delta": reasoning})
    renderer.update("text_delta", {"delta": "我来数一下"})
    renderer.update(
        "assistant_message",
        {"content": "我来数一下", "reasoning": reasoning, "tool_calls": []},
    )

    out = buf.getvalue()
    assert "✻ 思考 80 字" in out  # 思考折叠为一行 dim italic 摘要（Claude Code ✻ 语汇）
    assert out.count("长") == 40  # 只见摘要里的前 40 字，全文不出现
    assert out.count("我来数一下") == 1  # 正文恰好提交一次（live 区不向滚动区泄漏）


def test_commit_clears_state_for_next_iteration():
    renderer, _buf = make_renderer()
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    renderer.update("text_delta", {"delta": "答案"})
    renderer.update("assistant_message", {"content": "答案", "reasoning": None, "tool_calls": []})
    assert renderer._text_buf == []

    renderer.update("iteration", {"step": 2, "max_steps": 25})
    assert renderer.phase == "thinking"  # 新一轮回到思考阶段


# ---------- 工具头行：按工具特化的人话参数 ----------


def test_header_arg_specializes_by_tool():
    from polya.render import _header_arg

    assert _header_arg("bash", {"command": "pytest -q"}) == "$ pytest -q"
    assert _header_arg("bash", {"command": "a &&\nb"}) == "$ a && ⏎ b"
    assert _header_arg("read_file", {"path": "polya/ui.py", "start_line": 10, "end_line": 50}) == (
        "polya/ui.py:10-50"
    )
    assert _header_arg("multi_edit", {"path": "a.py", "edits": [{}, {}]}) == "a.py (2 edits)"
    assert _header_arg("grep", {"pattern": "def run", "glob": "*.py"}) == "def run  ·  glob *.py"
    assert _header_arg("web_fetch", {"url": "https://example.com"}) == "https://example.com"
    assert _header_arg("todo_write", {"items": [{}]}) == "1 items"
    # 兜底：陌生工具仍是紧凑 JSON
    assert _header_arg("custom", {"a": 1}) == '{"a": 1}'


# ---------- 状态标签与尾窗 ----------


def test_status_label_migrates_with_phase():
    renderer, _buf = make_renderer()
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    assert renderer._status_label() == "Waiting for model"

    renderer.update("text_delta", {"delta": "答案"})
    assert renderer._status_label() == "Responding"

    feed_tool_call(renderer)
    assert renderer._status_label() == "Running Bash"


def test_usage_feeds_context_occupancy():
    renderer, _buf = make_renderer(context_window=128000)
    renderer.update("usage", {"last": {"prompt_tokens": 12800, "cached_tokens": 3200}})
    assert renderer._ctx_used == 12800
    assert renderer._ctx_cached == 3200


def test_tool_output_delta_shows_tail_then_clears():
    renderer, _buf = make_renderer(tool_output_tail_lines=3)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    feed_tool_call(renderer)
    for i in range(5):
        renderer.update("tool_output_delta", {"name": "bash", "line": f"out{i}"})

    preview = renderer.preview(80)
    assert "out4" in preview and "out3" in preview
    assert "out0" not in preview  # 只保留尾 3 行

    feed_tool_result(renderer, "out0\nout1\nout2\nout3\nout4")
    assert renderer.preview(80) == ""  # 尾窗已清


# ---------- 生命周期 ----------


def test_context_manager_is_reentrant_noop():
    renderer, _buf = make_renderer()
    with renderer:
        renderer.update("text_delta", {"delta": "x"})
    assert renderer._text_buf == []


# ---------- /expand：块留档与全文展开 ----------


def test_expand_blocks_outputs_full_content():
    renderer, buf = make_renderer()
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    long_result = "\n".join(f"行{i}" for i in range(20))
    feed_tool_call(renderer)
    feed_tool_result(renderer, long_result, duration_s=0.4)
    renderer.update(
        "assistant_message",
        {"content": "答案", "reasoning": "思考全文内容", "tool_calls": []},
    )

    expanded = renderer.expand_blocks(5)
    assert "Ran Bash · 0.4s" in expanded
    assert "行19" in expanded  # 折叠隐藏的行在展开里可见
    assert "✻ 思考全文（6 字）" in expanded and "思考全文内容" in expanded
    assert renderer.expand_blocks(1).startswith("✻")  # count 只取最近 N 块


def test_expand_blocks_empty_hint():
    renderer, _buf = make_renderer()
    assert "暂无可展开" in renderer.expand_blocks()


def test_live_tool_tail_keeps_advancing_after_preview_limit():
    renderer, output = make_renderer(tool_output_tail_lines=3, min_render_interval=0)
    renderer.use_scrollback(renderer._console)
    feed_tool_call(renderer, arguments={"command": "pytest"})
    for i in range(15):
        renderer.update("tool_output_delta", {"line": f"progress-{i}"})
    preview = renderer.preview(40, max_lines=2)
    assert "progress-14" in preview and "progress-13" in preview
    assert "progress-12" not in preview
    assert len(preview.splitlines()) <= 3
    assert output.getvalue() == ""
    feed_tool_result(renderer, "all tests passed")
    assert renderer.preview(40) == ""
    assert "all tests passed" in output.getvalue()


def test_streamed_markdown_keeps_table_and_code_structure():
    renderer, output = make_renderer(min_render_interval=0)
    renderer.use_scrollback(renderer._console)
    content = "| Name | Value |\n| --- | --- |\n| answer | 42 |\n\n```python\nprint(42)\n```"
    for fragment in (content[:17], content[17:50], content[50:]):
        renderer.update("text_delta", {"delta": fragment})
    assert "print" in renderer.preview(60, max_lines=8)
    renderer.update("assistant_message", {"content": content})
    final = output.getvalue()
    assert (
        "─" in final and "answer" in final and "|" not in final
    )  # A table, not separate Markdown lines.
    assert "print(42)" in final
    assert "```" not in final
    assert final.count("answer") == 1
    assert renderer.preview(60) == ""


def test_interrupted_live_text_is_retained_without_leaking_into_next_task():
    renderer, output = make_renderer()
    renderer.use_scrollback(renderer._console)
    import pytest

    with pytest.raises(InterruptedError), renderer:
        renderer.update("text_delta", {"delta": "unfinished paragraph"})
        assert "unfinished paragraph" in renderer.preview(80)
        raise InterruptedError()
    assert output.getvalue().count("unfinished paragraph") == 1
    assert "Partial response" in output.getvalue()
    assert not renderer.has_preview
    with renderer:
        renderer.update("iteration", {"step": 1})
        renderer.update("assistant_message", {"content": "new answer"})
    assert output.getvalue().count("unfinished paragraph") == 1


def test_plan_submitted_prints_plan_to_scrollback():
    renderer, output = make_renderer()
    renderer.update("plan_submitted", {"plan": "## 步骤\n1. 读文件"})
    assert renderer._status_label() == "Waiting for model"
    assert "计划已提交" in output.getvalue()
    assert "读文件" in output.getvalue()


def test_pending_shell_result_does_not_claim_completion():
    renderer, buf = make_renderer()
    renderer.use_scrollback(renderer._console)
    feed_tool_call(renderer)
    feed_tool_result(renderer, "[命令 1]\n仍在运行；用 bash_output 等待，或 kill_bash 终止")
    assert "Running Bash" in buf.getvalue()
    assert "Ran Bash" not in buf.getvalue()
    buf.truncate(0)
    buf.seek(0)
    feed_tool_call(renderer)
    feed_tool_result(renderer, "日志中包含仍在运行这几个字\n退出码 0")
    assert "Ran Bash" in buf.getvalue()
