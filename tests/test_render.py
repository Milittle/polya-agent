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


def test_assistant_message_prints_thinking_tail_and_content_once():
    renderer, buf = make_renderer(min_render_interval=0, reasoning_tail_lines=2)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    reasoning = "第一行思考\n第二行思考\n第三行思考"
    renderer.update("reasoning_delta", {"delta": reasoning})
    renderer.update("text_delta", {"delta": "我来数一下"})
    renderer.update(
        "assistant_message",
        {"content": "我来数一下", "reasoning": reasoning, "tool_calls": []},
    )

    out = buf.getvalue()
    assert f"✻ 思考 {len(reasoning)} 字" in out  # 标题只报字数（Claude Code ✻ 语汇）
    assert "第三行思考" in out and "第二行思考" in out  # 尾窗大小的思考尾部
    assert "第一行思考" not in out  # 超出尾窗的行不落滚动区
    assert "…" in out  # 截断标记
    assert out.count("我来数一下") == 1  # 正文恰好提交一次（live 区不向滚动区泄漏）


def test_commit_clears_state_for_next_iteration():
    renderer, _buf = make_renderer()
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    renderer.update("text_delta", {"delta": "答案"})
    renderer.update("assistant_message", {"content": "答案", "reasoning": None, "tool_calls": []})
    assert renderer._text_buf == []

    renderer.update("iteration", {"step": 2, "max_steps": 25})
    assert renderer.phase == "thinking"  # 新一轮回到思考阶段


# ---------- 滚动区：正文流式提交（路线 B，段落边界切点） ----------


def test_answer_streams_to_scrollback_at_paragraph_boundary():
    renderer, output = make_renderer(min_render_interval=0)
    renderer.use_scrollback(renderer._console)
    renderer.update("iteration", {"step": 1, "max_steps": 25})

    renderer.update("text_delta", {"delta": "第一段"})
    assert output.getvalue() == ""  # 未到段落边界，先留在尾窗
    assert "第一段" in renderer.preview(80)

    renderer.update("text_delta", {"delta": "\n\n第二段"})
    assert "第一段" in output.getvalue()  # 空行即提交，尾窗只留未完成的段
    assert "第二段" not in output.getvalue()
    assert "第二段" in renderer.preview(80)

    renderer.update("assistant_message", {"content": "第一段\n\n第二段"})
    assert output.getvalue().count("第一段") == 1
    assert output.getvalue().count("第二段") == 1
    assert renderer.preview(80) == ""


def test_code_fence_is_not_split_by_line_budget():
    renderer, output = make_renderer(min_render_interval=0, text_flush_lines=2)
    renderer.use_scrollback(renderer._console)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    code = "```python\n" + "\n".join(f"line{i}" for i in range(6)) + "\n```"

    renderer.update("text_delta", {"delta": code})
    assert output.getvalue() == ""  # 围栏内即便超过行预算也不切
    renderer.update("assistant_message", {"content": code})
    assert "line0" in output.getvalue() and "line5" in output.getvalue()
    assert "```" not in output.getvalue()  # 整块渲染成代码，不是碎段落


def test_long_unbroken_block_streams_by_line_budget():
    renderer, output = make_renderer(min_render_interval=0, text_flush_lines=3)
    renderer.use_scrollback(renderer._console)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    block = "\n".join(f"row{i}" for i in range(6))

    renderer.update("text_delta", {"delta": block})
    assert "row0" in output.getvalue()  # 无空行但超预算 → 在围栏外换行处切一刀
    assert "row5" not in output.getvalue()
    renderer.update("assistant_message", {"content": block})
    assert output.getvalue().count("row0") == 1


def test_answer_header_printed_once_across_streamed_blocks():
    renderer, output = make_renderer(min_render_interval=0)
    renderer.use_scrollback(renderer._console)
    renderer.update("iteration", {"step": 1, "max_steps": 25})

    renderer.update("text_delta", {"delta": "一\n\n二\n\n三"})
    renderer.update("assistant_message", {"content": "一\n\n二\n\n三"})
    assert output.getvalue().count("⏺") == 1  # 多块正文共用一个 ⏺ 标题


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


def test_thinking_tail_renders_dim_italic_header_and_lines():
    renderer, _buf = make_renderer(min_render_interval=0, reasoning_tail_lines=3)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    for line in ("先读文件", "再看依赖", "然后验证", "最后给结论"):
        renderer.update("reasoning_delta", {"delta": line + "\n"})

    assert renderer.has_preview  # 思考阶段尾窗不再隐藏
    preview = renderer.preview(80)
    assert "✻ 思考中 ·" in preview
    assert "\x1b[2;3m" in preview  # dim italic
    assert "最后给结论" in preview and "然后验证" in preview
    assert "先读文件" not in preview  # 只保留尾 3 行
    assert "…" in preview  # 截断标记


def test_thinking_tail_yields_to_tool_then_text():
    renderer, _buf = make_renderer(min_render_interval=0)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    renderer.update("reasoning_delta", {"delta": "正在推理\n"})

    feed_tool_call(renderer, name="read_file", arguments={"path": "a.py"})
    preview = renderer.preview(80)
    assert "正在推理" not in preview  # 工具执行时不显示上一段思考
    assert "Running Read File" in preview

    renderer.update("text_delta", {"delta": "答案"})
    preview = renderer.preview(80)
    assert "答案" in preview
    assert "思考中" not in preview


def test_thinking_tail_throttles_but_updates_after_interval():
    renderer, _buf = make_renderer(min_render_interval=10)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    renderer.update("reasoning_delta", {"delta": "甲\n"})
    first = renderer.preview(80)
    renderer.update("reasoning_delta", {"delta": "乙\n"})
    assert renderer.preview(80) == first  # 间隔内复用帧（思考逐片增长不重渲）
    renderer._preview_at -= 11  # 手工跨过节流窗口
    assert renderer.preview(80) != first


def test_thinking_tail_clears_on_commit_and_collapses_to_scrollback():
    renderer, output = make_renderer()
    renderer.use_scrollback(renderer._console)
    renderer.update("iteration", {"step": 1, "max_steps": 25})
    reasoning = "思考内容第一行\n思考内容第二行"
    renderer.update("reasoning_delta", {"delta": reasoning})
    assert "思考内容" in renderer.preview(80)

    renderer.update(
        "assistant_message", {"content": "答案", "reasoning": reasoning, "tool_calls": []}
    )
    assert renderer.preview(80) == ""
    assert "✻ 思考 15 字" in output.getvalue()  # 折叠摘要进滚动区，全文不落


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


def test_interrupted_live_thinking_does_not_leak_into_next_task():
    renderer, _output = make_renderer(min_render_interval=0)
    import pytest

    with pytest.raises(InterruptedError), renderer:
        renderer.update("iteration", {"step": 1, "max_steps": 25})
        renderer.update("reasoning_delta", {"delta": "半截思考"})
        assert renderer.has_preview
        assert "半截思考" in renderer.preview(80)
        raise InterruptedError()

    # has_preview 现在包含 reasoning：中断后必须显式清，否则半截思考挂在尾窗。
    assert not renderer.has_preview
    assert renderer.preview(80) == ""
    assert renderer._status_label() == "Waiting for model"
    with renderer:
        renderer.update("iteration", {"step": 1})
        renderer.update("reasoning_delta", {"delta": "新一轮思考"})
        assert "半截思考" not in renderer.preview(80)


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
