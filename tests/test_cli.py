"""CLI 的测试：斜杠命令分发、终端审批回调、单任务模式与 REPL 主循环。

REPL 和 -p 模式通过 monkeypatch 注入假 LLM（绕过 API key 构造）与假 input 驱动。
"""

from __future__ import annotations

from types import SimpleNamespace

from mi_z import Agent, tool
from mi_z.cli import build_agent, handle_command, main, make_session, parse_args, terminal_approve
from mi_z.todos import TodoStore
from mi_z.ui import TerminalRenderer


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, messages, tools=None, on_delta=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=self._replies.pop(0))], usage=None)

    @property
    def model(self):
        return "fake-model"


def make_agent(llm=None):
    return Agent(llm=llm or ScriptedLLM([]), tools=[add], todos=TodoStore(), plan_capable=True)


@tool
def add(a: int, b: int) -> str:
    """把两个整数相加（测试用只读工具）。"""
    return str(a + b)


@tool(name="write_thing", dangerous=True)
def write_thing(content: str) -> str:
    """写点东西（测试用危险工具）。"""
    return f"已写入: {content}"


# ---------- 斜杠命令 ----------


def test_quit_commands_return_none():
    agent = make_agent()
    assert handle_command("/exit", agent) is None
    assert handle_command("/quit", agent) is None


def test_status_and_todos_rendering():
    agent = make_agent()
    agent.todos.rewrite([{"content": "修 bug", "status": "in_progress"}])
    assert "[1] [进行中] 修 bug" in handle_command("/todos", agent)
    assert "（TODO 清单为空）" in handle_command("/todos", make_agent())

    status = handle_command("/status", agent)
    assert "执行" in status and "token 用量" in status


def test_plan_toggle():
    agent = make_agent()
    assert agent.plan_mode is False
    assert "已进入规划模式" in handle_command("/plan on", agent)
    assert agent.plan_mode is True
    assert "规划" in handle_command("/status", agent)  # 状态反映模式
    assert "已退出规划模式" in handle_command("/plan off", agent)
    assert agent.plan_mode is False
    assert "用法" in handle_command("/plan", agent)  # 缺参数给出用法


def test_plan_toggle_without_registered_tool_warns():
    plain = Agent(llm=ScriptedLLM([]), tools=[add], todos=TodoStore())  # 未 plan_capable
    result = handle_command("/plan on", plain)
    assert plain.plan_mode is True
    assert "未注册 exit_plan_mode" in result


def test_reset_clears_session():
    agent = make_agent(ScriptedLLM([make_message("好")]))
    agent.run("hi")
    agent.todos.rewrite([{"content": "任务", "status": "pending"}])
    assert "已清空" in handle_command("/reset", agent)
    assert agent.history == [] and len(agent.todos) == 0


def test_unknown_command():
    assert "未知命令" in handle_command("/nope", make_agent())


# ---------- 终端审批 ----------


def test_terminal_approve_yes_no_always(monkeypatch):
    monkeypatch.setattr("mi_z.cli._select_option", lambda options, **kwargs: 0)
    approve = terminal_approve(interactive=True)
    assert approve(add, {}) is True  # 只读工具直接放行，不询问

    # 选择列表下标：0 允许 / 1 总是允许 / 2 拒绝
    answers = iter([2, 1])
    monkeypatch.setattr("mi_z.cli._select_option", lambda options, **kwargs: next(answers))
    approve = terminal_approve(interactive=True)
    assert approve(write_thing, {"content": "x"}) is False  # 拒绝
    assert approve(write_thing, {"content": "y"}) is True  # 总是允许 → 放行

    # 总是允许之后同工具不再询问：答案耗尽会抛 StopIteration，若被询问即测试失败
    assert approve(write_thing, {"content": "z"}) is True


def test_terminal_approve_noninteractive_rejects(monkeypatch):
    def fail(prompt=""):
        raise AssertionError("非交互环境不应交互询问")

    monkeypatch.setattr("builtins.input", fail)
    approve = terminal_approve(interactive=False)
    assert approve(write_thing, {"content": "x"}) is False
    assert approve(add, {}) is True  # 只读不受影响


# ---------- build_agent / main ----------


def test_build_agent_registers_exit_plan_mode(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path)]), llm=ScriptedLLM([]))
    assert "exit_plan_mode" in [t.name for t in agent.tools]
    assert agent.plan_mode is False
    assert agent.max_steps == 25

    planned = build_agent(parse_args(["--root", str(tmp_path), "--plan"]), llm=ScriptedLLM([]))
    assert planned.plan_mode is True


def test_yes_mode_disables_approval(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path), "--yes"]), llm=ScriptedLLM([]))
    assert agent.approve is None and agent.approve_plan is None  # None = 不拦截


def test_prompt_mode_prints_answer(monkeypatch, tmp_path, capsys):
    fake = ScriptedLLM([make_message("答案是 42")])
    monkeypatch.setattr("mi_z.cli.LLM", lambda **kwargs: fake)
    code = main(["--root", str(tmp_path), "-p", "终极问题的答案"])
    assert code == 0
    assert "答案是 42" in capsys.readouterr().out


def test_repl_loop_runs_and_exits(monkeypatch, tmp_path, capsys):
    fake = ScriptedLLM([make_message("1024")])
    monkeypatch.setattr("mi_z.cli.LLM", lambda **kwargs: fake)
    inputs = iter(["2 的 10 次方", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))

    assert main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1024" in out and "工具调用" in out  # 答案 + /status 输出


def test_no_stream_flag_disables_streaming(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path), "--no-stream"]), llm=ScriptedLLM([]))
    assert agent.stream is False


def test_expand_command_dispatches_to_renderer():
    from io import StringIO

    from rich.console import Console

    renderer = TerminalRenderer(Console(file=StringIO(), force_terminal=False, width=120))
    renderer.update("tool_call", {"name": "bash", "call_id": "c1", "arguments": {"command": "ls"}})
    renderer.update(
        "tool_result",
        {"name": "bash", "call_id": "c1", "result": "完整输出", "duration_s": 0.1, "error": False},
    )
    assert "完整输出" in handle_command("/expand", make_agent(), renderer)
    assert "⏺ bash 结果全文" in handle_command("/expand 1", make_agent(), renderer)
    # 无渲染器（非终端会话）：给出解释而不是炸
    assert "非终端" in handle_command("/expand", make_agent())
    assert "用法" in handle_command("/expand x", make_agent(), renderer)


def test_make_session_is_multiline_with_placeholder():
    from prompt_toolkit import PromptSession

    session = make_session()
    assert isinstance(session, PromptSession)  # 多行/按键/占位由 pty 冒烟端到端验证


def test_slugify_keeps_kebab_case_only():
    from mi_z.cli import _slugify

    assert _slugify("`Fix-Login-Bug`\n") == "fix-login-bug"
    assert _slugify("Count  README_words!") == "count-readme-words"
    assert _slugify("中文输入无英文") == ""


def test_topic_from_local_fallbacks():
    from mi_z.cli import _topic_from

    assert _topic_from("Count README words") == "count-readme-words"
    assert _topic_from("统计单词数") == "统计单词数"  # 纯中文退化为截断原文
    assert _topic_from("") == "new-session"


def test_rule_and_prompt_message_lay_out():
    from mi_z.cli import _prompt_message, _rule

    assert _rule("hi", 10) == "── hi ────"
    assert _rule("", 6) == "──────"
    message = _prompt_message({"topic": "count-readme-words"})
    text = "".join(fragment for _, fragment in message)
    assert "✳ count-readme-words" in text and text.endswith("❯ ")


def test_approve_cooperates_with_renderer_pause(monkeypatch):
    """审批询问前暂停渲染器、结束后恢复——非终端下 pause/resume 均 no-op，不炸。"""
    renderer = TerminalRenderer()
    assert renderer._live is None  # 非 tty：未进入 with 前本就无 Live
    renderer.pause()
    renderer.resume()

    monkeypatch.setattr("mi_z.cli._select_option", lambda options, **kwargs: 0)
    approve = terminal_approve(interactive=True, renderer=renderer)
    assert approve(write_thing, {"content": "x"}) is True


# ---------- 审批变更预览：diff / 命令 / 截断 ----------


def approve_and_capture(monkeypatch, capsys, root, name, arguments, choice=0):
    monkeypatch.setattr("mi_z.cli._select_option", lambda options, **kwargs: choice)
    approve = terminal_approve(interactive=True, root=root)
    approved = approve(
        SimpleNamespace(name=name, dangerous=True, fn=None),
        arguments,  # noqa: SLF001
    )
    return approved, capsys.readouterr().err


def test_approval_shows_diff_for_write_file(monkeypatch, tmp_path, capsys):
    (tmp_path / "app.py").write_text("old = 1\nprint(old)\n", encoding="utf-8")
    approved, err = approve_and_capture(
        monkeypatch,
        capsys,
        tmp_path,
        "write_file",
        {"path": "app.py", "content": "new = 2\nprint(new)\n"},
    )
    assert approved is True
    assert "--- a/app.py" in err and "+++ b/app.py" in err
    assert "-old = 1" in err and "+new = 2" in err  # 红删绿增的原料行


def test_approval_new_file_diff_is_all_additions(monkeypatch, tmp_path, capsys):
    approved, err = approve_and_capture(
        monkeypatch, capsys, tmp_path, "write_file", {"path": "new.py", "content": "x = 1\n"}
    )
    assert approved is True
    assert "-old" not in err and "+x = 1" in err


def test_approval_bash_shows_full_command(monkeypatch, tmp_path, capsys):
    approved, err = approve_and_capture(
        monkeypatch, capsys, tmp_path, "bash", {"command": "pytest -q tests/"}
    )
    assert approved is True
    assert "$ pytest -q tests/" in err


def test_approval_reject_and_always(monkeypatch, tmp_path, capsys):
    approved, _ = approve_and_capture(
        monkeypatch, capsys, tmp_path, "bash", {"command": "ls"}, choice=2
    )
    assert approved is False

    approved, err = approve_and_capture(
        monkeypatch, capsys, tmp_path, "bash", {"command": "ls"}, choice=1
    )
    assert approved is True
    assert "不再询问" in err


def test_diff_lines_truncates_with_note():
    from mi_z.cli import _diff_lines

    old = "\n".join(f"old{i}" for i in range(60))
    new = "\n".join(f"new{i}" for i in range(60))
    diff = _diff_lines(old, new, "big.txt", max_lines=10)
    assert len(diff) == 11 and diff[-1].startswith("… 还有 ")
    assert _diff_lines("同", "同", "same.txt") == ["（内容无变化）"]


def test_apply_edits_draft_flags_future_failures():
    from mi_z.cli import _apply_edits_draft

    text = "alpha beta\n"
    draft, error = _apply_edits_draft(text, [{"old_string": "alpha", "new_string": "gamma"}])
    assert draft == "gamma beta\n" and error is None

    draft, error = _apply_edits_draft(text, [{"old_string": "zeta", "new_string": "x"}])
    assert error and "未找到" in error

    draft, error = _apply_edits_draft("a a a", [{"old_string": "a", "new_string": "b"}])
    assert "不唯一" in error
