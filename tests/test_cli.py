"""CLI 的测试：斜杠命令分发、终端审批回调、单任务模式与 REPL 主循环。

REPL 和 -p 模式通过 monkeypatch 注入假 LLM（绕过 API key 构造）与假 input 驱动。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from polya import Agent, tool
from polya.cli import build_agent, main, parse_args
from polya.loop import handle_command
from polya.models import ModelsConfig, ProviderEntry
from polya.render import TerminalRenderer
from polya.todos import TodoStore


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


@tool(name="write_thing", kind="write")
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


def test_new_clear_reset_all_start_fresh_session():
    agent = make_agent(ScriptedLLM([make_message("好")] * 4))
    agent.run("hi")
    agent.todos.rewrite([{"content": "任务", "status": "pending"}])
    for command in ("/clear", "/new", "/reset"):
        assert "已开始新会话" in handle_command(command, agent)  # 非交互无会话级状态，仅换会话
        assert agent.history == [] and len(agent.todos) == 0
        agent.run("again")


def test_unknown_command():
    assert "未知命令" in handle_command("/nope", make_agent())


# ---------- build_agent / main ----------


def test_build_agent_registers_exit_plan_mode(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path)]), llm=ScriptedLLM([]))
    assert "exit_plan_mode" in [t.name for t in agent.tools]
    assert agent.plan_mode is False
    assert agent.max_steps == 100
    assert agent.max_continuations == 4

    planned = build_agent(parse_args(["--root", str(tmp_path), "--plan"]), llm=ScriptedLLM([]))
    assert planned.plan_mode is True


def test_loop_guard_flags(tmp_path):
    """票 07：CLI 默认开启无进展熔断；--loop-repeat-limit 调阈值，0 关闭。"""
    agent = build_agent(parse_args(["--root", str(tmp_path)]), llm=ScriptedLLM([]))
    assert agent.loop_guard is True and agent.loop_repeat_limit == 3

    off = build_agent(
        parse_args(["--root", str(tmp_path), "--loop-repeat-limit", "0"]),
        llm=ScriptedLLM([]),
    )
    assert off.loop_guard is False

    tuned = build_agent(
        parse_args(["--root", str(tmp_path), "--loop-repeat-limit", "5"]),
        llm=ScriptedLLM([]),
    )
    assert tuned.loop_guard is True and tuned.loop_repeat_limit == 5


def test_prompt_mode_prints_answer(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    fake = ScriptedLLM([make_message("答案是 42")])
    monkeypatch.setattr("polya.cli.LLM", lambda **kwargs: fake)
    code = main(["--root", str(tmp_path), "-p", "终极问题的答案"])
    assert code == 0
    assert "答案是 42" in capsys.readouterr().out


def test_prompt_mode_saves_session(monkeypatch, tmp_path, capsys):
    """票 03：-p 结束也落盘，供 /resume 与审计。"""
    from polya import session as session_store

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    fake = ScriptedLLM([make_message("完成")])
    monkeypatch.setattr("polya.cli.LLM", lambda **kwargs: fake)
    assert main(["--root", str(tmp_path), "-p", "随便"]) == 0
    assert [m.name for m in session_store.list_metas()]


def test_prompt_mode_budget_exhausted_returns_unfinished(monkeypatch, tmp_path, capsys):
    """票 03：达续跑上限 -> [未完成] + 退出码 1，且仍落盘。"""
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="add", arguments='{"a": 1, "b": 2}')
    )
    fake = ScriptedLLM([make_message(tool_calls=[call])])
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("polya.cli.LLM", lambda **kwargs: fake)
    code = main(
        [
            "--root", str(tmp_path), "-p", "循环",
            "--max-steps", "1", "--max-continuations", "0",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "[未完成]" in captured.err
    assert (tmp_path / ".polya" / "sessions").exists()


def test_prompt_mode_loop_guard_returns_unfinished(monkeypatch, tmp_path, capsys):
    """票 07：-p 熔断收尾复用 last_run_exhausted -> [未完成] + 退出码 1。"""
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="add", arguments='{"a": 1, "b": 2}')
    )
    fake = ScriptedLLM([make_message(tool_calls=[call]) for _ in range(2)])
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("polya.cli.LLM", lambda **kwargs: fake)
    code = main(
        [
            "--root", str(tmp_path), "-p", "循环",
            "--max-steps", "0", "--loop-repeat-limit", "1",
        ]
    )
    assert code == 1
    assert "[未完成]" in capsys.readouterr().err
    assert (tmp_path / ".polya" / "sessions").exists()


def test_repl_loop_runs_and_exits(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    fake = ScriptedLLM([make_message("1024")])
    monkeypatch.setattr("polya.cli.LLM", lambda **kwargs: fake)
    inputs = iter(["2 的 10 次方", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))

    assert main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1024" in out and "工具调用" in out  # 答案 + /status 输出


def test_no_stream_flag_disables_streaming(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path), "--no-stream"]), llm=ScriptedLLM([]))
    assert agent.stream is False


def test_details_command_dispatches_to_renderer():
    from io import StringIO

    from rich.console import Console

    renderer = TerminalRenderer(Console(file=StringIO(), force_terminal=False, width=120))
    renderer.update("tool_call", {"name": "bash", "call_id": "c1", "arguments": {"command": "ls"}})
    renderer.update(
        "tool_result",
        {"name": "bash", "call_id": "c1", "result": "完整输出", "duration_s": 0.1, "error": False},
    )
    # 无参：最近 5 块；带 ID：指定块
    assert "完整输出" in handle_command("/details", make_agent(), renderer)
    assert "Ran Bash" in handle_command("/details 1", make_agent(), renderer)
    # 无渲染器（非终端会话）：给出解释而不是炸
    assert "No details" in handle_command("/details", make_agent())
    assert "用法" in handle_command("/details x", make_agent(), renderer)


def test_input_box_builds_multiline_session(tmp_path):
    from prompt_toolkit import PromptSession

    from polya.input import InputBox

    box = InputBox(history_path=tmp_path / "history")
    assert isinstance(box._session, PromptSession)  # 多行/按键/占位由 pty 冒烟端到端验证


def test_topic_is_bounded_readable_and_has_no_control_characters():
    from polya.loop import _topic_from

    assert _topic_from("Fix 输入框 / footer") == "Fix 输入框 / footer"
    assert len(_topic_from("长" * 100)) == 48
    assert all(char.isprintable() for char in _topic_from("hello\x1b\x07\nworld"))


def test_topic_from_local_fallbacks():
    from polya.loop import _topic_from

    assert _topic_from("Count README words") == "Count README words"
    assert _topic_from("统计单词数") == "统计单词数"  # 中文与英文都保留自然语言
    assert _topic_from("") == "新会话"
    assert _topic_from("fix\n  输入框") == "fix 输入框"


def test_rule_and_prompt_message_lay_out():
    from polya.input import _rule, prompt_message

    assert _rule("hi", 10) == "── hi ────"
    assert _rule("", 6) == "──────"
    message = prompt_message({"topic": "count-readme-words"})
    text = "".join(fragment for _, fragment in message)
    assert text == "❯ "


# ---------- 启动解析（票 14）：active profile 与旗标覆盖 ----------


def test_build_agent_resolves_active_profile(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    config = ModelsConfig()
    config.add("p", ProviderEntry("https://x.example/v1", "sk-profile-key-1234", "m-x"))
    config.save(path)
    monkeypatch.setattr("polya.models.default_path", lambda: path)
    agent = build_agent(parse_args(["--root", str(tmp_path)]))
    assert agent.llm.model == "m-x" and agent.llm.profile_name == "p"
    assert "x.example" in str(agent.llm.client.base_url)
    assert agent.context_window == agent.profile.context_window


def test_build_agent_flags_override_active_profile_fieldwise(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    config = ModelsConfig()
    config.add("p", ProviderEntry("https://x.example/v1", "sk-profile-key-1234", "m-x"))
    config.save(path)
    monkeypatch.setattr("polya.models.default_path", lambda: path)
    agent = build_agent(parse_args(["--root", str(tmp_path), "--model", "flag-model"]))
    assert agent.llm.model == "flag-model"
    assert agent.llm.profile_name == "p"  # base_url/key 仍来自 profile
    assert "x.example" in str(agent.llm.client.base_url)


def test_main_reports_corrupt_models_config(tmp_path, monkeypatch, capsys):
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("polya.models.default_path", lambda: path)
    assert main(["--root", str(tmp_path)]) == 2
    assert "配置错误" in capsys.readouterr().err
