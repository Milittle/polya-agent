"""loop.py 的 06/07 号票行为：四选项审批、授权规则、! / # 分流、项目记忆。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from polya.loop import ApprovalGate, _append_project_memory, _run_shell_bang
from polya.permissions import Rule
from polya.tools import tool


@tool(name="bash", kind="exec")
def bash(command: str) -> str:
    """执行（测试用）。"""
    return ""


@tool(name="write_thing", kind="write")
def write_thing(path: str, content: str) -> str:
    """写（测试用）。"""
    return ""


class OptionSpy:
    """替身 _select_option：记录选项与落点，返回预设选择。"""

    def __init__(self, choice=0):
        self.choice = choice
        self.seen_options = None
        self.seen_kwargs = None

    def __call__(self, options, **kwargs):
        self.seen_options = options
        self.seen_kwargs = kwargs
        return self.choice


def screen_with(monkeypatch, choice, tool_obj, arguments, high_risk=False):
    spy = OptionSpy(choice)
    monkeypatch.setattr("polya.approval._select_option", spy)
    monkeypatch.setattr("polya.approval._ask_line", lambda label, default=None: "")
    gate = ApprovalGate(interactive=True)
    return gate.screen(tool_obj, arguments, high_risk=high_risk), gate, spy


def test_low_risk_defaults_to_allow_but_escape_denies(monkeypatch):
    outcome, _, spy = screen_with(monkeypatch, choice=3, tool_obj=bash, arguments={"command": "ls"})
    assert outcome.approved is False
    deny_index = len(spy.seen_options) - 1
    assert spy.seen_kwargs["cancel_index"] == deny_index
    assert spy.seen_kwargs["initial"] == 0


def test_bash_options_offer_prefix_and_modify(monkeypatch):
    _, _, spy = screen_with(
        monkeypatch, choice=0, tool_obj=bash, arguments={"command": "pytest tests/test_a.py -q"}
    )
    labels = [label for label, _ in spy.seen_options]
    assert labels == [
        "Allow once",
        "Allow prefix for session",
        "Edit command",
        "Allow all for session",
        "Deny",
    ]


def test_high_risk_hides_prefix_option(monkeypatch):
    outcome, _, spy = screen_with(
        monkeypatch,
        choice=2,
        tool_obj=bash,
        arguments={"command": "git push -f origin"},
        high_risk=True,
    )
    labels = [label for label, _ in spy.seen_options]
    assert labels == ["Allow once", "Edit command", "Deny"]  # 高危无授权出口（Q9）
    assert outcome.approved is False
    assert spy.seen_kwargs["cancel_index"] == 2


def test_compound_command_hides_prefix_option(monkeypatch):
    _, _, spy = screen_with(
        monkeypatch, choice=0, tool_obj=bash, arguments={"command": "cd tests && ls"}
    )
    labels = [label for label, _ in spy.seen_options]
    assert labels == [
        "Allow once",
        "Edit command",
        "Allow all for session",
        "Deny",
    ]  # Q14：复合命令不给前缀授权


def test_prefix_choice_creates_rule_and_feed_decide(monkeypatch):
    outcome, gate, _ = screen_with(
        monkeypatch, choice=1, tool_obj=bash, arguments={"command": "pytest tests/a.py -q"}
    )
    assert outcome.approved is True
    assert outcome.rule == Rule("bash", "pytest tests/a.py")
    gate.rules.append(outcome.rule)
    # 规则进会话集后，decide() 第 ④ 步放行同前缀命令（不再询问）
    from polya.permissions import Context, Decision, decide

    assert decide(
        bash, {"command": "pytest tests/a.py -x"}, Context(rules=tuple(gate.rules))
    ) == Decision("allow", "rule")


def test_modify_choice_returns_command(monkeypatch):
    monkeypatch.setattr("polya.approval._select_option", OptionSpy(2))
    monkeypatch.setattr("polya.approval._ask_line", lambda label, default=None: default + " -q")
    gate = ApprovalGate(interactive=True)
    outcome = gate.screen(bash, {"command": "pytest tests/"})
    assert outcome.approved is True
    assert outcome.command == "pytest tests/ -q"


def test_deny_with_reason_carries_it(monkeypatch):
    monkeypatch.setattr("polya.approval._select_option", OptionSpy(4))
    monkeypatch.setattr("polya.approval._ask_line", lambda label, default=None: "太危险")
    gate = ApprovalGate(interactive=True)
    outcome = gate.screen(bash, {"command": "reboot"})
    assert outcome.approved is False
    assert outcome.reason == "太危险"


def test_noninteractive_screens_reject_with_preview(monkeypatch, capsys):
    monkeypatch.setattr(
        "polya.approval._select_option", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    gate = ApprovalGate(interactive=False)
    outcome = gate.screen(bash, {"command": "ls"})
    assert outcome.approved is False
    assert "$ ls" in capsys.readouterr().err


# ---------- 07：! / # 分流 ----------


def test_shell_bang_appends_context_and_prints(capsys, tmp_path):
    agent = SimpleNamespace(history=[])
    _run_shell_bang(agent, str(tmp_path), "echo hello", lambda msg, style: None)
    assert len(agent.history) == 1
    assert agent.history[0]["role"] == "user"
    assert "$ echo hello" in agent.history[0]["content"]
    assert "hello" in agent.history[0]["content"]


def test_shell_bang_truncates_long_output(tmp_path):
    agent = SimpleNamespace(history=[])
    _run_shell_bang(agent, str(tmp_path), "seq 1 10000", lambda msg, style: None)
    assert len(agent.history[0]["content"]) < 9000


def test_project_memory_append_creates_and_appends(tmp_path, capsys):
    say = lambda msg, style: None  # noqa: E731
    _append_project_memory(str(tmp_path), "构建用 uv run pytest", say)
    _append_project_memory(str(tmp_path), "发布走 gh", say)
    content = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert content.startswith("# 项目记忆")
    assert "构建用 uv run pytest" in content and "发布走 gh" in content


def test_build_agent_injects_project_memory(tmp_path, monkeypatch):
    from polya.agent import DEFAULT_SYSTEM_PROMPT  # noqa: F401 - 保持导入面一致
    from polya.builtin import CODING_SYSTEM_PROMPT
    from polya.cli import _project_memory, build_agent, parse_args

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")  # 隔离用户级技能目录
    (tmp_path / "AGENTS.md").write_text("测试锚点：polya-agent 项目记忆", encoding="utf-8")
    assert _project_memory(str(tmp_path)) == "测试锚点：polya-agent 项目记忆"

    class FakeLLM:
        model = "fake"

    agent = build_agent(parse_args(["--root", str(tmp_path)]), llm=FakeLLM())
    assert agent.system_prompt.startswith(CODING_SYSTEM_PROMPT)
    assert "测试锚点：polya-agent 项目记忆" in agent.system_prompt
    assert "<project_memory>" in agent.system_prompt  # 项目记忆是独立 section

    empty = tmp_path / "empty"
    empty.mkdir()
    agent2 = build_agent(parse_args(["--root", str(empty)]), llm=FakeLLM())
    assert agent2.system_prompt.startswith(CODING_SYSTEM_PROMPT)
    assert "<project_memory>" not in agent2.system_prompt  # 缺失即不产生该段
