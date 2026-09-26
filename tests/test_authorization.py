"""Authorization behavior across edited commands, drivers and execution context."""

from io import StringIO

import pytest
from rich.console import Console

from polya.agent import Agent, ToolCall
from polya.approval import ApprovalGate, ApprovalOutcome, _approval_body
from polya.loop import _run_tool
from polya.permissions import Context, Rule, assess, decide, rule_for
from polya.render import TerminalRenderer
from polya.tools import tool


@tool(kind="exec")
def bash(command: str) -> str:
    """Return the command instead of executing it."""
    return command


@tool(kind="write")
def write_file(path: str) -> str:
    """Fake file writer."""
    return path


@pytest.mark.parametrize(
    "command",
    [
        "pytest $(rm notes.txt)",
        "pytest > out",
        "pytest $FLAGS",
        "pytest *.py",
        "python -c 'print(1)'",
        "bash -c 'echo x'",
        "pytest; echo x",
        "pytest `echo x`",
        "ENV=x pytest",
        "pytest 'unterminated",
    ],
)
def test_dynamic_commands_never_gain_prefix_authorization(command):
    assert rule_for(bash, {"command": command}) is None
    if not command.startswith(("python", "bash")):
        assert not Rule("bash", "pytest").matches(bash, {"command": command})


def test_edited_command_is_reviewed_again_and_never_runs_if_denied():
    gate = ApprovalGate(True)
    calls = []

    def screen(item, args, high_risk=False):
        calls.append((args["command"], high_risk))
        if len(calls) == 1:
            return ApprovalOutcome(True, command="rm -rf important")
        return ApprovalOutcome(False, reason="do not delete")

    gate.screen = screen
    agent = Agent(llm=object(), tools=[bash], approve=lambda *_: False)
    output = StringIO()
    result = _run_tool(
        agent,
        TerminalRenderer(Console(file=output)),
        ToolCall(name="bash", call_id="a", arguments={"command": "pytest tests"}),
        True,
        gate,
    )
    assert calls == [("pytest tests", False), ("rm -rf important", True)]
    assert "do not delete" in result
    assert "Running" not in output.getvalue()
    assert "Denied Bash" in output.getvalue()


def test_builtin_hook_executes_only_reapproved_arguments():
    gate = ApprovalGate(True)
    answers = iter([ApprovalOutcome(True, command="pytest other"), ApprovalOutcome(True)])
    gate.screen = lambda *a, **kw: next(answers)
    agent = Agent(llm=object(), tools=[bash], approve=gate.as_approve())
    result = agent._builtin_tool(
        ToolCall(name="bash", call_id="a", arguments={"command": "pytest tests"})
    )
    assert result == "pytest other"


def test_write_cannot_escape_workspace_even_with_session_allow(tmp_path):
    gate = ApprovalGate(True, root=tmp_path)
    gate.allow_all = True
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    for path in ["../other.txt", "outside/other.txt"]:
        outcome = gate.authorize(write_file, {"path": path})
        assert not outcome.approved and "outside the workspace" in outcome.reason
    assert not Rule("write_file", "src").matches(write_file, {"path": "src/../other.txt"})


@pytest.mark.parametrize(
    ("command", "risk"),
    [
        ("git status", "low"),
        ("pytest tests", "medium"),
        ("git reset --hard", "high"),
        ("python script.py", "unknown"),
        ("git diff --output=out", "unknown"),
        ("cat /etc/passwd", "unknown"),
        ("echo x && pytest", "unknown"),
    ],
)
def test_risk_changes_prompt_default_without_automatically_allowing_shell(command, risk):
    args = {"command": command}
    assert assess(bash, args).risk == risk
    assert decide(bash, args, Context()).verdict == "ask"


@pytest.mark.parametrize(
    ("command", "allow"),
    [
        ("git status", True),
        ("pytest tests", True),
        ("rm -rf build", False),
        ("python custom.py", False),
    ],
)
def test_prompt_initial_selection_tracks_risk_and_escape_always_denies(monkeypatch, command, allow):
    def select(options, *, cancel_index, initial):
        assert options[cancel_index][0] == "Deny"
        assert options[initial][0] == ("Allow once" if allow else "Deny")
        return cancel_index

    monkeypatch.setattr("polya.approval._select_option", select)
    monkeypatch.setattr("polya.approval._ask_line", lambda *a: "")
    assert not ApprovalGate(True).screen(bash, {"command": command}).approved


def test_approval_command_is_not_cut_after_eight_lines():
    command = "\n".join(f"echo {i}" for i in range(12))
    body = "\n".join(line.plain for line in _approval_body("bash", {"command": command}, None))
    assert "echo 11" in body


def test_stream_preview_is_bounded_and_details_keep_full_result():
    output = StringIO()
    renderer = TerminalRenderer()
    renderer.use_scrollback(Console(file=output, width=100))
    renderer.update("tool_call", {"name": "bash", "arguments": {"command": "pytest tests"}})
    lines = [f"line-{i}" for i in range(10)]
    for line in lines:
        renderer.update("tool_output_delta", {"line": line})
    preview = renderer.preview(100, max_lines=3)
    assert "line-9" in preview and "line-6" not in preview
    assert output.getvalue() == ""
    renderer.update("tool_result", {"name": "bash", "result": "\n".join(lines)})
    assert "line-2" in output.getvalue() and "line-3" not in output.getvalue()
    assert output.getvalue().count("line-0") == 1
    assert "Ran Bash" in output.getvalue()
    assert "line-9" in renderer.expand_blocks(1)
    assert "pytest tests" in renderer.expand_blocks(1)


def test_long_single_line_has_details_and_is_bounded():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, width=1000))
    renderer.update("tool_call", {"name": "read_file", "arguments": {"path": "x"}})
    renderer.update("tool_result", {"name": "read_file", "result": "x" * 1000})
    assert "x" * 401 not in output.getvalue()
    assert "Show details" in output.getvalue()
    assert "x" * 1000 in renderer.expand_blocks(1)


def test_approval_receipt_only_records_explicit_final_approval():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, width=120))
    gate = ApprovalGate(True, renderer)
    replies = iter(
        [
            ApprovalOutcome(True, command="pytest revised"),
            ApprovalOutcome(True, rule=Rule("bash", "pytest revised")),
        ]
    )
    gate.screen = lambda *a, **kw: next(replies)
    outcome = gate.authorize(bash, {"command": "pytest original"})
    assert outcome.arguments == {"command": "pytest revised"}
    assert output.getvalue().count("You approved polya to run") == 1
    assert "pytest original" not in output.getvalue()
    assert "+ Show details: /details 1" in output.getvalue()
    # A subsequent match reuses the grant without inventing a second user approval.
    gate.authorize(bash, {"command": "pytest revised -q"})
    assert output.getvalue().count("You approved polya to run") == 1
    renderer.update("tool_call", {"name": "bash", "arguments": outcome.arguments})
    renderer.update("tool_result", {"name": "bash", "result": "done"})
    assert "Approved Bash" in renderer.show_details(1)
    assert "pytest revised" in renderer.show_details(1)
    assert "Scope: bash(pytest revised:*)" in renderer.show_details(1)
    assert "Ran Bash" in renderer.show_details(2)


def test_denial_has_no_approval_receipt():
    output = StringIO()
    gate = ApprovalGate(True, TerminalRenderer(Console(file=output)))
    gate.screen = lambda *a, **kw: ApprovalOutcome(False)
    assert not gate.authorize(bash, {"command": "pytest tests"}).approved
    assert "You approved" not in output.getvalue()


def test_expired_detail_id_never_resolves_to_another_block():
    from polya.loop import handle_command

    renderer = TerminalRenderer(Console(file=StringIO()))
    for i in range(25):
        renderer.update("tool_result", {"name": "bash", "result": f"output {i}"})
    assert "unavailable" in renderer.show_details(1)
    assert "output 24" in renderer.show_details(25)
    assert "output 24" in handle_command("/details 25", None, renderer)
    assert "Usage" in handle_command("/details bad", None, renderer)
    assert "Usage" in handle_command("/details -1", None, renderer)
