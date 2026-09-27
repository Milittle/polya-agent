"""loop.py：! / # 分流与项目记忆。"""

from __future__ import annotations

from pathlib import Path

from polya.loop import _append_project_memory, _run_shell_bang


class _FakeAgent:
    """替身 Agent：只提供 _run_shell_bang 需要的注入面。"""

    def __init__(self):
        self.history: list[dict] = []

    def append_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})


def test_shell_bang_appends_context_and_prints(capsys, tmp_path):
    agent = _FakeAgent()
    _run_shell_bang(agent, str(tmp_path), "echo hello", lambda msg, style: None)
    assert len(agent.history) == 1
    assert agent.history[0]["role"] == "user"
    assert "$ echo hello" in agent.history[0]["content"]
    assert "hello" in agent.history[0]["content"]


def test_shell_bang_truncates_long_output(tmp_path):
    agent = _FakeAgent()
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

    agent = build_agent(parse_args(["--root", str(tmp_path), "--trust"]), llm=FakeLLM())
    assert agent.system_prompt.startswith(CODING_SYSTEM_PROMPT)
    assert "测试锚点：polya-agent 项目记忆" in agent.system_prompt
    assert "<project_memory>" in agent.system_prompt  # 项目记忆是独立 section

    empty = tmp_path / "empty"
    empty.mkdir()
    agent2 = build_agent(parse_args(["--root", str(empty), "--trust"]), llm=FakeLLM())
    assert agent2.system_prompt.startswith(CODING_SYSTEM_PROMPT)
    assert "<project_memory>" not in agent2.system_prompt  # 缺失即不产生该段
