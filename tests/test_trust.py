"""信任门：true/false/null 存储、父目录继承、资源探测与 /trust 命令。"""

from __future__ import annotations

import json

from polya import Agent, trust
from polya.commands import CommandContext, dispatch_command


def test_trust_roundtrip(tmp_path):
    path = tmp_path / "trust.json"
    assert not trust.is_trusted(tmp_path, path)
    trust.trust(tmp_path, path)
    assert trust.is_trusted(tmp_path, path)
    assert trust.load(path) == {str(tmp_path.resolve()): True}
    assert (path.stat().st_mode & 0o777) == 0o600


def test_untrust_and_forget(tmp_path):
    path = tmp_path / "trust.json"
    trust.trust(tmp_path, path)
    trust.untrust(tmp_path, path)
    assert not trust.is_trusted(tmp_path, path)
    assert trust.load(path)[str(tmp_path.resolve())] is False
    trust.forget(tmp_path, path)
    assert trust.load(path) == {}


def test_parent_inheritance_and_override(tmp_path):
    path = tmp_path / "trust.json"
    parent = tmp_path / "repo"
    child = parent / "pkg"
    child.mkdir(parents=True)
    trust.trust(parent, path)
    assert trust.is_trusted(child, path)  # 继承父目录
    trust.untrust(child, path)  # 子目录 false 覆盖继承
    assert not trust.is_trusted(child, path)
    assert trust.is_trusted(parent, path)


def test_nearest_reports_source(tmp_path):
    path = tmp_path / "trust.json"
    parent = tmp_path / "a"
    child = parent / "b"
    child.mkdir(parents=True)
    trust.trust(parent, path)
    assert trust.nearest(child, path) == (str(parent.resolve()), True)


def test_legacy_format_migrates(tmp_path):
    path = tmp_path / "trust.json"
    path.write_text(json.dumps({"trusted": [str(tmp_path.resolve())]}), encoding="utf-8")
    assert trust.is_trusted(tmp_path, path)
    assert trust.load(path) == {str(tmp_path.resolve()): True}


def test_null_value_falls_through_to_parent(tmp_path):
    path = tmp_path / "trust.json"
    parent = tmp_path / "r"
    child = parent / "c"
    child.mkdir(parents=True)
    trust.trust(parent, path)
    store = trust.load(path)
    store[str(child.resolve())] = None  # 显式「无决定」
    trust.save(store, path)
    assert trust.is_trusted(child, path)  # null 跳过 → 回退父目录


def test_corrupt_trust_file_is_empty(tmp_path):
    path = tmp_path / "trust.json"
    path.write_text("{not json", encoding="utf-8")
    assert trust.load(path) == {}
    assert not trust.is_trusted(tmp_path, path)


def test_has_trust_requiring_resources(tmp_path):
    assert not trust.has_trust_requiring_resources(tmp_path)
    (tmp_path / "AGENTS.md").write_text("x", encoding="utf-8")
    assert trust.has_trust_requiring_resources(tmp_path)
    (tmp_path / "AGENTS.md").unlink()
    (tmp_path / ".polya" / "skills").mkdir(parents=True)
    assert trust.has_trust_requiring_resources(tmp_path)


def test_has_trust_requiring_resources_from_ancestor_agents_skills(tmp_path):
    child = tmp_path / "repo" / "pkg"
    child.mkdir(parents=True)
    (tmp_path / "repo" / ".agents" / "skills").mkdir(parents=True)
    assert trust.has_trust_requiring_resources(child)


def test_trust_command_saves_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(trust, "default_path", lambda: tmp_path / "trust.json")
    project = tmp_path / "proj"
    project.mkdir()
    agent = Agent(llm=object(), tools=[])
    agent.cwd = str(project)
    agent.trusted = False

    status = dispatch_command("/trust", CommandContext(agent))
    assert "当前会话: 不信任" in status and "已存决定: 无" in status

    assert "已信任" in dispatch_command("/trust trust", CommandContext(agent))
    assert trust.is_trusted(project)
    assert "不信任" in dispatch_command("/trust untrust", CommandContext(agent))
    assert not trust.is_trusted(project)
    assert "已清除" in dispatch_command("/trust clear", CommandContext(agent))
    assert trust.load() == {}


def test_build_agent_gates_project_memory_on_trust(tmp_path, monkeypatch):
    from polya.cli import build_agent, parse_args

    monkeypatch.setattr("polya.trust.default_path", lambda: tmp_path / "trust.json")
    (tmp_path / "AGENTS.md").write_text("项目指令", encoding="utf-8")

    class FakeLLM:
        model = "fake"

    untrusted = build_agent(parse_args(["--root", str(tmp_path)]), llm=FakeLLM())
    assert "项目指令" not in untrusted.system_prompt

    trusted = build_agent(parse_args(["--root", str(tmp_path), "--trust"]), llm=FakeLLM())
    assert "项目指令" in trusted.system_prompt

    # 落盘后二次启动（无 --trust）也加载
    trust.trust(tmp_path, tmp_path / "trust.json")
    again = build_agent(parse_args(["--root", str(tmp_path)]), llm=FakeLLM())
    assert "项目指令" in again.system_prompt

    # --no-trust 覆盖已存 true
    denied = build_agent(parse_args(["--root", str(tmp_path), "--no-trust"]), llm=FakeLLM())
    assert "项目指令" not in denied.system_prompt


def test_parent_trust_gates_child_without_own_entry(tmp_path, monkeypatch):
    from polya.cli import build_agent, parse_args

    monkeypatch.setattr("polya.trust.default_path", lambda: tmp_path / "trust.json")
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "AGENTS.md").write_text("继承来的指令", encoding="utf-8")
    trust.trust(repo, tmp_path / "trust.json")

    class FakeLLM:
        model = "fake"

    agent = build_agent(parse_args(["--root", str(pkg)]), llm=FakeLLM())
    assert "继承来的指令" in agent.system_prompt
