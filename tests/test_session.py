"""Session identity, persistence, fork/clone and export (session-lifecycle)."""

from __future__ import annotations

import json
from pathlib import Path

from polya import Agent
from polya import session as session_store
from polya.commands import CommandContext, dispatch_command
from polya.tree import SessionTree


def _agent():
    return Agent(llm=object(), tools=[])


def _seed(agent, text="任务一", answer="答一"):
    agent.append_user_message(text)
    agent.tree.append("assistant", {"content": answer})
    return agent


def test_save_writes_metadata_header(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _agent()
    _seed(agent)
    agent.set_session_title("修复登录")

    assert "已保存会话 demo" in agent.save_session("demo")
    raw = (tmp_path / ".polya" / "sessions" / "demo.jsonl").read_text(encoding="utf-8")
    header = json.loads(raw.splitlines()[0])
    assert header["type"] == "session"
    assert header["name"] == "demo"
    assert header["title"] == "修复登录"
    assert header["created"] and header["updated"]


def test_legacy_header_without_metadata_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / ".polya" / "sessions"
    directory.mkdir(parents=True)
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("user", {"content": "旧任务"})
    body = [json.dumps({"type": "session", "version": 1, "cwd": "/old"}), *tree.to_jsonl()]
    (directory / "legacy.jsonl").write_text("\n".join(body) + "\n", encoding="utf-8")

    agent = _agent()
    assert "已恢复会话 legacy" in agent.load_session("legacy")
    assert agent.session_name == "legacy"
    assert agent.session_title is None
    assert agent.history == [{"role": "user", "content": "旧任务"}]


def test_load_session_resets_derived_state(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = _seed(_agent())
    source.save_session("alpha")
    source.total_usage["total_tokens"] = 999
    source.tool_counts["bash"] = 3
    source.todos.rewrite([{"content": "旧 TODO", "status": "pending"}])

    target = _agent()
    target.load_session("alpha")
    assert target.total_usage["total_tokens"] == 0
    assert dict(target.tool_counts) == {}
    assert target.todos.as_dicts() == []


def test_new_session_assigns_name_and_clear_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    context = CommandContext(agent, restart=lambda: "")

    name = dispatch_command("/new", context)
    assert "新会话" in name
    assert agent.history == []
    assert agent.session_name and agent.session_title is None
    first = agent.session_name

    assert "已清空" in dispatch_command("/clear", context)
    assert agent.session_name == first  # /clear 不改会话身份


def test_resume_lists_and_switches(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _seed(_agent()).save_session("alpha")
    _seed(_agent(), "二", "答二").save_session("beta")

    listing = dispatch_command("/resume", CommandContext(_agent()))
    assert "alpha" in listing and "beta" in listing

    target = _agent()
    assert "已恢复会话 beta" in dispatch_command("/resume beta", CommandContext(target))
    assert target.history == [
        {"role": "user", "content": "二"},
        {"role": "assistant", "content": "答二"},
    ]


def test_session_choices_expose_metas(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.set_session_title("标题")
    agent.save_session("named")

    from polya.commands import _session_choices

    choices = dict(_session_choices())
    assert "named" in choices
    assert "标题" in choices["named"]


def test_fork_session_copies_ancestor_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.tree.append("assistant", {"content": "答二"})

    assert "分叉出新会话" in dispatch_command("/fork 2", CommandContext(agent))
    assert len(agent.tree) == 2  # system + user
    assert agent.history == [{"role": "user", "content": "任务一"}]


def test_clone_session_duplicates_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    before = agent.history

    assert "已复制当前会话" in dispatch_command("/clone", CommandContext(agent))
    assert len(agent.tree) == 3
    assert agent.history == before


def test_export_markdown_contains_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.tree.append(
        "assistant",
        {
            "content": "调用工具",
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}],
        },
    )
    agent.tree.append("tool", {"tool_call_id": "c1", "content": "命令输出"})

    target = tmp_path / "out" / "session.md"
    assert "已导出会话" in dispatch_command(f"/export {target}", CommandContext(agent))
    text = target.read_text(encoding="utf-8")
    assert "## 用户" in text and "任务一" in text
    assert "## 助手" in text and "调用工具" in text
    assert "`bash`" in text
    assert "## 工具结果" in text and "命令输出" in text


def test_autosave_creates_resumable_session(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.autosave()

    assert agent.session_name is not None
    assert [m.name for m in session_store.list_metas()] == [agent.session_name]
    restored = _agent()
    assert "已恢复会话" in restored.load_session(agent.session_name)
    assert restored.history == agent.history


def test_tree_copy_branch_upto_and_copy():
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("user", {"content": "u1"})
    tree.append("assistant", {"content": "a1"})
    tree.append("user", {"content": "u2"})

    prefix = tree.copy_branch_upto(2)
    assert [e.id for e in prefix.active_branch()] == [1, 2]
    assert prefix.project() == tree.project()[:2]

    clone = tree.copy()
    assert clone.active_id == tree.active_id
    assert clone.project() == tree.project()


def test_unique_name_avoids_same_second_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first = session_store.unique_name("20260101-000000")
    second = session_store.unique_name("20260101-000000")
    assert first == "20260101-000000"
    assert second == "20260101-000000-2"
