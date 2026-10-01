"""Session identity, persistence, fork/clone and export (session-lifecycle)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from polya import Agent
from polya import session as session_store
from polya.commands import CommandContext, dispatch_command
from polya.render import TerminalRenderer
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


def test_new_session_assigns_name_and_clear_also_starts_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    context = CommandContext(agent, restart=lambda *, wipe_scrollback=False: "")

    name = dispatch_command("/new", context)
    assert "新会话" in name
    assert agent.history == []
    assert agent.session_name and agent.session_title is None
    first = agent.session_name

    result = dispatch_command("/clear", context)
    assert "新会话" in result
    assert agent.session_name != first  # /clear 同为开新会话，旧会话可 /resume 找回


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


def test_session_label_shows_topic_not_timestamp():
    """展示层只看主题：时间戳会话名与 ISO 日期不进 /resume 标签。"""
    meta = session_store.SessionMeta(
        name="20260928-234105",
        title="修复登录页面",
        updated="2026-09-28T23:41:05",
    )
    assert meta.label() == "修复登录页面 · 09-28 23:41"
    # 无主题时回落显示名字（仍不拼 ISO 日期）
    assert session_store.SessionMeta(name="20260928-234105").label() == "20260928-234105"


def test_auto_session_renamed_to_topic_slug(tmp_path, monkeypatch):
    """自动会话：主题确定后会话名重命名为主题 slug，name 与 topic 对得上。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _agent()
    stale = agent.new_session()
    _seed(agent)  # 有历史才能落盘
    agent.set_session_title("修复登录页面")
    assert agent.session_name == agent.session_title == "修复登录页面"
    agent.autosave()
    sessions = tmp_path / ".polya" / "sessions"
    assert (sessions / "修复登录页面.jsonl").exists()
    assert not (sessions / f"{stale}.jsonl").exists()  # 不残留时间戳幽灵会话


def test_generated_title_renames_and_removes_provisional(tmp_path, monkeypatch):
    """模型标题替换临时主题时，会话文件跟着改名并清掉旧的临时名文件。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _agent()
    agent.new_session()
    _seed(agent)
    agent.set_session_title("修复登录")  # 临时主题 → 立即落盘
    agent.set_session_title("修复登录页样式")  # 模型标题 → 改名
    sessions = tmp_path / ".polya" / "sessions"
    assert agent.session_name == "修复登录页样式"
    assert (sessions / "修复登录页样式.jsonl").exists()
    assert not (sessions / "修复登录.jsonl").exists()


def test_explicit_name_and_resume_are_locked(tmp_path, monkeypatch):
    """显式 /save 命名与 /resume 恢复的会话不随主题改名。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _agent()
    _seed(agent)
    agent.save_session("我的会话")
    agent.set_session_title("换个主题")
    assert agent.session_name == "我的会话"

    restored = _agent()
    restored.load_session("我的会话")
    restored.set_session_title("又一个主题")
    assert restored.session_name == "我的会话"


def test_slug_sanitizes_title_for_filenames():
    assert session_store.slug("修复 登录/页面") == "修复-登录-页面"
    assert session_store.slug("a:*?b") == "a-b"
    assert session_store.slug("")  # 空主题回落时间戳名


def test_unique_name_avoids_same_second_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first = session_store.unique_name("20260101-000000")
    second = session_store.unique_name("20260101-000000")
    assert first == "20260101-000000"
    assert second == "20260101-000000-2"


def _scrollback_renderer() -> tuple[TerminalRenderer, StringIO]:
    buf = StringIO()
    renderer = TerminalRenderer(Console(file=buf, force_terminal=False, width=120))
    renderer.use_scrollback(renderer._console)
    return renderer, buf


def test_resume_replays_history_into_scrollback(tmp_path, monkeypatch):
    """用户报告：/resume 恢复后要看得见会话内容，而不只是一行「已恢复」。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = _seed(_agent())
    source.tree.append(
        "assistant",
        {
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ],
        },
    )
    source.tree.append("tool", {"tool_call_id": "c1", "content": "列目录输出"})
    source.save_session("alpha")

    target = _agent()
    renderer, buf = _scrollback_renderer()
    result = dispatch_command("/resume alpha", CommandContext(target, renderer=renderer))

    assert "已恢复会话" in result
    out = buf.getvalue()
    assert "❯ 任务一" in out and "答一" in out
    assert "Ran Bash" in out and "ls" in out and "列目录输出" in out


def test_failed_resume_does_not_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    renderer, buf = _scrollback_renderer()
    result = dispatch_command("/resume missing", CommandContext(_agent(), renderer=renderer))
    assert result.startswith("无法")
    assert buf.getvalue() == ""


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.save_session("atomic")
    names = sorted(p.name for p in (tmp_path / ".polya" / "sessions").iterdir())
    assert names == ["atomic.jsonl"]


def test_truncated_tail_line_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.save_session("crashy")
    path = tmp_path / ".polya" / "sessions" / "crashy.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    # 模拟崩溃：丢掉尾行 active，并留一段半截 JSON
    path.write_text("\n".join(lines[:-1]) + '\n{"type": "entry", "id": 9', encoding="utf-8")

    fresh = _agent()
    assert "已恢复会话" in fresh.load_session("crashy")
    assert [m["content"] for m in fresh.history] == ["任务一", "答一"]  # 退回最新入口


def test_corrupt_middle_line_is_still_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _seed(_agent())
    agent.save_session("broken")
    path = tmp_path / ".polya" / "sessions" / "broken.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(2, "{not json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert "无法读取会话" in _agent().load_session("broken")
