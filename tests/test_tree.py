"""会话树与投影的纯函数测试（ADR 0005 / ticket 01）。"""

from __future__ import annotations

import pytest

from polya.tree import (
    Entry,
    SessionTree,
    conversation,
    project,
)


def test_append_builds_linear_branch_in_order():
    tree = SessionTree()
    tree.append("system", {"content": "sys"})
    tree.append("user", {"content": "hi"})
    tree.append("assistant", {"content": "hello"})

    branch = tree.active_branch()
    assert [e.kind for e in branch] == ["system", "user", "assistant"]
    assert [e.id for e in branch] == [1, 2, 3]
    assert tree.active_id == 3


def test_move_to_backtracks_and_forks():
    tree = SessionTree()
    a = tree.append("user", {"content": "a"})
    tree.append("assistant", {"content": "branch-1"})
    tree.move_to(a.id)
    tree.append("assistant", {"content": "branch-2"})

    branch = tree.active_branch()
    assert [(e.kind, e.payload["content"]) for e in branch] == [
        ("user", "a"),
        ("assistant", "branch-2"),
    ]
    # 分叉后树里有 3 个入口：a、branch-1、branch-2
    assert len(tree) == 3


def test_move_to_unknown_id_raises():
    tree = SessionTree()
    tree.append("user", {"content": "x"})
    with pytest.raises(ValueError, match="未知入口"):
        tree.move_to(999)


def test_rewind_moves_pointer_back_and_forks():
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("user", {"content": "task"})
    tree.append("assistant", {"content": "answer1"})
    tree.rewind(1)  # 回到 user
    tree.append("assistant", {"content": "answer2"})
    assert [e.kind for e in tree.active_branch()] == ["system", "user", "assistant"]
    assert tree.conversation()[-1] == {"role": "assistant", "content": "answer2"}
    assert len(tree) == 4  # system + user + answer1(孤儿) + answer2


def test_rewind_out_of_range_raises():
    tree = SessionTree()
    tree.reset_with_system("sys")
    with pytest.raises(ValueError, match="无法回退"):
        tree.rewind(1)


def test_jsonl_roundtrip_preserves_tree_and_overrides():
    tree = SessionTree()
    tree.reset_with_system("sys", {"preamble": "P"})
    tool_entry = tree.append("tool", {"tool_call_id": "a", "content": "原始输出"})
    tree.append("user", {"content": "hi"})
    tree.override(tool_entry.id, {**tool_entry.payload, "content": "被清理"})

    restored = SessionTree.from_jsonl(tree.to_jsonl())
    assert restored.project() == tree.project()
    assert restored.read(tool_entry.id) == tree.read(tool_entry.id)
    assert restored.active_id == tree.active_id


def test_jsonl_bad_line_raises():
    with pytest.raises(ValueError, match="坏行"):
        SessionTree.from_jsonl(["{", "not json"])


def test_project_is_deterministic_and_maps_roles():
    tree = SessionTree()
    tree.append("system", {"content": "sys"})
    tree.append("user", {"content": "hi"})
    tree.append("assistant", {"content": "hello"})
    tree.append("tool", {"tool_call_id": "c1", "content": "out"})

    branch = tree.active_branch()
    assert project(branch) == project(branch)  # 确定性：同输入必同输出
    assert project(branch) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]


def test_project_passes_reasoning_and_tool_calls_through():
    entry = Entry(
        id=1,
        parent_id=None,
        kind="assistant",
        payload={
            "content": None,
            "reasoning_content": "think",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{}"}}
            ],
        },
    )
    assert project([entry]) == [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "think",
            "tool_calls": entry.payload["tool_calls"],
        }
    ]


def test_summary_entry_projects_as_user_message():
    tree = SessionTree()
    tree.append("system", {"content": "sys"})
    tree.append(
        "summary",
        {"content": "<session_summary>…</session_summary>", "first_kept_entry_id": 1},
    )
    assert project(tree.active_branch())[1] == {
        "role": "user",
        "content": "<session_summary>…</session_summary>",
    }


def test_conversation_drops_system_entry():
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("user", {"content": "hi"})
    assert conversation(tree.active_branch()) == [{"role": "user", "content": "hi"}]


def test_reset_with_system_seeds_root_entry():
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("user", {"content": "x"})
    assert [e.kind for e in tree.active_branch()] == ["system", "user"]
    tree.reset_with_system("new")
    assert [e.kind for e in tree.active_branch()] == ["system"]
    assert project(tree.active_branch()) == [{"role": "system", "content": "new"}]


def test_replace_conversation_keeps_system_and_rebuilds_from_dicts():
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("user", {"content": "old"})
    tree.replace_conversation(
        [
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        ]
    )
    assert project(tree.active_branch()) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]


def test_system_sections_are_replayed_with_patches():
    tree = SessionTree()
    tree.reset_with_system(
        "rendered",
        {
            "preamble": "P",
            "tools": "<tools>t</tools>",
            "skills": "<skills>s1</skills>",
        },
    )
    tree.append("user", {"content": "hi"})
    tree.append("system", {"sections": {"skills": "<skills>s2</skills>"}})
    assert tree.project()[0] == {
        "role": "system",
        "content": "P\n\n<tools>t</tools>\n\n<skills>s2</skills>",
    }
    assert tree.project()[1] == {"role": "user", "content": "hi"}


def test_system_sections_patch_removes_with_none():
    tree = SessionTree()
    tree.reset_with_system("r", {"preamble": "P", "skills": "<skills>s1</skills>"})
    tree.append("system", {"sections": {"skills": None}})
    assert tree.project()[0] == {"role": "system", "content": "P"}


def test_strip_reasoning_rebuilds_assistant_without_reasoning():
    tree = SessionTree()
    tree.reset_with_system("sys")
    tree.append("assistant", {"content": "hi", "reasoning_content": "think"})
    tree.strip_reasoning()
    assert project(tree.active_branch())[1] == {"role": "assistant", "content": "hi"}
    # id 保持不变（换模型是重启点，只换 payload，不换寻址）
    assert tree.active_id == 2
