"""Session import: polya JSONL and pi session format; JSONL export round-trip."""

from __future__ import annotations

import json
from pathlib import Path

from polya import Agent
from polya.commands import CommandContext, dispatch_command


def _agent():
    return Agent(llm=object(), tools=[])


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )


def test_export_jsonl_and_import_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = _agent()
    agent.append_user_message("任务")
    agent.tree.append("assistant", {"content": "答案"})
    out = tmp_path / "s.jsonl"
    assert "已导出会话" in dispatch_command(f"/export {out}", CommandContext(agent))

    restored = _agent()
    assert "已导入" in dispatch_command(f"/import {out}", CommandContext(restored))
    assert restored.history == [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "答案"},
    ]


def test_import_pi_message_roles(tmp_path):
    records = [
        {
            "type": "session",
            "version": 3,
            "id": "s1",
            "timestamp": "2024-12-03T14:00:00.000Z",
            "cwd": "/proj",
        },
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "timestamp": "t",
            "message": {
                "role": "system",
                "content": "",
                "sections": {"preamble": "PI"},
                "timestamp": 1,
            },
        },
        {
            "type": "message",
            "id": "b",
            "parentId": "a",
            "timestamp": "t",
            "message": {"role": "user", "content": "你好", "timestamp": 2},
        },
        {
            "type": "message",
            "id": "c",
            "parentId": "b",
            "timestamp": "t",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "想"},
                    {"type": "text", "text": "回复"},
                    {
                        "type": "toolCall",
                        "id": "tc1",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    },
                ],
                "timestamp": 3,
            },
        },
        {
            "type": "message",
            "id": "d",
            "parentId": "c",
            "timestamp": "t",
            "message": {
                "role": "toolResult",
                "toolCallId": "tc1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "out"}],
                "isError": False,
                "timestamp": 4,
            },
        },
    ]
    path = tmp_path / "pi.jsonl"
    _write(path, records)

    agent = _agent()
    assert "已导入" in dispatch_command(f"/import {path}", CommandContext(agent))
    # pi 的 system 消息被丢弃，投影首条是 polya 自己的 system 提示词。
    projected = agent.tree.project()
    assert projected[0]["role"] == "system" and "PI" not in projected[0]["content"]
    history = agent.history
    assert history[0] == {"role": "user", "content": "你好"}
    assistant = history[1]
    assert assistant["content"] == "回复"
    assert assistant["reasoning_content"] == "想"
    assert assistant["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command": "ls"}',
    }
    assert history[2] == {"role": "tool", "tool_call_id": "tc1", "content": "out"}


def test_import_pi_honors_compaction(tmp_path):
    records = [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "t", "cwd": "/p"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "t",
            "message": {"role": "user", "content": "旧问题"},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "t",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "旧回答"}]},
        },
        {
            "type": "message",
            "id": "u2",
            "parentId": "a1",
            "timestamp": "t",
            "message": {"role": "user", "content": "保留"},
        },
        {
            "type": "compaction",
            "id": "cp",
            "parentId": "u2",
            "timestamp": "t",
            "summary": "摘要",
            "firstKeptEntryId": "u2",
        },
        {
            "type": "message",
            "id": "a2",
            "parentId": "cp",
            "timestamp": "t",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "之后"}]},
        },
    ]
    path = tmp_path / "pi.jsonl"
    _write(path, records)

    agent = _agent()
    dispatch_command(f"/import {path}", CommandContext(agent))
    contents = [m.get("content") for m in agent.history]
    assert any("旧问题" in c for c in contents) is False  # compaction 之前的入口被摘要替换
    assert any("摘要" in c for c in contents)
    assert any("保留" in c for c in contents)
    assert any("之后" in c for c in contents)


def test_import_reports_bad_file(tmp_path):
    path = tmp_path / "nope.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    agent = _agent()
    result = dispatch_command(f"/import {path}", CommandContext(agent))
    assert "无法导入" in result
