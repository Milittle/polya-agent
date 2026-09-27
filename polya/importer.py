"""导入外部会话：polya 原始 JSONL 与 pi 会话格式（v1–v3）。

pi 的会话树用字符串 ``id``/``parentId``；本模块按 pi 的上下文构建规则取
「当前叶 → 根」路径，套用最新 ``compaction``（``firstKeptEntryId`` 截断）与
``context_edit``（映射为 polya 的投影覆盖），再转成**线性**的 polya SessionTree。
导入的是 pi 会发给模型的上下文，而非全部历史树——分支不保留。

pi 的 system 消息（含各 section 与工具声明）在导入时**丢弃**，改用 polya 当前的
系统提示词，保证导入后可继续对话且工具声明一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .session import SessionMeta
from .tree import (
    KIND_ASSISTANT,
    KIND_SUMMARY,
    KIND_SYSTEM,
    KIND_TOOL,
    KIND_USER,
    SessionTree,
)

_PI_ENTRY_TYPES = {
    "message",
    "compaction",
    "branch_summary",
    "custom_message",
    "context_edit",
    "model_change",
    "thinking_level_change",
    "usage",
    "custom",
    "label",
    "session_info",
}


def import_file(
    path: str | Path, system_payload: dict[str, Any] | None = None
) -> tuple[SessionMeta, SessionTree]:
    """读取并识别会话文件，返回 (元数据, 树)；格式不符抛 ``ValueError``。"""
    target = Path(path).expanduser()
    raw = target.read_text(encoding="utf-8")  # OSError 由调用方处理
    records: list[dict] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"坏行：{exc}") from exc
        if isinstance(record, dict):
            records.append(record)
    if not records:
        raise ValueError("空文件")
    if any(record.get("type") == "entry" for record in records):
        return _import_polya(records, target.stem)
    if records[0].get("type") == "session" or any(
        record.get("type") in _PI_ENTRY_TYPES for record in records
    ):
        return _import_pi(records, target.stem, system_payload)
    raise ValueError("无法识别的会话格式（既非 polya 也非 pi）")


def _import_polya(records: list[dict], fallback_name: str) -> tuple[SessionMeta, SessionTree]:
    header = records[0] if records[0].get("type") == "session" else {}
    entry_lines = [
        json.dumps(record, ensure_ascii=False)
        for record in records
        if record.get("type") != "session"
    ]
    tree = SessionTree.from_jsonl(entry_lines)
    meta = SessionMeta.from_header(header, fallback_name)
    return meta, tree


def _import_pi(
    records: list[dict], fallback_name: str, system_payload: dict[str, Any] | None
) -> tuple[SessionMeta, SessionTree]:
    header = records[0] if records[0].get("type") == "session" else {}
    path_entries = _select_path(_active_path(records))
    path_ids = {str(entry.get("id")) for entry in path_entries if entry.get("id")}
    edits = _context_edits(records, path_ids)

    tree = SessionTree()
    if system_payload is not None:
        tree.append(KIND_SYSTEM, dict(system_payload))
    id_map: dict[str, int | None] = {}
    for entry in path_entries:
        kind, payload = _convert_entry(entry)
        if kind is None:
            if entry.get("id"):
                id_map[str(entry["id"])] = tree.active_id  # 跳过节点：子节点挂到最近入口
            continue
        assert payload is not None
        node = tree.append(kind, payload)
        if entry.get("id"):
            id_map[str(entry["id"])] = node.id
    for target, replacement in edits.items():
        entry_id = id_map.get(target)
        if entry_id is None:
            continue
        if replacement is None:
            tree.override(entry_id, None)
        else:
            node = tree.get(entry_id)
            tree.override(entry_id, {**node.payload, "content": _flatten(replacement)})

    title = next(
        (str(r["name"]) for r in records if r.get("type") == "session_info" and r.get("name")),
        None,
    )
    meta = SessionMeta(
        name=fallback_name,
        title=title,
        cwd=str(header.get("cwd") or ""),
        created=str(header.get("timestamp") or ""),
        updated="",
    )
    return meta, tree


def _active_path(records: list[dict]) -> list[dict]:
    """当前叶 → 根（反转为根 → 叶）；叶取文件里最后一个带 id 的条目。"""
    by_id = {str(r["id"]): r for r in records if r.get("id")}
    leaf: str | None = None
    for record in records:
        if record.get("id"):
            leaf = str(record["id"])
    path: list[dict] = []
    while leaf is not None:
        entry = by_id.get(leaf)
        if entry is None:
            break
        path.append(entry)
        parent = entry.get("parentId")
        leaf = str(parent) if parent is not None else None
    path.reverse()
    return path


def _is_system(entry: dict) -> bool:
    return entry.get("type") == "message" and (entry.get("message") or {}).get("role") == "system"


def _select_path(path: list[dict]) -> list[dict]:
    """套用最新 compaction：摘要入口 + 保留区非 system + 压缩后入口（pi buildContextEntries）。"""
    compactions = [entry for entry in path if entry.get("type") == "compaction"]
    if not compactions:
        return path
    comp = compactions[-1]
    index = path.index(comp)
    first_kept = comp.get("firstKeptEntryId")
    kept: list[dict] = []
    if first_kept:
        ids = [str(entry.get("id")) for entry in path[:index]]
        if str(first_kept) in ids:
            kept = path[ids.index(str(first_kept)) : index]
    kept = [entry for entry in kept if not _is_system(entry)]
    return [comp, *kept, *path[index + 1 :]]


def _context_edits(records: list[dict], path_ids: set[str]) -> dict[str, Any]:
    """选中路径上的 context_edit：目标 id → replacement（文件序后者胜）。"""
    edits: dict[str, Any] = {}
    for record in records:
        if record.get("type") == "context_edit" and str(record.get("targetId")) in path_ids:
            edits[str(record["targetId"])] = record.get("replacement")
    return edits


def _convert_entry(entry: dict) -> tuple[str | None, dict[str, Any] | None]:
    etype = entry.get("type")
    if etype == "message":
        return _convert_message(entry.get("message") or {})
    if etype in ("compaction", "branch_summary"):
        return KIND_SUMMARY, {"content": entry.get("summary", ""), "first_kept_entry_id": 0}
    if etype == "custom_message":
        return KIND_USER, {"content": _flatten(entry.get("content"))}
    return None, None


def _convert_message(message: dict) -> tuple[str | None, dict[str, Any] | None]:
    role = message.get("role")
    if role == "system":
        # pi 的 system 提示词不入 polya 上下文：导入恒用 polya 当前系统提示词。
        return None, None
    if role == "user":
        return KIND_USER, {"content": _flatten(message.get("content"))}
    if role == "assistant":
        return KIND_ASSISTANT, _assistant_payload(message)
    if role == "toolResult":
        return KIND_TOOL, {
            "tool_call_id": message.get("toolCallId", ""),
            "content": _flatten(message.get("content")),
        }
    if role == "custom":
        return KIND_USER, {"content": _flatten(message.get("content"))}
    if role == "bashExecution":
        if message.get("excludeFromContext"):
            return None, None
        command = message.get("command", "")
        output = message.get("output", "")
        code = message.get("exitCode")
        return KIND_USER, {
            "content": f"$ {command}\n{output}".rstrip() + (f"\n[exit {code}]" if code else "")
        }
    if role == "branchSummary":
        return KIND_SUMMARY, {"content": message.get("summary", ""), "first_kept_entry_id": 0}
    if role == "compactionSummary":
        return KIND_SUMMARY, {"content": message.get("summary", ""), "first_kept_entry_id": 0}
    return None, None


def _assistant_payload(message: dict) -> dict[str, Any]:
    """pi 的 content blocks（text/thinking/toolCall）→ polya assistant payload。"""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "thinking":
                reasoning_parts.append(str(block.get("thinking", "")))
            elif btype == "toolCall":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("arguments") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
    elif content is not None:
        text_parts.append(str(content))
    payload: dict[str, Any] = {"content": "".join(text_parts) or None}
    if reasoning_parts:
        payload["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _flatten(content: Any) -> str:
    """content 字符串或内容块数组 → 纯文本（图片降级为占位符）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image":
                    parts.append("[图片]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return str(content)
