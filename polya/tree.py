"""会话树：不可变入口 + 确定性投影（ADR 0005）。

原始历史是一棵不可变树：每条记录是带稳定 id 的 :class:`Entry`，按 ``parent_id`` 连成
树，当前指针所在的「根 → 叶」路径即当前分支。发给模型的 messages 由 :meth:`SessionTree.project`
从当前分支确定性派生——原始入口永不原地改写；微压缩等「只改投影不改原文」的操作经
:meth:`SessionTree.override` 挂在入口上，原文仍可经 :meth:`SessionTree.read` 回查。

前缀友好由「入口不可变 + 投影确定性」保证：常见路径（只追加入口）下，投影是上一次的
严格扩展；压缩 / 换模型 / 分支切换等是合法重启点，前缀基线清零后重新起算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Entry.kind 取值：与 provider role 一一对应，另有 summary 表达压缩摘要。
KIND_SYSTEM = "system"
KIND_USER = "user"
KIND_ASSISTANT = "assistant"
KIND_TOOL = "tool"
KIND_SUMMARY = "summary"

READ_PAGE = 8000  # history_read 每页返回的字符数


@dataclass(frozen=True)
class Entry:
    """会话树中一条不可变、带稳定 id 的记录。

    ``frozen`` 冻结属性绑定；``payload`` 内部按约定视作不可变（append-only 树从不
    原地改它）。payload 字段直接是投影需要的 provider 形态：

    - system: ``{"content": str}``（后续 system 入口可 patch 具名 section，见 ticket 03）
    - user: ``{"content": str}``
    - assistant: ``{"content": str | None, "reasoning_content"?: str, "tool_calls"?: list}``
    - tool: ``{"tool_call_id": str, "content": str}``
    - summary: ``{"content": str, "first_kept_entry_id": int}``
    """

    id: int
    parent_id: int | None
    kind: str
    payload: dict[str, Any]


class SessionTree:
    """按 id 索引入口的树；``append`` 在当前叶下挂新入口，``move_to`` 回溯（分叉）。"""

    def __init__(self) -> None:
        self._entries: dict[int, Entry] = {}
        self._active_id: int | None = None
        self._next_id = 1
        # 投影覆盖：entry_id -> 替换 payload。微压缩等「只改投影」的操作挂这里，
        # 原文仍在 entry.payload 里，经 read() 回查。
        self._overrides: dict[int, dict[str, Any] | None] = {}

    @property
    def active_id(self) -> int | None:
        """当前叶入口 id；空树为 None。"""
        return self._active_id

    def append(self, kind: str, payload: dict[str, Any]) -> Entry:
        """在当前叶下追加一个入口并成为新叶，返回它。"""
        entry = Entry(id=self._next_id, parent_id=self._active_id, kind=kind, payload=payload)
        self._entries[entry.id] = entry
        self._next_id += 1
        self._active_id = entry.id
        return entry

    def move_to(self, entry_id: int) -> Entry:
        """回溯：把当前指针移到历史入口（后续 append 即从此分叉）。"""
        entry = self._entries.get(entry_id)
        if entry is None:
            raise ValueError(f"未知入口 id：{entry_id}")
        self._active_id = entry_id
        return entry

    def rewind(self, steps: int = 1) -> Entry:
        """沿当前分支回退 steps 个入口（合法重启点），返回新的当前入口。"""
        branch = self.active_branch()
        if steps < 1:
            raise ValueError("回退步数至少为 1")
        if steps >= len(branch):
            raise ValueError(f"无法回退 {steps} 步：当前分支仅 {len(branch)} 个入口")
        return self.move_to(branch[-steps - 1].id)

    def leaves(self) -> list[Entry]:
        """所有叶入口（无子节点者）。"""
        parents = {e.parent_id for e in self._entries.values() if e.parent_id is not None}
        return [e for e in self._entries.values() if e.id not in parents]

    def branches(self) -> list[list[Entry]]:
        """所有分支（每条 = 根 → 某叶，按叶 id 升序）。"""
        result: list[list[Entry]] = []
        for leaf in sorted(self.leaves(), key=lambda e: e.id):
            branch: list[Entry] = []
            current: Entry | None = leaf
            while current is not None:
                branch.append(current)
                parent = current.parent_id
                current = self._entries.get(parent) if parent is not None else None
            branch.reverse()
            result.append(branch)
        return result

    def fork_points(self) -> list[Entry]:
        """分叉点（有多个子节点的入口）。"""
        child_counts: dict[int, int] = {}
        for entry in self._entries.values():
            if entry.parent_id is not None:
                child_counts[entry.parent_id] = child_counts.get(entry.parent_id, 0) + 1
        return [e for e in self._entries.values() if child_counts.get(e.id, 0) > 1]

    def get(self, entry_id: int) -> Entry:
        entry = self._entries.get(entry_id)
        if entry is None:
            raise ValueError(f"未知入口 id：{entry_id}")
        return entry

    def override(self, entry_id: int, payload: dict[str, Any] | None) -> None:
        """为入口挂投影覆盖（只改投影，不改原文）；``None`` 表示从投影移除。"""
        if entry_id not in self._entries:
            raise ValueError(f"未知入口 id：{entry_id}")
        self._overrides[entry_id] = payload

    def effective_payload(self, entry_id: int) -> dict[str, Any] | None:
        """投影生效的 payload（有覆盖则用覆盖，否则原文）；被移除时为 None。"""
        entry = self.get(entry_id)
        return self._overrides.get(entry_id, entry.payload)

    def read(self, entry_id: int, offset: int = 0) -> str:
        """按 id 回查入口**原文**（投影覆盖不影响），分页返回。"""
        entry = self.get(entry_id)
        if offset < 0:
            raise ValueError("offset 不能为负")
        content = json.dumps(entry.payload, ensure_ascii=False)
        next_offset = min(offset + READ_PAGE, len(content))
        header = f"[入口 {entry_id}; {len(content)} 字符; 下一偏移 {next_offset}]\n"
        return header + content[offset : offset + READ_PAGE]

    def active_branch(self) -> list[Entry]:
        """根 → 当前叶 的有序入口序列。"""
        branch: list[Entry] = []
        current = self._active_id
        while current is not None:
            entry = self._entries[current]
            branch.append(entry)
            current = entry.parent_id
        branch.reverse()
        return branch

    def ancestry(self, entry_id: int) -> list[Entry]:
        """根 → 指定入口 的祖先路径（该入口必须存在）。"""
        chain: list[Entry] = []
        current: Entry | None = self.get(entry_id)
        while current is not None:
            chain.append(current)
            parent = current.parent_id
            current = self._entries.get(parent) if parent is not None else None
        chain.reverse()
        return chain

    def copy_branch_upto(self, entry_id: int) -> SessionTree:
        """从根到 entry_id 的祖先路径复制成新树（新连续 id），active = 末入口。"""
        clone = SessionTree()
        for entry in self.ancestry(entry_id):
            clone.append(entry.kind, dict(entry.payload))
        return clone

    def copy(self) -> SessionTree:
        """整棵树复制（含投影覆盖与 active 指针），保 id。"""
        return SessionTree.from_jsonl(self.to_jsonl())

    def project(self) -> list[dict]:
        """当前分支 → provider messages（应用投影覆盖），纯函数、同状态必同输出。"""
        return _project(self.active_branch(), self._overrides)

    def conversation(self) -> list[dict]:
        """当前分支 → 会话消息（去掉 system 入口），供 ``Agent.history`` 兼容视图。"""
        return [m for m in self.project() if m["role"] != "system"]

    def clear(self) -> None:
        """清空整棵树（system 入口也移除，含投影覆盖）。"""
        self._entries.clear()
        self._overrides.clear()
        self._active_id = None
        self._next_id = 1

    def reset_with_system(
        self, system_content: str, sections: dict[str, str] | None = None
    ) -> Entry:
        """清空后重新挂上 system 入口，返回它。

        带 ``sections`` 时支持后续 system 入口按名 patch（见 :func:`_project` 的重放）；
        不带则退回单一 content（测试与简单用法）。
        """
        self.clear()
        payload: dict[str, Any] = {"content": system_content}
        if sections is not None:
            payload["sections"] = sections
        return self.append(KIND_SYSTEM, payload)

    def replace_conversation(self, messages: list[dict]) -> None:
        """从 system 入口重建会话（旧入口保留在树里、可经 id 回查，不再参与投影）。

        ticket 02 的过渡形态：压缩仍以「重建投影」表达，但原文不再丢失。ticket 05
        引入 context_edit 后，压缩改为追加 summary 入口 + cut point，此处收敛。
        """
        system = [e for e in self._entries.values() if e.kind == KIND_SYSTEM]
        self._active_id = system[-1].id if system else None
        for message in messages:
            self.append(_kind_for_role(message["role"]), _payload_for(message))

    def strip_reasoning(self) -> None:
        """从 assistant 入口剥离 reasoning_content（换模型时回传会撞陌生端点）。

        换模型是合法重启点：重建入口（保 id）而非原地改 payload，旧推理随前缀一起作废。
        """
        rebuilt: dict[int, Entry] = {}
        for entry_id, entry in self._entries.items():
            if entry.kind == KIND_ASSISTANT and "reasoning_content" in entry.payload:
                payload = {k: v for k, v in entry.payload.items() if k != "reasoning_content"}
                rebuilt[entry_id] = Entry(entry.id, entry.parent_id, entry.kind, payload)
            else:
                rebuilt[entry_id] = entry
        self._entries = rebuilt

    def __len__(self) -> int:
        return len(self._entries)

    def to_jsonl(self) -> list[str]:
        """序列化为 JSONL（不含 session 头）：入口按 id 升序，覆盖随后，尾行 active。"""
        lines: list[str] = []
        for entry in sorted(self._entries.values(), key=lambda e: e.id):
            lines.append(
                json.dumps(
                    {
                        "type": "entry",
                        "id": entry.id,
                        "parentId": entry.parent_id,
                        "kind": entry.kind,
                        "payload": entry.payload,
                    },
                    ensure_ascii=False,
                )
            )
        for entry_id, payload in self._overrides.items():
            lines.append(
                json.dumps(
                    {"type": "override", "id": entry_id, "payload": payload},
                    ensure_ascii=False,
                )
            )
        lines.append(json.dumps({"type": "active", "id": self._active_id}, ensure_ascii=False))
        return lines

    @classmethod
    def from_jsonl(cls, lines: list[str]) -> SessionTree:
        """从 :meth:`to_jsonl` 的产物重建树；坏行即报，不静默吞。"""
        tree = cls()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"会话文件坏行：{exc}") from exc
            kind = record.get("type")
            if kind == "entry":
                try:
                    entry = Entry(
                        id=record["id"],
                        parent_id=record.get("parentId"),
                        kind=record["kind"],
                        payload=record["payload"],
                    )
                except KeyError as exc:
                    raise ValueError(f"会话入口缺字段：{exc}") from exc
                tree._entries[entry.id] = entry
                tree._next_id = max(tree._next_id, entry.id + 1)
            elif kind == "override":
                tree._overrides[record["id"]] = record["payload"]
            elif kind == "active":
                tree._active_id = record.get("id")
            # 未知 type：跳过（前向兼容）
        return tree


def _kind_for_role(role: str) -> str:
    return {
        "system": KIND_SYSTEM,
        "user": KIND_USER,
        "assistant": KIND_ASSISTANT,
        "tool": KIND_TOOL,
    }[role]


def _payload_for(message: dict[str, Any]) -> dict[str, Any]:
    payload = dict(message)
    payload.pop("role", None)
    return payload


def _assistant_message(payload: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": payload.get("content")}
    reasoning = payload.get("reasoning_content")
    if reasoning:
        message["reasoning_content"] = reasoning
    tool_calls = payload.get("tool_calls")
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


def _render_sections(sections: dict[str, str]) -> str:
    """渲染合并后的具名 section（preamble 在前，其余按插入序，与 SystemPrompt.render 同规）。"""
    return "\n\n".join(v for v in sections.values() if v)


def _project(branch: list[Entry], overrides: dict[int, dict[str, Any] | None]) -> list[dict]:
    """分支 + 投影覆盖 → provider messages，纯函数、同输入必同输出。

    system 入口按序重放：首个带 ``sections`` 的为基础，后续带 ``sections`` 的为 patch
    （``None`` 值删除该段）；全程无 sections 则退回单一 content。最终只在开头发一条
    system 消息，其余入口按序投影。
    """
    base_sections: dict[str, str] | None = None
    content_fallback: str | None = None
    has_sections = False
    for entry in branch:
        if entry.kind != KIND_SYSTEM:
            continue
        payload = overrides.get(entry.id, entry.payload)
        if payload is None:
            continue
        sections = payload.get("sections")
        if sections is not None:
            has_sections = True
            if base_sections is None:
                base_sections = dict(sections)
            else:
                for name, value in sections.items():
                    if value is None:
                        base_sections.pop(name, None)
                    else:
                        base_sections[name] = value
        elif payload.get("content") is not None:
            content_fallback = payload["content"]

    messages: list[dict] = []
    if has_sections:
        messages.append({"role": "system", "content": _render_sections(base_sections or {})})
    elif content_fallback is not None:
        messages.append({"role": "system", "content": content_fallback})

    for entry in branch:
        if entry.kind == KIND_SYSTEM:
            continue
        payload = overrides.get(entry.id, entry.payload)
        if payload is None:
            continue  # 被 context_edit 删除
        if entry.kind == KIND_USER:
            messages.append({"role": "user", "content": payload["content"]})
        elif entry.kind == KIND_ASSISTANT:
            messages.append(_assistant_message(payload))
        elif entry.kind == KIND_TOOL:
            messages.append(
                {
                    "role": "tool",
                    # 合成/测试数据可能缺 tool_call_id；真实入口总带，宽容只影响测试。
                    "tool_call_id": payload.get("tool_call_id", ""),
                    "content": payload["content"],
                }
            )
        elif entry.kind == KIND_SUMMARY:
            messages.append({"role": "user", "content": payload["content"]})
        # 未知 kind：跳过（不静默伪造 role）
    return messages


def project(branch: list[Entry]) -> list[dict]:
    """分支 → provider messages（无覆盖，供纯函数测试与旧调用方）。"""
    return _project(branch, {})


def conversation(branch: list[Entry]) -> list[dict]:
    """分支 → 会话消息（去掉 system 入口），供纯函数测试。"""
    return [m for m in project(branch) if m["role"] != "system"]
