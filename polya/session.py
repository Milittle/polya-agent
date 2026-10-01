"""会话身份与持久化：元数据 + JSONL 文件 IO（session-lifecycle 票 01）。

会话 = 稳定名字 + 元数据（标题 / 创建 / 更新 / cwd）+ 会话树。树体沿用
``tree.to_jsonl``；文件头在旧格式（``version`` / ``cwd``）之上新增
``name`` / ``title`` / ``created`` / ``updated``——缺字段的旧文件按文件名补 name、
标题置空，向后兼容。

本模块只操作 JSONL 行，不依赖 :mod:`polya.tree`，避免循环导入。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

SESSION_VERSION = 1


def sessions_dir() -> Path:
    return Path.home() / ".polya" / "sessions"


def exports_dir() -> Path:
    return Path.home() / ".polya" / "exports"


def timestamp(now: float | None = None) -> str:
    """自动会话名的默认形态（本地时间，秒级）。"""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now))


_allocated: set[str] = set()


def unique_name(base: str | None = None) -> str:
    """自动会话名：同一秒内多次 new/fork/clone 也互不撞（内存 + 磁盘去重）。"""
    candidate = (base or timestamp()).strip() or timestamp()
    root = candidate
    index = 2
    while candidate in _allocated or session_path(candidate).exists():
        candidate = f"{root}-{index}"
        index += 1
    _allocated.add(candidate)
    return candidate


def iso_now(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))


def valid_name(name: str) -> bool:
    """会话名：非空、无空白、无路径分隔符（与旧 ``save_session`` 一致）。"""
    return (
        bool(name) and not any(c.isspace() for c in name) and "/" not in name and "\\" not in name
    )


def slug(title: str) -> str:
    """把主题转成合法会话名（slug）：空白与路径/保留字符 → ``-``，限长 48。

    中文主题原样保留（文件系统支持 Unicode）；只确保无空白与分隔符，
    避免主题与文件名“对不上”。
    """
    cleaned = "".join(
        "-" if (char.isspace() or char in '/\\:*?"<>|') else char for char in title.strip()
    )
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:48] or timestamp()


def delete(name: str) -> None:
    """删除会话文件（自动命名重命名时清理旧的临时名文件）。"""
    session_path(name).unlink(missing_ok=True)


@dataclass
class SessionMeta:
    name: str
    title: str | None = None
    cwd: str = ""
    created: str = ""
    updated: str = ""

    def header(self) -> dict:
        return {
            "type": "session",
            "version": SESSION_VERSION,
            "name": self.name,
            "title": self.title or "",
            "cwd": self.cwd or "",
            "created": self.created or iso_now(),
            "updated": self.updated or self.created or iso_now(),
        }

    @classmethod
    def from_header(cls, record: dict, fallback_name: str) -> SessionMeta:
        name = str(record.get("name") or fallback_name)
        title = record.get("title") or None
        return cls(
            name=name,
            title=str(title) if title else None,
            cwd=str(record.get("cwd") or ""),
            created=str(record.get("created") or ""),
            updated=str(record.get("updated") or ""),
        )

    def label(self) -> str:
        """选项器 / 列表用的一行摘要：主题优先，时间只保留 MM-DD HH:MM。

        时间戳会话名是内部稳定 id（文件名 / ``/resume`` 取值），不进展示——
        否则会出现「主题 · 日期 · 日期」这类与用户无关的杂讯。无主题时
        才回落显示名字。
        """
        when = f"{self.updated[5:10]} {self.updated[11:16]}" if len(self.updated) >= 16 else ""
        primary = self.title or self.name
        return f"{primary} · {when}" if when else primary


def session_path(name: str) -> Path:
    return sessions_dir() / f"{name}.jsonl"


def read(name: str) -> tuple[SessionMeta, list[str]] | None:
    """读一个会话文件 → (元数据, 树入口行)。不存在返回 ``None``。"""
    path = session_path(name)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    header: dict = {}
    entries: list[str] = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for index, stripped in enumerate(lines):
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            # 只容忍尾部半行（写入中途崩溃的典型残留）；中段坏行说明文件真损坏
            if index == len(lines) - 1:
                break
            return None
        if isinstance(record, dict) and record.get("type") == "session":
            header = record
        else:
            entries.append(stripped)
    meta = SessionMeta.from_header(header, name)
    return meta, entries


def write(meta: SessionMeta, entry_lines: list[str]) -> Path:
    """写会话文件（头部 + 树行），更新 ``updated`` 并收紧权限；返回路径。"""
    directory = sessions_dir()
    directory.mkdir(parents=True, exist_ok=True)
    meta.updated = iso_now()
    if not meta.created:
        meta.created = meta.updated
    path = directory / f"{meta.name}.jsonl"
    body = [json.dumps(meta.header(), ensure_ascii=False), *entry_lines]
    # 原子落盘：先写临时文件再 replace，写到一半被杀也不会留下半截会话文件
    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(body) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:  # Windows 等不支持时不强求
        pass
    return path


def list_metas() -> list[SessionMeta]:
    """所有已存会话的元数据，按 updated 倒序（最近的在前）。"""
    directory = sessions_dir()
    if not directory.exists():
        return []
    metas: list[SessionMeta] = []
    for path in directory.glob("*.jsonl"):
        result = read(path.stem)
        if result is not None:
            metas.append(result[0])
    return sorted(metas, key=lambda m: (m.updated, m.name), reverse=True)
