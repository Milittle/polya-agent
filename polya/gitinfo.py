"""读取当前 git 分支（状态栏用）。

直接解析 ``.git/HEAD``：不起子进程、不依赖 git 二进制，非仓库或读取失败一律
静默返回 ``None``。按 HEAD 的 mtime 缓存，状态栏每帧刷新也不会反复读盘。
"""

from __future__ import annotations

from pathlib import Path

_cache: dict[str, tuple[int, str | None]] = {}


def _git_dir(root: Path) -> Path | None:
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        # worktree / submodule：.git 文件内容为 "gitdir: <path>"（可为相对路径）。
        try:
            line = dot_git.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if line.startswith("gitdir:"):
            target = Path(line[len("gitdir:") :].strip())
            return target if target.is_absolute() else (root / target)
    return None


def current_branch(root: str | Path) -> str | None:
    """返回分支名（detached HEAD 返回短哈希）；非仓库或失败返回 ``None``。"""
    root = Path(root)
    git_dir = _git_dir(root)
    if git_dir is None:
        return None
    head = git_dir / "HEAD"
    try:
        stamp = head.stat().st_mtime_ns
    except OSError:
        return None
    key = str(head)
    cached = _cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        raw = head.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    branch: str | None
    if raw.startswith("ref:"):
        ref = raw[len("ref:") :].strip()
        branch = (ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref) or None
    elif raw:
        branch = raw[:7]  # detached HEAD：短哈希
    else:
        branch = None
    _cache[key] = (stamp, branch)
    return branch


def _reset_cache() -> None:
    """测试用：清空 mtime 缓存。"""
    _cache.clear()
