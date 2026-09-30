"""``@`` 联想的文件索引与模糊打分（对齐 Claude Code / Codex 的全项目搜索）。

与工具层共用 rg 依赖：``rg --files`` 天然尊重 gitignore。索引懒构建、
过期后由后台线程重建，查询路径永不阻塞输入线程。
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

REFRESH_INTERVAL = 120.0  # 秒：两次后台重建的最小间隔
RESULT_LIMIT = 20


def fuzzy_score(query: str, path: str) -> int | None:
    """子序列打分：文件名命中优于仅路径段命中，词首与连续命中加分。

    query 不是 path（大小写不敏感）的子序列时返回 None。
    """
    query = query.lower()
    lowered = path.lower()
    base_at = lowered.rfind("/") + 1
    score = 0
    index = 0
    run = 0
    for char in query:
        found = lowered.find(char, index)
        if found < 0:
            return None
        run = run + 1 if found == index else 1
        score += 3 if found >= base_at else 1
        if found == 0 or lowered[found - 1] in "/_- ":
            score += 2  # 词首（路径段 / 连字符后）命中更像用户意图
        score += run
        index = found + 1
    return score


class ProjectFiles:
    """``rg --files`` 索引：首次查询同步构建，之后过期时后台单飞重建。"""

    def __init__(self, root: Path, *, interval: float = REFRESH_INTERVAL) -> None:
        self._root = root
        self._interval = interval
        self._paths: list[str] = []
        self._listed_at = 0.0
        self._lock = threading.Lock()
        self._refreshing = threading.Event()
        self._warmed = False

    @property
    def warmed(self) -> bool:
        """是否已请求过预热（交互会话启动时）。"""
        return self._warmed

    @property
    def ready(self) -> bool:
        """索引是否已就绪（至少完成过一次构建）。"""
        with self._lock:
            return bool(self._paths)

    def warmup(self) -> None:
        """启动时后台预热：首次 ``@`` 不在输入线程里同步建索引（票 09）。"""
        self._warmed = True
        if not self.ready:
            self.refresh_soon()

    def search(self, query: str) -> list[str]:
        """按分数降序返回至多 RESULT_LIMIT 条路径；空查询按字母序。"""
        self._ensure_index()
        with self._lock:
            paths = list(self._paths)
        if not query:
            return paths[:RESULT_LIMIT]
        ranked = sorted(
            ((fuzzy_score(query, path) or -1, path) for path in paths),
            key=lambda pair: (-pair[0], pair[1]),  # Codex 同款：分数降序、路径字母序
        )
        return [path for score, path in ranked if score >= 0][:RESULT_LIMIT]

    def refresh_soon(self) -> None:
        """请求后台重建（任务边界调用）；已有重建在飞则忽略。"""
        if self._refreshing.is_set():
            return
        self._refreshing.set()
        threading.Thread(target=self._rebuild, daemon=True).start()

    def _ensure_index(self) -> None:
        with self._lock:
            fresh = self._paths and time.monotonic() - self._listed_at < self._interval
        if fresh:
            return
        if not self._paths and not self._warmed:
            self._rebuild()  # 未预热的库/测试路径：首次同步构建，不能空手而归
            return
        self.refresh_soon()  # 已预热或已过期：拿现有索引先答，后台换新，不阻塞

    def _rebuild(self) -> None:
        try:
            completed = subprocess.run(
                ["rg", "--files", "--hidden", "--glob", "!.git"],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._refreshing.clear()  # rg 缺席或超时：保留旧索引，下次再试
            return
        paths = sorted(line for line in completed.stdout.splitlines() if line)
        with self._lock:
            self._paths = paths
            self._listed_at = time.monotonic()
        self._refreshing.clear()
