"""filefind.py 测试：模糊打分排序、rg 索引、后台刷新。"""

from __future__ import annotations

import threading
import time
from shutil import which

import pytest

from polya.filefind import RESULT_LIMIT, ProjectFiles, fuzzy_score

requires_rg = pytest.mark.skipif(which("rg") is None, reason="rg 不在 PATH（工具层同依赖）")


# ---------- 打分器（纯函数，无 rg 依赖） ----------


def test_fuzzy_score_requires_subsequence():
    assert fuzzy_score("app", "src/app.py") is not None
    assert fuzzy_score("zz", "src/app.py") is None


def test_fuzzy_score_prefers_basename_hits_over_dir_hits():
    # 文件名里命中 app 的应高于只在目录名里命中 app 的（CC 排序语义）
    assert fuzzy_score("app", "src/app.py") > fuzzy_score("app", "app/docs/other.txt")


def test_fuzzy_score_rewards_contiguous_runs():
    assert fuzzy_score("ab", "ab.txt") > fuzzy_score("ab", "a_dir/zzb.txt")


def test_fuzzy_score_is_case_insensitive():
    assert fuzzy_score("APP", "src/app.py") == fuzzy_score("app", "src/app.py")


# ---------- 索引与搜索（需要 rg） ----------


def _tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x", encoding="utf-8")
    (tmp_path / "src" / "api.py").write_text("x", encoding="utf-8")
    (tmp_path / "zzz").mkdir()
    for i in range(RESULT_LIMIT + 5):
        (tmp_path / "zzz" / f"noise-{i:02d}.txt").write_text("x", encoding="utf-8")
    return ProjectFiles(tmp_path)


@requires_rg
def test_search_ranks_by_score_then_path(tmp_path):
    for d in ("alpha", "beta"):  # 同分对：须在首次 search（即索引构建）前就位
        (tmp_path / d).mkdir()
        (tmp_path / d / "dup.txt").write_text("x", encoding="utf-8")
    files = _tree(tmp_path)
    assert files.search("app")[:2] == ["src/app.py", "src/api.py"]  # 三连命中分更高
    assert files.search("ap")[:2] == ["src/api.py", "src/app.py"]  # 同分按路径字母序
    assert files.search("dup")[:2] == ["alpha/dup.txt", "beta/dup.txt"]  # 同分字母序


@requires_rg
def test_search_caps_results(tmp_path):
    assert len(_tree(tmp_path).search("noise")) == RESULT_LIMIT


@requires_rg
def test_refresh_soon_reindexes_background_writes(tmp_path):
    files = ProjectFiles(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert files.search("a.txt") == ["a.txt"]  # 首次同步构建
    (tmp_path / "new_b.txt").write_text("x", encoding="utf-8")
    files.refresh_soon()  # 任务边界：后台单飞重建
    for _ in range(200):  # 等后台线程换新（不阻塞断言超时）
        if "new_b.txt" in files.search("b.txt"):
            break
        time.sleep(0.02)
    assert files.search("b.txt") == ["new_b.txt"]
    # 单飞标志必须复位，否则后续刷新永久静默（票 09 顺带修）
    (tmp_path / "later_c.txt").write_text("x", encoding="utf-8")
    files.refresh_soon()
    for _ in range(200):
        if "later_c.txt" in files.search("c.txt"):
            break
        time.sleep(0.02)
    assert files.search("c.txt") == ["later_c.txt"]


def test_warmup_keeps_first_search_non_blocking(tmp_path, monkeypatch):
    """预热后首次 search 不阻塞输入线程：未就绪先答空，后台填好后可命中（票 09）。"""
    files = ProjectFiles(tmp_path)
    release = threading.Event()

    def slow_rebuild(self):
        release.wait(2)
        with self._lock:
            self._paths = ["a.txt"]
            self._listed_at = time.monotonic()
        self._refreshing.clear()

    monkeypatch.setattr(ProjectFiles, "_rebuild", slow_rebuild)
    files.warmup()
    assert files.warmed and not files.ready
    started = time.monotonic()
    assert files.search("a") == []  # 不阻塞、先答空
    assert time.monotonic() - started < 0.5
    release.set()
    for _ in range(200):
        if files.ready:
            break
        time.sleep(0.01)
    assert files.ready and files.search("a") == ["a.txt"]
