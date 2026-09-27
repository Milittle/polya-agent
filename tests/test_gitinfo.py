"""gitinfo：直接解析 ``.git/HEAD`` 的分支读取，不起子进程。"""

from __future__ import annotations

from polya.gitinfo import current_branch


def test_reads_branch_from_head(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/feature/x\n")
    assert current_branch(tmp_path) == "feature/x"


def test_detached_head_returns_short_hash(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("0123456789abcdef\n")
    assert current_branch(tmp_path) == "0123456"


def test_non_repo_returns_none(tmp_path):
    assert current_branch(tmp_path) is None


def test_broken_head_returns_none(tmp_path):
    (tmp_path / ".git").mkdir()
    assert current_branch(tmp_path) is None


def test_worktree_gitdir_file(tmp_path):
    real = tmp_path / "real"
    (real / ".git").mkdir(parents=True)
    (real / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {real / '.git'}\n")
    assert current_branch(worktree) == "main"
