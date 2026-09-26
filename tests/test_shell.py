"""持久 shell 会话的测试：状态保持、超时重启、输出读取。"""

from __future__ import annotations

from polya.shell import ShellSession


def test_run_returns_exit_code_and_output(tmp_path):
    session = ShellSession(str(tmp_path))
    result = session.run("echo hi")
    assert "退出码 0" in result
    assert "hi" in result


def test_state_persists_across_calls(tmp_path):
    session = ShellSession(str(tmp_path))
    (tmp_path / "sub").mkdir()

    session.run("cd sub")
    assert "sub" in session.run("pwd")

    session.run("export MI_Z_TEST=42")
    assert "42" in session.run("echo $MI_Z_TEST")


def test_nonzero_exit_code_is_reported(tmp_path):
    session = ShellSession(str(tmp_path))
    assert "退出码 1" in session.run("false")


def test_timeout_kills_session_then_it_restarts(tmp_path):
    session = ShellSession(str(tmp_path))
    session.run("export MI_Z_TEST=42")

    result = session.run("sleep 5", timeout=1.0)

    assert "超时" in result
    assert not session.alive
    # 自动重启后能继续用，但环境状态已丢失
    assert "退出码 0" in session.run("echo ok")
    assert "42" not in session.run("echo $MI_Z_TEST")


def test_output_returns_pending_lines_without_waiting(tmp_path):
    session = ShellSession(str(tmp_path))
    session.run("echo before")
    assert "before" in session.output() or session.output() == "(暂无新输出)"
    assert session.output() == "(暂无新输出)"  # 取空后再取没有新输出


def test_kill_then_restart(tmp_path):
    session = ShellSession(str(tmp_path))
    assert "退出码 0" in session.run("echo a")
    session.kill()
    assert not session.alive
    assert "退出码 0" in session.run("echo b")  # 自动重启


def test_run_streams_lines_via_callback(tmp_path):
    """on_line 在命令运行期间逐行回调（UI 实时输出的接缝），哨兵行不外漏。"""
    session = ShellSession(str(tmp_path))
    seen: list[str] = []
    result = session.run("echo one; echo two", on_line=seen.append)
    assert seen == ["one", "two"]
    assert "退出码 0" in result and "one" in result
    assert all("__polya_done_" not in line for line in seen)
