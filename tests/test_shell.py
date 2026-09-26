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


def test_wait_timeout_keeps_command_and_environment(tmp_path):
    session = ShellSession(str(tmp_path))
    session.run("export MI_Z_TEST=42")

    result = session.run("sleep 0.15; echo finished", timeout=0.01)

    assert "仍在运行" in result
    assert session.alive
    assert session.run("echo must-not-run").startswith("Error:")
    result = session.output(timeout=2)
    assert "finished" in result and "退出码 0" in result
    assert "退出码 0" in session.run("echo ok")
    assert "42" in session.run("echo $MI_Z_TEST")
    session.kill()


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


def test_no_newline_output_and_real_failure_code(tmp_path):
    session = ShellSession(str(tmp_path))
    result = session.run("printf partial; false")
    assert "partial" in result and "退出码 1" in result
    assert "polya_" not in result
    session.kill()


def test_large_output_keeps_failure_tail_and_full_log(tmp_path):
    session = ShellSession(str(tmp_path))
    result = session.run("for i in {1..1800}; do echo line-$i; done; echo TEST_FAILED; false")
    assert "line-1\n" in result
    assert "TEST_FAILED" in result and result.endswith("退出码 1")
    assert len(result) < 8300
    middle = session.output(command_id=1, start_line=900, end_line=902)
    assert "line-900" in middle and "line-902" in middle
    assert "line-899" not in middle
    session.kill()


def test_wait_deadline_is_total_even_with_output(tmp_path):
    import time

    session = ShellSession(str(tmp_path))
    start = time.monotonic()
    result = session.run("for i in {1..30}; do echo tick; sleep 0.05; done", timeout=0.1)
    assert time.monotonic() - start < 1
    assert "仍在运行" in result
    session.kill()
    assert not session.alive
    assert "退出码 0" in session.run("echo restart")
    session.kill()


def test_explicit_kill_stops_children(tmp_path):
    import time

    session = ShellSession(str(tmp_path))
    session.run("sleep 0.3; echo bad > leaked", timeout=0.01)
    session.kill()
    time.sleep(0.35)
    assert not (tmp_path / "leaked").exists()


def test_long_line_log_has_character_pagination(tmp_path):
    session = ShellSession(str(tmp_path))
    session.run("printf '%12000sEND\\n' x")
    first = session.output(command_id=1, start_line=1, end_line=1)
    second = session.output(command_id=1, start_line=1, end_line=1, offset=8000)
    assert "offset=8000" in first
    assert "END" not in first and "END" in second
    session.kill()
