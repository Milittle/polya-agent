"""串行持久 bash：等待期限只控制返回，命令继续运行；完整输出保存在会话临时目录。"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def preview(text: str, limit: int = 8000) -> str:
    """保留头尾，尤其是测试结论与退出码；完整内容由日志入口回查。"""
    if len(text) <= limit:
        return text
    half = (limit - 100) // 2
    note = "\n... [middle omitted; page the full log with bash_output] ...\n"
    return text[:half] + note + text[-half:]


@dataclass
class Command:
    id: int
    marker: str
    log: Path
    code: str | None = None


class ShellSession:
    """同一会话只允许一个前台命令；run/output 由工具驱动串行调用。"""

    def __init__(self, cwd: str):
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._commands: dict[int, Command] = {}
        self._active: Command | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _ensure_started(self) -> None:
        if self.alive:
            return
        self._proc = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.cwd,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=os.name == "posix",
        )
        self._queue = queue.Queue()
        # 捕获本次进程和队列，旧 reader 不得向重启后的新队列写 EOF。
        self._reader = threading.Thread(
            target=self._drain, args=(self._proc, self._queue), daemon=True
        )
        self._reader.start()

    @staticmethod
    def _drain(proc, sink) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sink.put(line)
        finally:
            sink.put(None)
            proc.stdout.close()

    def run(
        self, command: str, timeout: float = 10.0, on_line: Callable[[str], None] | None = None
    ) -> str:
        if not 0 <= timeout <= 300:
            raise ValueError("timeout must be 0-300 seconds (it only caps this wait)")
        if self._active is not None and self._active.code is None:
            return (
                f"Error: command {self._active.id} is still running; "
                "wait via bash_output or kill_bash."
            )
        self._ensure_started()
        if self._directory is None:
            self._directory = tempfile.TemporaryDirectory(prefix="polya-shell-")
        identifier = len(self._commands) + 1
        record = Command(
            identifier,
            f"\x1epolya_{uuid.uuid4().hex}",
            Path(self._directory.name) / f"{identifier}.log",
        )
        record.log.touch(mode=0o600)
        self._commands[identifier] = self._active = record
        assert self._proc is not None and self._proc.stdin is not None
        # 随机哨兵可跟在无换行输出之后；不强行给真实输出加空行。
        self._proc.stdin.write(f"{command}\nprintf '{record.marker} %s\\n' \"$?\"\n")
        self._proc.stdin.flush()
        return self._collect(record, timeout, on_line)

    def _collect(
        self, record: Command, timeout: float, on_line: Callable[[str], None] | None = None
    ) -> str:
        deadline = time.monotonic() + max(timeout, 0.02)
        output = ""
        with record.log.open("a", encoding="utf-8") as log:
            while record.code is None:
                try:
                    wait = max(0, deadline - time.monotonic()) if timeout else 0
                    line = self._queue.get(timeout=wait)
                except queue.Empty:
                    break
                if line is None:
                    assert self._proc is not None
                    record.code = str(self._proc.wait())
                    break
                before, marker, after = line.partition(record.marker)
                if before:
                    log.write(before)
                    output = preview(output + before)
                    if on_line is not None:
                        on_line(before.rstrip("\n"))
                if marker:
                    record.code = after.strip()
                    break
                if time.monotonic() >= deadline:
                    break
        status = (
            f"Exit code {record.code}"
            if record.code is not None
            else "still running; wait with bash_output or kill_bash"
        )
        return f"[Command {record.id}]\n{output.strip() or '(no new output)'}\n{status}"

    def output(
        self,
        timeout: float = 0,
        command_id: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        offset: int = 0,
    ) -> str:
        if not 0 <= timeout <= 300:
            raise ValueError("timeout must be 0-300 seconds")
        record = self._commands.get(command_id) if command_id is not None else self._active
        if record is None:
            if command_id is not None:
                raise ValueError("Unknown command id")
            return "(no new output)"
        update = ""
        if record.code is None:
            update = self._collect(record, timeout)
        if start_line is not None:
            if start_line < 1 or offset < 0 or (end_line is not None and end_line < start_line):
                raise ValueError("Invalid line range")
            end = min(end_line or start_line + 199, start_line + 199)
            lines = []
            with record.log.open(encoding="utf-8") as log:
                for i, line in enumerate(log, 1):
                    if i > end:
                        break
                    if i >= start_line:
                        lines.append(f"{i:>6}\t{line.rstrip()}")
            status = f"Exit code {record.code}" if record.code is not None else "still running"
            text = "\n".join(lines)
            more = (
                f"\n[line range unfinished; keep the range and continue with "
                f"offset={offset + 8000}]"
                if len(text) > offset + 8000
                else ""
            )
            return (
                f"[Command {record.id}; lines {start_line}-{end}]\n"
                f"{text[offset : offset + 8000]}{more}\n{status}"
            )
        if update:
            return update
        if command_id is not None:
            return f"[Command {record.id}]\npage output with start_line=1\nExit code {record.code}"
        # 已完成前台命令后仍支持显式后台进程的增量输出。
        output = ""
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                break
            if pending is None:
                break
            output = preview(output + pending)
        return output.strip() or "(no new output)"

    def kill(self) -> None:
        """POSIX 下终止整个会话进程组（含子进程）；随后可重启。"""
        if self._proc is not None:
            try:
                if os.name == "posix":
                    os.killpg(self._proc.pid, signal.SIGKILL)
                else:
                    self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass  # SIGKILL 后仍未退出：不阻塞清理，进程组已尽力终止
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            if self._active is not None and self._active.code is None:
                self._collect(self._active, 1)
                self._active.code = self._active.code or str(self._proc.returncode)
            self._proc = None
            self._reader = None

    def __del__(self):
        try:
            self.kill()
            if self._directory is not None:
                self._directory.cleanup()
        except Exception:
            pass  # 解释器关闭时模块可能已卸载；显式 kill 仍报告错误。
