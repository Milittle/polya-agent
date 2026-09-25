"""持久 shell 会话：跨多次工具调用保持环境状态（cwd、环境变量、后台进程）。

一次性的 ``subprocess.run`` 每次调用都是全新进程——上次 ``cd`` 的目录、
``export`` 的变量全部丢失，也无法启动 dev server 后回头读输出。持久会话
维持一个常驻 bash 进程，命令通过 stdin 写入、输出经哨兵标记定界读回。

实现要点：
- 读线程把 stdout 逐行放进队列，命令执行以 ``echo <marker> $?`` 结尾定界；
- ``output()`` 只取队列里现成的行（不等待），用于读后台/慢速输出；
- 超时视为命令失控：终止整个会话（环境状态随之丢失），下次调用自动重启；
- 命令必须是非交互的（不能等 stdin 输入），交互命令会一直等到超时。
"""

from __future__ import annotations

import queue
import subprocess
import threading
import uuid

_MARKER_PREFIX = "__mi_z_done_"


class ShellSession:
    """一个常驻 bash 进程。线程不安全：同一会话的命令串行执行。"""

    def __init__(self, cwd: str):
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None

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
        )
        self._queue = queue.Queue()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._queue.put(line)
        # 进程退出时放一个 None 哨兵，让等待者不会永远阻塞
        self._queue.put(None)

    def run(self, command: str, timeout: float = 10.0) -> str:
        """执行命令并等待完成，返回 `退出码 N\\n输出` 格式的结果。"""
        self._ensure_started()
        assert self._proc is not None and self._proc.stdin is not None
        marker = f"{_MARKER_PREFIX}{uuid.uuid4().hex[:8]}"
        self._proc.stdin.write(f"{command}\necho {marker} $?\n")
        self._proc.stdin.flush()

        lines: list[str] = []
        exit_code: str | None = None
        while True:
            try:
                line = self._queue.get(timeout=timeout)
            except queue.Empty:
                self.kill()
                return (
                    "退出码 -\n"
                    f"[命令超时（{timeout:.0f} 秒），会话已终止；环境状态丢失，下次调用将自动重启。"
                    "命令可能正在等待交互输入——请改用非交互命令]"
                )
            if line is None:  # bash 进程已退出
                return f"退出码 {self._proc.returncode}\n{''.join(lines).strip()}"
            if line.startswith(marker):
                exit_code = line[len(marker) :].strip()
                break
            lines.append(line)

        output = "".join(lines).strip()
        return f"退出码 {exit_code}\n{output if output else '(无输出)'}"

    def output(self) -> str:
        """不等待地取走目前已产生的输出（后台/慢速命令的增量读取）。"""
        lines: list[str] = []
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                lines.append("[会话进程已退出]")
                break
            lines.append(line)
        return "".join(lines).strip() or "(暂无新输出)"

    def kill(self) -> None:
        """终止会话进程（不影响下次 run 自动重启）。"""
        if self._proc is not None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)  # 收尸，避免僵尸进程堆积
            except subprocess.TimeoutExpired:  # pragma: no cover - kill 后几乎不会发生
                pass
            self._proc = None
            self._reader = None
