"""真实伪终端冒烟：启动、流式输出、默认放行工具执行和退出。"""

import fcntl
import os
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("width", [40, 100])
def test_terminal_session_runs_tool_and_exits(tmp_path, width):
    script = tmp_path / "terminal_demo.py"
    script.write_text(
        '''
import time
from pathlib import Path
from types import SimpleNamespace as NS
from polya import Agent, tool
from polya.input import InputBox
from polya.render import TerminalRenderer
import polya.loop as loop

@tool(kind="write")
def change() -> str:
    """A fake write tool; no filesystem side effect."""
    return "changed"

class LLM:
    model = "demo-model"
    count = 0
    def chat_iter(self, messages, schemas):
        self.count += 1
        if self.count == 1:
            yield "text", "Checking file."
            # Completion is gated by the parent observing the newline-free preview.
            deadline = time.monotonic() + 5
            while not Path(__file__).with_name("preview_seen").exists():
                assert time.monotonic() < deadline, "preview was not displayed"
                time.sleep(0.01)
            time.sleep(0.2)
            calls = [NS(id="one", function=NS(name="change", arguments="{}"))]
            message = NS(content="Checking file.\\n", tool_calls=calls)
        else:
            yield "text", "Finished.\\n"
            message = NS(content="Finished.\\n", tool_calls=None)
        return NS(choices=[NS(message=message)], usage=None)

loop.InputBox = lambda **kw: InputBox(Path(__file__).with_name("history"), **kw)
agent = Agent(llm=LLM(), tools=[change], status_bar=False)
loop.run_repl(agent, str(Path(__file__).parent), TerminalRenderer())
''',
        encoding="utf-8",
    )
    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "TERM": "xterm-256color",
    }
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    transcript = bytearray()

    def read_until(text, offset=0):
        deadline = time.monotonic() + 6
        while text.encode() not in transcript[offset:]:
            assert time.monotonic() < deadline, transcript.decode(errors="replace")
            readable, _, _ = select.select([master], [], [], 0.05)
            if readable:
                chunk = os.read(master, 65536)
                transcript.extend(chunk)
                if b"\x1b[6n" in chunk:
                    os.write(master, b"\x1b[1;1R")
        return len(transcript)

    try:
        read_until("polya · v")  # banner 版本行：REPL 已启动
        read_until("❯")
        os.write(master, b"hello\r")
        read_until("esc to interrupt)")
        read_until("Checking file.")
        (tmp_path / "preview_seen").touch()
        read_until("changed")  # 默认放行：工具无需审批直接执行并落滚动区
        read_until("Finished.")  # 第二轮正文
        os.write(master, b"\x03\x04")
        process.wait(timeout=6)
        assert process.returncode == 0
        assert b"Traceback" not in transcript
        assert b"\x1b[?1049h" not in transcript  # Native scrollback, no alternate screen.
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)
        (tmp_path / "terminal.ansi").write_bytes(transcript)


def test_completion_menu_renders_above_input(tmp_path):
    """管道测试只查补全状态；菜单在真实终端画出来由本测试锁定（票 12 修复）。"""
    script = tmp_path / "menu_demo.py"
    script.write_text(
        """
from pathlib import Path
from types import SimpleNamespace as NS
from polya import Agent
from polya.input import InputBox
from polya.render import TerminalRenderer
import polya.loop as loop

class LLM:
    model = "demo-model"
    def chat_iter(self, messages, schemas):
        return NS(choices=[NS(message=NS(content="ok", tool_calls=None))], usage=None)

loop.InputBox = lambda **kw: InputBox(Path(__file__).with_name("history"), **kw)
agent = Agent(llm=LLM(), tools=[], status_bar=False)
loop.run_repl(agent, str(Path(__file__).parent), TerminalRenderer())
""",
        encoding="utf-8",
    )
    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "TERM": "xterm-256color",
    }
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    transcript = bytearray()

    def read_until(text, offset=0):
        deadline = time.monotonic() + 6
        while text.encode() not in transcript[offset:]:
            assert time.monotonic() < deadline, transcript.decode(errors="replace")
            readable, _, _ = select.select([master], [], [], 0.05)
            if readable:
                chunk = os.read(master, 65536)
                transcript.extend(chunk)
                if b"\x1b[6n" in chunk:
                    os.write(master, b"\x1b[1;1R")
        return len(transcript)

    try:
        read_until("❯")
        os.write(master, b"/")
        read_until("/todos")  # 菜单行（与 banner / 底栏不混淆）
        read_until("显示当前 TODO 清单")  # 描述列同屏渲染
        os.write(master, b"he")
        read_until("显示命令帮助")  # 过滤收敛到 /help
        os.write(master, b"\x1b")  # Esc 关菜单
        os.write(master, b"\x03\x04")  # 清空草稿并退出
        process.wait(timeout=6)
        assert process.returncode == 0
        assert b"Traceback" not in transcript
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)
        (tmp_path / "menu.ansi").write_bytes(transcript)
