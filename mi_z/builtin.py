"""内置工具：一个本地编码代理该有的读写 / 搜索 / 执行能力。

所有文件操作都被限制在 :func:`default_tools` 传入的 ``root`` 目录内，
解析后校验，防止 ``../`` 或绝对路径穿越到目录外。

``write_file`` / ``edit_file`` / ``run_shell`` 标记为 ``dangerous``，
可交给 :class:`~mi_z.agent.Agent` 的 ``approve`` 钩子在执行前拦截。
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from .tools import Tool, tool

MAX_OUTPUT = 8000


def _truncate(text: str) -> str:
    """工具结果会进上下文，长输出必须截断。"""
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... [已截断，完整输出共 {len(text)} 字符]"


def _resolve(root: Path, path: str) -> Path:
    """把相对路径解析到 root 下，并确保没有越界。"""
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"路径超出工作目录: {path}")
    return target


def default_tools(root: str | os.PathLike[str] = ".") -> list[Tool]:
    """构造一组受限在 ``root`` 目录内的编码工具。"""
    base = Path(root).resolve()

    @tool
    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """读取工作目录内某个文本文件。可用 start_line / end_line（从 1 开始、含两端）只读片段。"""
        target = _resolve(base, path)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if start_line is not None or end_line is not None:
            lines = lines[(start_line or 1) - 1 : end_line]
        return _truncate("\n".join(lines))

    @tool(name="list_dir")
    def list_dir(path: str = ".") -> str:
        """列出工作目录内某个目录下的条目，子目录以 / 结尾。"""
        target = _resolve(base, path)
        if not target.is_dir():
            raise NotADirectoryError(f"不是目录: {path}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        listing = "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
        return _truncate(listing or "(空目录)")

    @tool(name="grep")
    def grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
        """在工作目录内按正则搜索文件内容，返回 `文件:行号: 内容`。glob 可限定文件名，如 '*.py'。"""
        target = _resolve(base, path)
        regex = re.compile(pattern)
        files = [target] if target.is_file() else sorted(target.rglob("*"))
        hits: list[str] = []
        for file in files:
            if not file.is_file() or (glob and not fnmatch.fnmatch(file.name, glob)):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{file.relative_to(base)}:{number}: {line.strip()}")
                    if len(hits) >= 200:
                        return _truncate("\n".join(hits) + "\n... [命中过多，已截断]")
        return "\n".join(hits) or "(无匹配)"

    @tool(name="write_file", dangerous=True)
    def write_file(path: str, content: str) -> str:
        """把 content 写入工作目录内的文件，已存在则覆盖，父目录会自动创建。"""
        target = _resolve(base, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {target.relative_to(base)}（{len(content)} 字符）"

    @tool(name="edit_file", dangerous=True)
    def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """把文件中的 old_string 替换成 new_string；默认要求唯一，replace_all=True 则全部替换。"""
        target = _resolve(base, path)
        text = target.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ValueError("未找到 old_string")
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string 出现 {count} 次，不唯一；请补充上下文或传 replace_all=True"
            )
        target.write_text(text.replace(old_string, new_string), encoding="utf-8")
        return f"已修改 {target.relative_to(base)}（{count} 处）"

    @tool(name="run_shell", dangerous=True)
    def run_shell(command: str, timeout: int = 30) -> str:
        """在工作目录下执行一条 shell 命令，返回退出码与合并后的 stdout/stderr。"""
        proc = subprocess.run(
            command, shell=True, cwd=base, capture_output=True, text=True, timeout=timeout
        )
        output = (proc.stdout + proc.stderr).strip()
        return f"退出码 {proc.returncode}\n{_truncate(output) if output else '(无输出)'}"

    return [read_file, list_dir, grep, write_file, edit_file, run_shell]
