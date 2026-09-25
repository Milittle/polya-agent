"""内置编码工具的测试：全部在 tmp_path 沙箱里跑，不触碰项目文件。"""

from __future__ import annotations

import pytest

from mi_z.builtin import default_tools
from mi_z.tools import ToolRegistry


@pytest.fixture
def tools(tmp_path) -> ToolRegistry:
    return ToolRegistry(default_tools(tmp_path))


def test_write_then_read_roundtrip(tools, tmp_path):
    assert (
        tools.call("write_file", {"path": "a/b.txt", "content": "hello"})
        == "已写入 a/b.txt（5 字符）"
    )
    assert tools.call("read_file", {"path": "a/b.txt"}) == "hello"


def test_read_file_line_range(tools):
    tools.call("write_file", {"path": "f.txt", "content": "1\n2\n3\n4"})
    assert tools.call("read_file", {"path": "f.txt", "start_line": 2, "end_line": 3}) == "2\n3"


def test_read_missing_file_returns_error_to_model(tools):
    assert "Error" in tools.call("read_file", {"path": "nope.txt"})


def test_relative_path_escape_is_rejected(tools, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    result = tools.call("write_file", {"path": "../outside.txt", "content": "x"})

    assert "Error" in result and "工作目录" in result
    assert not outside.exists()


def test_absolute_path_is_rejected(tools):
    result = tools.call("read_file", {"path": "/etc/passwd"})
    assert "Error" in result and "工作目录" in result


def test_list_dir_marks_subdirectories(tools):
    (tools.call("write_file", {"path": "pkg/mod.py", "content": ""}))
    listing = tools.call("list_dir", {})

    assert "pkg/" in listing
    assert "pkg/mod.py" not in listing  # 只列一层


def test_grep_filters_by_glob_and_reports_line(tools):
    tools.call("write_file", {"path": "a.py", "content": "import os\nx = 1\n"})
    tools.call("write_file", {"path": "b.txt", "content": "import sys\n"})

    result = tools.call("grep", {"pattern": "import", "glob": "*.py"})

    assert "a.py:1: import os" in result
    assert "b.txt" not in result


def test_edit_file_requires_unique_match(tools):
    tools.call("write_file", {"path": "f.txt", "content": "a b a"})

    assert "Error" in tools.call(
        "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "c"}
    )

    assert tools.call(
        "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "c", "replace_all": True}
    )
    assert tools.call("read_file", {"path": "f.txt"}) == "c b c"


def test_run_shell_reports_exit_code_and_output(tools):
    result = tools.call("run_shell", {"command": "echo hi"})
    assert "退出码 0" in result
    assert "hi" in result


def test_dangerous_flags():
    by_name = {item.name: item for item in default_tools(".")}

    assert by_name["write_file"].dangerous is True
    assert by_name["edit_file"].dangerous is True
    assert by_name["run_shell"].dangerous is True
    assert by_name["read_file"].dangerous is False
    assert by_name["list_dir"].dangerous is False
    assert by_name["grep"].dangerous is False
