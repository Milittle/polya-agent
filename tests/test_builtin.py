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


def test_bash_runs_and_reports_exit_code(tools):
    result = tools.call("bash", {"command": "echo hi"})
    assert "退出码 0" in result
    assert "hi" in result


def test_glob_finds_files_by_pattern(tools):
    tools.call("write_file", {"path": "a/x.py", "content": ""})
    tools.call("write_file", {"path": "b/y.txt", "content": ""})

    result = tools.call("glob", {"pattern": "**/*.py"})

    assert "a/x.py" in result
    assert "b/y.txt" not in result


def test_multi_edit_applies_all_edits_atomically(tools):
    tools.call("write_file", {"path": "f.txt", "content": "one two three"})

    result = tools.call(
        "multi_edit",
        {
            "path": "f.txt",
            "edits": [
                {"old_string": "one", "new_string": "1"},
                {"old_string": "three", "new_string": "3"},
            ],
        },
    )

    assert "2 处" in result
    assert tools.call("read_file", {"path": "f.txt"}) == "1 two 3"


def test_multi_edit_failure_leaves_file_untouched(tools):
    """原子性：任何一处失败，前面的编辑也不落盘。"""
    tools.call("write_file", {"path": "f.txt", "content": "one two three"})

    result = tools.call(
        "multi_edit",
        {
            "path": "f.txt",
            "edits": [
                {"old_string": "one", "new_string": "1"},
                {"old_string": "missing", "new_string": "X"},
            ],
        },
    )

    assert "Error" in result and "第 2 处" in result
    assert tools.call("read_file", {"path": "f.txt"}) == "one two three"


def test_web_fetch_wraps_content_with_source_marker(tools, monkeypatch):
    class FakeHeaders:
        def get(self, name, default=""):
            return "text/html" if name == "content-type" else default

        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        status = 200
        headers = FakeHeaders()

        def read(self, limit=-1):
            return (
                b"<html><head><style>.x{}</style></head><body>"
                b"<p>Hello docs</p><script>evil_instruction()</script></body></html>"
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("mi_z.web.urllib.request.urlopen", lambda req, timeout: FakeResponse())
    result = tools.call("web_fetch", {"url": "https://example.com/docs"})

    assert '<external_content source="webpage" url="https://example.com/docs">' in result
    assert "Hello docs" in result
    assert "evil_instruction" not in result  # script 内容被剥掉


def test_web_fetch_rejects_non_http_scheme(tools):
    result = tools.call("web_fetch", {"url": "file:///etc/passwd"})
    assert "Error" in result


def test_dangerous_flags():
    by_name = {item.name: item for item in default_tools(".")}

    for name in ("write_file", "edit_file", "multi_edit", "bash", "kill_bash"):
        assert by_name[name].dangerous is True, name
    for name in ("read_file", "list_dir", "glob", "grep", "bash_output", "web_fetch"):
        assert by_name[name].dangerous is False, name


def test_tool_descriptions_carry_usage_guidance():
    """描述不是摆设：关键的使用边界和协作关系要在（书实验 2-4：去掉描述 → 错误率 +45%）。"""
    by_name = {item.name: item for item in default_tools(".")}
    keywords = {
        "read_file": "编辑",  # 提示「编辑前必读」
        "write_file": "覆盖",  # 提示整文件覆盖的边界
        "edit_file": "唯一",  # 提示唯一性约束
        "multi_edit": "原子",  # 提示原子性保证
        "bash": "非交互",  # 提示交互命令会超时
        "glob": "不读内容",  # 提示按名字而非内容找
        "web_fetch": "不可信",  # 提示外部数据的注入风险
        "grep": "正则",  # 提示 pattern 是正则
        "list_dir": "不递归",  # 提示只列一层
    }
    for name, word in keywords.items():
        assert word in by_name[name].description, f"{name} 的描述缺少关键信息「{word}」"
        assert len(by_name[name].description) >= 30
