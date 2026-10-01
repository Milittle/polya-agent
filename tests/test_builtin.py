"""内置编码工具的测试：全部在 tmp_path 沙箱里跑，不触碰项目文件。"""

from __future__ import annotations

import pytest

from polya.builtin import default_tools
from polya.todos import TodoStore
from polya.tools import ToolRegistry, tool


@pytest.fixture
def tools(tmp_path) -> ToolRegistry:
    return ToolRegistry(default_tools(tmp_path))


def test_write_then_read_roundtrip(tools, tmp_path):
    assert (
        tools.call("write_file", {"path": "a/b.txt", "content": "hello"})
        == "Wrote a/b.txt (5 chars)"
    )
    assert tools.call("read_file", {"path": "a/b.txt"}) == "     1\thello"  # 输出带行号


def test_read_file_line_range(tools):
    tools.call("write_file", {"path": "f.txt", "content": "1\n2\n3\n4"})
    result = tools.call("read_file", {"path": "f.txt", "start_line": 2, "end_line": 3})
    assert result == "     2\t2\n     3\t3"


def test_read_file_rejects_binary_and_oversized(tools):
    tools.call("bash", {"command": "printf '\\x00\\x01bin' > bin.dat"})
    assert "Binary file" in tools.call("read_file", {"path": "bin.dat"})

    tools.call("bash", {"command": "head -c 3000000 /dev/zero | tr '\\0' 'x' > big.txt"})
    assert "too large" in tools.call("read_file", {"path": "big.txt"})


def test_read_missing_file_returns_error_to_model(tools):
    assert "Error" in tools.call("read_file", {"path": "nope.txt"})


def test_relative_path_escape_is_rejected(tools, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    result = tools.call("write_file", {"path": "../outside.txt", "content": "x"})

    assert "Error" in result and "working directory" in result
    assert not outside.exists()


def test_absolute_path_is_rejected(tools):
    result = tools.call("read_file", {"path": "/etc/passwd"})
    assert "Error" in result and "working directory" in result


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


def test_grep_skips_ignored_dirs(tools):
    """grep 不应扫描 .venv/.git 等目录——那会把第三方库的结果混进上下文。"""
    tools.call("write_file", {"path": "src/app.py", "content": "needle\n"})
    tools.call("write_file", {"path": ".venv/lib/site.py", "content": "needle\n"})
    tools.call("write_file", {"path": ".git/config", "content": "needle\n"})

    result = tools.call("grep", {"pattern": "needle"})

    assert "src/app.py:1: needle" in result
    assert ".venv" not in result and ".git" not in result


def test_grep_context_lines_and_ignore_case(tools):
    tools.call("write_file", {"path": "f.py", "content": "a\nb\nTARGET\nc\nd\n"})

    result = tools.call("grep", {"pattern": "target", "ignore_case": True, "context_lines": 1})

    assert "f.py:3: TARGET" in result  # 命中行用 :
    assert "f.py-2- b" in result  # 上下文行用 -
    assert "f.py-4- c" in result
    assert "a" not in result.replace("a\n", "")  # 超出上下文的不出现


def test_glob_skips_ignored_dirs_and_prefix_optional(tools):
    tools.call("write_file", {"path": "src/app.py", "content": ""})
    tools.call("write_file", {"path": ".venv/lib/site.py", "content": ""})

    result = tools.call("glob", {"pattern": "*.py"})  # 无 **/ 前缀也应递归匹配

    assert "src/app.py" in result
    assert ".venv" not in result


def test_edit_file_requires_unique_match(tools):
    tools.call("write_file", {"path": "f.txt", "content": "a b a"})

    assert "Error" in tools.call(
        "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "c"}
    )

    assert tools.call(
        "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "c", "replace_all": True}
    )
    assert tools.call("read_file", {"path": "f.txt"}) == "     1\tc b c"


def test_bash_runs_and_reports_exit_code(tools):
    result = tools.call("bash", {"command": "echo hi"})
    assert "Exit code 0" in result
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

    assert "(2 edits)" in result
    assert tools.call("read_file", {"path": "f.txt"}) == "     1\t1 two 3"


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

    assert "Error" in result and "Edit #2" in result
    assert tools.call("read_file", {"path": "f.txt"}) == "     1\tone two three"  # 原子性：未落盘


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

    monkeypatch.setattr("polya.web._open_url", lambda req, timeout: FakeResponse())
    monkeypatch.setattr("polya.web._assert_public_url", lambda url: None)  # 测试不做 DNS 解析
    result = tools.call("web_fetch", {"url": "https://example.com/docs"})

    assert '<external_content source="webpage" url="https://example.com/docs">' in result
    assert "Hello docs" in result
    assert "evil_instruction" not in result  # script 内容被剥掉


def test_web_fetch_rejects_non_http_scheme(tools):
    result = tools.call("web_fetch", {"url": "file:///etc/passwd"})
    assert "Error" in result


def test_web_fetch_blocks_ssrf_targets(tools):
    """localhost / 内网 IP / .local 域一律拒绝——模型不能借本工具探测内网。"""
    for url in (
        "http://localhost:8500/admin",
        "http://127.0.0.1:8080/",
        "http://192.168.1.1/",
        "http://10.0.0.5:3000/",
        "http://printer.local/",
    ):
        result = tools.call("web_fetch", {"url": url})
        assert "Error" in result, url
        assert "拒绝" in result, url


def test_web_fetch_redirect_to_internal_is_rejected():
    """重定向目标逐跳校验：公开 URL 302 到内网必须在跟随前拒绝。"""
    from urllib.request import Request

    from polya.web import _ValidatingRedirectHandler

    handler = _ValidatingRedirectHandler()
    with pytest.raises(ValueError, match="拒绝"):
        handler.redirect_request(
            Request("https://example.com"), None, 302, "Found", {}, "http://127.0.0.1/x"
        )


def test_web_fetch_revalidates_final_url_against_rebinding(tools, monkeypatch):
    """连接后最终地址再校验：opener 返回的 response.url 指向内网也要拒绝。"""

    class FakeHeaders:
        def get(self, name, default=""):
            return "text/plain" if name == "content-type" else default

        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        status = 200
        url = "http://127.0.0.1/x"
        headers = FakeHeaders()

        def read(self, limit=-1):
            return b"secret"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("polya.web._open_url", lambda req, timeout: FakeResponse())
    result = tools.call("web_fetch", {"url": "https://8.8.8.8/"})
    assert "Error" in result
    assert "拒绝" in result


def test_kinds():
    """kind 分类表锁定（spec：read 无副作用直接放行 / write 写文件 / exec 执行命令）。"""
    expected = {
        "read_file": "read",
        "list_dir": "read",
        "glob": "read",
        "grep": "read",
        "bash_output": "read",
        "web_fetch": "read",
        "write_file": "write",
        "edit_file": "write",
        "multi_edit": "write",
        "bash": "exec",
        "kill_bash": "exec",
    }
    by_name = {item.name: item for item in default_tools(".")}
    for name, kind in expected.items():
        assert by_name[name].kind == kind, name
    # todo_write 可选启用，归类 read（外部记忆，无文件系统副作用）
    with_todos = {item.name: item.kind for item in default_tools(".", todos=TodoStore())}
    assert with_todos["todo_write"] == "read"


def test_poll_tools_are_marked():
    """票 05：等待类工具 bash_output 标 poll=True（无进展熔断跳过它）。"""
    by_name = {item.name: item for item in default_tools(".")}
    assert by_name["bash_output"].poll is True
    assert by_name["read_file"].poll is False
    assert by_name["bash"].poll is False
    assert by_name["write_file"].poll is False


def test_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):

        @tool(name="bad", kind="wat")
        def bad() -> str:
            return ""


def test_tool_descriptions_carry_usage_guidance():
    """描述不是摆设：关键的使用边界和协作关系要在（书实验 2-4：去掉描述 → 错误率 +45%）。"""
    by_name = {item.name: item for item in default_tools(".")}
    keywords = {
        "read_file": "editing",  # 提示「编辑前必读」
        "write_file": "overwrit",  # 提示整文件覆盖的边界
        "edit_file": "unique",  # 提示唯一性约束
        "multi_edit": "atomically",  # 提示原子性保证
        "bash": "non-interactive",  # 提示交互命令会超时
        "glob": "does not read contents",  # 提示按名字而非内容找
        "web_fetch": "untrusted",  # 提示外部数据的注入风险
        "grep": "regex",  # 提示 pattern 是正则
        "list_dir": "one level",  # 提示只列一层
    }
    for name, word in keywords.items():
        assert word in by_name[name].description, f"{name} 的描述缺少关键信息「{word}」"
        assert len(by_name[name].description) >= 30


def test_read_only_tools_is_read_subset_without_shell():
    """票 02：只读子集是真子集、全 kind=read、不含 shell 工具。"""
    from polya.builtin import read_only_tools

    readonly = read_only_tools(".")
    names = {item.name for item in readonly}
    full = {item.name for item in default_tools(".")}
    assert names < full  # 真子集
    assert all(item.kind == "read" for item in readonly)
    assert not names & {"bash", "bash_output", "kill_bash", "write_file", "edit_file"}
    assert names == {"read_file", "list_dir", "glob", "grep", "web_fetch"}


def test_edit_preserves_crlf_line_endings(tools, tmp_path):
    (tmp_path / "w.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    # 模型给的 old_string 用 \n，也要能匹配 CRLF 文件
    tools.call("edit_file", {"path": "w.txt", "old_string": "a\nb", "new_string": "x\ny"})
    assert (tmp_path / "w.txt").read_bytes() == b"x\r\ny\r\nc\r\n"
    tools.call("multi_edit", {"path": "w.txt", "edits": [{"old_string": "c", "new_string": "z"}]})
    assert (tmp_path / "w.txt").read_bytes() == b"x\r\ny\r\nz\r\n"


def test_edit_preserves_lf_files(tools, tmp_path):
    (tmp_path / "l.txt").write_bytes(b"a\nb\n")
    tools.call("edit_file", {"path": "l.txt", "old_string": "b", "new_string": "c"})
    assert (tmp_path / "l.txt").read_bytes() == b"a\nc\n"


def test_edit_rejected_when_file_changed_since_read(tools, tmp_path):
    (tmp_path / "s.txt").write_text("one\n")
    tools.call("read_file", {"path": "s.txt"})
    (tmp_path / "s.txt").write_text("one\nexternally added\n")  # bash / 编辑器改动
    result = tools.call("edit_file", {"path": "s.txt", "old_string": "one", "new_string": "two"})
    assert result.startswith("Error:") and "changed since" in result
    assert "externally added" in (tmp_path / "s.txt").read_text()
    # 重读后可编辑
    tools.call("read_file", {"path": "s.txt"})
    assert "Edited" in tools.call(
        "edit_file", {"path": "s.txt", "old_string": "one", "new_string": "two"}
    )


def test_consecutive_edits_do_not_trip_freshness(tools, tmp_path):
    (tmp_path / "c.txt").write_text("a b c\n")
    tools.call("read_file", {"path": "c.txt"})
    tools.call("edit_file", {"path": "c.txt", "old_string": "a", "new_string": "x"})
    assert "Edited" in tools.call(
        "edit_file", {"path": "c.txt", "old_string": "b", "new_string": "y"}
    )


def test_read_file_line_numbers_ignore_exotic_separators(tools, tmp_path):
    (tmp_path / "x.txt").write_text("a\x0cb\nc\n", encoding="utf-8")
    assert tools.call("read_file", {"path": "x.txt"}) == "     1\ta\x0cb\n     2\tc"
