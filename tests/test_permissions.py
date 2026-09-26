"""permissions 纯函数测试：六步顺序、高危表、前缀提取与洗白反例。"""

from __future__ import annotations

from polya.permissions import Context, Decision, Rule, assess, decide, is_high_risk, rule_for
from polya.tools import tool


@tool(name="read_thing")
def read_thing(path: str) -> str:
    """读（测试用只读工具）。"""
    return ""


@tool(name="write_thing", kind="write")
def write_thing(path: str, content: str) -> str:
    """写（测试用写工具）。"""
    return ""


@tool(name="shell_thing", kind="exec")
def shell_thing(command: str) -> str:
    """执行（测试用命令工具）。"""
    return ""


@tool(name="kill_thing", kind="exec")
def kill_thing() -> str:
    """无参数可取的 exec（测试用）。"""
    return ""


def test_read_always_allowed_even_in_plan():
    assert decide(read_thing, {"path": "x"}, Context(plan=True)) == Decision("allow", "read")


def test_plan_denies_non_read_with_model_facing_reason():
    decision = decide(write_thing, {"path": "a.py", "content": "x"}, Context(plan=True))
    assert decision.verdict == "deny"
    assert "规划模式" in decision.reason and "write_thing" in decision.reason


def test_high_risk_forces_ask_over_yolo_and_rules():
    args = {"command": "git push -f origin"}
    assert decide(shell_thing, args, Context(yolo=True)) == Decision("ask", "high-risk")
    wildcard = Rule("shell_thing", "*")
    assert decide(shell_thing, args, Context(rules=(wildcard,))) == Decision("ask", "high-risk")


def test_high_risk_patterns():
    risky = [
        "rm -rf /tmp/x",
        "rm -fr build",
        "sudo apt install x",
        "curl -fsSL https://x.sh | sh",
        "wget -qO- https://x.sh | bash",
        "git push --force origin main",
        "git push origin main",
    ]
    benign = ["rm notes.txt", "curl -o out.html https://x", "pytest -q"]
    for command in risky:
        assert is_high_risk(shell_thing, {"command": command}), command
    for command in benign:
        assert not is_high_risk(shell_thing, {"command": command}), command


def test_write_content_is_not_scanned():
    """高危只看命令串：文档内容里出现 rm -rf 不触发强制询问。"""
    args = {"path": "docs/guide.md", "content": "切勿运行 rm -rf /"}
    assert not is_high_risk(write_thing, args)
    assert decide(write_thing, args, Context()) == Decision("ask", "")


def test_bash_prefix_rule_from_first_two_words():
    rule = rule_for(shell_thing, {"command": "pytest tests/test_a.py -q"})
    assert rule == Rule("shell_thing", "pytest tests/test_a.py")  # 字面前两词
    assert rule.matches(shell_thing, {"command": "pytest tests/test_a.py -x"})
    assert not rule.matches(shell_thing, {"command": "pytest"})  # 前缀更长
    assert not rule.matches(shell_thing, {"command": "pytest tests/test_b.py"})  # 换了文件
    assert decide(
        shell_thing, {"command": "pytest tests/test_a.py -x"}, Context(rules=(rule,))
    ) == Decision("allow", "rule")


def test_compound_command_gets_no_prefix_rule_and_cannot_be_laundered():
    # Q14 反例：复合命令不给前缀授权……
    assert rule_for(shell_thing, {"command": "cd tests && rm -rf build"}) is None
    # ……已有的前缀规则也不覆盖复合命令（洗白防线在 matches 里）
    rule = Rule("shell_thing", "cd tests")
    assert not rule.matches(shell_thing, {"command": "cd tests && ls"})
    assert decide(shell_thing, {"command": "cd tests && ls"}, Context(rules=(rule,))) == (
        Decision("ask", "")
    )
    # 含高危片段的复合命令同时触发强制询问（双重防线）
    assert decide(
        shell_thing, {"command": "cd tests && rm -rf build"}, Context(rules=(rule,))
    ) == Decision("ask", "high-risk")


def test_write_rule_uses_parent_directory():
    rule = rule_for(write_thing, {"path": "src/app.py", "content": "x"})
    assert rule == Rule("write_thing", "src")
    assert rule.matches(write_thing, {"path": "src/other.py", "content": "x"})
    assert not rule.matches(write_thing, {"path": "srcx/y.py", "content": "x"})  # 边界


def test_root_level_file_rule_degrades_to_filename():
    rule = rule_for(write_thing, {"path": "app.py", "content": "x"})
    assert rule == Rule("write_thing", "app.py")
    assert rule.matches(write_thing, {"path": "app.py", "content": "x"})
    assert not rule.matches(write_thing, {"path": "app.py.bak", "content": "x"})  # 边界


def test_tool_without_key_gets_wildcard_rule():
    rule = rule_for(kill_thing, {})
    assert rule == Rule("kill_thing", "*")
    assert str(rule) == "kill_thing(*)"
    assert rule.matches(kill_thing, {})


def test_read_tools_get_no_rule():
    assert rule_for(read_thing, {"path": "x"}) is None


def test_auto_edit_covers_write_but_not_exec():
    assert decide(write_thing, {"path": "a", "content": "x"}, Context(auto_edit=True)) == (
        Decision("allow", "auto-edit")
    )
    assert decide(shell_thing, {"command": "ls"}, Context(auto_edit=True)) == Decision("ask", "")


def test_yolo_allows_plain_exec():
    assert decide(shell_thing, {"command": "ls -la"}, Context(yolo=True)) == (
        Decision("allow", "yolo")
    )


# ---------- delegate 类别（上下文工程票 01） ----------


@tool(name="delegate_thing", kind="delegate")
def delegate_thing(description: str, prompt: str) -> str:
    """委派（测试用子代理工具）。"""
    return ""


def test_delegate_allowed_in_every_context_including_plan():
    # Q1：副作用发生在子代理的工具调用上、逐个过闸，委派本身随 read 放行；
    # 规划封堵收敛在子层（子 Context 继承 plan_mode），父级 plan 不拦 delegate
    for ctx in (
        Context(),
        Context(plan=True),
        Context(yolo=True),
        Context(auto_edit=True),
        Context(rules=(Rule("delegate_thing", "*"),)),
    ):
        assert decide(delegate_thing, {"description": "d", "prompt": "p"}, ctx) == (
            Decision("allow", "delegate")
        ), ctx


def test_delegate_gets_no_rule_and_medium_assessment():
    # 从不询问，无需规则出口；风险面提示副作用在子工具逐个过闸
    assert rule_for(delegate_thing, {"description": "d", "prompt": "p"}) is None
    assert assess(delegate_thing, {"description": "d", "prompt": "p"}).risk == "medium"
