"""review.py 的审查器缝：默认放行、plan 只读、批准白名单。"""

from __future__ import annotations

from polya.review import AllowAllReviewer, is_plan_approval
from polya.tools import tool


@tool(name="read_thing", kind="read")
def read_thing(path: str) -> str:
    """读（测试用）。"""
    return ""


@tool(name="write_thing", kind="write")
def write_thing(path: str, content: str) -> str:
    """写（测试用）。"""
    return ""


@tool(name="delegate_thing", kind="delegate")
def delegate_thing(description: str, prompt: str) -> str:
    """委派（测试用）。"""
    return ""


def test_default_allows_everything_without_plan():
    reviewer = AllowAllReviewer()
    for item in (read_thing, write_thing, delegate_thing):
        assert reviewer.review(item, {}).verdict == "allow"


def test_plan_denies_non_read_but_allows_read_and_delegate():
    reviewer = AllowAllReviewer()
    assert reviewer.review(read_thing, {"path": "x"}, plan=True).verdict == "allow"
    assert reviewer.review(delegate_thing, {}, plan=True).verdict == "allow"
    denied = reviewer.review(write_thing, {"path": "x", "content": "y"}, plan=True)
    assert denied.verdict == "deny"
    assert "规划模式" in denied.reason and "exit_plan_mode" in denied.reason


def test_plan_approval_whitelist_is_exact_match():
    for text in ("批准", " 批准。", "go", " OK ", "yes!", "继续"):
        assert is_plan_approval(text), text
    # 含批准二字的长句 / 非等值短语不得误触发
    for text in ("批准，但先跑测试", "go ahead please", "批准执行计划", "", "开始写代码"):
        assert not is_plan_approval(text), text
