"""工具调用审查器缝：默认放行，未来可接入模型审查（Jev 类）。

审批已从默认路径移除（见 ``.scratch/pi-interaction/spec.md``）。审查发生在
驱动层 :func:`polya.loop.run_tool_call` 与内置驱动 :meth:`Agent._builtin_tool`，
二者共用一个 :class:`Reviewer`。默认 :class:`AllowAllReviewer` 只在规划模式下
拒绝非只读工具（保住 plan 的只读约束），其余一律放行——与 pi「默认以进程权限
运行」一致。

审查结果只有 ``allow`` / ``deny`` 两态；``deny`` 只作为工具结果回传模型，
不弹窗、不让位终端、不停止整轮。未来模型审查器实现同一协议即可接入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .tools import Tool

# plan 模式拒绝文案：只读约束的唯一实现点（审批栈删除后迁到这里）
PLAN_DENY_TEMPLATE = (
    "Error: 规划模式下只能使用只读工具（{name} 被拒绝）。"
    "完成计划后调用 exit_plan_mode 提交，批准后进入执行模式。"
)

# 计划批准白名单（05 号票）：整条消息去首尾空白与尾标点、英文小写后精确匹配，
# 只做等值不做包含/模糊，避免正文里的「批准」二字误触发。
PLAN_APPROVALS = frozenset(
    {
        "批准",
        "同意",
        "继续",
        "执行",
        "开始",
        "可以",
        "好的",
        "好",
        "行",
        "approve",
        "approved",
        "go",
        "ok",
        "okay",
        "yes",
        "proceed",
        "continue",
        "lgtm",
    }
)

_APPROVAL_STRIP = " \t\r\n。.!！"


def is_plan_approval(text: str) -> bool:
    """整条消息是否恰为一次计划批准（精确匹配，不做包含）。"""
    return text.strip(_APPROVAL_STRIP).lower() in PLAN_APPROVALS


@dataclass(frozen=True)
class Review:
    """一次审查结果。``reason`` 在 deny 时是回传模型的完整文案。"""

    verdict: Literal["allow", "deny"]
    reason: str = ""


class Reviewer(Protocol):
    def review(
        self,
        tool: Tool,
        arguments: dict,
        *,
        plan: bool = False,
        origin: str | None = None,
    ) -> Review: ...


class AllowAllReviewer:
    """默认审查器：plan 模式拒绝非只读（delegate 例外），其余放行。

    read / delegate 放行与旧 ``decide`` 步 ① 一致；delegate 的副作用不在委派
    工具本身，而在子代理的工具调用上——子代理继承父 plan_mode，逐个再过审查。
    """

    def review(
        self,
        tool: Tool,
        arguments: dict,
        *,
        plan: bool = False,
        origin: str | None = None,
    ) -> Review:
        if plan and tool.kind not in ("read", "delegate"):
            return Review("deny", PLAN_DENY_TEMPLATE.format(name=tool.name))
        return Review("allow")
