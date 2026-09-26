"""权限判定：纯函数、零 IO。

判定分三层（见 ``.scratch/interaction-v2/spec.md``「权限判定」）：

- 分类：``Tool.kind``（read / write / exec，tools.py）
- 判定：:func:`decide` 六步顺序，本模块的核心
- 审批：四选项交互（默认拒绝），在驱动层实现

高危是**打断提醒层，不是安全边界**：正则只看命令串、承认不全，真正的边界是
root 路径限制 + 审批默认拒绝 + plan 模式。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .tools import Tool

# 高危启发式表：只对 exec 类工具的命令串匹配（不碰写类工具的 content 参数）。
# 明确承认不全（git reset --hard / dd / chmod -R 均不在表上）——它是「让用户抬眼」
# 的提醒层；扩充条目随用例迭代，行为由 tests/test_permissions.py 锁定。
HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-\w*r\w*f"),  # rm -rf / -fr / -rdf（单簇旗标）
    re.compile(r"\brm\s+-\w*f\w*r"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\b(?:curl|wget)\b[^|;&]*\|\s*(?:ba|z|fi)?sh\b"),  # curl | sh
    re.compile(r"\bgit\s+push\b[^;&|]*\s(?:-f|--force)\b"),  # push -f
)

# 复合命令判定：出现这些元字符即视为多段命令（Q14——前缀授权不覆盖复合命令，
# 防「cd tests && 危险命令」借 cd tests:* 洗白）
_COMPOUND = re.compile(r"[;|&\n]")

# plan 模式拒绝文案（与 agent.py 现行分发层拒绝一致，05 号票统一到此处）
PLAN_DENY_TEMPLATE = (
    "Error: 规划模式下只能使用只读工具（{name} 被拒绝）。"
    "完成计划后调用 exit_plan_mode 提交，批准后进入执行模式。"
)


@dataclass(frozen=True)
class Rule:
    """授权规则 ``tool(prefix:*)``：命中即放行。工具级通配（前缀 ``*``）即「放行」。

    内部用工具真名（小写、下划线）；``Bash(pytest:*)`` 是展示层的借形写法。
    """

    tool: str
    prefix: str

    def __str__(self) -> str:
        if self.prefix == "*":
            return f"{self.tool}(*)"
        return f"{self.tool}({self.prefix}:*)"

    def matches(self, tool: Tool, arguments: dict) -> bool:
        if tool.name != self.tool:
            return False
        if self.prefix == "*":
            return True
        key = match_key(tool, arguments)
        if tool.kind == "exec" and _is_compound(key):
            return False  # 复合命令不参与前缀匹配
        return _has_boundary(key, self.prefix)


@dataclass(frozen=True)
class Context:
    """一次判定的决策上下文，由驱动层组装。"""

    plan: bool = False
    yolo: bool = False
    auto_edit: bool = False
    rules: tuple[Rule, ...] = ()


@dataclass(frozen=True)
class Decision:
    """判定结果。``reason`` 双语义：deny 时是回传模型的完整文案；allow / ask 时是
    来源标签（read / rule / yolo / auto-edit / high-risk / 空=常规询问）。"""

    verdict: str  # "allow" | "deny" | "ask"
    reason: str = ""


def match_key(tool: Tool, arguments: dict) -> str:
    """调用的匹配键：exec 类取命令串、write 类取路径，其余为空（只有通配能命中）。"""
    if "command" in arguments:
        return str(arguments["command"])
    if "path" in arguments:
        return str(arguments["path"])
    return ""


def is_high_risk(tool: Tool, arguments: dict) -> bool:
    """高危启发式：只看 exec 类工具的命令串，不碰写类工具的 content 参数。"""
    if tool.kind != "exec" or "command" not in arguments:
        return False
    command = str(arguments["command"])
    return any(p.search(command) for p in HIGH_RISK_PATTERNS)


def rule_for(tool: Tool, arguments: dict) -> Rule | None:
    """审批选项 2（本会话前缀授权）生成规则；取不出安全前缀时返回 None（不提供该选项）。

    exec 取命令前两词（复合命令直接 None，Q14）；write 取父目录（根级文件退化为
    文件名，比目录更窄）；无键可取的工具给工具级通配（即「放行」）。
    """
    if tool.kind == "read":
        return None
    command = arguments.get("command")
    if tool.kind == "exec" and isinstance(command, str):
        if _is_compound(command):
            return None
        words = command.split()
        return Rule(tool.name, " ".join(words[:2])) if words else None
    path = arguments.get("path")
    if tool.kind == "write" and isinstance(path, str) and path:
        p = PurePosixPath(path.replace("\\", "/"))
        prefix = p.name if p.parent == PurePosixPath(".") else str(p.parent)
        return Rule(tool.name, prefix)
    return Rule(tool.name, "*")


def decide(tool: Tool, arguments: dict, ctx: Context) -> Decision:
    """六步判定（顺序即语义，勿调换）：

    ① read 放行 → ② plan 且非 read 拒绝 → ③ 高危强制询问（无授权出口，
    yolo / 规则都压不过）→ ④ 会话规则 / yolo 放行 → ⑤ auto-edit 放行 write →
    ⑥ 其余询问。
    """
    if tool.kind == "read":
        return Decision("allow", "read")
    if ctx.plan:
        return Decision("deny", PLAN_DENY_TEMPLATE.format(name=tool.name))
    if is_high_risk(tool, arguments):
        return Decision("ask", "high-risk")
    if any(r.matches(tool, arguments) for r in ctx.rules):
        return Decision("allow", "rule")
    if ctx.yolo:
        return Decision("allow", "yolo")
    if ctx.auto_edit and tool.kind == "write":
        return Decision("allow", "auto-edit")
    return Decision("ask")


def _is_compound(command: str) -> bool:
    return bool(_COMPOUND.search(command))


def _has_boundary(key: str, prefix: str) -> bool:
    """前缀命中且后随边界（串尾 / 空白 / 路径分隔），防 ``tests`` 误吞 ``tests-x``。"""
    return key.startswith(prefix) and (len(key) == len(prefix) or key[len(prefix)] in " /")
