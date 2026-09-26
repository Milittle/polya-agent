"""权限判定：纯函数、零 IO。

判定分三层（见 ``.scratch/interaction-v2/spec.md``「权限判定」）：

- 分类：``Tool.kind``（read / write / exec / delegate，tools.py）
- 判定：:func:`decide` 六步顺序，本模块的核心
- 审批：按风险选择默认项，在 approval.py 实现

高危是打断提醒层：正则只看命令串、承认不全。
shell 无系统级隔离；规则与人工审批不能替代沙箱。
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from .tools import Tool

# 高危启发式表：只对 exec 类工具的命令串匹配（不碰写类工具的 content 参数）。
# 只用于提醒，不能证明未命中的命令安全。
HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-\w*r\w*f"),  # rm -rf / -fr / -rdf（单簇旗标）
    re.compile(r"\brm\s+-\w*f\w*r"),
    re.compile(r"\b(?:sudo|dd|mkfs|reboot|shutdown)\b"),
    re.compile(r"\bgit\s+(?:reset\s+--hard|clean\s+-\w*f|push\b)"),
    re.compile(r"\brm\b[^;&|]*\s-\w*[rf]"),
    re.compile(r"\b(?:chmod|chown)\b[^;&|]*\s-[^\s]*R"),
    re.compile(r"\b(?:curl|wget)\b[^|;&]*\|\s*(?:ba|z|fi)?sh\b"),  # curl | sh
    re.compile(r"\bgit\s+push\b[^;&|]*\s(?:-f|--force)\b"),  # push -f
)

# 复合命令判定：出现这些元字符即视为多段命令（Q14——前缀授权不覆盖复合命令，
# 防「cd tests && 危险命令」借 cd tests:* 洗白）
_COMPOUND = re.compile(r"[;|&\n<>$`*?(){}\[\]~!#\\]|(?:^|\s)\w+=")

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
        if tool.kind == "exec":
            words, prefix = _simple_words(key), _simple_words(self.prefix)
            return bool(words and prefix and words[: len(prefix)] == prefix)
        if ".." in PurePosixPath(key).parts:
            return False
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
    if tool.kind in ("read", "delegate"):
        return None  # read 从不询问；delegate 从不询问（副作用在子工具逐个过闸）
    command = arguments.get("command")
    if tool.kind == "exec" and isinstance(command, str):
        if _is_compound(command):
            return None
        words = _simple_words(command)
        if not words or words[0] not in {"git", "pytest", "uv", "npm", "cargo", "ls", "rg"}:
            return None
        # Never grant an interpreter or a generic script runner by executable alone.
        if len(words) < 2 or words[:2] == ["uv", "run"]:
            return None
        if words[0] in {"git", "cargo", "npm", "uv"} and words[1].startswith("-"):
            return None
        return Rule(tool.name, shlex.join(words[:2]))
    path = arguments.get("path")
    if tool.kind == "write" and isinstance(path, str) and path:
        p = PurePosixPath(path.replace("\\", "/"))
        prefix = p.name if p.parent == PurePosixPath(".") else str(p.parent)
        return Rule(tool.name, prefix)
    return Rule(tool.name, "*")


def decide(tool: Tool, arguments: dict, ctx: Context) -> Decision:
    """六步判定（顺序即语义，勿调换）：

    ① read / delegate 放行 → ② plan 且非 read 拒绝 → ③ 高危强制询问（无授权
    出口，yolo / 规则都压不过）→ ④ 会话规则 / yolo 放行 → ⑤ auto-edit 放行
    write → ⑥ 其余询问。delegate 在此放行是因为副作用不发生在委派工具本身，
    而在子代理的工具调用上、逐个再过本判定；规划封堵同理收敛在子层（子 Context
    继承 plan_mode，子代理的 write/exec 会在第②步被拒）。
    """
    if tool.kind in ("read", "delegate"):
        return Decision("allow", tool.kind)
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


@dataclass(frozen=True)
class Assessment:
    """Risk informs the prompt; it never grants permission by itself."""

    risk: str
    reason: str

    @property
    def default_allow(self) -> bool:
        return self.risk in ("low", "medium")


def _simple_words(command: str) -> list[str]:
    if _is_compound(command):
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []


def assess(tool: Tool, arguments: dict) -> Assessment:
    """Conservative classification for prompts, not a shell sandbox or allowlist."""
    if tool.kind == "read":
        return Assessment("low", "Read-only tool")
    if tool.kind == "delegate":
        return Assessment("medium", "Delegates a bounded subtask; child tools still pass the gate")
    if is_high_risk(tool, arguments):
        return Assessment("high", "Destructive, privileged, or remote-changing operation")
    if tool.kind == "write":
        path = arguments.get("path")
        if (
            path
            and not PurePosixPath(str(path)).is_absolute()
            and ".." not in PurePosixPath(str(path)).parts
        ):
            return Assessment("medium", "Local file change; review the diff before allowing")
        return Assessment("unknown", "The target or impact needs review")
    words = _simple_words(str(arguments.get("command", "")))
    if not words:
        return Assessment("unknown", "Dynamic or compound command; review every operation")
    if any(arg.startswith(("--output", "--ext-diff", "--textconv", "--exec")) for arg in words):
        return Assessment("unknown", "Arguments can write output or invoke external programs")
    if any(".." in PurePosixPath(arg).parts or arg.startswith("/") for arg in words[1:]):
        return Assessment("unknown", "Command may access paths outside the workspace")
    if words[0] in {"ls", "pwd", "rg", "cat", "head", "tail", "wc"}:
        return Assessment(
            "low", "Inspection command; shell state and arguments still need approval"
        )
    if words[0] == "git" and len(words) > 1 and words[1] in {"status", "diff", "log", "show"}:
        return Assessment("low", "Git inspection; arguments still need approval")
    if words[0] == "pytest" or words[:2] in (
        ["cargo", "test"],
        ["cargo", "check"],
        ["cargo", "build"],
        ["uv", "run"],
        ["npm", "test"],
        ["npm", "run"],
    ):
        return Assessment("medium", "Runs project code; may write files or start processes")
    return Assessment("unknown", "Command effects are not covered by the built-in rules")
