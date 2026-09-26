"""Agent 状态栏：把分散在轨迹里的隐式状态提炼成显式元信息，注入上下文末尾。

模型「检索强、归纳弱」（书 2.6：上下文窗口是一台只有一半的检索引擎）——
「某工具已经调了几次」这类结论若靠模型从原始轨迹现数，代价随上下文增长
且容易数错。状态栏用代码提前算好，模型瞥一眼就能直接用。

缓存纪律（书 2.3/2.6）：状态以 user 角色消息**追加**在上下文末尾，从不修改
system 或历史消息。更新采用持久追加（Claude Code ``<system-reminder>`` 同款）：
旧状态永久留在轨迹里，前缀始终字节稳定——状态消息小、会话长度受控时，
这比每轮替换缓存更便宜（αSN/2 < (1-α)R）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

_STATUS_LABELS = {
    "pending": "待办",
    "in_progress": "进行中",
    "completed": "已完成",
    "cancelled": "已取消",
}


@dataclass
class StatusSnapshot:
    """一次状态快照：由代码在每轮迭代前采集，交给渲染器生成状态栏文本。"""

    iteration: int  # 本次 run() 的第几轮迭代（从 1 开始）
    max_steps: int
    tool_calls: dict[str, int] = field(default_factory=dict)  # 工具名 -> 会话累计调用次数
    usage: dict = field(default_factory=dict)  # 累计 token 用量
    todos: list[dict] = field(default_factory=list)  # TODO 清单（任务规划组件）
    plan_mode: bool = False  # 规划模式（只读；环境状态组件）
    now: datetime = field(default_factory=datetime.now)


def render_status(snapshot: StatusSnapshot) -> str:
    """默认渲染器：生成 <agent_status> 包裹的状态栏文本。

    注意「本条为最新状态」的提示：持久追加模式下历史里会有多条状态栏，
    需要明确告诉模型以最后一条为准。
    """
    if snapshot.tool_calls:
        calls = "\n".join(f"  - {name}: {count} 次" for name, count in snapshot.tool_calls.items())
    else:
        calls = "  - （尚未调用工具）"
    usage = snapshot.usage
    mode_line = (
        "- 模式: 规划中（只读；完成计划后调用 exit_plan_mode 提交）\n" if snapshot.plan_mode else ""
    )
    todo_lines = ""
    if snapshot.todos:
        items = "\n".join(
            f"  [{index}] [{_STATUS_LABELS.get(item['status'], item['status'])}] {item['content']}"
            for index, item in enumerate(snapshot.todos, 1)
        )
        todo_lines = f"- TODO 清单:\n{items}\n"
    return (
        "<agent_status>\n"
        f"当前状态（第 {snapshot.iteration}/{snapshot.max_steps} 轮迭代；"
        "历史中若有多条状态，以最后一条为准）：\n"
        f"- 时间: {snapshot.now:%Y-%m-%d %H:%M:%S}\n"
        f"{mode_line}"
        f"- 工具调用累计:\n{calls}\n"
        f"{todo_lines}"
        f"- token 用量: prompt {usage.get('prompt_tokens', 0)},"
        f" completion {usage.get('completion_tokens', 0)}\n"
        "</agent_status>"
    )
