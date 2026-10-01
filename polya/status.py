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


@dataclass
class StatusSnapshot:
    """一次状态快照：由代码在每轮迭代前采集，交给渲染器生成状态栏文本。"""

    iteration: int  # 本次 run() 的第几轮迭代（从 1 开始）
    max_steps: int
    tool_calls: dict[str, int] = field(default_factory=dict)  # 工具名 -> 会话累计调用次数
    usage: dict = field(default_factory=dict)  # 累计 token 用量
    todos: list[dict] = field(default_factory=list)  # TODO 清单（任务规划组件）
    plan_mode: bool = False  # 规划模式（只读；环境状态组件）


def render_status(snapshot: StatusSnapshot) -> str:
    """默认渲染器：生成 <agent_status> 包裹的状态栏文本（固定英文，模型侧）。

    注意「本条为最新状态」的提示：持久追加模式下历史里会有多条状态栏，
    需要明确告诉模型以最后一条为准。TODO 状态用原词（pending /
    in_progress / completed / cancelled），不做中文映射。
    """
    if snapshot.tool_calls:
        calls = "\n".join(f"  - {name}: {count}" for name, count in snapshot.tool_calls.items())
    else:
        calls = "  - (no tool calls yet)"
    usage = snapshot.usage
    mode_line = (
        "- mode: planning (read-only; submit via exit_plan_mode when the plan is complete)\n"
        if snapshot.plan_mode
        else ""
    )
    todo_lines = ""
    if snapshot.todos:
        items = "\n".join(
            f"  [{index}] [{item['status']}] {item['content']}"
            for index, item in enumerate(snapshot.todos, 1)
        )
        todo_lines = f"- TODO list:\n{items}\n"
    # max_steps==0 表示无界：不显示分母，避免「turn N/0」。
    iteration_line = (
        f"turn {snapshot.iteration}, unbounded"
        if snapshot.max_steps == 0
        else f"turn {snapshot.iteration}/{snapshot.max_steps}"
    )
    return (
        "<agent_status>\n"
        f"Current state ({iteration_line}; "
        "if multiple snapshots exist, the last one wins):\n"
        f"{mode_line}"
        f"- tool calls:\n{calls}\n"
        f"{todo_lines}"
        f"- token usage: prompt {usage.get('prompt_tokens', 0)},"
        f" completion {usage.get('completion_tokens', 0)},"
        f" cached {usage.get('cached_tokens', 0)}\n"
        "</agent_status>"
    )
