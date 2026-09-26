"""TODO 清单存储：todo_write 工具写入，状态栏每轮渲染到上下文末尾。

TODO 是外部记忆（书实验 2-9：启用后平均 15 轮完成任务，禁用 21 轮且常漏子任务）。
它属于状态栏的「任务规划」组件（书 2.6）：清单不靠模型从历史里回忆，
而是每轮迭代以 user 消息追加在上下文末尾，持续把注意力拉回剩余目标。

设计取舍：单一全量重写工具（todo_write），不做逐项 update——ID 制在模型
手里容易出现陈旧 ID；全量重写无状态歧义，代价是多写几行清单文本。
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")

_STATUS_LABELS = {
    "pending": "待办",
    "in_progress": "进行中",
    "completed": "已完成",
    "cancelled": "已取消",
}


@dataclass
class Todo:
    content: str
    status: str = "pending"


class TodoStore:
    """当前会话的 TODO 清单。线程模型与 Agent 一致：单线程串行使用。"""

    def __init__(self) -> None:
        self._items: list[Todo] = []

    def rewrite(self, items: list[dict]) -> int:
        """用 items 全量替换清单，返回项数。任何一项非法则整体不变（原子）。"""
        cleaned: list[Todo] = []
        for index, item in enumerate(items or [], 1):
            content = str(item.get("content", "")).strip()
            status = item.get("status", "pending")
            if not content:
                raise ValueError(f"第 {index} 项缺少 content")
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"第 {index} 项 status 非法: {status!r}（可选 {', '.join(VALID_STATUSES)}）"
                )
            cleaned.append(Todo(content=content, status=status))
        self._items = cleaned
        return len(cleaned)

    def as_dicts(self) -> list[dict]:
        return [{"content": item.content, "status": item.status} for item in self._items]

    def __len__(self) -> int:
        return len(self._items)
