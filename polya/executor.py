"""工具执行器：驱动层共用（loop 交互驱动与 run() 内置驱动）。

执行属于驱动层的职责（ADR 0002：``result = yield ToolCall`` 把执行权交给消费
方），因此 bash 实时输出这类「执行期流」由驱动层经 ``default_tools(on_shell_output=)``
注入渲染器——这是常态通道，不是绕过引擎事件的例外（引擎已无事件旁路）。
"""

from __future__ import annotations

import time

from .tools import Tool, run_tool


def execute(tool: Tool, arguments: dict) -> tuple[str, float]:
    """执行一次工具调用，返回 ``(result, duration_s)``。

    异常不外抛，转成 ``Error:`` 文本交回模型（经 :func:`tools.run_tool`，与
    ToolRegistry.call 同一执行点），让模型有机会自行调整重试；审批阻塞的耗时
    由调用方计时，不在这里。
    """
    start = time.monotonic()
    result = run_tool(tool, arguments)
    return result, time.monotonic() - start
