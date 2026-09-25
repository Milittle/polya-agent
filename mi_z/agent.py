"""Agent 核心循环：模型思考 → 调用工具 → 回填结果 → 直到给出最终答案。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from .tools import Tool, ToolRegistry

logger = logging.getLogger("mi_z.agent")

DEFAULT_SYSTEM_PROMPT = (
    "你是一个可以调用工具来解决问题的助手。"
    "需要外部信息或计算时，优先调用合适的工具，不要凭空猜测。"
    "拿到工具结果后用简洁的中文回答用户。"
)


class Agent:
    def __init__(
        self,
        llm,
        tools: list[Tool] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        approve: Callable[[Tool, dict], bool] | None = None,
    ):
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self.approve = approve
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def run(self, user_input: str) -> str:
        """处理一条用户输入，返回最终答案；对话历史会被保留以便多轮对话。"""
        self.history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": self.system_prompt}, *self.history]
        schemas = self.tools.schemas() or None

        for _ in range(self.max_steps):
            response = self.llm.chat(messages, tools=schemas)
            message = response.choices[0].message

            assistant = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(assistant)
            self.history.append(assistant)

            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                logger.info("调用工具 %s(%s)", name, arguments)

                item = self.tools.get(name)
                if (
                    item is not None
                    and self.approve is not None
                    and not self.approve(item, arguments)
                ):
                    result = f"Error: 用户拒绝了工具调用 {name}"
                else:
                    result = self.tools.call(name, arguments)

                logger.info("工具 %s 返回: %s", name, result)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
                messages.append(tool_message)
                self.history.append(tool_message)

        raise RuntimeError(f"超过最大步数 {self.max_steps}，仍未得到最终答案")
