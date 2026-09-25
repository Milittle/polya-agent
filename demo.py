"""本地编码代理示例：让 agent 用内置工具浏览、阅读、修改代码。

运行前在 .env 里配置好模型（见 .env.example），然后：

    python demo.py

有副作用的工具（write_file / edit_file / run_shell）执行前会要求确认；
在非交互环境（比如管道、CI）下默认拒绝，避免静默执行破坏性命令。
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from mi_z import LLM, Agent, default_tools

load_dotenv()

QUESTION = (
    "先列出当前目录，再读一下 mi_z/agent.py，"
    "然后用三五句话说明这个文件负责什么、核心循环是怎么跑的。"
)


def approve(tool, arguments) -> bool:
    """副作用工具的审批钩子：只读工具直接放行，危险工具先问一句。"""
    if not tool.dangerous:
        return True
    if not sys.stdin.isatty():
        print(f"[非交互环境，默认拒绝] {tool.name}({arguments})")
        return False
    answer = input(f"允许执行 {tool.name}({arguments})? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    agent = Agent(llm=LLM(), tools=default_tools(), approve=approve)
    print(agent.run(QUESTION))


if __name__ == "__main__":
    main()
