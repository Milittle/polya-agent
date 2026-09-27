"""子代理（ADR 0004 / context-engineering 票 03）：``task`` 工具引擎级实现。

``SubagentRunner`` 把一个有界的探查/执行子任务交给隔离上下文的子 Agent：独立
history、独立 shell 会话，只有最终报告回父上下文（主上下文只进报告，探查链不占
主窗口）。同步运行、深度 1（子 Agent 工具集不含 ``task``）。

审查不在这里做：``dispatch`` 可注入，引擎级默认走子 Agent 的内置驱动；驱动层
（票 04）换成经共享 :class:`~polya.review.Reviewer` 的版本。引擎不 import 驱动层，
避免循环。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .agent import Agent, Compaction, Iteration, PlanSubmitted, ToolCall
from .builtin import default_tools
from .i18n import t, tool_text
from .todos import TodoStore
from .tools import Tool, tool

logger = logging.getLogger("polya.subagent")

SUBAGENT_SYSTEM_PROMPT = t("prompt.subagent")

# 默认 dispatch：子 Agent 内置驱动（引擎级测试用；驱动层换闸门版）
Dispatch = Callable[[Agent, ToolCall], str]


class SubagentRunner:
    """``task`` 工具的引擎级实现；生命周期由驱动层 attach / bind。"""

    def __init__(
        self,
        *,
        root: str,
        memory: str | None = None,
        renderer=None,
        max_steps: int = 20,
        result_limit: int = 12000,
    ):
        self.root = root
        self.memory = memory
        self.renderer = renderer
        self.max_steps = max_steps
        self.result_limit = result_limit
        # 进度行回调（驱动层注入；进父 task 尾窗）
        self.progress: Callable[[str], None] | None = None
        # 工具调用分派：默认内置驱动；驱动层 bind 后换经闸门的版本
        self.dispatch: Dispatch = lambda child, ev: child._builtin_tool(ev)
        self.parent: Agent | None = None
        self.stop: threading.Event | None = None
        self.interactive = False
        self.last_child: Agent | None = None  # 留档供测试回查子历史 / 压缩

    # ---------- 绑定（构建期 attach，运行期 bind） ----------

    def attach(self, parent: Agent) -> None:
        """cli 在 Agent 构建后调用；挂 ``agent.subagent`` 供驱动层发现。"""
        self.parent = parent
        parent.subagent = self

    def bind(
        self,
        stop: threading.Event | None,
        interactive: bool,
    ) -> None:
        """驱动层注入停止事件与交互标记（票 04）。"""
        self.stop = stop
        self.interactive = interactive

    def task_tool(self) -> Tool:
        """构造 ``task`` 工具（构建期注册，先于 registry.freeze）。"""

        @tool(kind="delegate", **tool_text("task"))
        def task(description: str, prompt: str) -> str:
            """把一个有界的探查子任务交给隔离上下文的子代理，只有最终报告回到当前上下文。
            用于把探查链（多轮 grep / read_file）挡在主上下文之外。description 是简短标签，
            prompt 是完整指令。子代理的工具调用仍逐个经过审批；它没有自己的审批权。"""
            return self.run(description, prompt)

        return task

    # ---------- 运行 ----------

    def _make_child(self) -> Agent:
        parent = self.parent
        assert parent is not None, "SubagentRunner.attach(parent) 未调用"
        return Agent(
            llm=parent.llm,  # 同实例：同一 worker 线程串行，安全
            tools=default_tools(self.root, todos=TodoStore()),  # 无 task ⇒ 深度 1
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
            project_memory=self.memory,  # 子系统提示词含 AGENTS.md 项目记忆
            cwd=self.root,
            max_steps=self.max_steps,
            status_bar=None,
            plan_mode=parent.plan_mode,  # 子 Context 继承父规划模式
            plan_capable=False,  # 子无 exit_plan_mode
            reviewer=parent.reviewer,  # 共享父会话审查器
            compress=True,  # 子代理总是启用压缩（ticket 03）
            context_window=parent.context_window,
            compress_threshold=parent.compress_threshold,
            keep_recent=parent.keep_recent,
            keep_recent_tokens=parent.keep_recent_tokens,
            micro_threshold=parent.micro_threshold,
            micro_min_chars=parent.micro_min_chars,
            profile=parent.profile,
            stream=parent.stream,
        )

    def run(self, description: str, prompt: str) -> str:
        """同步驱动一个子 Agent，返回最终报告（含用量尾注）。"""
        parent = self.parent
        assert parent is not None, "SubagentRunner.attach(parent) 未调用"
        if self.progress:
            self.progress(f"子任务：{description}")
        child = self._make_child()
        self.last_child = child
        gen = child.steps(prompt)
        to_send: str | None = None
        rounds = 0
        interrupted = False
        report = ""
        try:
            while True:
                if self.stop is not None and self.stop.is_set():
                    interrupted = True
                    break
                try:
                    ev = gen.send(to_send)
                except StopIteration as stop:
                    report = stop.value or ""
                    break
                if isinstance(ev, ToolCall):
                    # 进度行由 dispatch 侧（驱动层 run_tool_call）输出，避免重复
                    to_send = self.dispatch(child, ev)
                elif isinstance(ev, PlanSubmitted):
                    # 子无 plan_capable，正常不会走到；防御性回绝
                    to_send = "Error: 子任务不含计划审批；把需要的变更写进最终报告。"
                elif isinstance(ev, Iteration):
                    rounds = ev.step
                    to_send = None
                elif isinstance(ev, Compaction):
                    if self.progress:
                        self.progress(f"（子任务压缩：{ev.before} → {ev.after} 条）")
                    to_send = None
                else:
                    to_send = None  # 吞掉流式片段与其余事件
        except RuntimeError as exc:  # 子超步数等
            report = f"Error: 子代理未能完成（{exc}）"
        finally:
            gen.close()  # GeneratorExit 回填保证子历史合法
        self._merge_usage(parent, child)
        if interrupted:
            head = "（子任务已被用户中断；以下为截至中断的进展）"
            report = f"{head}\n{report}" if report else head
        report = report or "(子代理未返回结果)"
        if len(report) > self.result_limit:
            report = report[: self.result_limit] + f"\n…[已截断，共 {len(report)} 字符]"
        usage = child.total_usage
        report += (
            f"\n（子任务 {rounds} 轮 · prompt {usage.get('prompt_tokens', 0)}"
            f" · completion {usage.get('completion_tokens', 0)} token）"
        )
        return report

    @staticmethod
    def _merge_usage(parent: Agent, child: Agent) -> None:
        """子用量并入父 total_usage（Q7）；父 last_usage 不动，压缩触发不受污染。"""
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            parent.total_usage[field] = parent.total_usage.get(field, 0) + child.total_usage.get(
                field, 0
            )
