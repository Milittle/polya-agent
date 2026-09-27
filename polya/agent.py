"""Agent 核心循环：模型思考 → 调用工具 → 回填结果 → 直到给出最终答案。

生成器协议（ADR 0002）：:meth:`Agent.steps` 是一个生成器，``yield`` 统一事件
（词表见 :class:`Event` 各子类），``result = yield ToolCall(...)`` 把工具的执行权
与审查交给消费方（驱动层）。:meth:`Agent.run` 是内置驱动——消费生成器 + 审查器
+ 执行器，库用法、``-p`` 模式与测试复用之；需要自定义审查或实时渲染的前端直接
消费 ``steps()``。压缩、状态栏注入、历史管理等上下文管理仍属 agent。
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

from . import session as session_store
from .compact import (
    COMPRESS_MARKER,
    MICRO_MIN_CHARS,
    compact_messages,
    compact_restart,
    effective_keep,
    extract_file_operations,
    microcompact,
)
from .executor import execute
from .i18n import t, tool_text
from .llm import is_context_overflow
from .prompt import SystemPrompt, diff_sections, tool_guidelines, tool_snippets
from .providers import ModelProfile, profile_for
from .review import AllowAllReviewer, Reviewer
from .skills import SkillCatalog
from .status import StatusSnapshot, render_status
from .todos import TodoStore
from .tools import Tool, ToolRegistry, tool
from .tree import (
    KIND_ASSISTANT,
    KIND_SUMMARY,
    KIND_SYSTEM,
    KIND_TOOL,
    KIND_USER,
    SessionTree,
)

logger = logging.getLogger("polya.agent")

DEFAULT_SYSTEM_PROMPT = t("prompt.default")


# ---------- 事件联合类型（词表契约，测试锁定） ----------


@dataclass(frozen=True)
class Event:
    """生成器事件基类：``event`` 是词表名，子类字段名即载荷键（event_payload）。"""

    event: ClassVar[str]


@dataclass(frozen=True)
class Iteration(Event):
    event: ClassVar[str] = "iteration"
    step: int
    max_steps: int


@dataclass(frozen=True)
class BudgetCheckpoint(Event):
    """软检查点：走满 max_steps 轮但模型仍要继续时发出，随后自动续跑。

    ``step`` 是已完成的轮数（max_steps 的整数倍），``continuation`` 是第几次续跑
    （1 起）。检查点不是错误——驱动层只做 dim 提示，回合不结束。
    """

    event: ClassVar[str] = "budget_checkpoint"
    step: int
    limit: int
    continuation: int


@dataclass(frozen=True)
class BudgetExhausted(Event):
    """连跳 max_continuations 次后仍要继续：本轮收尾，历史保留，可再发消息继续。"""

    event: ClassVar[str] = "budget_exhausted"
    step: int
    limit: int
    continuations: int


@dataclass(frozen=True)
class NoProgress(Event):
    """无进展熔断（票 06）：同一工具同参数连续重复到达阈值。

    ``phase="nudged"`` 是第 limit 次，劝告已拼进工具结果，回合继续；
    ``phase="stopped"`` 是再犯，收尾（可续）。
    """

    event: ClassVar[str] = "no_progress"
    tool: str
    arguments: dict
    count: int
    phase: str


@dataclass(frozen=True)
class ReasoningDelta(Event):
    event: ClassVar[str] = "reasoning_delta"
    delta: str


@dataclass(frozen=True)
class Text(Event):
    event: ClassVar[str] = "text_delta"
    delta: str


@dataclass(frozen=True)
class Usage(Event):
    event: ClassVar[str] = "usage"
    last: dict
    total: dict


@dataclass(frozen=True)
class AssistantMessage(Event):
    event: ClassVar[str] = "assistant_message"
    content: str
    reasoning: str | None
    tool_calls: list


@dataclass(frozen=True)
class ToolCall(Event):
    """请求工具执行：驱动层判定 / 审批 / 执行后 ``send`` 回结果字符串。"""

    event: ClassVar[str] = "tool_call"
    name: str
    call_id: str = ""
    arguments: dict = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Compaction(Event):
    event: ClassVar[str] = "compaction"
    before: int
    after: int
    mode: str = "full"  # "full"=LLM 摘要压缩；"micro"=无 LLM 的指针折叠
    cleared: int = 0  # 微压缩折叠的条数（全量压缩为 0）


@dataclass(frozen=True)
class PlanSubmitted(Event):
    """规划模式提交计划：驱动层展示并结束本轮（批准由调用方翻转 plan_mode）。"""

    event: ClassVar[str] = "plan_submitted"
    plan: str


def event_payload(ev: Event) -> dict:
    """事件对象 → 载荷 dict（字段名即键，与渲染器词表同名同键）。"""
    return {f.name: getattr(ev, f.name) for f in fields(ev)}


# ---------- 驱动 ----------


def drive(gen, handle: Callable[[Event], str | None]) -> str:
    """驱动一个 ``steps()`` 生成器：事件交 ``handle``，其返回值成为下一个
    ``send``（ToolCall / PlanSubmitted 期待结果字符串，其余事件返回 None）。"""
    to_send = None
    while True:
        try:
            ev = gen.send(to_send)
        except StopIteration as stop:
            return stop.value
        to_send = handle(ev)


class Agent:
    def __init__(
        self,
        llm,
        tools: list[Tool] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        max_continuations: int = 0,
        loop_guard: bool = False,
        loop_repeat_limit: int = 3,
        status_bar: bool | Callable[[StatusSnapshot], str] | None = None,
        todos: TodoStore | None = None,
        plan_mode: bool = False,
        plan_capable: bool = False,
        reviewer: Reviewer | None = None,
        compress: bool = False,
        context_window: int | None = None,
        compress_threshold: float | None = None,
        keep_recent: int = 30,
        keep_recent_tokens: int | None = None,
        micro_threshold: float | None = 0.6,
        micro_min_chars: int = MICRO_MIN_CHARS,
        reserve_tokens: int = 16384,
        profile: ModelProfile | None = None,
        prefix_check: bool = False,
        stream: bool = True,
        skills: SkillCatalog | None = None,
        project_memory: str | None = None,
        cwd: str | None = None,
    ):
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self._base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.project_memory = project_memory
        self.cwd = str(cwd) if cwd is not None else None
        self.skills = skills
        if skills is not None:
            self.tools.add(skills.tool())
        if compress:
            self.tools.add(self._history_read_tool())
        self.max_steps = max_steps
        # 软检查点：max_steps>0 时每走满 max_steps 轮发一次 BudgetCheckpoint 并自动
        # 续跑，最多连跳 max_continuations 次后以 BudgetExhausted 收尾。max_steps==0
        # 表示无界（pi 语义）。库默认 0 连跳 = 有界硬停（不抛异常，返回提示串）。
        self.max_continuations = max_continuations
        # 无进展熔断（票 06）：同一工具同参数连续重复的护栏。库默认关（与 compress
        # 同款，库保守、CLI 开启）；loop_repeat_limit 次先 nudge，再犯即可续收尾。
        self.loop_guard = loop_guard
        self.loop_repeat_limit = loop_repeat_limit
        self.last_run_exhausted = False  # 本次 run 是否因预算/熔断收尾（驱动判定未完成）
        self._last_signature: str | None = None  # 上一个工具调用签名（无进展检测）
        self._streak = 0  # 相同签名连续出现次数
        self._nudged = False  # 本 streak 是否已 nudge 过
        # 审查器缝（review.py）：默认放行，plan 只读约束在 AllowAllReviewer 内。
        # 未来模型审查器（Jev 类）实现同一协议即可接入。
        self.reviewer: Reviewer = reviewer or AllowAllReviewer()
        # 状态栏渲染器：True 用默认渲染，callable 自定义，None/False 关闭。
        # 状态以 user 消息追加在上下文末尾（书 2.6），绝不修改已有消息。
        self.status_bar = render_status if status_bar is True else status_bar or None
        # 状态栏去重键：实质字段（工具计数/TODO/模式）不变时跳过追加——截断续跑等
        # 「无工具调用的空转轮」不再重复塞入相同的状态消息。
        self._last_status_key: tuple | None = None
        # 两阶段模式：plan_mode 是驱动层拥有的运行时状态（Q12——agent 的控制流
        # 不再分支于它；decide() 判定、状态栏显示、驱动层翻转），构造时给定初值。
        # plan_capable 只控制 exit_plan_mode 工具的注册（构造时一次）——两者解耦，
        # CLI 可以随时切换模式而不动工具数组（缓存纪律）。
        self.plan_mode = plan_mode
        if plan_mode or plan_capable:
            self._register_exit_plan_mode()
        # TODO 存储：todo_write 工具写入（default_tools(todos=...) 接同一个实例），
        # 状态栏每轮把它渲染到上下文末尾——外部记忆，不靠模型回忆。
        self.todos = todos if todos is not None else TodoStore()
        self.tree = SessionTree()
        # 会话身份（session-lifecycle 票 01）：当前会话名 + 元数据。autosave/`/save`
        # 写头，`/resume`/`/fork`/`/clone` 读改。
        self.session_name: str | None = None
        self.session_title: str | None = None
        self.session_created: str = ""
        self.session_updated: str = ""
        # 项目信任态（CLI 门控的元信息，供 /trust 状态查询；引擎本身不读）。
        self.trusted: bool = True
        # token 用量统计：只做记录，不进消息历史（保持前缀字节稳定）
        self.last_usage: dict | None = None
        self.total_usage: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        self.tool_counts: Counter[str] = Counter()
        # 上下文压缩（书 2.7）：80% 阈值触发、批量压 tool 结果。压缩策略由
        # 模型能力决定（providers.ModelProfile）：支持原地替换 tool content 的
        # 用 compact_messages；thinking 签名绑定前缀的用 compact_restart
        # （整段历史压成一条摘要，从摘要冷启动）。触发判据是**最近一次调用**
        # 的 prompt_tokens——绝不能用 total_usage：累计 counter 每轮重复计入
        # 共享前缀，随轮数二次增长，会过早触发。连续 3 次失败熔断（书 5 章）。
        self.profile = profile or ModelProfile()
        self.compress = compress
        self.context_window = (
            context_window if context_window is not None else self.profile.context_window
        )
        self.compress_threshold = (
            compress_threshold
            if compress_threshold is not None
            else self.profile.compress_threshold
        )
        self.keep_recent = keep_recent
        self.keep_recent_tokens = keep_recent_tokens
        self.micro_threshold = micro_threshold
        self.micro_min_chars = micro_min_chars
        # 全量压缩的绝对预留（对齐 pi 的 reserveTokens）：小窗口下先于百分比
        # 阈值触发，保证触发时至少留出一轮响应的空间（见 _full_trigger）。
        self.reserve_tokens = reserve_tokens
        self._compress_failures = 0
        self._request_size = 0
        # 压缩累积追踪：跨多次压缩记住读过/改过的文件，摘要后仍可引用。
        self._read_files: set[str] = set()
        self._modified_files: set[str] = set()
        self._last_summary: str | None = None
        # 估算缓存：system_prompt + 工具 schema 是静态的，只算一次；
        # 历史按条命中缓存（压缩产出新对象，天然失效）。
        self._static_size: int | None = None
        self._size_cache: dict[int, int] = {}
        self._size_cache_len = -1
        self.system_prompt = self._build_system_prompt()
        self.tree.reset_with_system(self.system_prompt, self._prompt.sections())
        # 运行时前缀不变量（可选）：两次压缩点之间请求序列必须严格 append-only。
        # 破坏前缀对纯 KV Cache 只是缓存变贵（2.3）；对回传 thinking 的模型是
        # 推理连续性断裂（2.7）——所以提供可开启的运行时断言。
        self.prefix_check = prefix_check
        self._last_prefix: list[dict] | None = None
        # 流式开关：steps() 有 chat_iter（迭代器流式）就走流式让 UI 逐段渲染；
        # --no-stream 逃生口给不支持流式的端点。
        self.stream = stream
        # 子代理渲染/绑定槽（ADR 0004）：驱动层 attach 时挂上 SubagentRunner；
        # agent 不依赖 subagent 模块，避免循环导入。
        self.subagent: object | None = None

    @property
    def history(self) -> list[dict]:
        """会话消息（不含 system 入口）的兼容视图：从树确定性投影。

        只读——驱动层注入消息用 :meth:`append_user_message`，压缩用树方法。
        """
        return self.tree.conversation()

    def append_user_message(self, content: str) -> None:
        """驱动层在迭代边界注入一条 user 消息（shell 输出 / 排队输入）。"""
        self.tree.append(KIND_USER, {"content": content})

    def _build_system_prompt(self) -> str:
        """具名 section 装配系统提示词（prompt.SystemPrompt）。

        preamble 是基础指令（编码工作流），工具贡献 ``<tools>`` / ``<rules>`` 两段，
        技能目录、项目记忆、工作目录各占一段。section 化让「工具描述 / 工具指引 /
        技能目录」三面分离，且渲染确定——同样输入永远同样字节，前缀不变量成立。
        """
        prompt = SystemPrompt(self._base_prompt)
        prompt.set("tools", tool_snippets(self.tools))
        prompt.set("rules", tool_guidelines(self.tools))
        if self.skills is not None:
            prompt.set("skills", self.skills.prompt().strip())
        if self.project_memory:
            prompt.set("project_memory", self.project_memory)
        if self.cwd:
            prompt.set("cwd", self.cwd.replace("\\", "/"))
        self._prompt = prompt
        return prompt.render()

    def reload_skills(self) -> str:
        """热加载技能目录：重扫 + 追加 patch ``<skills>`` section 的 system 入口。

        合法重启点：前缀基线 / usage 校准 / 尺寸缓存全部重置，投影一次有界 miss。
        """
        if self.skills is None:
            return "未启用技能目录。"
        previous = self._prompt.sections()
        self.skills.reload()
        self._prompt.set("skills", self.skills.prompt().strip())
        patch = diff_sections(previous, self._prompt.sections())
        if not patch:
            return "技能无变化。"
        self.tree.append(KIND_SYSTEM, {"sections": patch})
        self.system_prompt = self._prompt.render()
        self._reset_restart_point()
        names = "、".join(sorted(self.skills.skills)) or "（无）"
        return f"已重载技能：{names}"

    def rewind(self, steps: int = 1) -> str:
        """回退当前分支 steps 步（合法重启点），后续输入从此分叉。"""
        entry = self.tree.rewind(steps)
        self._reset_restart_point()
        return f"已回退到入口 #{entry.id}（{entry.kind}）；后续输入将从此分叉。"

    def jump(self, entry_id: int) -> str:
        """跳到任意历史入口（合法重启点），后续输入从此分叉。"""
        entry = self.tree.move_to(entry_id)
        self._reset_restart_point()
        return f"已跳到入口 #{entry.id}（{entry.kind}）。"

    def branch_overview(self) -> str:
        """整棵树概览：分叉点 + 所有分支（当前分支标 →），供 /tree 导航。"""
        lines: list[str] = []
        forks = self.tree.fork_points()
        if forks:
            lines.append("分叉点：" + "、".join(f"#{e.id}({e.kind})" for e in forks))
        for index, branch in enumerate(self.tree.branches(), 1):
            leaf = branch[-1]
            active = leaf.id == self.tree.active_id
            marker = "→" if active else " "
            summary = str(leaf.payload.get("content") or "").replace("\n", " ")[:40]
            lines.append(
                f"{marker} 分支{index}：叶 #{leaf.id} {leaf.kind} · {summary} · {len(branch)} 入口"
            )
        return "\n".join(lines) or "（空树）"

    def edit_entry(self, entry_id: int, new_content: str) -> str:
        """在投影里替换某入口内容（原文保留，可 history_read 回查）。"""
        entry = self.tree.get(entry_id)
        if entry.kind == "system":
            return "不能编辑 system 入口。"
        self.tree.override(entry.id, {**entry.payload, "content": new_content})
        self._reset_restart_point()
        return f"已编辑入口 #{entry.id}（投影）；原文仍可 history_read(entry_id={entry.id}) 回查。"

    def remove_entry(self, entry_id: int) -> str:
        """从投影移除某入口（原文保留，可 history_read 回查）。"""
        entry = self.tree.get(entry_id)
        if entry.kind == "system":
            return "不能移除 system 入口。"
        self.tree.override(entry.id, None)
        self._reset_restart_point()
        return f"已从投影移除入口 #{entry.id}；原文仍可 history_read(entry_id={entry.id}) 回查。"

    def _reset_restart_point(self) -> None:
        """合法重启点共用的状态重置（回退 / 跳转 / 换模型 / skills 热加载 / 恢复）。"""
        self._last_prefix = None
        self.last_usage = None
        self._request_size = 0
        self._reset_size_cache()

    # ---------- 会话身份与持久化（session-lifecycle 票 01） ----------

    def session_meta(self) -> session_store.SessionMeta:
        """当前会话元数据（落盘 / 导出用）。"""
        return session_store.SessionMeta(
            name=self.session_name or "",
            title=self.session_title,
            cwd=self.cwd or "",
            created=self.session_created,
            updated=self.session_updated,
        )

    def set_session_title(self, title: str) -> None:
        self.session_title = title
        self.session_updated = session_store.iso_now()

    def _ensure_session_name(self) -> str:
        """无名字时生成自动名（不重置会话），供 autosave / 导出用。"""
        if self.session_name is None:
            self.session_name = session_store.unique_name()
            self.session_created = self.session_created or session_store.iso_now()
        return self.session_name

    def save_session(self, name: str | None = None) -> str:
        """把会话树与元数据持久化到 ``~/.polya/sessions/<name>.jsonl``。"""
        if name is None:
            name = self._ensure_session_name()
        name = name.strip()
        if not session_store.valid_name(name):
            return "会话名不能含空白或路径分隔符。"
        self.session_name = name
        meta = session_store.SessionMeta(
            name=name,
            title=self.session_title,
            cwd=self.cwd or "",
            created=self.session_created or session_store.iso_now(),
        )
        path = session_store.write(meta, self.tree.to_jsonl())
        self.session_created, self.session_updated = meta.created, meta.updated
        return f"已保存会话 {name} 到 {path}"

    def autosave(self) -> None:
        """任务收尾静默落盘：有内容才写；无名字则生成。失败不打断会话。"""
        if not self.history:
            return
        self._ensure_session_name()
        try:
            self.save_session()
        except OSError:
            pass

    def load_session(self, name: str) -> str:
        """从 ``~/.polya/sessions/<name>.jsonl`` 恢复会话（合法重启点）。"""
        result = session_store.read(name)
        if result is None:
            return f"无法读取会话 {name}：文件不存在或格式错误。"
        meta, entry_lines = result
        try:
            self.tree = SessionTree.from_jsonl(entry_lines)
        except ValueError as exc:
            return f"无法读取会话 {name}：{exc}"
        projected = self.tree.project()
        if projected and projected[0]["role"] == "system":
            self.system_prompt = projected[0]["content"]
        self.session_name = meta.name
        self.session_title = meta.title
        self.session_created = meta.created
        self.session_updated = meta.updated
        self._adopt_session()
        return f"已恢复会话 {name}（{len(self.tree)} 个入口）"

    def new_session(self, name: str | None = None) -> str:
        """开新会话：清树与派生状态，分配新名字（不落盘，首次 autosave 写）。"""
        self.reset()
        self.session_name = name.strip() if name else session_store.unique_name()
        self.session_title = None
        self.session_created = session_store.iso_now()
        self.session_updated = self.session_created
        return self.session_name

    def fork_session(self, entry_id: int, name: str | None = None) -> str:
        """从根到入口 entry_id 的祖先路径派生新会话（合法重启点）。"""
        try:
            forked = self.tree.copy_branch_upto(entry_id)
        except ValueError as exc:
            return f"无法分叉：{exc}"
        self.tree = forked
        self.session_name = name.strip() if name else session_store.unique_name()
        self.session_title = None
        self.session_created = session_store.iso_now()
        self.session_updated = self.session_created
        self._adopt_session()
        self.autosave()  # 分叉结果立刻可见于 /resume
        return f"已从入口 #{entry_id} 分叉出新会话 {self.session_name}（{len(self.tree)} 个入口）。"

    def clone_session(self, name: str | None = None) -> str:
        """复制整棵当前会话（含投影覆盖）为新会话，主题沿用。"""
        self.tree = self.tree.copy()
        self.session_name = name.strip() if name else session_store.unique_name()
        self.session_created = session_store.iso_now()
        self.session_updated = self.session_created
        self._adopt_session()
        self.autosave()  # 克隆结果立刻可见于 /resume
        return f"已复制当前会话为 {self.session_name}（{len(self.tree)} 个入口）。"

    def render_transcript(self) -> str:
        """按 active 分支渲染 Markdown 转录（走投影覆盖，反映 /edit）。"""
        lines = [f"# 会话 {self.session_name or '(未命名)'}", ""]
        if self.session_title:
            lines.append(f"> 主题：{self.session_title}")
        if self.cwd:
            lines.append(f"> 工作目录：{self.cwd}")
        lines += [f"> 入口数：{len(self.tree)}", ""]
        for entry in self.tree.active_branch():
            if entry.kind == KIND_SYSTEM:
                continue
            payload = self.tree.effective_payload(entry.id)
            if payload is None:
                continue
            if entry.kind == KIND_USER:
                lines += ["## 用户", "", str(payload.get("content", "")), ""]
            elif entry.kind == KIND_ASSISTANT:
                lines.append("## 助手")
                lines.append("")
                if payload.get("reasoning_content"):
                    lines += [
                        "<details><summary>推理</summary>",
                        "",
                        str(payload["reasoning_content"]),
                        "",
                        "</details>",
                        "",
                    ]
                lines += [str(payload.get("content") or "（仅工具调用）"), ""]
                for call in payload.get("tool_calls") or []:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    name = function.get("name", "?")
                    arguments = function.get("arguments", "")
                    lines.append(f"- 工具调用 `{name}`：`{arguments}`")
                if payload.get("tool_calls"):
                    lines.append("")
            elif entry.kind == KIND_TOOL:
                lines += ["## 工具结果", "", "```", str(payload.get("content", "")), "```", ""]
            elif entry.kind == KIND_SUMMARY:
                lines += ["## 摘要", "", str(payload.get("content", "")), ""]
        return "\n".join(lines)

    def export_session(self, target: str | None = None) -> str:
        """导出会话：``.jsonl`` 落原始会话（可被 /import、/load 读），否则 Markdown。

        默认 ``~/.polya/exports/<name>.md``。
        """
        name = self._ensure_session_name()
        path = Path(target).expanduser() if target else session_store.exports_dir() / f"{name}.md"
        if path.suffix == ".jsonl":
            lines = [
                json.dumps(self.session_meta().header(), ensure_ascii=False),
                *self.tree.to_jsonl(),
            ]
            content = "\n".join(lines) + "\n"
        else:
            content = self.render_transcript()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"无法导出：{exc}"
        return f"已导出会话到 {path}"

    def import_session(self, path: str) -> str:
        """从任意路径导入会话（polya JSONL / pi 格式）为新会话（session-lifecycle）。"""
        from . import importer  # 局部导入：仅导入路径需要，避免启动期开销

        target = Path(path).expanduser()
        payload = {"content": self.system_prompt, "sections": self._prompt.sections()}
        try:
            meta, tree = importer.import_file(target, system_payload=payload)
        except OSError as exc:
            return f"无法导入 {path}：{exc}"
        except ValueError as exc:
            return f"无法导入 {path}：{exc}"
        if len(tree) == 0:
            return f"无法导入 {path}：没有任何可用的会话入口。"
        self.tree = tree
        projected = self.tree.project()
        if projected and projected[0]["role"] == "system":
            self.system_prompt = projected[0]["content"]
        base = target.stem if session_store.valid_name(target.stem) else None
        self.session_name = session_store.unique_name(base)
        self.session_title = meta.title
        self.session_created = meta.created or session_store.iso_now()
        self.session_updated = self.session_created
        self._adopt_session()
        self.autosave()
        return f"已导入 {target.name} → 新会话 {self.session_name}（{len(self.tree)} 个入口）。"

    def _adopt_session(self) -> None:
        """切换/加载会话后重置派生状态（统计 / TODO / 读改追踪 / 压缩累积）。"""
        self.last_usage = None
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        self.tool_counts.clear()
        self.todos.rewrite([])  # 清单随会话一起重置
        self._last_prefix = None  # 前缀基线随之失效
        self._compress_failures = 0
        self._request_size = 0
        self._read_files.clear()
        self._modified_files.clear()
        self._last_summary = None
        self._reset_size_cache()
        self._last_status_key = None
        if self.skills is not None:
            self.skills.reset()

    def reset(self) -> None:
        self.tree.reset_with_system(self.system_prompt, self._prompt.sections())
        self._adopt_session()

    def switch_model(
        self,
        llm,
        profile: ModelProfile | None = None,
        context_window: int | None = None,
    ) -> None:
        """会话中换模型（/model，票 05）：只在迭代边界调用（驱动层 busy 语义保证）。

        端点实例、能力档案与解析窗口一起换——压缩策略、温度纪律、窗口都按新模型走
        （providers 按名匹配，无需厂商特判）。历史与统计保留，但旧模型的
        ``reasoning_content`` 跨模型不连续，回传陌生端点可能被拒，一律剥离；
        前缀基线随之作废。``context_window`` 由驱动层用解析链（条目发现值 → 静态表
        → 默认）算好后传入；不给则回落能力档案的静态值。
        """
        self.llm = llm
        self.profile = profile or profile_for(getattr(llm, "model", None))
        self.context_window = (
            context_window if context_window is not None else self.profile.context_window
        )
        self.compress_threshold = self.profile.compress_threshold
        self.tree.strip_reasoning()
        self._last_prefix = None
        self.last_usage = None
        self._request_size = 0
        self._compress_failures = 0
        self._reset_size_cache()

    def set_thinking(self, level: str) -> str:
        """设置推理档位（/thinking，一家一策；见 providers.reasoning_params）。"""
        style = getattr(self.llm, "reasoning_style", "none")
        if style == "none":
            return f"当前模型（{self.llm.model}）没有可切换的推理档位。"
        self.llm.thinking_level = level
        self._reset_restart_point()  # 请求参数变更：前缀基线重算（保守）
        return f"推理档位已设为 {level}（{style}）。"

    # ---------- 生成器协议 ----------

    def steps(self, user_input: str):
        """处理一条用户输入的生成器：yield 事件，ToolCall/PlanSubmitted 期待
        ``send`` 回结果字符串；StopIteration.value 是最终答案。"""
        # 工具定义位于上下文前部，首次请求后必须保持字节级不变（KV Cache 前缀
        # 复用的前提），因此在这里冻结注册表，防止运行中途增删工具。
        self.tools.freeze()
        self._truncation_continues = 0
        self.last_run_exhausted = False
        self._last_signature = None
        self._streak = 0
        self._nudged = False
        self.tree.append(KIND_USER, {"content": user_input})
        messages = self.tree.project()
        schemas = self.tools.schemas() or None

        step = 0
        continuations = 0
        while True:
            step += 1
            yield Iteration(step=step, max_steps=self.max_steps)
            # 驱动可在迭代边界追加排队输入。此时上一批 tool_call 已全部回填，
            # 从树重建请求，确保新消息进入本轮且保留前缀不变量。
            messages = self.tree.project()
            full_applied = False
            if self._should_compress():
                compacted = self._try_compress(user_input)
                if compacted is not None:
                    before = self._apply_compaction(compacted)
                    messages = self.tree.project()
                    full_applied = True
                    yield Compaction(mode="full", before=before, after=len(compacted))
            # 微压缩兜底（Q8）：全量未触发但过 0.6 阈值，或全量失败时，用无 LLM 的
            # 指针清理推迟/减轻全量压缩；不受熔断计数限制。
            if not full_applied and self._should_microcompress():
                micro = self._try_microcompress()
                if micro is not None:
                    before, cleared = micro
                    messages = self.tree.project()
                    yield Compaction(
                        mode="micro",
                        before=before,
                        after=len(self.history),
                        cleared=cleared,
                    )

            if self.prefix_check:
                self._check_prefix(messages)
            if self.status_bar is not None:
                # 状态栏：以 user 角色追加在末尾（书 2.6）。持久追加模式——
                # 旧状态留在轨迹里不删改，前缀保持字节稳定。
                # 去重：实质字段（工具计数/TODO/模式）不变时跳过追加，避免空转轮
                # 重复塞入相同状态（时间戳已移除，否则每轮都必然「变化」）。
                key = (
                    tuple(sorted(self.tool_counts.items())),
                    tuple((item["content"], item["status"]) for item in self.todos.as_dicts()),
                    self.plan_mode,
                )
                if key != self._last_status_key:
                    status_content = self.status_bar(
                        StatusSnapshot(
                            iteration=step,
                            max_steps=self.max_steps,
                            tool_calls=dict(self.tool_counts),
                            usage=dict(self.total_usage),
                            todos=self.todos.as_dicts(),
                            plan_mode=self.plan_mode,
                        )
                    )
                    self.tree.append(KIND_USER, {"content": status_content})
                    messages = self.tree.project()
                    self._last_status_key = key

            if self.compress and self._context_size() >= self.context_window * 0.95:
                raise RuntimeError(t("agent.context_limit"))
            self._request_size = self._estimated_size()

            # 超窗恢复（pi 同款语义，每任务一次）：provider 拒绝超长请求时，压缩
            # 释放空间后原地重试；仅未吐出任何片段才允许重试（避免重复文本）。
            # 无可压目标或已恢复过一次 → 原样上抛。
            overflow_recovered = False
            response = None
            for attempt in (0, 1):
                emitted = False
                try:
                    # 流式：迭代器形态（chat_iter）逐段实时 yield；回调式客户端（测试
                    # 假件）片段收齐后统一 yield——事件序列不变，只丢实时性。
                    if not self.stream:
                        response = self.llm.chat(messages, tools=schemas)
                    else:
                        chat_iter = getattr(self.llm, "chat_iter", None)
                        if chat_iter is not None:
                            chunks = chat_iter(messages, schemas)
                            try:
                                while True:
                                    try:
                                        kind, delta = next(chunks)
                                    except StopIteration as stop:
                                        response = stop.value
                                        break
                                    emitted = True
                                    yield (
                                        ReasoningDelta(delta)
                                        if kind == "reasoning"
                                        else Text(delta)
                                    )
                            finally:
                                close = getattr(chunks, "close", None)
                                if close is not None:
                                    close()
                        else:
                            box: list[tuple[str, str]] = []

                            def _tap(kind: str, delta: str, sink: list = box) -> None:
                                sink.append((kind, delta))

                            response = self.llm.chat(messages, tools=schemas, on_delta=_tap)
                            for kind, delta in box:
                                emitted = True
                                yield (
                                    ReasoningDelta(delta) if kind == "reasoning" else Text(delta)
                                )
                    break
                except Exception as exc:
                    if not (
                        attempt == 0
                        and not emitted
                        and not overflow_recovered
                        and self.compress
                        and self._compress_failures < 3
                        and is_context_overflow(exc)
                    ):
                        raise
                    compacted = self._try_compress(user_input)
                    if compacted is not None:
                        before = self._apply_compaction(compacted)
                    elif self._try_microcompress() is not None:
                        before = len(self.history)
                    else:
                        raise
                    overflow_recovered = True
                    messages = self.tree.project()
                    yield Compaction(mode="full", before=before, after=len(self.history))
            # 重试循环成功退出后 response 必已赋值（异常路径都已 raise）。
            assert response is not None
            usage_present = getattr(response, "usage", None) is not None
            self._record_usage(response)
            if usage_present:
                yield Usage(last=dict(self.last_usage or {}), total=dict(self.total_usage))
            message = response.choices[0].message
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            # 输出被长度上限截断（且没有工具调用可继续）——不能当成功返回，
            # 下面会在本轮结束时决定「压缩后继续」或明确报错。
            truncated = finish_reason == "length" and not message.tool_calls

            assistant = {"role": "assistant", "content": message.content}
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning and self.profile.reasoning_passthrough:
                # DeepSeek interleaved thinking 等扩展字段：原样保存并随消息回传
                # （只追加不改写）。回传的 thinking 与产生它的前缀绑定——这是
                # 「两个压缩点之间必须 append-only」的根源。
                assistant["reasoning_content"] = reasoning
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
            self.tree.append(KIND_ASSISTANT, {k: v for k, v in assistant.items() if k != "role"})
            # reasoning 原样给 UI——显示它与是否随历史回传（profile 的
            # reasoning_passthrough）是两回事
            try:
                yield AssistantMessage(
                    content=message.content or "",
                    reasoning=reasoning,
                    tool_calls=[
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                        for call in (message.tool_calls or [])
                    ],
                )

                if not message.tool_calls:
                    if truncated and self._truncation_continues < 2:
                        # 截断恢复：压缩释放空间后让模型从中断处继续（最多 2 次，
                        # 由 max_steps 与计数器共同限幅）。
                        self._truncation_continues += 1
                        if self.compress:
                            compacted = self._try_compress(user_input)
                            if compacted is not None:
                                self._apply_compaction(compacted)
                        self.tree.append(
                            KIND_USER,
                            {"content": t("agent.truncation_continue")},
                        )
                        continue
                    if truncated:
                        raise RuntimeError(t("agent.truncated"))
                    return message.content or ""

                for call in message.tool_calls:
                    name = call.function.name
                    raw_arguments = call.function.arguments
                    parse_error: str | None = None
                    try:
                        arguments = json.loads(raw_arguments or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError("参数必须是 JSON 对象")
                    except (json.JSONDecodeError, ValueError) as exc:
                        arguments = {}
                        parse_error = (
                            f"Error: 工具 {name} 的参数不是合法 JSON 对象（{exc}）。"
                            f"收到的原始参数：{raw_arguments!r}。请修正为合法 JSON 后重新调用。"
                        )
                    if parse_error is None:
                        logger.debug("调用工具 %s(%s)", name, arguments)
                        if name == "exit_plan_mode":
                            # 规划提交交驱动层审批（Q12）；批准与否由回传文本表达
                            verdict = yield PlanSubmitted(plan=str(arguments.get("plan", "")))
                        else:
                            verdict = yield ToolCall(
                                name=name, call_id=call.id, arguments=arguments
                            )
                    else:
                        # 参数解析失败：不执行工具，把明确错误回传模型——它需要
                        # 知道自己的 JSON 坏了，才能修正（静默置空会断掉反馈环）。
                        logger.debug("工具 %s 参数解析失败：%s", name, parse_error)
                        verdict = parse_error
                    result = verdict if isinstance(verdict, str) else "Error: 工具结果缺失"
                    self.tool_counts[name] += 1
                    # 无进展熔断（票 06）：同工具同参数的连续重复检测。坏 JSON、计划
                    # 提交、poll 等待类都重置 streak（合法重复不算打转）。
                    nudge = guard_stop = False
                    if self.loop_guard:
                        tool_item = self.tools.get(name)
                        if (
                            parse_error is not None
                            or name == "exit_plan_mode"
                            or (tool_item is not None and tool_item.poll)
                        ):
                            self._last_signature, self._streak, self._nudged = None, 0, False
                        else:
                            signature = json.dumps(
                                arguments, sort_keys=True, ensure_ascii=False
                            )
                            if signature == self._last_signature:
                                self._streak += 1
                            else:
                                self._last_signature = signature
                                self._streak = 1
                                self._nudged = False
                            if self._streak == self.loop_repeat_limit:
                                self._nudged = True
                                nudge = True
                            elif self._nudged and self._streak > self.loop_repeat_limit:
                                guard_stop = True
                    annotated = result
                    if self.status_bar is not None:
                        # 调用计数标注（书实验 2-9）：显式次数触发模型的模式识别——
                        # 第 3 次失败后主动换路，而不是无限重试。只进 tool 消息，
                        # tool_result 事件发原始结果（UI 不该看到给模型的标注）。
                        annotated = f"（{name} 第 {self.tool_counts[name]} 次调用）\n{result}"
                    if nudge:
                        # 独立于 status_bar：guard 必须始终生效。
                        annotated = (
                            f"{annotated}\n\n"
                            + t("agent.no_progress_nudge", tool=name, count=self._streak)
                        )

                    logger.debug("工具 %s 返回: %.200s", name, result)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": annotated,
                    }
                    messages.append(tool_message)
                    self.tree.append(KIND_TOOL, {"tool_call_id": call.id, "content": annotated})
                    if nudge:
                        yield NoProgress(
                            tool=name,
                            arguments=arguments,
                            count=self._streak,
                            phase="nudged",
                        )
                    if guard_stop:
                        # 本消息可能还声明了其他 tool_call：先回填，保证历史合法。
                        self._backfill_tool_results(messages, message.tool_calls or [])
                        yield NoProgress(
                            tool=name,
                            arguments=arguments,
                            count=self._streak,
                            phase="stopped",
                        )
                        self.last_run_exhausted = True
                        return t(
                            "agent.no_progress_stopped", tool=name, count=self._streak
                        )
            except (KeyboardInterrupt, GeneratorExit):
                # 中断可能落在工具序列中间：assistant 已声明 N 个 tool_call，
                # 只回填一部分的话，下一轮请求的序列残缺会被 API 拒绝（每个
                # tool_call_id 都必须有对应的 tool 消息）。给尚未回填的调用
                # 补上中断结果，让历史保持合法，再向上传播中断。
                self._backfill_tool_results(messages, message.tool_calls or [])
                raise

            # 软检查点（while 内、中断回填的 try 之外）：一轮工具回填完毕后判定。
            # 无界模式（max_steps==0）跳过；到点先发检查点并原地续跑，连跳上限后
            # 发 BudgetExhausted 收尾——不再是 raise，历史完整保留。
            if self.max_steps > 0 and step % self.max_steps == 0:
                if continuations < self.max_continuations:
                    continuations += 1
                    yield BudgetCheckpoint(
                        step=step, limit=self.max_steps, continuation=continuations
                    )
                else:
                    yield BudgetExhausted(
                        step=step, limit=self.max_steps, continuations=continuations
                    )
                    self.last_run_exhausted = True
                    return t("agent.budget_exhausted")

    # ---------- 内置驱动 ----------

    def run(self, user_input: str) -> str:
        """内置驱动：消费 steps() + 审查器 + 执行器（库 / -p / 测试）。

        流式片段照常 yield，但内置驱动不渲染、安静忽略。计划提交即本轮结束，
        计划文本作为返回值（无交互，展示即结束）。"""

        gen = self.steps(user_input)
        to_send: str | None = None
        while True:
            try:
                ev = gen.send(to_send)
            except StopIteration as stop:
                return stop.value
            if isinstance(ev, PlanSubmitted):
                self.leave_plan_mode()
                return self._end_turn_after_plan(gen, ev.plan)
            if isinstance(ev, ToolCall):
                to_send = self._builtin_tool(ev)
            else:
                to_send = None

    @staticmethod
    def _end_turn_after_plan(gen, plan: str) -> str:
        """回填计划结果并结束本轮：先 send 让工具结果落历史，再 close。"""
        try:
            gen.send("计划已展示；本轮结束，等待用户指示。")
        except StopIteration:
            pass
        finally:
            gen.close()
        return plan

    def leave_plan_mode(self) -> None:
        """驱动层翻转规划模式（Q12）：批准或内置驱动展示计划后调用。"""
        self.plan_mode = False

    def _builtin_tool(self, ev: ToolCall) -> str:
        item = self.tools.get(ev.name)
        if item is None:
            return f"Error: unknown tool '{ev.name}'"
        outcome = self.reviewer.review(item, ev.arguments, plan=self.plan_mode)
        if outcome.verdict == "deny":
            return outcome.reason
        result, _ = execute(item, ev.arguments)
        return result

    # ---------- 内部 ----------

    def _check_prefix(self, messages: list[dict]) -> None:
        """前缀不变量的运行时断言：本次请求必须是上一次的严格扩展。

        基线用深拷贝保存（浅引用会与被改写的消息同源，检测不到篡改）；
        比较用 ==。开销随上下文增长（每轮一次全量内容比较），因此默认关闭，
        需要抓前缀回归时开启。压缩/重启/reset 后基线清空——压缩点是合法的
        推理重启点（checkpoint），不变量只在两个压缩点之间成立。
        """
        if self._last_prefix is None:
            self._last_prefix = copy.deepcopy(messages)
            return
        shared = len(self._last_prefix)
        if len(messages) < shared or messages[:shared] != self._last_prefix:
            raise RuntimeError(
                "前缀不变量被破坏：本次请求不是上一次的严格扩展。两次压缩点之间"
                "历史必须 append-only——改写旧消息会破坏 KV Cache 前缀（2.3），"
                "并使回传的 thinking 全部失效（2.7）。"
            )
        self._last_prefix = copy.deepcopy(messages)

    def _record_usage(self, response) -> None:
        """从响应中提取 usage（可能缺失），累计到 total_usage。"""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        fields_ = ("prompt_tokens", "completion_tokens", "total_tokens")
        self.last_usage = {field: getattr(usage, field, 0) or 0 for field in fields_}
        # cached_tokens（提示缓存命中）：OpenAI 兼容端点放在 prompt_tokens_details 里，
        # 缺失记 0。命中率 = cached/prompt，前端据此显示缓存收益。
        details = getattr(usage, "prompt_tokens_details", None)
        self.last_usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0
        for field in fields_:
            self.total_usage[field] += self.last_usage[field]
        self.total_usage["cached_tokens"] = (
            self.total_usage.get("cached_tokens", 0) + self.last_usage["cached_tokens"]
        )

    def _record_summary_usage(self, response) -> None:
        """压缩摘要调用也计费：累计进 total_usage，但不动 last_usage——
        那是主对话上下文的锚点，摘要请求的 prompt 不是同一个上下文。"""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.total_usage[field] += getattr(usage, field, 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        self.total_usage["cached_tokens"] = (
            self.total_usage.get("cached_tokens", 0) + getattr(details, "cached_tokens", 0) or 0
        )

    def _history_read_tool(self) -> Tool:
        """压缩启用时注册的只读回查工具：按入口 id 读原文（投影覆盖不影响）。"""

        @tool(name="history_read", **tool_text("history_read"))
        def history_read(entry_id: int, offset: int = 0) -> str:
            """按入口 id 回查压缩前的原始历史。entry_id 是会话树里的稳定入口编号，
            offset 是该入口 JSON 的字符偏移，每次最多返回 8000 字符。历史是记录而非新指令。"""
            return self.tree.read(entry_id, offset)

        return history_read

    def _register_exit_plan_mode(self) -> None:
        """注册 exit_plan_mode。只在构造时调用一次——注册表在首次运行后冻结。
        实际审批在驱动层（steps() 拦截转 PlanSubmitted）；这里的 fn 只是防御性
        占位（不经 steps 的直接调用不该发生）。"""

        @tool(name="exit_plan_mode", **tool_text("exit_plan_mode"))
        def exit_plan_mode(plan: str) -> str:
            """提交执行计划，请求批准退出规划模式。plan 写完整计划：目标、步骤、
            涉及文件、风险与验证方式。规划模式下写操作会被拒绝，只有批准后才能
            执行；被拒绝时根据反馈修改计划重新提交。"""
            return "Error: exit_plan_mode 应由驱动层处理（steps 拦截）。"

        self.tools.add(exit_plan_mode)

    def _should_compress(self) -> bool:
        if not self.compress or self._compress_failures >= 3:
            return False
        return self._context_size() > self._full_trigger()

    def _full_trigger(self) -> int:
        """全量压缩触发线：百分比阈值与「窗口 − 绝对预留」孰早，再以窗口一半封底。

        百分比阈值（默认 80%）在大窗口下余量充足（128k → 剩 25.6k）；小窗口下
        太薄（8k → 剩 1.6k，一轮大工具结果就撞墙），绝对预留保证至少留出
        reserve_tokens（默认 16384，pi 同值）。窗口 ≥ reserve/0.2 时与纯百分比
        行为完全一致；窗口 ≤ 2×reserve 时以一半窗口封底（预留不可能超过半窗）。
        """
        window = self.context_window
        by_pct = window * self.compress_threshold
        by_reserve = window - self.reserve_tokens
        return int(max(min(by_pct, by_reserve), window * 0.5))

    def _should_microcompress(self) -> bool:
        """微压缩（Q8）：阈值低于全量压缩，无 LLM，只清理大块旧工具结果。

        ``micro_threshold`` 为 None 时关闭；thinking 绑定前缀的档案跳过（原地改写
        会使保留区的 reasoning 失效）。不受熔断计数限制（无 LLM，不会失败）。
        """
        if not self.compress or self.micro_threshold is None:
            return False
        if not self.profile.supports_inplace_tool_edit:
            return False
        if self.micro_min_chars <= 0:
            return False
        return self._context_size() > self.context_window * self.micro_threshold

    def _try_microcompress(self) -> tuple[int, int] | None:
        """无 LLM 的微压缩：大块旧工具结果换成 ``history_read`` 回查指针。

        返回 ``(压缩前条数, 清理字符数)``；无候选返回 None。压缩点是合法重启点：
        前缀基线、usage 校准与估算缓存全部重置（与全量压缩同规）。
        """
        keep = effective_keep(self.tree.conversation(), self.keep_recent, self.keep_recent_tokens)
        result = microcompact(self.tree, keep, self.micro_min_chars)
        if result is None:
            return None
        before, cleared = result
        self._last_prefix = None
        self.last_usage = None
        self._request_size = 0
        self._reset_size_cache()
        logger.info("微压缩：清理旧工具结果约 %d 字符", cleared)
        return before, cleared

    def _reset_size_cache(self) -> None:
        """丢弃逐条尺寸缓存——压缩 / 换模型 / 重置后历史对象已换，旧缓存失效。

        全量压缩与微压缩共用此入口，避免「压缩后恰好同长而沿用陈旧尺寸」的不对称。
        """
        self._size_cache.clear()
        self._size_cache_len = -1

    def _estimated_size(self) -> int:
        # 非 tokenizer 精确计数：UTF-8 / 3 估算，配合服务器 usage 校准增量。
        # system_prompt + 工具 schema 静态，只算一次；历史按条缓存（压缩产出新对象）。
        if self._static_size is None:
            static = json.dumps([self.system_prompt, self.tools.schemas()], ensure_ascii=False)
            self._static_size = len(static.encode("utf-8")) // 3
        if len(self.history) != self._size_cache_len:
            self._size_cache.clear()
            self._size_cache_len = len(self.history)
        total = self._static_size
        for index, message in enumerate(self.history):
            size = self._size_cache.get(index)
            if size is None:
                size = len(json.dumps(message, ensure_ascii=False).encode("utf-8")) // 3
                self._size_cache[index] = size
            total += size
        return total

    def _context_size(self) -> int:
        estimate = self._estimated_size()
        used = (self.last_usage or {}).get("prompt_tokens", 0)
        if used and self._request_size:
            return max(estimate, used + max(0, estimate - self._request_size))
        return max(estimate, used)

    def _try_compress(self, query: str) -> list[dict] | None:
        """压缩历史（发生在两次 API 调用之间，书 2.7 的时机定义）。

        压缩失败不能拖垮主任务：计数并继续用原历史，连续 3 次后熔断。
        """
        try:
            original_ids = [e.id for e in self.tree.active_branch() if e.kind != KIND_SYSTEM]
            read_files, modified_files = extract_file_operations(self.history)
            self._read_files.update(read_files)
            self._modified_files.update(modified_files)
            file_lines = ""
            if self._read_files:
                file_lines += "已读文件（压缩累积）：" + ", ".join(sorted(self._read_files)) + "\n"
            if self._modified_files:
                file_lines += (
                    "已改文件（压缩累积）：" + ", ".join(sorted(self._modified_files)) + "\n"
                )
            if original_ids:
                readback = (
                    f"压缩前原始历史入口 id：{original_ids[0]}–{original_ids[-1]}，"
                    f"共 {len(original_ids)} 条，需要原文时用 history_read 逐条回查。\n"
                )
            else:
                readback = "（压缩前无会话历史）\n"
            checkpoint = (
                readback
                + f"当前 TODO：{json.dumps(self.todos.as_dicts(), ensure_ascii=False)}\n"
                + file_lines
                + (self.skills.checkpoint() if self.skills is not None else "")
            )
            updates = [
                str(m.get("content") or "")
                for m in self.history
                if m.get("role") == "user"
                and not str(m.get("content") or "").startswith(
                    ("<agent_status>", "<compaction_context>")
                )
            ]
            query += "\n最近用户要求（按时间顺序）：\n" + "\n".join(u[:4000] for u in updates[-6:])
            query += "\n" + checkpoint
            keep = effective_keep(self.history, self.keep_recent, self.keep_recent_tokens)
            previous = self._last_summary
            if self.profile.supports_inplace_tool_edit:
                compacted = compact_messages(
                    self.llm,
                    self.history,
                    keep,
                    query,
                    previous_summary=previous,
                    on_usage=self._record_summary_usage,
                )
                if compacted is None:
                    compacted = compact_restart(
                        self.llm,
                        self.history,
                        keep,
                        query,
                        previous_summary=previous,
                        on_usage=self._record_summary_usage,
                    )
            else:
                compacted = compact_restart(
                    self.llm,
                    self.history,
                    keep,
                    query,
                    previous_summary=previous,
                    on_usage=self._record_summary_usage,
                )
        except Exception:  # noqa: BLE001 - 摘要调用失败不该让任务失败
            self._compress_failures += 1
            logger.warning(
                "上下文压缩失败（连续第 %d 次，达到 3 次后本次 Agent 不再尝试）",
                self._compress_failures,
            )
            return None
        if compacted is None:
            return None
        compacted = [
            m
            for m in compacted
            if not (
                m.get("role") == "user"
                and str(m.get("content") or "").startswith("<compaction_context>")
            )
        ]
        compacted.append(
            {
                "role": "user",
                "content": f"<compaction_context>\n{checkpoint}\n</compaction_context>",
            }
        )
        self._remember_summary(compacted)
        # 使用新的估算基线，不能以压缩前 usage 再次触发相同压缩。
        self.last_usage = None
        self._request_size = 0
        self._compress_failures = 0
        logger.info(
            "上下文已压缩：%d 条消息 → %d 条（旧 tool 结果替换为摘要，旧状态栏已清理）",
            len(self.history),
            len(compacted),
        )
        return compacted

    def _remember_summary(self, compacted: list[dict]) -> None:
        """记下本次压缩产出的摘要文本，供下一次迭代压缩合并（不从头重写）。"""
        pieces = []
        for message in compacted:
            content = str(message.get("content") or "")
            if message.get("role") == "user" and content.startswith("<session_summary>"):
                pieces.append(content)
            elif content.startswith(COMPRESS_MARKER):
                pieces.append(content)
        if pieces:
            self._last_summary = "\n".join(pieces)[:4000]

    def _apply_compaction(self, compacted: list[dict]) -> int:
        """应用全量压缩结果并重置推理基线（steps 与 compact_now 共用）。

        返回压缩前的历史条数。压缩点是合法推理重启点：前缀基线清空，下一轮重新起算。
        """
        before = len(self.history)
        self.tree.replace_conversation(compacted)
        self._last_prefix = None
        self._reset_size_cache()  # 历史对象已换，逐条尺寸缓存随之失效
        return before

    def compact_now(self, instructions: str | None = None) -> str:
        """手动压缩（/compact [instructions]）：无视阈值立即执行。

        返回给用户的结论文案；压缩失败或无可压内容时不抛异常。
        """
        query = instructions or "（用户通过 /compact 手动请求压缩）"
        before = len(self.history)
        if self.tools.get("history_read") is None:
            return t("agent.compact_disabled")
        compacted = self._try_compress(query)
        if compacted is None:
            return t("agent.compact_empty")
        before = self._apply_compaction(compacted)
        return t("agent.compact_done", before=before, after=len(compacted))

    def _backfill_tool_results(self, messages: list[dict], tool_calls) -> None:
        answered = {
            m["tool_call_id"]
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id") in {c.id for c in tool_calls}
        }
        for call in tool_calls:
            if call.id in answered:
                continue
            interrupted = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": t("agent.interrupted"),
            }
            messages.append(interrupted)
            self.tree.append(
                KIND_TOOL,
                {"tool_call_id": call.id, "content": t("agent.interrupted")},
            )
