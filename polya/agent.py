"""Agent 核心循环：模型思考 → 调用工具 → 回填结果 → 直到给出最终答案。

生成器协议（ADR 0002）：:meth:`Agent.steps` 是一个生成器，``yield`` 统一事件
（词表见 :class:`Event` 各子类），``result = yield ToolCall(...)`` 把工具的执行权
与审批交给消费方（驱动层）。:meth:`Agent.run` 是内置驱动——消费生成器 + approve
审批策略 + 执行器，库用法、``-p`` 模式与测试复用之；需要自定义审批或实时渲染的
前端直接消费 ``steps()``。压缩、状态栏注入、历史管理等上下文管理仍属 agent。
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import ClassVar

from .compact import (
    COMPRESS_MARKER,
    MICRO_MIN_CHARS,
    clearable_indices,
    compact_messages,
    compact_restart,
    effective_keep,
    extract_file_operations,
    microcompact,
)
from .executor import execute
from .history import HistoryArchive
from .i18n import t, tool_text
from .permissions import Context, decide
from .prompt import SystemPrompt, tool_guidelines, tool_snippets
from .providers import ModelProfile, profile_for
from .skills import SkillCatalog
from .status import StatusSnapshot, render_status
from .todos import TodoStore
from .tools import Tool, ToolRegistry, tool

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
    """规划模式提交计划：驱动层审批（批准则翻转 agent.plan_mode）后回传结果。"""

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
        approve: Callable[[Tool, dict], bool] | None = None,
        status_bar: bool | Callable[[StatusSnapshot], str] | None = None,
        todos: TodoStore | None = None,
        plan_mode: bool = False,
        plan_capable: bool = False,
        approve_plan: Callable[[str], bool] | None = None,
        compress: bool = False,
        context_window: int | None = None,
        compress_threshold: float | None = None,
        keep_recent: int = 30,
        keep_recent_tokens: int | None = None,
        micro_threshold: float | None = 0.6,
        micro_min_chars: int = MICRO_MIN_CHARS,
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
        self.archive = HistoryArchive()
        if compress:
            self.tools.add(self.archive.tool())
        self.max_steps = max_steps
        self.approve = approve
        # 状态栏渲染器：True 用默认渲染，callable 自定义，None/False 关闭。
        # 状态以 user 消息追加在上下文末尾（书 2.6），绝不修改已有消息。
        self.status_bar = render_status if status_bar is True else status_bar or None
        # 两阶段模式：plan_mode 是驱动层拥有的运行时状态（Q12——agent 的控制流
        # 不再分支于它；decide() 判定、状态栏显示、驱动层翻转），构造时给定初值。
        # plan_capable 只控制 exit_plan_mode 工具的注册（构造时一次）——两者解耦，
        # CLI 可以随时切换模式而不动工具数组（缓存纪律）。
        self.plan_mode = plan_mode
        self.approve_plan = approve_plan
        if plan_mode or plan_capable:
            self._register_exit_plan_mode()
        # TODO 存储：todo_write 工具写入（default_tools(todos=...) 接同一个实例），
        # 状态栏每轮把它渲染到上下文末尾——外部记忆，不靠模型回忆。
        self.todos = todos if todos is not None else TodoStore()
        self.history: list[dict] = []
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
        return prompt.render()

    def reset(self) -> None:
        self.history.clear()
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
        self._size_cache.clear()
        self._size_cache_len = -1
        self.archive.reset()
        if self.skills is not None:
            self.skills.reset()

    def switch_model(self, llm, profile: ModelProfile | None = None) -> None:
        """会话中换模型（/models，票 14）：只在迭代边界调用（驱动层 busy 语义保证）。

        端点实例与能力档案一起换——压缩策略、温度纪律、窗口都按新模型走
        （providers 按名匹配，无需厂商特判）。历史与统计保留，但旧模型的
        ``reasoning_content`` 跨模型不连续，回传陌生端点可能被拒，一律剥离；
        前缀基线随之作废。``--context-window`` 显式给过的窗口会被新档案覆盖
        （切换即按新模型档案重置）。
        """
        self.llm = llm
        self.profile = profile or profile_for(getattr(llm, "model", None))
        self.context_window = self.profile.context_window
        self.compress_threshold = self.profile.compress_threshold
        for message in self.history:
            message.pop("reasoning_content", None)
        self._last_prefix = None
        self.last_usage = None
        self._request_size = 0
        self._compress_failures = 0
        self._size_cache.clear()
        self._size_cache_len = -1

    # ---------- 生成器协议 ----------

    def steps(self, user_input: str):
        """处理一条用户输入的生成器：yield 事件，ToolCall/PlanSubmitted 期待
        ``send`` 回结果字符串；StopIteration.value 是最终答案。"""
        # 工具定义位于上下文前部，首次请求后必须保持字节级不变（KV Cache 前缀
        # 复用的前提），因此在这里冻结注册表，防止运行中途增删工具。
        self.tools.freeze()
        self._truncation_continues = 0
        self.history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": self.system_prompt}, *self.history]
        schemas = self.tools.schemas() or None

        for step in range(1, self.max_steps + 1):
            yield Iteration(step=step, max_steps=self.max_steps)
            # 驱动可在迭代边界追加排队输入。此时上一批 tool_call 已全部回填，
            # 从 history 重建请求，确保新消息进入本轮且保留前缀不变量。
            messages = [{"role": "system", "content": self.system_prompt}, *self.history]
            full_applied = False
            if self._should_compress():
                compacted = self._try_compress(user_input)
                if compacted is not None:
                    before = self._apply_compaction(compacted)
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        *self.history,
                    ]
                    full_applied = True
                    yield Compaction(mode="full", before=before, after=len(compacted))
            # 微压缩兜底（Q8）：全量未触发但过 0.6 阈值，或全量失败时，用无 LLM 的
            # 指针清理推迟/减轻全量压缩；不受熔断计数限制。
            if not full_applied and self._should_microcompress():
                micro = self._try_microcompress()
                if micro is not None:
                    before, cleared = micro
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        *self.history,
                    ]
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
                status_message = {
                    "role": "user",
                    "content": self.status_bar(
                        StatusSnapshot(
                            iteration=step,
                            max_steps=self.max_steps,
                            tool_calls=dict(self.tool_counts),
                            usage=dict(self.total_usage),
                            todos=self.todos.as_dicts(),
                            plan_mode=self.plan_mode,
                        )
                    ),
                }
                messages.append(status_message)
                self.history.append(status_message)

            if self.compress and self._context_size() >= self.context_window * 0.95:
                raise RuntimeError(t("agent.context_limit"))
            self._request_size = self._estimated_size()

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
                            yield ReasoningDelta(delta) if kind == "reasoning" else Text(delta)
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
                        yield ReasoningDelta(delta) if kind == "reasoning" else Text(delta)
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
            self.history.append(assistant)
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
                                self.history[:] = compacted
                                self._last_prefix = None
                        self.history.append(
                            {
                                "role": "user",
                                "content": t("agent.truncation_continue"),
                            }
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
                    annotated = result
                    if self.status_bar is not None:
                        # 调用计数标注（书实验 2-9）：显式次数触发模型的模式识别——
                        # 第 3 次失败后主动换路，而不是无限重试。只进 tool 消息，
                        # tool_result 事件发原始结果（UI 不该看到给模型的标注）。
                        annotated = f"（{name} 第 {self.tool_counts[name]} 次调用）\n{result}"

                    logger.debug("工具 %s 返回: %.200s", name, result)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": annotated,
                    }
                    messages.append(tool_message)
                    self.history.append(tool_message)
            except (KeyboardInterrupt, GeneratorExit):
                # 中断可能落在工具序列中间：assistant 已声明 N 个 tool_call，
                # 只回填一部分的话，下一轮请求的序列残缺会被 API 拒绝（每个
                # tool_call_id 都必须有对应的 tool 消息）。给尚未回填的调用
                # 补上中断结果，让历史保持合法，再向上传播中断。
                self._backfill_tool_results(messages, message.tool_calls or [])
                raise

        raise RuntimeError(f"超过最大步数 {self.max_steps}，仍未得到最终答案")

    # ---------- 内置驱动 ----------

    def run(self, user_input: str) -> str:
        """内置驱动：消费 steps() + approve 审批策略 + 执行器（库 / -p / 测试）。

        流式片段照常 yield，但内置驱动不渲染、安静忽略（handle 返回 None）。"""

        def handle(ev: Event) -> str | None:
            if isinstance(ev, PlanSubmitted):
                return self._handle_plan(ev.plan)
            if isinstance(ev, ToolCall):
                return self._builtin_tool(ev)
            return None

        return drive(self.steps(user_input), handle)

    def _builtin_tool(self, ev: ToolCall) -> str:
        item = self.tools.get(ev.name)
        if item is None:
            return f"Error: unknown tool '{ev.name}'"
        decision = decide(item, ev.arguments, Context(plan=self.plan_mode))
        if decision.verdict == "deny":
            return decision.reason
        if self.approve is not None and not self.approve(item, ev.arguments):
            return t("agent.rejected", name=ev.name)
        result, _ = execute(item, ev.arguments)
        return result

    def _handle_plan(self, plan: str) -> str:
        if not self.plan_mode:
            return "已处于执行模式，无需再调用 exit_plan_mode。"
        approved = self.approve_plan(plan) if self.approve_plan is not None else True
        if approved:
            self.plan_mode = False
            return "计划已批准，进入执行模式：现在可以执行写操作（仍受审批钩子约束）。"
        return "计划被拒绝：请根据用户反馈修改计划，重新调用 exit_plan_mode 提交。"

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
        used = self._context_size()
        return used > self.context_window * self.compress_threshold

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
        keep = effective_keep(self.history, self.keep_recent, self.keep_recent_tokens)
        if not clearable_indices(self.history, keep, self.micro_min_chars):
            return None
        snapshot = self.archive.save(self.history)
        result = microcompact(self.history, keep, snapshot, self.micro_min_chars)
        if result is None:
            return None
        new_history, cleared = result
        before = len(self.history)
        self.history[:] = new_history
        self._last_prefix = None
        self.last_usage = None
        self._request_size = 0
        self._size_cache.clear()
        self._size_cache_len = -1
        logger.info("微压缩：清理旧工具结果约 %d 字符（快照 %s）", cleared, snapshot)
        return before, cleared

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
            snapshot = self.archive.save(self.history)
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
            checkpoint = (
                f"压缩前原始历史：history_read(snapshot={snapshot!r}, message=1)，"
                f"共 {len(self.history)} 条消息，可按编号与字符偏移回查。\n"
                f"当前 TODO：{json.dumps(self.todos.as_dicts(), ensure_ascii=False)}\n"
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
                    self.llm, self.history, keep, query, previous_summary=previous
                )
                if compacted is None:
                    compacted = compact_restart(
                        self.llm, self.history, keep, query, previous_summary=previous
                    )
            else:
                compacted = compact_restart(
                    self.llm, self.history, keep, query, previous_summary=previous
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
        self.history[:] = compacted
        self._last_prefix = None
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
            self.history.append(interrupted)
