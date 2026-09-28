"""Built-in command registry shared by help, completion, validation and dispatch.

命令是平面注册表：`name + handler + 参数`。唯一的安全属性是 `idle`——会改
agent 会话树 / 历史的命令标 `idle=True`，任务运行中拒绝执行并提示先按 Esc 中断；
其余命令随到随执行。调度语义（busy 三态 / 队列恢复）已删除。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from getpass import getpass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from . import session as session_store
from . import trust as trust_store
from .llm import LLM
from .models import (
    PROVIDERS,
    ModelEntry,
    ModelsConfig,
    ProviderEntry,
    discover_models,
    format_context_window,
    host_of,
    resolve_context_window,
)
from .providers import profile_for
from .todos import _STATUS_LABELS

if TYPE_CHECKING:
    from .agent import Agent
    from .render import TerminalRenderer


_T = TypeVar("_T")


@dataclass
class CommandContext:
    agent: Agent
    renderer: TerminalRenderer | None = None
    restart: Callable[[], str] | None = None
    # 终端让位（/login 向导）：交互会话借道挂起机制运行交互闭包
    # （输入框让位、key 走 getpass 不进屏幕与输入历史）；缺省直接跑（管道 stdin）。
    in_terminal: Callable[[Callable[[], _T]], _T] | None = None
    rename: Callable[[str], str] | None = None


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    handler: Callable[[CommandContext, str], str | None]
    aliases: tuple[str, ...] = ()
    argument_hint: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    # 动态 choices：/model /login /logout 的选项来自 ~/.polya/models.json，运行时取数。
    # 校验（error）、参数补全与选项器统一走 effective_choices()，三处同源。
    choices_provider: Callable[[], tuple[tuple[str, str], ...]] | None = None
    # 参数面宽于静态 choices 的命令（/login /logout /model）自带 validator，
    # 提供 validator 时接管 error() 的全部判定。
    validator: Callable[[str], str | None] | None = None
    integer: bool = False
    required: bool = False
    # 需要无存活任务（会改会话树 / 历史 / llm）；忙时拒绝并提示先 Esc 中断。
    idle: bool = False

    @property
    def usage(self) -> str:
        return f"{self.name} {self.argument_hint}".rstrip()

    def effective_choices(self) -> tuple[tuple[str, str], ...]:
        return (
            self.choices
            if self.choices
            else (self.choices_provider() if self.choices_provider else ())
        )

    def error(self, argument: str, *, allow_picker: bool = False) -> str | None:
        if self.validator is not None:
            # 校验器自决全部参数面（含空参是否放行），choices 仅供补全与选项器
            return self.validator(argument)
        choices = self.effective_choices()
        if choices:
            # 动态 choices（/model /login /logout）无参放行到处理器：交互层先开选项器拦住
            # （input._submit），非交互落到处理器出清单；静态 choices 维持
            # 「无参即用法错误」（非交互 /plan 不该误触发切换）。
            if argument in dict(choices) or (
                not argument and (allow_picker or self.choices_provider)
            ):
                return None
        elif self.integer:
            if not argument and not self.required:
                return None
            try:
                if int(argument) > 0:
                    return None
            except ValueError:
                pass
        elif not argument:
            return None
        return f"用法 (Usage): {self.usage}"


def _help(ctx: CommandContext, arg: str) -> str:
    return HELP_TEXT


def _exit(ctx: CommandContext, arg: str) -> None:
    return None


def _todos(ctx: CommandContext, arg: str) -> str:
    items = ctx.agent.todos.as_dicts()
    return (
        "\n".join(
            f"[{i}] [{_STATUS_LABELS[item['status']]}] {item['content']}"
            for i, item in enumerate(items, 1)
        )
        or "（TODO 清单为空）"
    )


def _status(ctx: CommandContext, arg: str) -> str:
    agent = ctx.agent
    lines = [
        f"模式: {'规划中（只读）' if agent.plan_mode else '执行'}",
        f"历史消息: {len(agent.history)} 条",
        f"工具调用: {dict(agent.tool_counts) or '（无）'}",
        f"token 用量: {agent.total_usage}",
    ]
    thinking = getattr(agent.llm, "thinking_level", None)
    if thinking:
        lines.append(f"推理档位: {thinking}")
    return "\n".join(lines)


def _plan(ctx: CommandContext, arg: str) -> str:
    if arg == "go":
        ctx.agent.leave_plan_mode()
        return "已批准计划，进入执行模式。"
    ctx.agent.plan_mode = arg == "on"
    if arg == "off":
        return "已退出规划模式。"
    if ctx.agent.tools.get("exit_plan_mode") is None:
        return "已进入规划模式（注意：未注册 exit_plan_mode 工具，计划无法提交批准）。"
    return "已进入规划模式：只读探查，模型完成计划后会调用 exit_plan_mode 提交。"


def _details(ctx: CommandContext, arg: str) -> str:
    """查看留档块：带 ID 看指定块；无参看最近 5 块（合并旧 /expand）。"""
    if ctx.renderer is None:
        return "No details recorded in this session."
    if arg:
        return ctx.renderer.show_details(int(arg))
    return ctx.renderer.expand_blocks(5)


def _rename(ctx: CommandContext, arg: str) -> str:
    ctx.agent.set_session_title(arg)
    return ctx.rename(arg) if ctx.rename else f"已更新会话主题：{arg}"


def _rename_error(argument: str) -> str | None:
    if argument and len(argument) <= 120 and all(c.isprintable() for c in argument):
        return None
    return "用法 (Usage): /rename <主题>（1–120 字，单行）"


def _new(ctx: CommandContext, arg: str) -> str:
    """统一 /new /clear /reset：开新会话，旧会话保留可 /resume 找回。"""
    name = ctx.agent.new_session()
    note = "（旧会话保留，/resume 找回）"
    if ctx.restart is None:
        return f"已开始新会话 {name}{note}。"
    # 会话级重置（主题、排队消息）由驱动层提供；返回附注（如丢弃条数）
    return f"已开始新会话 {name}{note}：主题与排队消息已重置。" + ctx.restart()


def _model_choices() -> tuple[tuple[str, str], ...]:
    """/model 的动态选项：全部已登录 provider 的模型（值 = ``provider/模型``）。"""
    try:
        config = ModelsConfig.load()
    except ValueError:
        return ()
    return tuple((ref, _model_label(config, ref)) for ref in config.refs())


def _model_label(config: ModelsConfig, ref: str) -> str:
    provider_id, model = ref.split("/", 1)
    window = resolve_context_window(model, config.get(provider_id))
    return f"{model} @ {provider_id} · {format_context_window(window)}"


def _login_choices() -> tuple[tuple[str, str], ...]:
    """/login 的 provider 列表 + 自定义端点。"""
    entries = [(pid, f"{p.label} · {host_of(p.base_url)}") for pid, p in PROVIDERS.items()]
    entries.append(("custom", "自定义端点（自填 base_url / 模型 / key）"))
    return tuple(entries)


def _logout_choices() -> tuple[tuple[str, str], ...]:
    """/logout 的已登录 provider；坏配置当空。"""
    try:
        config = ModelsConfig.load()
    except ValueError:
        return ()
    return tuple(
        (pid, f"{entry.model} @ {host_of(entry.base_url)}")
        for pid, entry in config.providers.items()
    )


_LOGIN_USAGE = "用法 (Usage): /login [provider|custom]（无参数打开 provider 列表）"
_LOGOUT_USAGE = "用法 (Usage): /logout <provider>（无参数打开已登录列表）"
_MODEL_USAGE = "用法 (Usage): /model [provider/模型]（无参数打开模型列表，Ctrl+S 设为默认）"


def _login_validate(argument: str) -> str | None:
    if not argument:
        return None  # 空参进选项器
    parts = argument.split()
    if len(parts) == 1 and (parts[0] in PROVIDERS or parts[0] == "custom"):
        return None
    return _LOGIN_USAGE


def _logout_validate(argument: str) -> str | None:
    if not argument:
        return None
    try:
        config = ModelsConfig.load()
    except ValueError:
        return None
    return None if argument.strip() in config.providers else _LOGOUT_USAGE


def _model_validate(argument: str) -> str | None:
    if not argument:
        return None
    try:
        config = ModelsConfig.load()
    except ValueError:
        return None
    ref = argument.strip()
    if "/" not in ref:
        return None if config.get(ref) is not None else _MODEL_USAGE
    provider_id, model = ref.split("/", 1)
    entry = config.get(provider_id)
    if entry is None:
        return _MODEL_USAGE
    if model != entry.model and all(item.id != model for item in entry.models):
        return _MODEL_USAGE
    return None


def _refresh_catalog(provider_id: str, base_url: str, api_key: str) -> str:
    """后台刷新模型目录，返回一句提示（空串 = 无需提示）。

    对齐 pi：登录本身不等 `models.list()`；这里在独立线程里跑，15s 超时、不重试。
    只替换 `models[]`，不动已选的当前模型与 `active`（重登不擅自换模型，active 不悬空）；
    期间若 provider 已被 `/logout`，过期结果丢弃。
    """
    try:
        entries = discover_models(base_url, api_key, timeout=15.0)
    except Exception as exc:  # noqa: BLE001 - 后台任务，任何失败都只提示不阻断
        return f"未能获取模型列表（{type(exc).__name__}: {exc}），沿用静态窗口。"
    try:
        config = ModelsConfig.load()
        entry = config.get(provider_id)
        if entry is None:
            return ""  # 登录后已被 /logout：丢弃过期结果
        if entry.model and all(item.id != entry.model for item in entries):
            entries.append(ModelEntry(entry.model))  # 保住当前模型，active 不悬空
        config.upsert(
            provider_id, ProviderEntry(entry.base_url, entry.api_key, entry.model, entries)
        )
        config.save()
    except (ValueError, OSError) as exc:
        return f"模型目录落盘失败（{type(exc).__name__}: {exc}）。"
    return f"模型目录已刷新（{len(entries)} 个模型）。"


def _start_catalog_refresh(provider_id: str, base_url: str, api_key: str) -> None:
    """后台线程跑目录刷新；daemon 化，不影响会话退出（对齐 pi 的 `void refresh()`）。"""

    def run() -> None:
        note = _refresh_catalog(provider_id, base_url, api_key)
        if note:
            print(f"{provider_id}: {note}")

    threading.Thread(target=run, name=f"polya-refresh-{provider_id}", daemon=True).start()


def _login(ctx: CommandContext, arg: str) -> str:
    run = ctx.in_terminal or (lambda go: go())  # 非交互/测试：直接跑（管道 stdin）
    try:
        return run(lambda: _login_flow(arg.strip()))
    except (InterruptedError, KeyboardInterrupt):
        return "已取消登录。"


def _login_flow(provider_id: str) -> str:
    """借道终端窗口的登录：base_url → 隐藏 key → 立即落盘 → 目录后台刷新。

    对齐 pi：`models.list()` 不再阻塞登录；先用预置默认模型（自定义端点则手输）
    建条目落盘，发现目录交给 `_start_catalog_refresh` 在后台补齐。
    """
    preset = PROVIDERS.get(provider_id)
    if preset is None and provider_id != "custom":
        return _LOGIN_USAGE
    if provider_id == "custom":
        provider_id = input("provider 名（如 my-gateway）: ").strip()
        if not provider_id:
            print("[取消] provider 名不能为空。")
            return "已取消登录。"
        if provider_id in PROVIDERS:
            return f"[错误] provider 名与预置重复：{provider_id}"
    default_url = preset.base_url if preset else ""
    suffix = f"（回车用 {default_url}）" if default_url else ""
    base_url = input(f"base_url{suffix}: ").strip() or default_url
    if not base_url:
        print("[取消] base_url 不能为空。")
        return "已取消登录。"
    api_key = getpass("api_key（输入不回显）: ").strip()
    if not api_key:
        print("[取消] api_key 不能为空。")
        return "已取消登录。"
    config = ModelsConfig.load()
    existing = config.get(provider_id)
    default_model = preset.model if preset else ""
    if not default_model:
        # 无预置默认模型（自定义端点 / Nvidia 等）：定下当前模型才能落盘并设 active
        default_model = input("模型名: ").strip()
        if not default_model:
            print("[取消] 模型名不能为空。")
            return "已取消登录。"
    # 重登保留用户已选模型（不擅自切回预置默认），并把它放进目录以免 active 悬空
    current = existing.model if existing and existing.model else default_model
    models = [ModelEntry(current)]
    if default_model != current:
        models.append(ModelEntry(default_model))
    try:
        config.upsert(provider_id, ProviderEntry(base_url, api_key, current, models))
        config.save()
    except ValueError as exc:
        return f"[错误] {exc}"
    _start_catalog_refresh(provider_id, base_url, api_key)
    window = resolve_context_window(current, config.get(provider_id))
    default_note = "，已设为默认启动模型" if config.active == f"{provider_id}/{current}" else ""
    return (
        f"已登录 {provider_id}：{current} @ {host_of(base_url)} · "
        f"窗口 {format_context_window(window)}{default_note}；模型目录后台刷新中。"
        "/model 切换，Ctrl+S 设默认。"
    )


def _logout(ctx: CommandContext, arg: str) -> str:
    provider_id = arg.strip()
    if not provider_id:
        return _LOGOUT_USAGE  # 无参：交互层先开选项器，非交互落到这里
    config = ModelsConfig.load()
    try:
        removed = config.remove(provider_id)
        config.save()
    except ValueError as exc:
        return f"[错误] {exc}"
    note = ""
    if getattr(ctx.agent.llm, "profile_name", None) == provider_id:
        note = "；当前会话仍在使用该端点，重启后需重新 /login"
    if config.active is None:
        note += "；已无已登录 provider，请 /login"
    return f"已登出 {provider_id}（{removed.model} @ {host_of(removed.base_url)}）{note}。"


def _model(ctx: CommandContext, arg: str) -> str:
    config = ModelsConfig.load()
    ref = arg.strip()
    if not ref:
        if not config.providers:
            return "还没有登录任何 provider：/login 选厂商并输入 key（隐藏输入）。"
        rows = [
            f"{'●' if item == config.active else '○'} {_model_label(config, item)}"
            for item in config.refs()
        ]
        return "已登录模型（/model <provider/模型> 切换，Ctrl+S 设默认）：\n" + "\n".join(rows)
    if "/" in ref:
        provider_id, model = ref.split("/", 1)
    else:
        provider_id, model = ref, ""
    entry = config.get(provider_id)
    if entry is None:
        return _MODEL_USAGE
    model = model or entry.model
    profile = profile_for(model)
    window = resolve_context_window(model, entry)
    # 动态 choices（/model）：无参走选项器/清单，生成器存活时不触碰会话
    ctx.agent.switch_model(
        LLM(
            model=model,
            base_url=entry.base_url,
            api_key=entry.api_key,
            temperature=profile.temperature,
            profile_name=provider_id,
        ),
        profile,
        window,
    )
    if ctx.renderer is not None:
        ctx.renderer.context_window = ctx.agent.context_window  # 状态行占用比例跟随
    return (
        f"已切换到 {provider_id}/{model}（窗口 {format_context_window(window)}）；"
        "对话保留，旧模型 thinking 已剥离。"
    )


def _compact(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.compact_now(arg or None)


def _reload(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.reload_skills()


def _rewind(ctx: CommandContext, arg: str) -> str:
    try:
        steps = int(arg) if arg else 1
    except ValueError:
        return "用法 (Usage): /rewind [N]"
    try:
        return ctx.agent.rewind(steps)
    except ValueError as exc:
        return f"无法回退：{exc}"


def _save(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.save_session(arg or None)


def _sessions(ctx: CommandContext, arg: str) -> str:
    metas = session_store.list_metas()
    if not metas:
        return "（暂无已保存会话）"
    return "已保存会话（/resume 打开选择器）：\n" + "\n".join(f"  {m.label()}" for m in metas)


def _session_choices() -> tuple[tuple[str, str], ...]:
    "/resume 的动态选项：已存会话，title · name · updated。"
    return tuple((m.name, m.label()) for m in session_store.list_metas())


def _resume_validate(argument: str) -> str | None:
    # 空参放行：交互层先开选项器，非交互落到处理器出清单。
    return _session_name_error(argument) if argument else None


def _resume(ctx: CommandContext, arg: str) -> str:
    if not arg:
        return _sessions(ctx, "")
    return ctx.agent.load_session(arg)


def _fork_validate(argument: str) -> str | None:
    try:
        entry_id = int(argument)
    except ValueError:
        return "用法 (Usage): /fork <入口 id>"
    if entry_id < 1:
        return "入口 id 必须为正整数。"
    return None


def _fork(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.fork_session(int(arg))


def _clone(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.clone_session()


def _export(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.export_session(arg or None)


def _export_validate(argument: str) -> str | None:
    # 自由路径参数（可空）；校验器接管 error()，放行任意单参数。
    return None


def _thinking(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.set_thinking(arg)


def _import_validate(argument: str) -> str | None:
    if not argument:
        return "用法 (Usage): /import <路径>"
    return None


def _import(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.import_session(arg)


_TRUST_CHOICES = (
    ("trust", "信任当前目录"),
    ("trust-parent", "信任父目录（清除本目录决定）"),
    ("untrust", "不信任当前目录"),
    ("clear", "清除本目录决定（回退继承）"),
)


def _trust_usage() -> str:
    return "用法 (Usage): /trust [trust|trust-parent|untrust|clear]"


def _trust_validate(argument: str) -> str | None:
    if not argument or argument in dict(_TRUST_CHOICES):
        return None
    return _trust_usage()


def _trust(ctx: CommandContext, arg: str) -> str:
    root = ctx.agent.cwd
    if not root:
        return "无法确定工作目录，不能处理信任。"
    target = Path(root).resolve()
    if not arg:
        found = trust_store.nearest(root)
        session = getattr(ctx.agent, "trusted", trust_store.is_trusted(root))
        lines = [f"目录: {target}"]
        if found:
            where = "本目录" if found[0] == str(target) else f"继承自 {found[0]}"
            lines.append(f"已存决定: {'信任' if found[1] else '不信任'}（{where}）")
        else:
            lines.append("已存决定: 无")
        lines.append(f"当前会话: {'信任' if session else '不信任'}")
        lines.append("可选: /trust trust | trust-parent | untrust | clear；保存后重启 polya 生效。")
        return "\n".join(lines)
    if arg == "trust":
        trust_store.trust(root)
        return f"已信任 {target}；重启 polya 生效。"
    if arg == "untrust":
        trust_store.untrust(root)
        return f"已标记不信任 {target}；重启后不加载项目资源。"
    if arg == "trust-parent":
        parent = target.parent
        trust_store.trust(parent)
        trust_store.forget(root)
        return f"已信任父目录 {parent}（本目录决定已清除）；重启 polya 生效。"
    trust_store.forget(root)
    return f"已清除 {target} 的决定（回退父目录继承）；重启 polya 生效。"


def _load(ctx: CommandContext, arg: str) -> str:
    if not arg:
        return "用法 (Usage): /load <会话名>"
    return ctx.agent.load_session(arg)


def _tree(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.branch_overview()


def _edit_validate(argument: str) -> str | None:
    parts = argument.split(maxsplit=1)
    if not parts:
        return "用法 (Usage): /edit <id> <新内容|remove>"
    try:
        entry_id = int(parts[0])
    except ValueError:
        return "入口 id 必须是整数。"
    if entry_id < 1:
        return "入口 id 必须为正整数。"
    if len(parts) == 1:
        return "需要新内容，或用 remove 删除。"
    return None


def _edit(ctx: CommandContext, arg: str) -> str:
    entry_id_str, rest = arg.split(maxsplit=1)
    entry_id = int(entry_id_str)
    if rest.strip() in ("remove", "删除"):
        try:
            return ctx.agent.remove_entry(entry_id)
        except ValueError as exc:
            return f"无法移除：{exc}"
    try:
        return ctx.agent.edit_entry(entry_id, rest)
    except ValueError as exc:
        return f"无法编辑：{exc}"


def _jump(ctx: CommandContext, arg: str) -> str:
    try:
        entry_id = int(arg)
    except ValueError:
        return "用法 (Usage): /jump <入口 id>"
    try:
        return ctx.agent.jump(entry_id)
    except ValueError as exc:
        return f"无法跳转：{exc}"


def _session_name_error(argument: str) -> str | None:
    if any(c.isspace() for c in argument) or "/" in argument or "\\" in argument:
        return "会话名不能含空白或路径分隔符。"
    return None


def _save_validate(argument: str) -> str | None:
    return _session_name_error(argument) if argument else None


def _load_validate(argument: str) -> str | None:
    if not argument:
        return "用法 (Usage): /load <会话名>"
    return _session_name_error(argument)


COMMANDS = (
    Command("/help", "显示命令帮助", _help),
    Command("/todos", "显示当前 TODO 清单", _todos),
    Command(
        "/compact",
        "立即压缩上下文（当前任务结束后执行）",
        _compact,
        argument_hint="[说明]",
        idle=True,
    ),
    Command("/status", "显示模式、用量与工具计数", _status),
    Command("/reload", "热加载技能目录（重扫 .polya/skills 等）", _reload, idle=True),
    Command(
        "/rewind",
        "回退到更早的入口（后续输入分叉）",
        _rewind,
        argument_hint="[N]",
        integer=True,
        idle=True,
    ),
    Command(
        "/save",
        "保存会话树到 ~/.polya/sessions",
        _save,
        argument_hint="[名称]",
        validator=_save_validate,
        idle=True,
    ),
    Command("/sessions", "列出已保存会话", _sessions),
    Command(
        "/resume",
        "恢复已保存会话（无参数打开选择器）",
        _resume,
        argument_hint="[名称]",
        choices_provider=_session_choices,
        validator=_resume_validate,
        idle=True,
    ),
    Command(
        "/trust",
        "查看/保存项目信任决定（重启生效）",
        _trust,
        argument_hint="[trust|trust-parent|untrust|clear]",
        choices=_TRUST_CHOICES,
        validator=_trust_validate,
    ),
    Command(
        "/import",
        "从路径导入会话（polya JSONL / pi 格式）",
        _import,
        argument_hint="<路径>",
        validator=_import_validate,
        idle=True,
    ),
    Command(
        "/fork",
        "从指定入口分叉出新会话",
        _fork,
        argument_hint="<id>",
        validator=_fork_validate,
        idle=True,
    ),
    Command("/clone", "复制当前会话为新会话", _clone, idle=True),
    Command(
        "/export",
        "导出当前会话为 Markdown（.jsonl 落原始会话）",
        _export,
        argument_hint="[路径]",
        validator=_export_validate,
    ),
    Command(
        "/thinking",
        "设置推理档位（一家一策）",
        _thinking,
        argument_hint="[off|low|medium|high]",
        choices=(
            ("off", "关闭推理（部分模型只能降到最低）"),
            ("low", "低"),
            ("medium", "中"),
            ("high", "高"),
        ),
    ),
    Command("/tree", "显示整棵树：分叉点与所有分支", _tree),
    Command(
        "/edit",
        "编辑某入口在投影里的内容（原文保留，remove 删除）",
        _edit,
        argument_hint="<id> <新内容|remove>",
        validator=_edit_validate,
        idle=True,
    ),
    Command(
        "/jump",
        "跳到指定入口（后续输入分叉）",
        _jump,
        argument_hint="<id>",
        integer=True,
        required=True,
        idle=True,
    ),
    Command(
        "/load",
        "恢复已保存会话到当前 Agent",
        _load,
        argument_hint="<名称>",
        validator=_load_validate,
        idle=True,
    ),
    Command(
        "/plan",
        "切换规划模式（无参数打开选项）",
        _plan,
        argument_hint="[on|off|go]",
        choices=(
            ("on", "规划：只读探查，完成后提交计划"),
            ("go", "批准当前计划，进入执行"),
            ("off", "执行：退出规划模式"),
        ),
    ),
    Command(
        "/details",
        "查看留档块详情（无参数=最近 5 块）",
        _details,
        argument_hint="[ID]",
        integer=True,
    ),
    Command(
        "/rename",
        "重命名当前会话主题",
        _rename,
        argument_hint="<主题>",
        validator=_rename_error,
    ),
    Command(
        "/login",
        "登录 provider（选厂商 → base_url → 隐藏输 key → 发现模型）",
        _login,
        argument_hint="[provider|custom]",
        choices_provider=_login_choices,
        validator=_login_validate,
        idle=True,
    ),
    Command(
        "/logout",
        "登出并移除 provider 凭据",
        _logout,
        argument_hint="<provider>",
        choices_provider=_logout_choices,
        validator=_logout_validate,
        idle=True,
    ),
    Command(
        "/model",
        "切换模型（Ctrl+S 设为默认启动模型）",
        _model,
        aliases=("/models",),
        argument_hint="[provider/模型]",
        choices_provider=_model_choices,
        validator=_model_validate,
        idle=True,
    ),
    Command(
        "/new",
        "开新会话：清空上下文，旧会话保留可 /resume 找回",
        _new,
        aliases=("/clear", "/reset"),
        idle=True,
    ),
    Command("/exit", "退出", _exit, aliases=("/quit",), idle=True),
)
BY_NAME = {name: command for command in COMMANDS for name in (command.name, *command.aliases)}


def parse_command(text: str) -> tuple[Command | None, str]:
    parts = text.split(maxsplit=1)
    return BY_NAME.get(parts[0] if parts else ""), parts[1].strip() if len(parts) > 1 else ""


def command_error(text: str, *, allow_picker: bool = False) -> str | None:
    command, argument = parse_command(text)
    if command is not None:
        return command.error(argument, allow_picker=allow_picker)
    name = text.split(maxsplit=1)[0] if text.strip() else "/"
    suggestions = get_close_matches(name, BY_NAME, n=3, cutoff=0.6)
    hint = f"，你可能想用 {'、'.join(suggestions)}" if suggestions else ""
    return f"未知命令 {name}{hint}；/help 查看可用命令。"


def dispatch_command(text: str, context: CommandContext) -> str | None:
    error = command_error(text)
    if error:
        return error
    command, argument = parse_command(text)
    assert command is not None
    return command.handler(context, argument)


def handle_command(cmd: str, agent: Agent, renderer: TerminalRenderer | None = None) -> str | None:
    """Compatibility entry point, also used by the noninteractive REPL."""
    return dispatch_command(cmd, CommandContext(agent, renderer))


HELP_TEXT = "命令：\n" + "\n".join(
    f"  {command.usage}"
    + (f"（别名 {'、'.join(command.aliases)}）" if command.aliases else "")
    + f"  {command.description}"
    for command in COMMANDS
)
