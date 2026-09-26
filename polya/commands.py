"""Built-in command registry shared by help, completion, validation and dispatch.

Handlers run on the session worker. Only the queue's resume control runs on the
input thread; task-end commands must never run inside a live agent generator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from getpass import getpass
from typing import TYPE_CHECKING, Literal, TypeVar

from .llm import LLM
from .models import PRESETS, ModelsConfig, Profile, host_of, mask_key
from .providers import profile_for
from .todos import _STATUS_LABELS

if TYPE_CHECKING:
    from .agent import Agent
    from .approval import ApprovalGate
    from .render import TerminalRenderer


_T = TypeVar("_T")


@dataclass
class CommandContext:
    agent: Agent
    renderer: TerminalRenderer | None = None
    gate: ApprovalGate | None = None
    resume: Callable[[], str] | None = None
    restart: Callable[[], str] | None = None
    # 终端让位（/models add 向导）：交互会话借道审批的挂起机制运行交互闭包
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
    # 动态 choices（票 14）：/models 的选项来自 ~/.polya/models.json，运行时取数。
    # 校验（error）、参数补全与选项器统一走 effective_choices()，三处同源。
    choices_provider: Callable[[], tuple[tuple[str, str], ...]] | None = None
    # 参数面比 choices 宽的命令（/models 的 add/remove 子动词）自带校验，
    # 提供 validator 时接管 error() 的全部判定。
    validator: Callable[[str], str | None] | None = None
    integer: bool = False
    required: bool = False
    busy: Literal["boundary", "task_end", "control"] = "boundary"

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
            # 动态 choices（/models）无参放行到处理器：交互层先开选项器拦住
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
    if ctx.gate is not None:
        mode = (
            "全部允许（高危仍询问）" if ctx.gate.allow_all or agent.approve is None else "逐次审批"
        )
        lines.extend([f"审批模式: {mode}", f"会话授权规则: {len(ctx.gate.rules)} 条"])
        lines.extend(str(rule) for rule in ctx.gate.rules)
    return "\n".join(lines)


def _plan(ctx: CommandContext, arg: str) -> str:
    ctx.agent.plan_mode = arg == "on"
    if arg == "off":
        return "已退出规划模式。"
    if ctx.agent.tools.get("exit_plan_mode") is None:
        return "已进入规划模式（注意：未注册 exit_plan_mode 工具，计划无法提交批准）。"
    return "已进入规划模式：只读探查，模型完成计划后会调用 exit_plan_mode 提交。"


def _details(ctx: CommandContext, arg: str) -> str:
    if ctx.renderer is None:
        return "No details recorded in this session."
    return ctx.renderer.show_details(int(arg))


def _expand(ctx: CommandContext, arg: str) -> str:
    if ctx.renderer is None:
        return "（非终端会话不记录块，无法展开）"
    return ctx.renderer.expand_blocks(int(arg) if arg else 5)


def _permissions(ctx: CommandContext, arg: str) -> str:
    if ctx.gate is None:
        return "此会话不支持切换审批模式，请在交互终端中使用 /permissions。"
    ctx.gate.allow_all = arg == "all"
    if arg == "ask":
        ctx.agent.approve = ctx.gate.as_approve()
        ctx.gate.rules.clear()
    return "本会话全部允许（高危仍询问）" if arg == "all" else "已恢复逐次审批"


def _resume(ctx: CommandContext, arg: str) -> str:
    return ctx.resume() if ctx.resume is not None else "此会话没有可恢复的排队任务。"


def _rename(ctx: CommandContext, arg: str) -> str:
    return ctx.rename(arg) if ctx.rename else "此会话不支持主题重命名。"


def _rename_error(argument: str) -> str | None:
    if argument and len(argument) <= 120 and all(c.isprintable() for c in argument):
        return None
    return "用法 (Usage): /rename <主题>（1–120 字，单行）"


def _clear(ctx: CommandContext, arg: str) -> str:
    ctx.agent.reset()
    return "已清空对话历史、TODO 与统计。"


def _new(ctx: CommandContext, arg: str) -> str:
    ctx.agent.reset()
    if ctx.restart is None:
        return "已开始新会话。"
    # 会话级重置（主题、授权规则、排队消息）由驱动层提供；返回附注（如丢弃条数）
    return "已开始新会话：主题与授权规则已重置。" + ctx.restart()


def _model_choices() -> tuple[tuple[str, str], ...]:
    """/models 的动态选项：profile 名 + 「模型 @ 主机」标签 + add/remove 伪选项；
    坏配置当无 profile。"""
    try:
        config = ModelsConfig.load()
    except ValueError:
        config = ModelsConfig()
    entries = [(p.name, f"{p.model} @ {host_of(p.base_url)}") for p in config.profiles]
    entries.extend([("add", "录入新 profile（交互向导）"), ("remove", "移除已有 profile")])
    return tuple(entries)


_MODELS_USAGE = (
    "用法 (Usage): /models [profile]\n"
    f"  /models add                      （交互向导：预设选名字 → 隐藏输 key）\n"
    f"  /models add <名字> <预设: {'|'.join(PRESETS)}>\n"
    "  /models add <名字> <base_url> <模型>\n"
    "  /models remove <名字>"
)


def _models_validate(argument: str) -> str | None:
    """/models 的参数面比 choices 宽（add/remove 子动词），自带校验。"""
    if not argument:
        return None
    parts = argument.split()
    if parts[0] == "add":
        if len(parts) == 1:
            return None  # 裸 add 进向导
        if len(parts) == 3 and parts[2] in PRESETS:
            return None
        if len(parts) == 4 and parts[2].startswith(("http://", "https://")):
            return None
        return _MODELS_USAGE
    if parts[0] == "remove":
        return None if len(parts) == 2 else _MODELS_USAGE
    try:
        names = {p.name for p in ModelsConfig.load().profiles}
    except ValueError:
        names = set()
    return None if argument in names else _MODELS_USAGE


def _ask_profile(name: str, base_url: str, model: str) -> Profile | None:
    """补齐单行 add 缺的字段（预设无建议模型时问模型名），key 恒隐藏输入。"""
    if not model:
        model = input("模型名: ").strip()
    api_key = getpass("api_key（输入不回显）: ").strip()
    if not model or not api_key:
        print("[取消] 模型名与 api_key 不能为空。")
        return None
    return Profile(name=name, base_url=base_url, api_key=api_key, model=model)


def _models_wizard() -> Profile | None:
    """终端让位窗口里的录入向导：内置预设选个名字，未知信息逐项问，key 隐藏输。"""
    keys = list(PRESETS)
    print("可用预设（已知厂商内置，选名字即可）：")
    for index, key in enumerate(keys, 1):
        preset = PRESETS[key]
        model = preset.model or "（自填模型名）"
        print(f"  {index}. {preset.label} · {model} @ {host_of(preset.base_url)}")
    print(f"  {len(keys) + 1}. 自定义 OpenAI 兼容端点")
    choice = input(f"选择 [1-{len(keys) + 1}]（回车 1）: ").strip() or "1"
    try:
        index = int(choice)
        picked = 1 <= index <= len(keys) + 1
    except ValueError:
        picked = False
    if not picked:
        print("[取消] 无效选择。")
        return None
    if index == len(keys) + 1:
        default_name, base_url = "", input("base_url（OpenAI 兼容端点）: ").strip()
        model = input("模型名: ").strip()
    else:
        key = keys[index - 1]
        preset = PRESETS[key]
        default_name, base_url = key, preset.base_url
        model = preset.model or input("模型名: ").strip()
    if default_name:
        name = input(f"profile 名（回车用 {default_name}）: ").strip() or default_name
    else:
        name = input("profile 名: ").strip()
    if not name:
        print("[取消] profile 名不能为空。")
        return None
    return _ask_profile(name, base_url, model)


def _models_add(ctx: CommandContext, parts: list[str]) -> str:
    run = ctx.in_terminal or (lambda go: go())  # 非交互/测试：直接跑（管道 stdin）
    try:
        if not parts:
            profile = run(_models_wizard)
        else:  # 单行捷径：预设名或 base_url+模型（key 仍隐藏输入）
            if len(parts) == 2 and parts[1] in PRESETS:
                preset = PRESETS[parts[1]]
                base_url, model = preset.base_url, preset.model
            else:
                base_url, model = parts[1], parts[2]
            profile = run(lambda: _ask_profile(parts[0], base_url, model))
    except (InterruptedError, KeyboardInterrupt):
        return "已取消录入。"
    if profile is None:
        return "已取消录入。"
    config = ModelsConfig.load()
    try:
        config.add(profile)
        config.save()
    except ValueError as exc:
        return f"[错误] {exc}"
    note = "，已设为 active" if config.active == profile.name else ""
    endpoint = f"{profile.model} @ {host_of(profile.base_url)}"
    masked = mask_key(profile.api_key)
    return (
        f"已录入 {profile.name}：{endpoint} · key {masked}{note}。/models {profile.name} 即刻切换。"
    )


def _models_remove(parts: list[str]) -> str:
    config = ModelsConfig.load()
    try:
        removed = config.remove(parts[0])
        config.save()
    except (ValueError, IndexError) as exc:
        return f"[错误] {exc}"
    note = f"，active → {config.active}" if config.active else "（已无 profile）"
    return f"已移除 {removed.name}{note}。"


def _models(ctx: CommandContext, arg: str) -> str:
    parts = arg.split()
    if parts and parts[0] == "add":
        return _models_add(ctx, parts[1:])
    if parts and parts[0] == "remove":
        return _models_remove(parts[1:])
    config = ModelsConfig.load()
    if not arg:
        if not config.profiles:
            return "还没有模型 profile：/models add 进向导录入（预设选名字，key 隐藏输入）。"
        rows = [
            f"{'●' if p.name == config.active else '○'} {p.name} · {p.model} @ "
            f"{host_of(p.base_url)} · key {mask_key(p.api_key)}"
            for p in config.profiles
        ]
        return "模型 profile（/models <名字> 切换，对话保留；add/remove 录入移除）：\n" + "\n".join(
            rows
        )
    profile = config.find(arg)
    if profile is None:  # 双保险：_models_validate 已拦截未知名
        return _MODELS_USAGE
    # boundary 语义（busy="boundary"）：下一轮请求前生效，生成器存活时不碰
    new_profile = profile_for(profile.model)
    ctx.agent.switch_model(
        LLM(
            model=profile.model,
            base_url=profile.base_url,
            api_key=profile.api_key,
            temperature=new_profile.temperature,
            profile_name=profile.name,
        ),
        new_profile,
    )
    config.use(arg)
    config.save()  # 写回 active：下次启动沿用
    if ctx.renderer is not None:
        ctx.renderer.context_window = ctx.agent.context_window  # 状态行占用比例跟随
    endpoint = f"{profile.name}（{profile.model} @ {host_of(profile.base_url)}）"
    return f"已切换到 {endpoint}；对话保留，旧模型 thinking 已剥离。"


def _compact(ctx: CommandContext, arg: str) -> str:
    return ctx.agent.compact_now(arg or None)


COMMANDS = (
    Command("/help", "显示命令帮助", _help),
    Command("/todos", "显示当前 TODO 清单", _todos),
    Command(
        "/compact",
        "立即压缩上下文（当前任务结束后执行）",
        _compact,
        argument_hint="[说明]",
        busy="task_end",
    ),
    Command("/status", "显示模式、用量与工具计数", _status),
    Command(
        "/plan",
        "切换规划模式（无参数打开选项）",
        _plan,
        argument_hint="[on|off]",
        choices=(("on", "规划：只读探查与计划审批"), ("off", "执行：退出规划模式")),
    ),
    Command(
        "/details",
        "查看审批或工具调用的完整详情",
        _details,
        argument_hint="ID",
        integer=True,
        required=True,
    ),
    Command(
        "/expand",
        "展开最近 N 块工具结果 / 思考（默认 5）",
        _expand,
        argument_hint="[N]",
        integer=True,
    ),
    Command("/resume", "恢复中断或拒绝后暂停的排队任务", _resume, busy="control"),
    Command(
        "/rename",
        "重命名当前会话主题",
        _rename,
        argument_hint="<主题>",
        validator=_rename_error,
    ),
    Command(
        "/permissions",
        "切换审批模式（无参数打开选项）",
        _permissions,
        argument_hint="[ask|all]",
        choices=(("ask", "逐次审批，并清除会话授权规则"), ("all", "本会话全部允许，高危仍询问")),
    ),
    Command(
        "/models",
        "查看/切换/录入模型 profile（add 进交互向导）",
        _models,
        argument_hint="[profile|add|remove]",
        choices_provider=_model_choices,
        validator=_models_validate,
    ),
    Command(
        "/clear",
        "清空对话历史、TODO 与统计",
        _clear,
        aliases=("/reset",),
        busy="task_end",
    ),
    Command("/new", "开新会话：另清主题、授权规则与排队消息", _new, busy="task_end"),
    Command("/exit", "退出", _exit, aliases=("/quit",), busy="task_end"),
)
BY_NAME = {name: command for command in COMMANDS for name in (command.name, *command.aliases)}
BUSY_HINTS = {
    "boundary": "下一轮请求前执行",
    "task_end": "当前任务结束后执行",
    "control": "立即恢复队列",
}


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
    + f"  {command.description} · {BUSY_HINTS[command.busy]}"
    for command in COMMANDS
)
