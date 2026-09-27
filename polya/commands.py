"""Built-in command registry shared by help, completion, validation and dispatch.

命令是平面注册表：`name + handler + 参数`。唯一的安全属性是 `idle`——会改
agent 会话树 / 历史的命令标 `idle=True`，任务运行中拒绝执行并提示先按 Esc 中断；
其余命令随到随执行。调度语义（busy 三态 / 队列恢复）已删除。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from getpass import getpass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from . import session as session_store
from . import trust as trust_store
from .llm import LLM
from .models import PRESETS, ModelsConfig, Profile, host_of, mask_key
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
    # 终端让位（/models add 向导）：交互会话借道挂起机制运行交互闭包
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


def _clear(ctx: CommandContext, arg: str) -> str:
    ctx.agent.reset()
    return "已清空对话历史、TODO 与统计。"


def _new(ctx: CommandContext, arg: str) -> str:
    name = ctx.agent.new_session()
    if ctx.restart is None:
        return f"已开始新会话 {name}。"
    # 会话级重置（主题、排队消息）由驱动层提供；返回附注（如丢弃条数）
    return f"已开始新会话 {name}：主题与排队消息已重置。" + ctx.restart()


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
    # 动态 choices（/models）：无参走选项器/清单，生成器存活时不触碰会话
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
    "/resume 的动态选项：已存会话，name · title · updated。"
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
        "/models",
        "查看/切换/录入模型 profile（add 进交互向导）",
        _models,
        argument_hint="[profile|add|remove]",
        choices_provider=_model_choices,
        validator=_models_validate,
        idle=True,
    ),
    Command(
        "/clear",
        "清空对话历史、TODO 与统计",
        _clear,
        aliases=("/reset",),
        idle=True,
    ),
    Command("/new", "开新会话：分配新名、重置主题与排队消息", _new, idle=True),
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
