"""Command UI and dispatch agree on inputs and protect session state."""

from io import StringIO

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from polya import Agent
from polya.commands import (
    BY_NAME,
    COMMANDS,
    HELP_TEXT,
    CommandContext,
    dispatch_command,
    handle_command,
)
from polya.input import InputBox, SlashCompleter
from polya.llm import LLM
from polya.models import ModelsConfig, Profile
from polya.render import TerminalRenderer


@pytest.fixture
def box(tmp_path):
    with create_pipe_input() as pipe:
        yield InputBox(tmp_path / "history", input=pipe, output=DummyOutput())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/plan ", ["on", "go", "off"]),
        ("/plan o", ["on", "off"]),
        ("/plan on ", []),
        ("/plan on off", []),
        ("hello /pl", []),
        ("/pl\n", []),
    ],
)
def test_argument_completions(text, expected):
    assert [c.text for c in SlashCompleter().get_completions(Document(text), None)] == expected


def test_help_lists_commands_and_menu_omits_alias_rows():
    completions = {c.text for c in SlashCompleter().get_completions(Document("/"), None)}
    for command in COMMANDS:
        assert command.usage in HELP_TEXT
        assert command.name in completions and command.name in HELP_TEXT
    assert "/quit" not in completions  # 别名不单列菜单行
    assert "/quit" in BY_NAME  # 提交侧仍按别名解析


@pytest.mark.parametrize(
    "text",
    [
        "/reset extra",
        "/clear extra",
        "/new extra",
        "/exit now",
        "/plan yes",
        "/details 0",
        "/details -3",
    ],
)
def test_bad_arguments_do_not_change_state(text, box):
    agent = Agent(llm=object(), tools=[])
    agent.append_user_message("keep")
    assert "用法" in handle_command(text, agent)
    assert len(agent.history) == 1 and not agent.plan_mode
    sent = []
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document(text)
    box._submit(buffer)
    assert buffer.text == text and not sent
    assert "用法" in box._hint


def test_typo_is_suggested_but_never_executed(box):
    sent = []
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document("/rest")
    box._submit(buffer)
    assert not sent and buffer.text == "/rest"
    assert "/reset" in box._hint


@pytest.mark.parametrize("name,choice", [("/plan", "off")])
def test_bare_command_picker_selection_then_submission(box, name, choice):
    sent = []
    box._state = {"mode": "plan"}
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document(name)
    box._submit(buffer)
    assert buffer.text == name + " " and not sent
    state = buffer.complete_state
    assert any("当前" in str(c.display_meta) for c in state.completions)
    index = next(i for i, c in enumerate(state.completions) if c.text == choice)
    buffer.go_to_completion(index)
    box._submit(buffer)
    # 对齐 CC：选项器里选中后一次 Enter 即执行，不再二段式
    assert sent == [f"{name} {choice}"]


def test_cancel_picker_keeps_command_without_submission(box):
    sent = []
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document("/plan")
    box._submit(buffer)
    buffer.go_to_completion(1)
    buffer.cancel_completion()
    assert buffer.text == "/plan " and not sent


def test_complete_command_is_not_stuck_in_exact_match_menu(box):
    sent = []
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document("/status")
    buffer._set_completions(list(SlashCompleter().get_completions(buffer.document, None)))
    box._submit(buffer)
    assert sent == ["/status"]


def test_rewind_save_sessions_load_roundtrip(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent = Agent(llm=object(), tools=[])
    agent.append_user_message("任务一")
    agent.tree.append("assistant", {"content": "答一"})

    assert "已回退" in dispatch_command("/rewind 1", CommandContext(agent))
    assert agent.history == [{"role": "user", "content": "任务一"}]  # assistant 被孤立

    assert "已保存" in dispatch_command("/save demo", CommandContext(agent))
    assert "demo" in dispatch_command("/sessions", CommandContext(agent))

    restored = Agent(llm=object(), tools=[])
    assert "已恢复" in dispatch_command("/load demo", CommandContext(restored))
    assert restored.history == agent.history


def test_tree_and_jump_navigate_branch():
    agent = Agent(llm=object(), tools=[])
    agent.append_user_message("任务一")
    agent.tree.append("assistant", {"content": "答一"})

    overview = dispatch_command("/tree", CommandContext(agent))
    assert "→ 分支1" in overview and "#3 assistant" in overview

    assert "已跳到入口 #2" in dispatch_command("/jump 2", CommandContext(agent))
    assert agent.history == [{"role": "user", "content": "任务一"}]  # assistant 被孤立


def test_tree_shows_forks_and_branches_after_rewind():
    agent = Agent(llm=object(), tools=[])
    agent.append_user_message("任务一")
    agent.tree.append("assistant", {"content": "答一"})
    agent.rewind(1)  # 回到 user，答一 成孤儿分支
    agent.tree.append("assistant", {"content": "答二"})  # 新分支

    overview = dispatch_command("/tree", CommandContext(agent))
    assert "分叉点：#2" in overview
    assert "分支1" in overview and "分支2" in overview
    assert "→ 分支2" in overview  # 当前分支是新的


def test_edit_and_remove_entry_change_projection_only():
    agent = Agent(llm=object(), tools=[])
    agent.append_user_message("原始任务")
    assert "已编辑" in dispatch_command("/edit 2 新任务", CommandContext(agent))
    assert agent.history == [{"role": "user", "content": "新任务"}]
    assert "原始任务" in agent.tree.read(2)  # 原文保留

    assert "已从投影移除" in dispatch_command("/edit 2 remove", CommandContext(agent))
    assert agent.history == []
    assert "原始任务" in agent.tree.read(2)


def test_clear_new_and_reset_alias_split_session_scopes():
    """/clear 只清 agent；/new 另触发会话级 restart；/reset 是 /clear 的别名。"""
    agent = Agent(llm=object(), tools=[])
    agent.append_user_message("keep")
    calls = []

    def restart() -> str:
        calls.append(1)
        return "（已丢弃 2 条排队消息）"

    context = CommandContext(agent, restart=restart)
    assert "已清空" in dispatch_command("/clear", context)
    assert agent.history == [] and calls == []  # /clear 不动会话级状态
    agent.append_user_message("again")
    result = dispatch_command("/new", context)
    assert agent.history == [] and calls == [1]
    assert "新会话" in result and "已丢弃 2 条排队消息" in result
    agent.append_user_message("once more")
    assert "已清空" in dispatch_command("/reset", context)  # 别名走同一处理器
    assert agent.history == [] and calls == [1]


def test_invalid_details_never_reaches_renderer():
    renderer = TerminalRenderer(Console(file=StringIO()))
    renderer.expand_blocks = lambda *_: pytest.fail("invalid id must not render")
    assert "用法" in handle_command("/details -1", None, renderer)


# ---------- /models：动态 choices + 向导录入 + 会话中切换（票 14） ----------


def _write_models(tmp_path, monkeypatch, profiles, active=None):
    """落一份 models.json 并把 polya.models.default_path 指过去（命令侧真实取数）。"""
    config = ModelsConfig()
    for profile in profiles:
        config.add(profile)
    if active:
        config.use(active)
    path = tmp_path / "models.json"
    config.save(path)
    monkeypatch.setattr("polya.models.default_path", lambda: path)
    return config


GLM = Profile("glm-plan", "https://api.z.ai/api/coding/paas/v4", "sk-abcd1234efgh", "glm-4.7")
CLAUDE = Profile("claude-max", "https://proxy.example/v1", "sk-claude-key-9876", "claude-opus-4-5")


def _fake_io(monkeypatch, inputs, key):
    """向导/单行 add 的交互替身：input 按队列出队，key 走 getpass 替身。"""
    monkeypatch.setattr("builtins.input", lambda prompt="": inputs.pop(0))
    monkeypatch.setattr("polya.commands.getpass", lambda prompt="": key)


def _ctx(agent, in_terminal=True):
    return (
        CommandContext(agent, in_terminal=lambda go: go()) if in_terminal else CommandContext(agent)
    )


def test_models_without_profiles_offers_wizard(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "none.json")
    agent = Agent(llm=object(), tools=[])
    result = handle_command("/models", agent)
    assert "/models add" in result and "向导" in result


def test_models_listing_marks_active_and_masks_keys(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, [GLM, CLAUDE], active="glm-plan")
    agent = Agent(llm=object(), tools=[])
    old_llm = agent.llm
    result = handle_command("/models", agent)
    assert "● glm-plan" in result and "○ claude-max" in result
    assert "sk-…efgh" in result and "sk-abcd1234efgh" not in result
    assert agent.llm is old_llm  # 仅查看不切换


def test_models_switch_replaces_llm_strips_reasoning_writes_active(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, [GLM, CLAUDE], active="glm-plan")
    agent = Agent(llm=object(), tools=[])
    agent.tree.replace_conversation(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a", "reasoning_content": "thinking"},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "b", "tool_calls": [], "reasoning_content": "t2"},
        ]
    )
    agent._last_prefix = [{"role": "system", "content": "x"}]
    renderer = TerminalRenderer(Console(file=StringIO()))

    result = handle_command("/models claude-max", agent, renderer)

    assert "已切换到 claude-max" in result
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "claude-opus-4-5" and agent.llm.profile_name == "claude-max"
    assert "proxy.example" in str(agent.llm.client.base_url)
    # 能力档案跟随：claude 档 200K 窗口，压缩策略/温度按新模型走
    assert agent.context_window == 200_000 and agent.profile.supports_inplace_tool_edit is False
    assert renderer.context_window == 200_000
    assert all("reasoning_content" not in m for m in agent.history)
    assert len(agent.history) == 4  # 对话保留
    assert agent._last_prefix is None  # 前缀基线作废
    assert ModelsConfig.load().active == "claude-max"  # 写回：下次启动沿用


def test_models_add_wizard_preset_flow(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    _fake_io(monkeypatch, ["2", ""], "sk-wizard-12345678")  # 选 zai-cn 预设，名字回车用默认
    result = dispatch_command("/models add", _ctx(Agent(llm=object(), tools=[])))
    assert "已录入 zai-cn" in result and "sk-…5678" in result
    config = ModelsConfig.load()
    assert config.active == "zai-cn"  # 首个 profile 自动 active
    profile = config.find("zai-cn")
    assert profile.base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert profile.model == "glm-5.3" and profile.api_key == "sk-wizard-12345678"
    out = capsys.readouterr().out
    assert "sk-wizard-12345678" not in out  # 明文 key 不落屏幕
    assert "国内" in out  # 向导菜单可见预设标签


def test_models_add_wizard_custom_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    _fake_io(
        monkeypatch,
        ["6", "http://localhost:8000/v1", "qwen3", "box"],
        "sk-local-999888777666",
    )
    result = dispatch_command("/models add", _ctx(Agent(llm=object(), tools=[])))
    assert "已录入 box" in result
    profile = ModelsConfig.load().find("box")
    assert (profile.base_url, profile.model) == ("http://localhost:8000/v1", "qwen3")


def test_models_add_wizard_bad_choice_cancels(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    _fake_io(monkeypatch, ["9"], "sk-never-used")
    result = dispatch_command("/models add", _ctx(Agent(llm=object(), tools=[])))
    assert result == "已取消录入。"
    assert ModelsConfig.load().profiles == []


def test_models_add_oneline_preset_and_duplicate(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, [GLM])
    _fake_io(monkeypatch, [], "sk-claude-key-9876")  # 预设连模型带端点，只差 key
    agent = Agent(llm=object(), tools=[])
    result = dispatch_command("/models add claude-max zai", _ctx(agent))
    assert "已录入 claude-max" in result and "glm-5.3 @ api.z.ai" in result
    assert "已设为 active" not in result  # active 仍是先录入的 glm-plan
    duplicate = dispatch_command("/models add claude-max zai", _ctx(agent))
    assert "已存在同名" in duplicate
    assert len(ModelsConfig.load().profiles) == 2


def test_models_add_falls_back_to_getpass_without_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    monkeypatch.setattr("polya.commands.getpass", lambda prompt="": "sk-fallback-9999")
    result = dispatch_command(
        "/models add glm zai", _ctx(Agent(llm=object(), tools=[]), in_terminal=False)
    )
    assert "已录入 glm" in result and ModelsConfig.load().active == "glm"


def test_models_remove_reassigns_active(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, [GLM, CLAUDE], active="claude-max")
    agent = Agent(llm=object(), tools=[])
    result = dispatch_command("/models remove claude-max", _ctx(agent))
    assert "已移除 claude-max" in result and "active → glm-plan" in result
    config = ModelsConfig.load()
    assert config.active == "glm-plan" and len(config.profiles) == 1
    assert "未找到" in dispatch_command("/models remove ghost", _ctx(agent))


@pytest.mark.parametrize(
    "text",
    [
        "/models add glm",  # 名字后既非预设也非 URL
        "/models add glm notapreset",
        "/models add glm ftp://x/v1 m",
        "/models remove",
        "/models remove a b",
        "/models ghost",
    ],
)
def test_models_bad_shapes_rejected(tmp_path, monkeypatch, text):
    _write_models(tmp_path, monkeypatch, [GLM])
    agent = Agent(llm=object(), tools=[])
    old_llm = agent.llm
    assert "用法" in handle_command(text, agent)
    assert agent.llm is old_llm


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/models ", ["glm-plan", "claude-max", "add", "remove"]),
        ("/models g", ["glm-plan"]),
        ("/models a", ["add"]),
        ("/models c", ["claude-max"]),
        ("/models x", []),
        ("/models glm-plan ", []),
    ],
)
def test_models_argument_completion_is_dynamic(tmp_path, monkeypatch, text, expected):
    _write_models(tmp_path, monkeypatch, [GLM, CLAUDE])
    completions = [c.text for c in SlashCompleter().get_completions(Document(text), None)]
    assert completions == expected


def test_models_bare_opens_picker_marking_active(tmp_path, monkeypatch, box):
    _write_models(tmp_path, monkeypatch, [GLM, CLAUDE], active="glm-plan")
    sent = []
    box._state = {"profile": "glm-plan"}
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document("/models")
    box._submit(buffer)
    assert buffer.text == "/models " and not sent
    state = buffer.complete_state
    assert [c.text for c in state.completions] == ["glm-plan", "claude-max", "add", "remove"]
    active_meta = next(str(c.display_meta) for c in state.completions if c.text == "glm-plan")
    assert "当前" in active_meta
    buffer.go_to_completion(1)
    box._submit(buffer)
    assert sent == ["/models claude-max"]  # 选中即一次 Enter 执行（对齐 CC）
