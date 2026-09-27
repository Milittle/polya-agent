"""Command UI and dispatch agree on inputs and protect session state."""

from io import StringIO

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from polya import Agent
from polya import commands as commands_mod
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
from polya.models import ModelEntry, ModelsConfig, ProviderEntry
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


def test_new_clear_reset_all_start_fresh_session():
    """/new /clear /reset 统一为开新会话：换名 + 清空 + 会话级 restart（旧会话可 /resume）。"""
    agent = Agent(llm=object(), tools=[])
    calls = []

    def restart() -> str:
        calls.append(1)
        return "（已丢弃 2 条排队消息）"

    context = CommandContext(agent, restart=restart)
    names = []
    for command in ("/clear", "/new", "/reset"):
        agent.append_user_message("stale")
        result = dispatch_command(command, context)
        assert agent.history == []
        assert "新会话" in result and "旧会话保留" in result
        names.append(agent.session_name)

    assert calls == [1, 1, 1]
    assert "已丢弃 2 条排队消息" in result
    assert len(set(names)) == 3  # 每次重新分配会话名，旧会话保留在 /resume


def test_invalid_details_never_reaches_renderer():
    renderer = TerminalRenderer(Console(file=StringIO()))
    renderer.expand_blocks = lambda *_: pytest.fail("invalid id must not render")
    assert "用法" in handle_command("/details -1", None, renderer)


# ---------- /login /logout /model：provider 认证与切换（票 04/05） ----------


def _write_models(tmp_path, monkeypatch, entries, active=None):
    """落一份 models.json 并把 polya.models.default_path 指过去（命令侧真实取数）。

    ``entries`` 是 ``{provider_id: ProviderEntry}``。
    """
    config = ModelsConfig()
    for provider_id, entry in entries.items():
        config.add(provider_id, entry)
    if active:
        config.use(active)
    path = tmp_path / "models.json"
    config.save(path)
    monkeypatch.setattr("polya.models.default_path", lambda: path)
    return config


GLM = ProviderEntry(
    "https://api.z.ai/api/coding/paas/v4",
    "sk-abcd1234efgh",
    "glm-4.7",
    [ModelEntry("glm-4.7", 200_000), ModelEntry("glm-5.3", 1_000_000)],
)
CLAUDE = ProviderEntry(
    "https://proxy.example/v1",
    "sk-claude-key-9876",
    "claude-opus-4-5",
    [ModelEntry("claude-opus-4-5")],
)
FIXTURES = {"glm-plan": GLM, "claude-max": CLAUDE}


def _fake_io(monkeypatch, inputs, key):
    """登录流程的交互替身：input 按队列出队，key 走 getpass 替身。"""
    monkeypatch.setattr("builtins.input", lambda prompt="": inputs.pop(0))
    monkeypatch.setattr("polya.commands.getpass", lambda prompt="": key)


def _ctx(agent, in_terminal=True):
    return (
        CommandContext(agent, in_terminal=lambda go: go()) if in_terminal else CommandContext(agent)
    )


def _run_refresh_inline(monkeypatch):
    """让登录后的后台目录刷新在测试里同步执行，便于断言（真实现走线程）。"""

    def start(provider_id, base_url, api_key):
        note = commands_mod._refresh_catalog(provider_id, base_url, api_key)
        if note:
            print(f"{provider_id}: {note}")

    monkeypatch.setattr(commands_mod, "_start_catalog_refresh", start)


def test_model_without_login_prompts_login(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "none.json")
    agent = Agent(llm=object(), tools=[])
    result = handle_command("/model", agent)
    assert "/login" in result


def test_model_listing_marks_active_and_shows_window(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, FIXTURES, active="glm-plan/glm-4.7")
    agent = Agent(llm=object(), tools=[])
    old_llm = agent.llm
    result = handle_command("/model", agent)
    assert "● glm-4.7 @ glm-plan · 200k" in result
    assert "○ claude-opus-4-5 @ claude-max · 200k" in result
    assert "sk-abcd1234efgh" not in result  # 明文 key 不出现
    assert agent.llm is old_llm  # 仅查看不切换


def test_model_switch_replaces_llm_strips_reasoning_without_changing_default(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, FIXTURES, active="glm-plan/glm-4.7")
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

    result = handle_command("/model claude-max/claude-opus-4-5", agent, renderer)

    assert "已切换到 claude-max/claude-opus-4-5" in result
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "claude-opus-4-5" and agent.llm.profile_name == "claude-max"
    assert "proxy.example" in str(agent.llm.client.base_url)
    # 能力档案跟随：claude 档 200K 窗口，压缩策略/温度按新模型走
    assert agent.context_window == 200_000 and agent.profile.supports_inplace_tool_edit is False
    assert renderer.context_window == 200_000
    assert all("reasoning_content" not in m for m in agent.history)
    assert len(agent.history) == 4  # 对话保留
    assert agent._last_prefix is None  # 前缀基线作废
    # 切换不写默认（Ctrl+S 才写）：票 05
    assert ModelsConfig.load().active == "glm-plan/glm-4.7"


def test_model_switch_uses_discovered_window(tmp_path, monkeypatch):
    entry = ProviderEntry(
        "https://openrouter.ai/api/v1",
        "sk-or-key-1234",
        "vendor/model",
        [ModelEntry("vendor/model", 1_000_000)],
    )
    _write_models(tmp_path, monkeypatch, {"openrouter": entry})
    agent = Agent(llm=object(), tools=[])
    result = handle_command("/model openrouter/vendor/model", agent)
    assert "窗口 1M" in result and agent.context_window == 1_000_000


def test_login_preset_flow_discovers_and_selects_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    monkeypatch.setattr(
        "polya.commands.discover_models",
        lambda base_url, api_key, timeout=10.0: [ModelEntry("glm-5.3", 1_000_000)],
    )
    _run_refresh_inline(monkeypatch)
    _fake_io(monkeypatch, [""], "sk-wizard-12345678")  # base_url 回车用预设默认
    result = dispatch_command("/login zai-coding-cn", _ctx(Agent(llm=object(), tools=[])))
    assert "已登录 zai-coding-cn" in result and "glm-5.3" in result and "1M" in result
    config = ModelsConfig.load()
    assert config.active == "zai-coding-cn/glm-5.3"  # 首个自动默认
    entry = config.get("zai-coding-cn")
    assert entry.base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert entry.api_key == "sk-wizard-12345678"
    assert entry.models[0].context_window == 1_000_000  # 后台刷新补上的窗口
    assert "sk-wizard-12345678" not in capsys.readouterr().out  # 明文 key 不落屏幕


def test_login_saves_key_without_blocking_on_discovery(tmp_path, monkeypatch):
    """登录向导不做同步 models.list()；只登记后台刷新（对齐 pi，不冻结登录）。"""
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    calls: list[object] = []
    monkeypatch.setattr(commands_mod, "discover_models", lambda *a, **k: calls.append("sync") or [])
    monkeypatch.setattr(commands_mod, "_start_catalog_refresh", lambda *a, **k: calls.append("bg"))
    _fake_io(monkeypatch, [""], "sk-wizard-12345678")
    result = dispatch_command("/login deepseek", _ctx(Agent(llm=object(), tools=[])))
    assert "已登录 deepseek" in result and "deepseek-flash" in result
    assert "后台刷新中" in result
    assert calls == ["bg"]  # 没有同步发现，只有后台任务
    entry = ModelsConfig.load().get("deepseek")
    assert (entry.model, entry.base_url) == ("deepseek-flash", "https://api.deepseek.com")


def test_refresh_catalog_keeps_current_model_and_drops_logged_out(tmp_path, monkeypatch):
    entry = ProviderEntry(
        "https://api.deepseek.com",
        "sk-x-12345678",
        "deepseek-flash",
        [ModelEntry("deepseek-flash")],
    )
    _write_models(tmp_path, monkeypatch, {"deepseek": entry})
    monkeypatch.setattr(
        "polya.commands.discover_models",
        lambda base_url, api_key, timeout=15.0: [ModelEntry("deepseek-v4-pro", 1_000_000)],
    )
    assert "已刷新" in commands_mod._refresh_catalog("deepseek", "https://api.deepseek.com", "k")
    ids = [m.id for m in ModelsConfig.load().get("deepseek").models]
    assert "deepseek-v4-pro" in ids and "deepseek-flash" in ids  # 当前模型不丢，active 不悬空
    # 期间已 /logout：过期结果丢弃，不复活 provider
    assert commands_mod._refresh_catalog("ghost", "https://x", "k") == ""


def test_login_custom_flow_falls_back_when_discovery_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")

    def _boom(base_url, api_key, timeout=10.0):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("polya.commands.discover_models", _boom)
    _run_refresh_inline(monkeypatch)
    _fake_io(
        monkeypatch,
        ["box", "http://localhost:8000/v1", "qwen3"],
        "sk-local-999888777666",
    )
    result = dispatch_command("/login custom", _ctx(Agent(llm=object(), tools=[])))
    assert "已登录 box" in result and "qwen3" in result
    entry = ModelsConfig.load().get("box")
    assert (entry.base_url, entry.model) == ("http://localhost:8000/v1", "qwen3")
    assert "未能获取模型列表" in capsys.readouterr().out


def test_login_requires_key_and_url(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    _fake_io(monkeypatch, [""], "")  # key 为空 → 取消
    result = dispatch_command("/login zai", _ctx(Agent(llm=object(), tools=[])))
    assert result == "已取消登录。"
    assert ModelsConfig.load().providers == {}


def test_login_rejects_custom_name_collision(tmp_path, monkeypatch):
    monkeypatch.setattr("polya.models.default_path", lambda: tmp_path / "models.json")
    _fake_io(monkeypatch, ["zai"], "sk-x-12345678")
    result = dispatch_command("/login custom", _ctx(Agent(llm=object(), tools=[])))
    assert "重复" in result


def test_logout_removes_and_reassigns_active(tmp_path, monkeypatch):
    _write_models(tmp_path, monkeypatch, FIXTURES, active="claude-max/claude-opus-4-5")
    agent = Agent(llm=object(), tools=[])
    result = dispatch_command("/logout claude-max", _ctx(agent))
    assert "已登出 claude-max" in result
    config = ModelsConfig.load()
    assert config.active == "glm-plan/glm-4.7" and list(config.providers) == ["glm-plan"]
    assert "用法" in dispatch_command("/logout ghost", _ctx(agent))


@pytest.mark.parametrize(
    "text",
    [
        "/login notaprovider",
        "/login zai extra",
        "/logout",
        "/logout a b",
        "/model ghost",
        "/model ghost/model",
    ],
)
def test_model_entry_bad_shapes_rejected(tmp_path, monkeypatch, text):
    _write_models(tmp_path, monkeypatch, FIXTURES)
    agent = Agent(llm=object(), tools=[])
    old_llm = agent.llm
    assert "用法" in handle_command(text, agent)
    assert agent.llm is old_llm


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "/model ",
            ["glm-plan/glm-4.7", "glm-plan/glm-5.3", "claude-max/claude-opus-4-5"],
        ),
        ("/model g", ["glm-plan/glm-4.7", "glm-plan/glm-5.3"]),
        ("/model c", ["claude-max/claude-opus-4-5"]),
        ("/model x", []),
        ("/model glm-plan ", []),
    ],
)
def test_model_argument_completion_is_dynamic(tmp_path, monkeypatch, text, expected):
    _write_models(tmp_path, monkeypatch, FIXTURES)
    completions = [c.text for c in SlashCompleter().get_completions(Document(text), None)]
    assert completions == expected


def test_model_bare_opens_picker_marking_current_and_hints_ctrl_s(tmp_path, monkeypatch, box):
    _write_models(tmp_path, monkeypatch, FIXTURES, active="glm-plan/glm-4.7")
    sent = []
    box._state = {"profile": "glm-plan", "model": "glm-4.7"}
    buffer = Buffer(accept_handler=lambda b: sent.append(b.text))
    buffer.document = Document("/model")
    box._submit(buffer)
    assert buffer.text == "/model " and not sent
    state = buffer.complete_state
    assert [c.text for c in state.completions] == [
        "glm-plan/glm-4.7",
        "glm-plan/glm-5.3",
        "claude-max/claude-opus-4-5",
    ]
    current_meta = next(
        str(c.display_meta) for c in state.completions if c.text == "glm-plan/glm-4.7"
    )
    assert "当前" in current_meta
    assert "Ctrl+S" in box._hint  # 提示设默认
    buffer.go_to_completion(1)
    box._submit(buffer)
    assert sent == ["/model glm-plan/glm-5.3"]  # 选中即一次 Enter 执行（对齐 CC）
