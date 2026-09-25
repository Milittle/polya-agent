"""CLI 的测试：斜杠命令分发、终端审批回调、单任务模式与 REPL 主循环。

REPL 和 -p 模式通过 monkeypatch 注入假 LLM（绕过 API key 构造）与假 input 驱动。
"""

from __future__ import annotations

from types import SimpleNamespace

from mi_z import Agent, tool
from mi_z.cli import LiveStatusBar, build_agent, handle_command, main, parse_args, terminal_approve
from mi_z.todos import TodoStore


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, messages, tools=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=self._replies.pop(0))], usage=None)

    @property
    def model(self):
        return "fake-model"


def make_agent(llm=None):
    return Agent(llm=llm or ScriptedLLM([]), tools=[add], todos=TodoStore(), plan_capable=True)


@tool
def add(a: int, b: int) -> str:
    """把两个整数相加（测试用只读工具）。"""
    return str(a + b)


@tool(name="write_thing", dangerous=True)
def write_thing(content: str) -> str:
    """写点东西（测试用危险工具）。"""
    return f"已写入: {content}"


# ---------- 斜杠命令 ----------


def test_quit_commands_return_none():
    agent = make_agent()
    assert handle_command("/exit", agent) is None
    assert handle_command("/quit", agent) is None


def test_status_and_todos_rendering():
    agent = make_agent()
    agent.todos.rewrite([{"content": "修 bug", "status": "in_progress"}])
    assert "[1] [in_progress] 修 bug" in handle_command("/todos", agent)
    assert "（TODO 清单为空）" in handle_command("/todos", make_agent())

    status = handle_command("/status", agent)
    assert "执行" in status and "token 用量" in status


def test_plan_toggle():
    agent = make_agent()
    assert agent.plan_mode is False
    assert "已进入规划模式" in handle_command("/plan on", agent)
    assert agent.plan_mode is True
    assert "规划" in handle_command("/status", agent)  # 状态反映模式
    assert "已退出规划模式" in handle_command("/plan off", agent)
    assert agent.plan_mode is False
    assert "用法" in handle_command("/plan", agent)  # 缺参数给出用法


def test_plan_toggle_without_registered_tool_warns():
    plain = Agent(llm=ScriptedLLM([]), tools=[add], todos=TodoStore())  # 未 plan_capable
    result = handle_command("/plan on", plain)
    assert plain.plan_mode is True
    assert "未注册 exit_plan_mode" in result


def test_reset_clears_session():
    agent = make_agent(ScriptedLLM([make_message("好")]))
    agent.run("hi")
    agent.todos.rewrite([{"content": "任务", "status": "pending"}])
    assert "已清空" in handle_command("/reset", agent)
    assert agent.history == [] and len(agent.todos) == 0


def test_unknown_command():
    assert "未知命令" in handle_command("/nope", make_agent())


# ---------- 终端审批 ----------


def test_terminal_approve_yes_no_always(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    approve = terminal_approve(interactive=True)
    assert approve(add, {}) is True  # 只读工具直接放行，不询问

    answers = iter(["n", "a"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    approve = terminal_approve(interactive=True)
    assert approve(write_thing, {"content": "x"}) is False  # n → 拒绝
    assert approve(write_thing, {"content": "y"}) is True  # a → 放行

    # a 之后同工具不再询问：input 耗尽会抛 StopIteration，若被询问即测试失败
    assert approve(write_thing, {"content": "z"}) is True


def test_terminal_approve_noninteractive_rejects(monkeypatch):
    def fail(prompt=""):
        raise AssertionError("非交互环境不应交互询问")

    monkeypatch.setattr("builtins.input", fail)
    approve = terminal_approve(interactive=False)
    assert approve(write_thing, {"content": "x"}) is False
    assert approve(add, {}) is True  # 只读不受影响


# ---------- build_agent / main ----------


def test_build_agent_registers_exit_plan_mode(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path)]), llm=ScriptedLLM([]))
    assert "exit_plan_mode" in [t.name for t in agent.tools]
    assert agent.plan_mode is False
    assert agent.max_steps == 25

    planned = build_agent(parse_args(["--root", str(tmp_path), "--plan"]), llm=ScriptedLLM([]))
    assert planned.plan_mode is True


def test_yes_mode_disables_approval(tmp_path):
    agent = build_agent(parse_args(["--root", str(tmp_path), "--yes"]), llm=ScriptedLLM([]))
    assert agent.approve is None and agent.approve_plan is None  # None = 不拦截


def test_prompt_mode_prints_answer(monkeypatch, tmp_path, capsys):
    fake = ScriptedLLM([make_message("答案是 42")])
    monkeypatch.setattr("mi_z.cli.LLM", lambda **kwargs: fake)
    code = main(["--root", str(tmp_path), "-p", "终极问题的答案"])
    assert code == 0
    assert "答案是 42" in capsys.readouterr().out


def test_repl_loop_runs_and_exits(monkeypatch, tmp_path, capsys):
    fake = ScriptedLLM([make_message("1024")])
    monkeypatch.setattr("mi_z.cli.LLM", lambda **kwargs: fake)
    inputs = iter(["2 的 10 次方", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))

    assert main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1024" in out and "工具调用" in out  # 答案 + /status 输出


def test_live_status_bar_tracks_events():
    """状态栏读事件更新轮次与当前工具；新一轮开始清掉上一轮的工具名。"""
    bar = LiveStatusBar()
    bar.update("iteration", {"step": 2, "max_steps": 25})
    bar.update("tool_call", {"name": "read_file"})
    label = bar.render().text.plain
    assert "第 2/25 轮" in label and "read_file" in label

    bar.update("iteration", {"step": 3, "max_steps": 25})
    assert "read_file" not in bar.render().text.plain
