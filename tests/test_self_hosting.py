"""真实文件与 bash + 确定性模型：技能 → 失败验证 → 压缩 → 修复 → 成功验证。"""

import copy
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

from polya import Agent
from polya.builtin import CODING_SYSTEM_PROMPT, default_tools
from polya.providers import ModelProfile
from polya.skills import SkillCatalog
from polya.todos import TodoStore


def test_development_survives_compaction_and_repairs_failed_check(tmp_path):
    skill = tmp_path / ".polya/skills/develop-polya/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(Path(".polya/skills/develop-polya/SKILL.md").read_text())
    target = tmp_path / "calc.py"
    target.write_text("def add(a, b):\n    return a - b\n")
    verification = (
        f"{shlex.quote(sys.executable)} -B -c 'from calc import add; assert add(2, 3) == 5'"
    )
    todo = {"content": "修复 add 并验证，保持函数签名", "status": "in_progress"}
    actions = [
        ("skill_read", {"name": "develop-polya"}),
        ("read_file", {"path": "calc.py"}),
        ("todo_write", {"items": [todo]}),
        ("bash", {"command": verification}),
        (
            "edit_file",
            {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"},
        ),
        ("bash", {"command": verification}),
        ("todo_write", {"items": [{**todo, "status": "completed"}]}),
    ]

    class DevelopmentModel:
        def __init__(self):
            self.step = 0
            self.compactions = 0
            self.requests = []
            self.schemas = None

        def chat(self, messages, tools=None, on_delta=None):
            if tools is None:
                request = messages[-1]["content"]
                assert "保持函数签名" in request
                assert "develop-polya" in request
                self.compactions += 1
                content = (
                    "目标：修复 calc.py 的 add；保持函数签名。已加载 develop-polya skill，"
                    "已读 calc.py，当前用减法。TODO 正在修复并验证。"
                    "下一步：读取验证结果，修复加法后重新运行检查；尚未验证成功。"
                )
                message = SimpleNamespace(content=content, tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
            self.requests.append(copy.deepcopy(messages))
            if self.schemas is None:
                self.schemas = copy.deepcopy(tools)
            assert tools == self.schemas  # 压缩与 skill 加载不改工具表
            pending = set()
            for msg in messages:
                if msg.get("role") == "assistant":
                    pending.update(call["id"] for call in msg.get("tool_calls", []))
                elif msg.get("role") == "tool":
                    assert msg["tool_call_id"] in pending
                    pending.remove(msg["tool_call_id"])
                elif msg.get("role") == "user":
                    assert not pending
            assert not pending
            if self.step == 4:
                assert self.compactions == 1
                context = json.dumps(messages, ensure_ascii=False)
                assert "develop-polya" in context and "保持函数签名" in context
                assert "退出码 1" in context and "AssertionError" in context
            if self.step == 6:
                assert messages[-1]["role"] == "tool"
                assert messages[-1]["content"].endswith("退出码 0")
            if self.step < len(actions):
                name, args = actions[self.step]
                call = SimpleNamespace(
                    id=f"c{self.step}",
                    function=SimpleNamespace(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    ),
                )
                message = SimpleNamespace(content=None, tool_calls=[call])
            else:
                message = SimpleNamespace(
                    content="add 已修复；检查由失败变为通过。", tool_calls=None
                )
            self.step += 1
            count = 81000 if self.step == 4 else 1000
            usage = SimpleNamespace(
                prompt_tokens=count, completion_tokens=10, total_tokens=count + 10
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    todos = TodoStore()
    model = DevelopmentModel()
    agent = Agent(
        llm=model,
        tools=default_tools(tmp_path, todos=todos),
        todos=todos,
        system_prompt=CODING_SYSTEM_PROMPT,
        skills=SkillCatalog.discover(tmp_path, tmp_path / "no-user-skills"),
        compress=True,
        context_window=100000,
        keep_recent=2,
        profile=ModelProfile(supports_inplace_tool_edit=False),
        prefix_check=True,
        max_steps=10,
    )
    result = agent.run("使用 $develop-polya 修复 calc.py 的加法，保持函数签名，并运行检查。")
    assert "检查由失败变为通过" in result
    assert "return a + b" in target.read_text()
    assert todos.as_dicts()[0]["status"] == "completed"
    assert model.compactions == 1
    assert "develop-polya" in agent.tools.call("history_read", {"snapshot": "1", "message": 3})
    agent.tools.call("kill_bash", {})
