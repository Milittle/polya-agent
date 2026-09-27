"""上下文压缩的测试：可压目标识别、合并压缩与回填、降级、Agent 触发与熔断。"""

from __future__ import annotations

from types import SimpleNamespace

from polya import Agent, tool
from polya.compact import (
    COMPRESS_MARKER,
    compact_messages,
    compact_restart,
    compressible_indices,
    find_restart_split,
    stale_status_indices,
)


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


@tool
def echo(text: str) -> str:
    """原样返回（测试用只读工具）。"""
    return f"echo: {text}"


def usage(prompt=0, completion=1):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


class ScriptedLLM:
    """replies 按序返回（压缩调用与主调用共用），并记录每次请求。"""

    def __init__(self, replies, usages=None):
        self._replies = list(replies)
        self._usages = list(usages) if usages is not None else [None] * len(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_delta=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        message = self._replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=self._usages.pop(0)
        )


def sample_history():
    """含两条 tool 结果与两条状态栏的 8 条历史（keep=3 时下标 0-4 在保留区外）。"""
    return [
        {"role": "user", "content": "任务"},  # 0
        {"role": "user", "content": "<agent_status>\n第1轮\n</agent_status>"},  # 1
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "a", "function": {"name": "read_file", "arguments": "{}"}}],
        },  # 2
        {"role": "tool", "tool_call_id": "a", "content": "x" * 5000},  # 3
        {"role": "user", "content": "<agent_status>\n第2轮\n</agent_status>"},  # 4
        {"role": "assistant", "content": "中间结论"},  # 5
        {"role": "user", "content": "<agent_status>\n第3轮\n</agent_status>"},  # 6
        {"role": "tool", "tool_call_id": "b", "content": "近期输出"},  # 7
    ]


# ---------- 目标识别 ----------


def test_compressible_indices_respects_marker_and_keep():
    history = sample_history()
    assert compressible_indices(history, keep=3) == [3]  # 下标 7 在保留区（7 >= 8-3）

    history[3]["content"] = COMPRESS_MARKER + " 已压过"
    assert compressible_indices(history, keep=3) == []  # 防重复：已压的跳过


def test_stale_status_indices_only_old_status():
    history = sample_history()
    assert stale_status_indices(history, keep=3) == [1, 4]  # 6 在保留区外？8-3=5，6≥5 保留
    assert stale_status_indices(history, keep=1) == [1, 4, 6]


# ---------- compact_messages ----------


def test_compact_replaces_and_preserves_structure():
    llm = ScriptedLLM([make_message(content="#3: 第一条工具结果的摘要")])
    new = compact_messages(llm, sample_history(), keep=3, query="测试任务")

    compress_call = llm.calls[0]
    assert compress_call["tools"] is None  # 压缩调用不带工具
    assert "测试任务" in compress_call["messages"][1]["content"]  # query 注入（任务感知）
    assert "中段截断" in compress_call["messages"][1]["content"]  # 超长内容截断

    tool_a = next(m for m in new if m.get("tool_call_id") == "a")
    assert tool_a["content"].startswith(COMPRESS_MARKER)
    assert "原 5000 字符" in tool_a["content"]
    assert tool_a["content"].endswith("第一条工具结果的摘要")

    # 结构与保留：条数只少在删掉的状态栏；tool_call_id 配对与近期 tool 原样
    assert len(new) == 6  # 8 条 - 2 条旧状态栏
    assert next(m for m in new if m.get("tool_call_id") == "b")["content"] == "近期输出"
    assert {"role": "user", "content": "任务"} in new
    assert {"role": "assistant", "content": "中间结论"} in new


def test_compact_parse_failure_merges_into_earliest():
    llm = ScriptedLLM([make_message(content="综合摘要：一切尽在不言中")])  # 无编号格式
    history = sample_history()
    new = compact_messages(llm, history, keep=0, query="q")  # 两条 tool 都出保留区

    tools = [m for m in new if m.get("role") == "tool"]
    assert "综合摘要" in tools[0]["content"]  # 并入最早一条
    assert tools[1]["content"] == f"{COMPRESS_MARKER} (内容已并入前述摘要)"
    assert history[3]["content"] == "x" * 5000  # 原列表不被修改


def test_compact_partial_parse_keeps_missing_original():
    llm = ScriptedLLM([make_message(content="#7: 只压了第二条")])  # 缺 #3
    new = compact_messages(llm, sample_history(), keep=0, query="q")
    tools = [m for m in new if m.get("role") == "tool"]
    assert tools[0]["content"] == "x" * 5000  # 缺编号的保留原内容，下次再压
    assert "只压了第二条" in tools[1]["content"]


def test_compact_without_targets_skips_llm():
    llm = ScriptedLLM([])
    assert compact_messages(llm, sample_history(), keep=8, query="q") is None
    assert llm.calls == []


# ---------- Agent 集成：触发 / 保留区 / 熔断 ----------


def test_agent_triggers_compression_on_threshold():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "echo", '{"text": "一"}')]),
            make_message(tool_calls=[make_tool_call("c2", "echo", '{"text": "二"}')]),
            make_message(content="#3: 已压缩的回显"),
            make_message(content="完成"),
        ],
        usages=[usage(prompt=8100), usage(prompt=8100), usage(prompt=50), usage(prompt=50)],
    )
    agent = Agent(
        llm=llm,
        tools=[echo],
        status_bar=True,
        compress=True,
        context_window=10000,
        keep_recent=2,
        max_steps=4,
    )
    assert agent.run("干活") == "完成"

    # 两次触发检查后，第 3 次迭代开头完成压缩：调用序列 主/主/压缩/主
    assert llm.calls[2]["tools"] is None

    after = llm.calls[3]["messages"]  # 压缩后的下一次主请求
    compressed = [
        m for m in after if m.get("role") == "tool" and "已压缩" in (m.get("content") or "")
    ]
    assert compressed and compressed[0]["content"].startswith(COMPRESS_MARKER)
    assert not any("<agent_status>" in (m.get("content") or "") for m in after[:-1])  # 旧状态栏已清

    # 保留区内的近期 tool 结果原样保留（状态栏开启时带调用计数前缀）
    kept = [m for m in agent.history if m.get("role") == "tool"]
    assert kept[-1]["content"].endswith("echo: 二")


def test_agent_below_threshold_or_missing_usage_never_compresses():
    llm = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "echo", '{"text": "一"}')]),
            make_message(content="完成"),
        ],
        usages=[usage(prompt=50), usage(prompt=50)],  # 50 < 100*0.8
    )
    agent = Agent(llm=llm, tools=[echo], compress=True, context_window=10000)
    agent.run("干活")
    assert all(call["tools"] is not None for call in llm.calls)

    no_usage = ScriptedLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "echo", '{"text": "一"}')]),
            make_message(content="完成"),
        ]
    )  # usage 全 None
    agent2 = Agent(llm=no_usage, tools=[echo], compress=True, context_window=10000)
    agent2.run("干活")
    assert all(call["tools"] is not None for call in no_usage.calls)


def test_agent_circuit_breaker_after_three_failures():
    class FailingCompactLLM(ScriptedLLM):
        def chat(self, messages, tools=None, on_delta=None):
            if tools is None:  # 压缩调用直接失败
                self.calls.append({"messages": list(messages), "tools": tools, "failed": True})
                raise RuntimeError("压缩服务不可用")
            return super().chat(messages, tools)

    llm = FailingCompactLLM(
        [
            make_message(tool_calls=[make_tool_call(f"c{i}", "echo", '{"text": "x"}')])
            for i in range(4)
        ]
        + [make_message(content="完成")],
        usages=[usage(prompt=8100)] * 5,
    )
    agent = Agent(
        llm=llm, tools=[echo], compress=True, context_window=10000, keep_recent=1, max_steps=5
    )
    assert agent.run("干活") == "完成"  # 压缩失败不拖垮主任务

    failed = [c for c in llm.calls if c.get("failed")]
    assert len(failed) == 3  # 连续 3 次后熔断，第 4/5 轮不再尝试
    assert agent._compress_failures == 3


# ---------- 摘要重启（thinking 绑定模型的压缩路径） ----------


def test_find_restart_split_lands_on_complete_tool_boundary():
    history = sample_history()  # 8 条，末尾 tool 在下标 7
    split = find_restart_split(history, keep=3)
    assert split == 5  # assistant 也可作为完整工具轮之后的边界
    assert history[split]["role"] == "assistant"

    assert find_restart_split(history, keep=9) is None  # 历史太短，保留区盖全


def test_compact_restart_folds_history_into_summary():
    llm = ScriptedLLM([make_message(content="任务进行到一半：已读 agent.py，核心是 run() 循环")])
    history = sample_history()
    new = compact_restart(llm, history, keep=3, query="理解代码库")

    request = llm.calls[0]["messages"][1]["content"]
    assert "理解代码库" in request
    assert "[assistant] [调用工具:" in request
    assert "agent_status" not in request  # 状态栏噪声不进

    assert new[0]["role"] == "user"
    assert new[0]["content"].startswith("<session_summary>")
    assert "已读 agent.py" in new[0]["content"]
    assert new[1:] == history[5:]  # 保留区始于已回填工具结果之后
    assert history == sample_history() or len(history) == 8  # 原列表不动


def test_agent_uses_restart_for_thinking_bound_models():
    from polya.providers import ModelProfile

    llm = ScriptedLLM(
        # keep_recent=4：第 3 轮开头历史 7 条，切点落在第 3 轮状态栏（user 边界）
        [
            make_message(tool_calls=[make_tool_call("c1", "echo", '{"text": "一"}')]),
            make_message(tool_calls=[make_tool_call("c2", "echo", '{"text": "二"}')]),
            make_message(content="重启摘要：已完成两步探查"),  # 压缩调用消费这条
            make_message(content="完成"),
        ],
        usages=[usage(prompt=8100), usage(prompt=8100), usage(prompt=0), usage(prompt=50)],
    )
    agent = Agent(
        llm=llm,
        tools=[echo],
        status_bar=True,
        compress=True,
        context_window=10000,
        keep_recent=4,
        max_steps=4,
        profile=ModelProfile(supports_inplace_tool_edit=False),
    )
    assert agent.run("干活") == "完成"

    assert llm.calls[2]["tools"] is None  # 压缩调用发生
    # thinking 绑定模型：历史折叠为 [摘要消息, 保留区...]，而非原地替换
    first = llm.calls[3]["messages"][1]
    assert first["content"].startswith("<session_summary>")
    assert "重启摘要" in first["content"]


def test_empty_summary_preserves_original():
    import pytest

    original = sample_history()
    for compact in (compact_messages, compact_restart):
        llm = ScriptedLLM([make_message(content=" \n ")])
        with pytest.raises(RuntimeError, match="空摘要"):
            compact(llm, original, keep=3, query="q")
        assert original == sample_history()


def test_restart_without_status_never_splits_tool_batch():
    history = [
        {"role": "user", "content": "fix"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "b", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
        {"role": "assistant", "content": "next", "reasoning_content": "old thinking"},
    ]
    assert find_restart_split(history, keep=2) == 4
    result = compact_restart(ScriptedLLM([make_message("read A and B")]), history, 2, "fix")
    assert result[-1] == {"role": "assistant", "content": "next"}
    assert "reasoning_content" in history[-1]
    assert find_restart_split(history[:3], keep=0) is None  # b 尚未回填


def test_compaction_checkpoint_preserves_skill_todo_and_raw_history(tmp_path):
    from polya.skills import SkillCatalog
    from tests.test_skills import make_skill

    make_skill(tmp_path / ".polya/skills")
    skills = SkillCatalog.discover(tmp_path, tmp_path / "empty")
    skills.tool().run({"name": "develop"})
    llm = ScriptedLLM([make_message("#3: read file")])
    agent = Agent(llm=llm, compress=True, skills=skills, keep_recent=3)
    history = sample_history()
    history[0]["content"] = "只改解析器，保持 API 不变"
    agent.tree.replace_conversation(history)
    agent.todos.rewrite([{"content": "验证解析器", "status": "in_progress"}])
    compacted = agent._try_compress("原始任务")
    checkpoint = compacted[-1]["content"]
    assert "develop" in checkpoint and "验证解析器" in checkpoint
    assert "只改解析器" in llm.calls[0]["messages"][1]["content"]
    original = agent.tools.call("history_read", {"entry_id": 5})
    assert "x" * 5000 in original
    agent.reset()
    assert agent.tools.call("history_read", {"entry_id": 5}).startswith("Error:")
    assert skills.checkpoint() == ""


def test_missing_usage_large_new_result_triggers_preflight():
    agent = Agent(llm=ScriptedLLM([]), compress=True, context_window=10000)
    agent.tree.replace_conversation([{"role": "user", "content": "x" * 30000}])
    assert agent._should_compress()


def test_full_context_fails_locally_and_preserves_history():
    import pytest

    llm = ScriptedLLM([])
    agent = Agent(llm=llm, compress=True, context_window=10000)
    agent._compress_failures = 3
    with pytest.raises(RuntimeError, match="上下文接近上限"):
        agent.run("x" * 32000)
    assert llm.calls == []
    assert agent.history[0]["content"] == "x" * 32000


# ---------- 触发线（绝对预留下限） / 超窗恢复 / 摘要计费 ----------


def test_full_trigger_absolute_reserve_floor():
    """触发线 = min(窗口×阈值, 窗口−reserve)，半窗封底；大窗口与纯百分比一致。"""
    llm = ScriptedLLM([])
    # 大窗口（20% 余量 ≥ reserve 16384）：行为与原百分比完全一致
    assert Agent(llm=llm, compress=True, context_window=128_000)._full_trigger() == 102_400
    assert Agent(llm=llm, compress=True, context_window=1_000_000)._full_trigger() == 800_000
    # 中小窗口：绝对预留先于百分比，保证触发时留出 reserve
    assert Agent(llm=llm, compress=True, context_window=64_000)._full_trigger() == 47_616
    # 窗口 < 2×reserve：预留放不进 20% 余量，以半窗封底
    assert Agent(llm=llm, compress=True, context_window=32_000)._full_trigger() == 16_000
    assert Agent(llm=llm, compress=True, context_window=8_000)._full_trigger() == 4_000
    # reserve 可调：大预留把触发线拉得更早
    agent = Agent(llm=llm, compress=True, context_window=100_000, reserve_tokens=50_000)
    assert agent._full_trigger() == 50_000


def test_overflow_error_compacts_and_retries_once():
    from polya.llm import is_context_overflow

    class OverflowScriptLLM(ScriptedLLM):
        """剧本里混入 Exception 时抛出而不返回（usage 不消费，保持对齐）。"""

        def chat(self, messages, tools=None, on_delta=None):
            self.calls.append({"messages": list(messages), "tools": tools})
            message = self._replies.pop(0)
            if isinstance(message, Exception):
                raise message
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=self._usages.pop(0)
            )

    overflow = RuntimeError(
        "Error code: 400 - {'error': {'message': \"This model's maximum context length "
        'is 8192 tokens. However, you requested 9000 tokens."}}'
    )
    assert is_context_overflow(overflow)

    llm = OverflowScriptLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "echo", '{"text": "x"}')]),
            overflow,  # 第二次主请求超窗
            make_message(content="#3: 大结果已压成一句"),
            make_message(content="完成"),
        ],
        usages=[usage(prompt=100), usage(prompt=40, completion=80), usage(prompt=60)],
    )
    agent = Agent(
        llm=llm, tools=[echo], compress=True, context_window=100_000, keep_recent=0, max_steps=4
    )
    assert agent.run("干活") == "完成"

    # 调用序列：主(工具) / 主(超窗) / 压缩 / 主(重试成功)
    assert [c["tools"] is None for c in llm.calls] == [False, False, True, False]
    # 摘要计费：压缩调用 40+80 计入 total_usage（连同主调用）
    assert agent.total_usage["prompt_tokens"] == 100 + 40 + 60
    assert agent.total_usage["completion_tokens"] == 1 + 80 + 1
    # 超窗的那次工具结果已替换为摘要
    assert any(
        m.get("role") == "tool" and (m.get("content") or "").startswith(COMPRESS_MARKER)
        for m in agent.history
    )


def test_second_overflow_raises_after_single_recovery():
    class OverflowScriptLLM(ScriptedLLM):
        def chat(self, messages, tools=None, on_delta=None):
            self.calls.append({"messages": list(messages), "tools": tools})
            message = self._replies.pop(0)
            if isinstance(message, Exception):
                raise message
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=self._usages.pop(0)
            )

    import pytest

    overflow = RuntimeError("Error: prompt is too long: 9000 tokens > 8192 maximum")
    llm = OverflowScriptLLM(
        [
            make_message(tool_calls=[make_tool_call("c1", "echo", '{"text": "x"}')]),
            overflow,
            make_message(content="#3: 摘要"),
            overflow,  # 重试后仍超窗：原样上抛，不再恢复
        ],
        usages=[usage(prompt=100), usage(prompt=40)],
    )
    agent = Agent(
        llm=llm, tools=[echo], compress=True, context_window=100_000, keep_recent=0, max_steps=4
    )
    with pytest.raises(RuntimeError, match="prompt is too long"):
        agent.run("干活")
    assert len(llm.calls) == 4
    assert agent._compress_failures == 0  # 压缩本身成功，不是熔断路径


def test_restart_summary_uses_fixed_sections():
    """摘要重启的压缩请求带固定小节模板（进展三态、关键决策与理由等）。"""
    reply = make_message(content="1. 目标与验收：修好解析器\n9. 未完成事项与下一步：补测试")
    llm = ScriptedLLM([reply])
    compact_restart(llm, sample_history(), keep=3, query="修复解析器")
    request = llm.calls[0]["messages"][1]["content"]
    for section in (
        "目标与验收",
        "用户约束与偏好",
        "已完成 / 进行中 / 受阻",
        "关键决策与理由",
        "已读/已改文件",
        "验证证据",
        "失败路径",
        "已加载技能",
        "未完成事项与下一步",
    ):
        assert section in request
