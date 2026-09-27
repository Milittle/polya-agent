"""Per-vendor reasoning levels: /thinking + request-param mapping."""

from __future__ import annotations

from polya import Agent
from polya.commands import CommandContext, dispatch_command
from polya.llm import LLM
from polya.providers import reasoning_params


class _FakeLLM:
    def __init__(self, model: str, style: str) -> None:
        self.model = model
        self.reasoning_style = style
        self.thinking_level: str | None = None


def test_reasoning_params_per_vendor():
    assert reasoning_params("reasoning_effort", "high") == {"reasoning_effort": "high"}
    assert reasoning_params("reasoning_effort", "off") == {"reasoning_effort": "minimal"}
    assert reasoning_params("thinking_toggle", "off") == {"thinking": {"type": "disabled"}}
    assert reasoning_params("thinking_toggle", "medium") == {"thinking": {"type": "enabled"}}
    assert reasoning_params("none", "high") == {}


def test_thinking_command_sets_level():
    llm = _FakeLLM("gpt-5", "reasoning_effort")
    agent = Agent(llm=llm, tools=[])
    assert "high" in dispatch_command("/thinking high", CommandContext(agent))
    assert llm.thinking_level == "high"


def test_thinking_command_reports_unsupported_model():
    llm = _FakeLLM("gpt-4o", "none")
    agent = Agent(llm=llm, tools=[])
    result = dispatch_command("/thinking high", CommandContext(agent))
    assert "没有可切换的推理档位" in result
    assert llm.thinking_level is None


def test_llm_injects_reasoning_kwargs():
    llm = LLM(model="gpt-5", api_key="test")
    assert llm.reasoning_style == "reasoning_effort"
    llm.thinking_level = "low"
    assert llm._base_kwargs([], None)["reasoning_effort"] == "low"

    toggle = LLM(model="deepseek-reasoner", api_key="test")
    toggle.thinking_level = "off"
    assert toggle._base_kwargs([], None)["thinking"] == {"type": "disabled"}

    plain = LLM(model="gpt-4o", api_key="test")
    plain.thinking_level = "high"  # 档案 style=none：不注入未知字段
    assert "reasoning_effort" not in plain._base_kwargs([], None)
    assert "thinking" not in plain._base_kwargs([], None)
