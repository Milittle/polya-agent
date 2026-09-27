"""模型能力声明的测试：档案查找与关键能力位的正确性。"""

from __future__ import annotations

import pytest

from polya.providers import (
    SUBSCRIPTION_PROVIDERS,
    ModelProfile,
    estimate_cost,
    profile_for,
)


def test_profile_for_matches_by_prefix():
    assert profile_for("claude-opus-4-5").context_window == 200_000
    assert profile_for("o3-mini").temperature is None
    assert profile_for("deepseek-reasoner").supports_inplace_tool_edit is False


def test_profile_for_falls_back_to_default():
    default = profile_for("totally-unknown-model")
    assert default == ModelProfile()
    assert profile_for(None) == ModelProfile()
    assert profile_for("").temperature == 0.0


def test_thinking_bound_models_reject_inplace_edit():
    """签名绑定前缀的模型（thinking 回传系）一律不支持原地压缩。"""
    for name in ("deepseek-reasoner", "claude-sonnet-4", "gemini-2.5-pro"):
        assert profile_for(name).supports_inplace_tool_edit is False, name


def test_openai_style_models_allow_inplace_edit():
    """不回传 reasoning 的模型原地替换安全。"""
    for name in ("deepseek-chat", "gpt-4o", "qwen-max"):
        assert profile_for(name).supports_inplace_tool_edit is True, name


def test_glm_prefix_splits_by_generation():
    # glm-5 系 1M（z.ai 官方，2026-09 核实）；glm-4 系 200K。前缀表按代分层，
    # "glm-5" 须声明在 "glm" 前（profile_for 按声明序首个命中）
    assert profile_for("glm-5.3").context_window == 1_000_000
    assert profile_for("glm-5.3-flash").context_window == 1_000_000
    assert profile_for("glm-4.7").context_window == 200_000
    assert profile_for("glm-4.6-air").context_window == 200_000


def test_deepseek_flash_gets_1m_window_and_conservative_compaction():
    # deepseek-flash（V4.1-Flash）：1M 窗口（官方 pricing 页）；thinking 默认开，
    # 压缩保守走摘要重启（同 reasoner 档，宁保守不赌原地替换）
    flash = profile_for("deepseek-flash")
    assert flash.context_window == 1_000_000
    assert flash.supports_inplace_tool_edit is False


def test_estimate_cost_uses_prefix_price_table():
    # deepseek-flash：input 0.3 / output 1.2 / cacheRead 0.006（USD per 1M）
    cost = estimate_cost("deepseek-flash", 1_000_000, 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.3 + 1.2 + 0.006)
    # 更具体的 glm-5.3-flash 不能被 glm-5.3 抢匹配
    flash = estimate_cost("glm-5.3-flash", 1_000_000, 0)
    assert flash == pytest.approx(0.15)
    plain = estimate_cost("glm-5.3", 1_000_000, 0)
    assert plain == pytest.approx(1.4)
    # 未列出的模型与空模型不估价
    assert estimate_cost("totally-unknown", 1_000_000, 0) is None
    assert estimate_cost(None, 10, 10) is None


def test_subscription_providers_marked():
    assert "zai" in SUBSCRIPTION_PROVIDERS
    assert "qwen-token-plan-cn" in SUBSCRIPTION_PROVIDERS
    assert "deepseek" not in SUBSCRIPTION_PROVIDERS
