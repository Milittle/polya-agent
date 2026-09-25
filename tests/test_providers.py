"""模型能力声明的测试：档案查找与关键能力位的正确性。"""

from __future__ import annotations

from mi_z.providers import ModelProfile, profile_for


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
