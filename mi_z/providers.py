"""模型能力声明：Agent 与压缩策略只问能力，不特判模型名。

特判模型名的代码会随模型迭代腐烂（一年前的型号表今天已失效），能力接口
稳定——"是否支持原地替换 tool 结果"比"是不是 claude-xxx"活得久。

能力项的含义（各厂商 thinking 机制的差异，书 2.7）：

- ``reasoning_passthrough``：assistant 回复带 ``reasoning_content``（DeepSeek
  interleaved thinking 等扩展字段）时，是否原样保存并在后续请求中回传。
  回传场景下它与 KV Cache 前缀绑定：**两个压缩点之间必须严格 append-only**，
  否则推理连续性断裂（不是缓存变贵，是 thinking 作废）。
- ``supports_inplace_tool_edit``：压缩能否原地替换旧 tool 消息的 content。
  OpenAI 系 Chat Completions 不回传 reasoning，原地替换安全；Anthropic/
  Gemini/DeepSeek-thinking 的签名绑定前缀，原地替换会使保留的 thinking 全部
  失效——这些模型用「摘要重启」（整段历史压成一条摘要消息，从摘要冷启动）。
- ``temperature=None`` 表示该模型不接受自定义温度（o 系列只允许默认 1），
  发请求时不传该参数。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """一个模型（族）的能力与默认参数。"""

    context_window: int = 128_000
    compress_threshold: float = 0.8
    reasoning_passthrough: bool = True  # 扩展字段，OpenAI 官方 API 不返回也无副作用
    supports_inplace_tool_edit: bool = True
    temperature: float | None = 0.0
    note: str = ""


_DEFAULT = ModelProfile()

# 预置档案：按模型名前缀匹配。不做穷举——未知模型回落默认值（安全侧：
# 原地压缩 + 温度 0，绝大多数 OpenAI 兼容服务的常态）。
PROFILES: dict[str, ModelProfile] = {
    # DeepSeek reasoner/V3.2-thinking：interleaved thinking 要求 reasoning_content
    # 原样回传（工具调用轮之间），回传即绑定前缀 → 压缩须摘要重启
    "deepseek-reasoner": ModelProfile(
        supports_inplace_tool_edit=False,
        note="interleaved thinking：reasoning_content 回传绑定前缀",
    ),
    # Claude 系：thinking block 密码学签名绑定前缀；200K 窗口；温度 1 起步。
    # 原生 Messages API 需渲染层（content blocks），走 OpenAI 兼容端点时无 thinking
    "claude": ModelProfile(
        context_window=200_000,
        supports_inplace_tool_edit=False,
        temperature=1.0,
        note="thinking 签名绑定前缀；原生 API 需中立轨迹渲染层",
    ),
    # OpenAI o 系 / gpt-5：Chat Completions 不回传 reasoning（原地压缩安全），
    # 但不接受自定义温度（None = 不传，用服务端默认）
    "o1": ModelProfile(temperature=None, note="仅默认温度"),
    "o3": ModelProfile(temperature=None, note="仅默认温度"),
    "gpt-5": ModelProfile(temperature=None, note="仅默认温度"),
    # Gemini：thought signature 绑定前缀（函数调用多轮需回传）
    "gemini": ModelProfile(supports_inplace_tool_edit=False, note="thought signature 绑定前缀"),
}


def profile_for(model: str | None) -> ModelProfile:
    """按模型名查档案：前缀匹配（如 'claude-opus-4-5' 命中 'claude'），未命中回落默认。"""
    if not model:
        return _DEFAULT
    for prefix, profile in PROFILES.items():
        if model.startswith(prefix):
            return profile
    return _DEFAULT
