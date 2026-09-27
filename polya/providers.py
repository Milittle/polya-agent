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
    # 推理档位的请求形态（一家一策，用户 2026-09-27 拍板；见 reasoning_params）：
    #   "none"             不发任何档位参数（默认；未知/不支持的端点安全侧）
    #   "reasoning_effort" OpenAI o 系 / gpt-5：reasoning_effort=low|medium|high
    #   "thinking_toggle"  GLM / DeepSeek：thinking={type: enabled|disabled}
    reasoning_style: str = "none"
    note: str = ""


_DEFAULT = ModelProfile()

# 预置档案：按模型名前缀匹配。不做穷举——未知模型回落默认值（安全侧：
# 原地压缩 + 温度 0，绝大多数 OpenAI 兼容服务的常态）。
PROFILES: dict[str, ModelProfile] = {
    # DeepSeek reasoner/V3.2-thinking：interleaved thinking 要求 reasoning_content
    # 原样回传（工具调用轮之间），回传即绑定前缀 → 压缩须摘要重启
    "deepseek-reasoner": ModelProfile(
        supports_inplace_tool_edit=False,
        reasoning_style="thinking_toggle",
        note="interleaved thinking：reasoning_content 回传绑定前缀",
    ),
    # GLM-5 系（z.ai / bigmodel coding plan 的 glm-5.x）：1M 窗口（官方，2026-09 核实）。
    # 前缀须排在 "glm" 前：profile_for 按声明序首个 startswith 命中
    "glm-5": ModelProfile(context_window=1_000_000, note="1M 窗口（z.ai 官方）"),
    # GLM-4 系：200K 窗口（官方，2026-09 核实）
    "glm": ModelProfile(
        context_window=200_000,
        reasoning_style="thinking_toggle",
        note="200K 窗口（z.ai 官方）；thinking 开关",
    ),
    # DeepSeek V4.1 Flash（deepseek-flash）：1M 窗口 + 384K 输出（官方 pricing 页，
    # 2026-09 核实；deepseek-chat/reasoner 已是遗留名）。thinking 默认开，按
    # reasoner 同款保守档：摘要重启压缩（宁保守不赌原地替换）
    "deepseek-flash": ModelProfile(
        context_window=1_000_000,
        supports_inplace_tool_edit=False,
        reasoning_style="thinking_toggle",
        note="1M 窗口；thinking 默认开，保守同 reasoner",
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
    # 但不接受自定义温度（None = 不传，用服务端默认）；支持 reasoning_effort。
    "o1": ModelProfile(
        temperature=None, reasoning_style="reasoning_effort", note="仅默认温度；reasoning_effort"
    ),
    "o3": ModelProfile(
        temperature=None, reasoning_style="reasoning_effort", note="仅默认温度；reasoning_effort"
    ),
    "gpt-5": ModelProfile(
        temperature=None, reasoning_style="reasoning_effort", note="仅默认温度；reasoning_effort"
    ),
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


# 可选的推理档位（/thinking）；"none" 表示模型不支持，命令会明确提示。
REASONING_LEVELS = ("off", "low", "medium", "high")


def reasoning_params(style: str, level: str) -> dict:
    """把统一档位翻译成厂商请求字段（一家一策）；纯函数，离线可测。

    level ∈ REASONING_LEVELS。style 为 ``ModelProfile.reasoning_style``；未知/"none"
    返回空（不污染请求）。
    """
    if style == "reasoning_effort":
        # OpenAI o 系 / gpt-5：off 映射到极低推理（无法真正关闭思考），gpt-5 支持 minimal。
        effort = {"off": "minimal", "low": "low", "medium": "medium", "high": "high"}
        return {"reasoning_effort": effort[level]}
    if style == "thinking_toggle":
        # GLM / DeepSeek：只有开/关，档位粒度忽略（非 off 视为开）。
        return {"thinking": {"type": "disabled" if level == "off" else "enabled"}}
    return {}


# 订阅制订阅 provider（coding plan / token plan）：按套餐计费，状态栏以 ``(sub)``
# 代替金额，不做 token 估价。
SUBSCRIPTION_PROVIDERS = frozenset(
    {"zai", "zai-coding-cn", "qwen-token-plan", "qwen-token-plan-cn"}
)

# 每 1M token 的美元价 ``(input, output, cacheRead, cacheWrite)``，按模型名前缀匹配。
# 值抄自 pi 内嵌的模型目录（``@earendil-works/pi-ai/dist/providers/data/*.json``，
# 2026-09 核实）；未列出的模型不显示金额。更具体的模型名要排在更靠前。
#
# 更新策略（决策）：静态表，随 polya 版本发布刷新，不做运行时更新管道。
# 后续若要动态化，参考 pi 的远程目录：``GET {pi.dev}/api/models/providers/<id>``，
# ETag 条件请求 + 本地缓存 lastModified 竞争（比内置目录新才生效）+ 4h 节流，
# 失败静默回退本表。
PRICES: dict[str, tuple[float, float, float, float]] = {
    # DeepSeek
    "deepseek-flash": (0.3, 1.2, 0.006, 0.0),
    "deepseek-v4-pro": (1.32, 3.96, 0.044, 0.0),
    # GLM（glm-5.3-flash 必须在 glm-5.3 前）
    "glm-5.3-flash": (0.15, 0.5, 0.03, 0.0),
    "glm-5.3": (1.4, 4.4, 0.26, 0.0),
    "glm-5.2": (1.4, 4.4, 0.26, 0.0),
    "glm-4.7": (0.6, 2.2, 0.11, 0.0),
    # Kimi / Moonshot（区分大小写：开放平台 vs Together）
    "moonshotai/kimi-k2.6": (0.95, 4.0, 0.16, 0.0),
    "moonshotai/Kimi-K2.6": (1.2, 4.5, 0.2, 0.0),
    "kimi-k2.6": (0.95, 4.0, 0.16, 0.0),
    "kimi-k2.7-code": (0.95, 4.0, 0.19, 0.0),
    # Groq
    "openai/gpt-oss-120b": (0.15, 0.6, 0.075, 0.0),
    "gpt-oss-120b": (0.15, 0.6, 0.075, 0.0),
}


def estimate_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """按 ``PRICES`` 估算累计花费；无匹配价格返回 ``None``（状态栏不显示金额）。

    ``input_tokens`` 是**未命中缓存**的输入（pi 同款语义：``prompt - cacheRead``）；
    缓存读取按 cacheRead 单价单独计，不按 input 单价重复计。
    """
    if not model:
        return None
    for prefix, (in_price, out_price, read_price, write_price) in PRICES.items():
        if model.startswith(prefix):
            return (
                input_tokens * in_price
                + output_tokens * out_price
                + cache_read_tokens * read_price
                + cache_write_tokens * write_price
            ) / 1_000_000
    return None
