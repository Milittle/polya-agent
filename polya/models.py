"""模型 provider 配置：``~/.polya/models.json`` 的读写、校验与启动解析（票 03）。

polya 只说 OpenAI Chat Completions 一种协议（coding plan 与普通 API 统一）。
配置按 **provider id** 键位：每个 provider 存 ``base_url`` / ``api_key`` / 当前模型 /
已发现的模型目录（``id`` + 可选的 ``context_window``）。密钥文件 0600。

认证录入在 ``/login``（票 04），切换与默认在 ``/model``（票 05）。设计决议见
``.scratch/branding/issues/02-dynamic-model-catalog.md``。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .providers import profile_for


@dataclass(frozen=True)
class Provider:
    """预置 provider：登录时只选名字，端点与建议模型免填（base_url 仍可覆盖）。"""

    label: str  # 选项器显示
    base_url: str
    model: str = ""  # 建议默认模型；空则登录时必填（如跨厂商聚合）


# 首批预置（2026-09-27 拍板 8 组，值抄自 pi 的 openai-completions 目录；其余
# fireworks / huggingface / cerebras / baseten / ant-ling / cloudflare-* / opencode* /
# xiaomi* 后补，各一行数据）。
PROVIDERS: dict[str, Provider] = {
    "openrouter": Provider(
        "OpenRouter（跨厂商聚合）", "https://openrouter.ai/api/v1", "moonshotai/kimi-k2.6"
    ),
    "deepseek": Provider("DeepSeek API", "https://api.deepseek.com", "deepseek-flash"),
    "zai": Provider("z.ai coding plan（国际）", "https://api.z.ai/api/coding/paas/v4", "glm-5.3"),
    "zai-coding-cn": Provider(
        "z.ai coding plan（国内 bigmodel）",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        "glm-5.3",
    ),
    "moonshotai": Provider("Moonshot Kimi（国际）", "https://api.moonshot.ai/v1", "kimi-k2.6"),
    "moonshotai-cn": Provider("Moonshot Kimi（国内）", "https://api.moonshot.cn/v1", "kimi-k2.6"),
    "groq": Provider("Groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "together": Provider("Together AI", "https://api.together.ai/v1", "moonshotai/Kimi-K2.6"),
    "nvidia": Provider("NVIDIA NIM", "https://integrate.api.nvidia.com/v1", ""),
    "qwen-token-plan": Provider(
        "Qwen Token Plan（国际）",
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "qwen3.7-max",
    ),
    "qwen-token-plan-cn": Provider(
        "Qwen Token Plan（国内）",
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "qwen3.7-max",
    ),
}


@dataclass
class ModelEntry:
    """目录里的一条模型：发现自端点，``context_window`` 仅在端点报了才有。"""

    id: str
    context_window: int | None = None


@dataclass
class ProviderEntry:
    """一个已登录 provider：端点、凭据、当前模型与已发现的模型目录。"""

    base_url: str
    api_key: str
    model: str
    models: list[ModelEntry] = field(default_factory=list)

    def window_for(self, model: str) -> int | None:
        for item in self.models:
            if item.id == model and item.context_window:
                return item.context_window
        return None


def default_path() -> Path:
    return Path.home() / ".polya" / "models.json"


def mask_key(key: str) -> str:
    """展示打码：尾四位（``sk-…abcd``），过短全遮。"""
    return f"{key[:3]}…{key[-4:]}" if len(key) > 8 else "…"


def host_of(base_url: str) -> str:
    return urlparse(base_url).netloc


def format_context_window(tokens: int) -> str:
    """128000 → 128k，1000000 → 1M（底栏与选项器展示用）。"""
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    if tokens >= 1000:
        return f"{tokens // 1000}k"
    return str(tokens)


def format_tokens(count: int) -> str:
    """token 数的人话格式：1234 → 1.2k、128000 → 128k、1500000 → 1.5M。"""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        value = count / 1000
        return f"{value:.0f}k" if value >= 100 else f"{value:.1f}k"
    value = count / 1_000_000
    return f"{value:.0f}M" if value >= 100 else f"{value:.1f}M"


def resolve_context_window(model: str | None, entry: ProviderEntry | None = None) -> int:
    """窗口解析（票 03）：条目发现值 → 静态前缀表 → 128k 默认。"""
    if entry is not None and model is not None:
        own = entry.window_for(model)
        if own:
            return own
    return profile_for(model).context_window


def discover_models(base_url: str, api_key: str, timeout: float = 10.0) -> list[ModelEntry]:
    """``GET {base_url}/models``：拿 id，顺手读 ``context_length`` / ``max_model_len``。

    只在端点真的报了窗口时才填 ``context_window``，让静态表保持兜底活性。失败抛
    异常（网络/401/超时），由调用方决定回退与提示（票 04 的失败语义）。
    """
    from openai import OpenAI  # 局部导入：仅发现路径需要，避免启动期开销

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    try:
        page = client.models.list()
    finally:
        client.close()
    entries: list[ModelEntry] = []
    for item in page.data:
        raw = getattr(item, "context_length", None) or getattr(item, "max_model_len", None)
        try:
            window = int(raw) if raw else None
        except (TypeError, ValueError):
            window = None
        if window is not None and window <= 0:
            window = None
        entries.append(ModelEntry(id=item.id, context_window=window))
    return entries


def _validate_provider_id(provider_id: str) -> None:
    if (
        not provider_id
        or provider_id.strip() != provider_id
        or any(c.isspace() for c in provider_id)
    ):
        raise ValueError("provider 名不能为空、含空白或首尾空白")


def _validate_entry(provider_id: str, entry: ProviderEntry) -> None:
    if not entry.base_url.startswith(("http://", "https://")) or not host_of(entry.base_url):
        raise ValueError(f"{provider_id}: base_url 须是带主机的 http(s) 地址：{entry.base_url}")
    if not entry.api_key:
        raise ValueError(f"{provider_id}: api_key 不能为空")
    if not entry.model:
        raise ValueError(f"{provider_id}: model 不能为空")
    seen: set[str] = set()
    for item in entry.models:
        if not item.id or any(c.isspace() for c in item.id):
            raise ValueError(f"{provider_id}: 模型名不能为空或含空白：{item.id!r}")
        if item.id in seen:
            raise ValueError(f"{provider_id}: 模型重复：{item.id}")
        seen.add(item.id)
        if item.context_window is not None and item.context_window <= 0:
            raise ValueError(f"{provider_id}: {item.id} 的 context_window 须为正数")


@dataclass
class ModelsConfig:
    providers: dict[str, ProviderEntry] = field(default_factory=dict)
    active: str | None = None  # "provider/model" 引用；启动默认模型

    # ---- 读写 ----

    @classmethod
    def load(cls, path: Path | None = None) -> ModelsConfig:
        """缺文件返回空配置；坏文件（JSON 坏、字段缺、active 悬空）启动即报。

        旧格式（``profiles``）自动迁移到 ``providers``：每个 profile 转成一条
        provider 条目，``active`` 从名字转成 ``name/model``。
        """
        path = path or default_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 {path}：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path} 的顶层结构不合法：应为对象")
        try:
            if "providers" in data:
                config = cls(
                    providers=_parse_providers(data["providers"]), active=data.get("active")
                )
            elif "profiles" in data:
                config = _migrate_profiles(data)
            else:
                return cls()
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"{path} 的 providers/profiles 字段不合法：{exc}") from exc
        config._validate_all()
        return config

    def save(self, path: Path | None = None) -> None:
        """原子落盘（临时文件 + replace），密钥文件首建即 0600、已存在也归位。"""
        path = path or default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {
                "active": self.active,
                "providers": {
                    pid: {
                        "base_url": entry.base_url,
                        "api_key": entry.api_key,
                        "model": entry.model,
                        "models": [asdict(item) for item in entry.models],
                    }
                    for pid, entry in self.providers.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _validate_all(self) -> None:
        for provider_id, entry in self.providers.items():
            _validate_provider_id(provider_id)
            _validate_entry(provider_id, entry)
        if self.active is not None:
            self._split_ref(self.active)  # 悬空即报

    def _split_ref(self, ref: str) -> tuple[str, str]:
        if "/" not in ref:
            raise ValueError(f"active 须是 provider/model 引用：{ref}")
        provider_id, model = ref.split("/", 1)
        entry = self.providers.get(provider_id)
        if entry is None:
            raise ValueError(f"active 指向不存在的 provider：{provider_id}")
        if model != entry.model and all(item.id != model for item in entry.models):
            raise ValueError(f"active 指向不存在的模型：{ref}")
        return provider_id, model

    # ---- 管理 ----

    def get(self, provider_id: str) -> ProviderEntry | None:
        return self.providers.get(provider_id)

    def add(self, provider_id: str, entry: ProviderEntry) -> None:
        """新增 provider；同名拒绝。首个自动成为 active（开箱即用）。"""
        _validate_provider_id(provider_id)
        if provider_id in self.providers:
            raise ValueError(f"已存在 provider：{provider_id}")
        _validate_entry(provider_id, entry)
        self.providers[provider_id] = entry
        if self.active is None:
            self.active = f"{provider_id}/{entry.model}"

    def upsert(self, provider_id: str, entry: ProviderEntry) -> None:
        """登录时新增或更新（重跑 ``/login`` 覆盖旧端点/凭据/目录）。"""
        _validate_provider_id(provider_id)
        _validate_entry(provider_id, entry)
        self.providers[provider_id] = entry
        if self.active is None:
            self.active = f"{provider_id}/{entry.model}"

    def remove(self, provider_id: str) -> ProviderEntry:
        entry = self.providers.pop(provider_id, None)
        if entry is None:
            raise ValueError(f"未找到 provider：{provider_id}")
        if self.active is not None and self.active.startswith(f"{provider_id}/"):
            first = next(iter(self.providers), None)
            self.active = f"{first}/{self.providers[first].model}" if first else None
        return entry

    def use(self, ref: str) -> tuple[str, str]:
        """设置默认启动模型（``provider/model``）；校验引用存在，返回拆分结果。"""
        provider_id, model = self._split_ref(ref)
        self.active = ref
        return provider_id, model

    def refs(self) -> list[str]:
        """全部 ``provider/model`` 引用（每 provider 的目录，目录空则用当前模型）。"""
        out: list[str] = []
        for provider_id, entry in self.providers.items():
            ids = [item.id for item in entry.models] or [entry.model]
            out.extend(f"{provider_id}/{model}" for model in ids)
        return out

    def active_ref(self) -> tuple[str, str] | None:
        if self.active is None:
            return None
        return self._split_ref(self.active)

    def active_entry(self) -> tuple[str, ProviderEntry] | None:
        pair = self.active_ref()
        if pair is None:
            return None
        return pair[0], self.providers[pair[0]]

    def active_model(self) -> str | None:
        pair = self.active_ref()
        return pair[1] if pair else None


def _parse_providers(raw: object) -> dict[str, ProviderEntry]:
    if not isinstance(raw, dict):
        raise TypeError("providers 应为对象")
    providers: dict[str, ProviderEntry] = {}
    for provider_id, item in raw.items():
        models = [
            ModelEntry(id=m["id"], context_window=m.get("context_window"))
            for m in item.get("models", [])
        ]
        providers[provider_id] = ProviderEntry(
            base_url=item["base_url"],
            api_key=item["api_key"],
            model=item["model"],
            models=models,
        )
    return providers


def _migrate_profiles(data: dict) -> ModelsConfig:
    """旧 ``profiles`` → 新 ``providers``（票 03 一次性迁移）。"""
    providers: dict[str, ProviderEntry] = {}
    for item in data.get("profiles", []):
        name = item["name"]
        if name in providers:
            raise ValueError(f"重复的 profile：{name}")
        providers[name] = ProviderEntry(
            base_url=item["base_url"],
            api_key=item["api_key"],
            model=item["model"],
            models=[ModelEntry(id=item["model"])],
        )
    old_active = data.get("active")
    active = f"{old_active}/{providers[old_active].model}" if old_active in providers else None
    return ModelsConfig(providers=providers, active=active)


def resolve_connection(
    model_flag: str | None,
    base_url_flag: str | None,
    api_key_flag: str | None,
    config: ModelsConfig | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """启动解析（票 14/03）：旗标 > active provider > 环境变量（由 LLM 内部兜底）。

    返回 ``(model, base_url, api_key, provider_id)``，未定字段为 None。active 存在
    时未给旗标的字段整体取自它，不与环境变量逐字段混搭——防跨端点串 key。旗标只
    一次性覆盖，不改写 active。
    """
    active = (config if config is not None else ModelsConfig.load()).active_entry()
    if active is None:
        return model_flag, base_url_flag, api_key_flag, None
    provider_id, entry = active
    return (
        model_flag or entry.model,
        base_url_flag or entry.base_url,
        api_key_flag or entry.api_key,
        provider_id,
    )
