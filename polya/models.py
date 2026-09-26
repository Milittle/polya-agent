"""模型 profile 配置：``~/.polya/models.json`` 的读写、校验与启动解析（票 14）。

polya 只说 OpenAI Chat Completions 一种协议，coding plan 与普通 API 统一为
``(name, base_url, api_key, model)``；厂商预设只是预填模板，不是 provider
抽象。密钥敏感面独立成文件（0600），不与阶段二授权规则的 settings.json 混放。
录入在 CLI（``polya models add``，key 走 getpass 不回显），切换在会话内
（``/models``，见 commands.py）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


# 内置预设（2026-09-26 拍板四家，同日扩 zai 国内线）：已知厂商只选名字，端点
# 与建议模型免填。z.ai 两条都是 coding plan 的 OpenAI 兼容端点（国际
# docs.z.ai/devpack/tool/others、国内 open.bigmodel.cn，均官方核实）。
@dataclass(frozen=True)
class Preset:
    label: str  # 向导菜单显示
    base_url: str
    model: str = ""  # 建议模型；空则向导必填（OpenRouter 跨厂商无默认）


PRESETS: dict[str, Preset] = {
    "zai": Preset("z.ai coding plan（国际）", "https://api.z.ai/api/coding/paas/v4", "glm-5.3"),
    "zai-cn": Preset(
        "z.ai coding plan（国内 bigmodel）",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        "glm-5.3",
    ),
    "deepseek": Preset("DeepSeek API", "https://api.deepseek.com", "deepseek-flash"),
    "openrouter": Preset("OpenRouter（跨厂商）", "https://openrouter.ai/api/v1"),
    "moonshot": Preset("Moonshot Kimi", "https://api.moonshot.cn/v1"),
}


@dataclass(frozen=True)
class Profile:
    name: str
    base_url: str
    api_key: str
    model: str


def default_path() -> Path:
    return Path.home() / ".polya" / "models.json"


def mask_key(key: str) -> str:
    """展示打码：尾四位（``sk-…abcd``），过短全遮。"""
    return f"{key[:3]}…{key[-4:]}" if len(key) > 8 else "…"


def host_of(base_url: str) -> str:
    return urlparse(base_url).netloc


def _validate(profile: Profile, existing: list[str]) -> None:
    # 名字是 /models <名字> 与 CLI 的索引，含空格会拆词，首尾空白是手滑
    if (
        not profile.name
        or profile.name.strip() != profile.name
        or any(c.isspace() for c in profile.name)
    ):
        raise ValueError("profile 名不能为空、含空白或首尾空白")
    if profile.name in existing:
        raise ValueError(f"已存在同名 profile：{profile.name}")
    if not profile.base_url.startswith(("http://", "https://")) or not host_of(profile.base_url):
        raise ValueError(f"base_url 须是带主机的 http(s) 地址：{profile.base_url}")
    if not profile.model:
        raise ValueError("model 不能为空")
    if not profile.api_key:
        raise ValueError("api_key 不能为空")


@dataclass
class ModelsConfig:
    profiles: list[Profile] = field(default_factory=list)
    active: str | None = None

    # ---- 读写 ----

    @classmethod
    def load(cls, path: Path | None = None) -> ModelsConfig:
        """缺文件返回空配置；坏文件（JSON 坏、字段缺、重名）启动即报，不静默吞。"""
        path = path or default_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 {path}：{exc}") from exc
        try:
            config = cls(
                profiles=[Profile(**item) for item in data.get("profiles", [])],
                active=data.get("active"),
            )
        except TypeError as exc:
            raise ValueError(f"{path} 的 profiles 字段不合法：{exc}") from exc
        names: list[str] = []
        for item in config.profiles:
            _validate(item, names)
            names.append(item.name)
        if config.active is not None and config.active not in names:
            raise ValueError(f"active 指向不存在的 profile：{config.active}")
        return config

    def save(self, path: Path | None = None) -> None:
        """原子落盘（临时文件 + replace），密钥文件首建即 0600、已存在也归位。"""
        path = path or default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {"active": self.active, "profiles": [asdict(p) for p in self.profiles]},
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

    # ---- 管理 ----

    def find(self, name: str) -> Profile | None:
        return next((p for p in self.profiles if p.name == name), None)

    def add(self, profile: Profile) -> None:
        _validate(profile, [p.name for p in self.profiles])
        self.profiles.append(profile)
        if self.active is None:
            self.active = profile.name  # 首个 profile 自动成为 active，开箱即用

    def remove(self, name: str) -> Profile:
        profile = self.find(name)
        if profile is None:
            raise ValueError(f"未找到 profile：{name}")
        self.profiles.remove(profile)
        if self.active == name:
            self.active = self.profiles[0].name if self.profiles else None
        return profile

    def use(self, name: str) -> Profile:
        profile = self.find(name)
        if profile is None:
            raise ValueError(f"未找到 profile：{name}")
        self.active = name
        return profile

    def active_profile(self) -> Profile | None:
        return self.find(self.active) if self.active else None


def resolve_connection(
    model_flag: str | None,
    base_url_flag: str | None,
    api_key_flag: str | None,
    config: ModelsConfig | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """启动解析（票 14）：旗标 > active profile > 环境变量（由 LLM 内部兜底）。

    返回 ``(model, base_url, api_key, profile_name)``，未定字段为 None。profile
    是完整三元组：active 存在时未给旗标的字段整体取自它，不与环境变量逐字段
    混搭——防跨端点串 key。旗标只一次性覆盖，不改写 active。
    """
    active = (config if config is not None else ModelsConfig.load()).active_profile()
    if active is None:
        return model_flag, base_url_flag, api_key_flag, None
    return (
        model_flag or active.model,
        base_url_flag or active.base_url,
        api_key_flag or active.api_key,
        active.name,
    )
