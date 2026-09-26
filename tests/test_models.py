"""票 14：models.json 的读写、校验、预设与启动解析。"""

import stat

import pytest

from polya.models import (
    PRESETS,
    ModelsConfig,
    Preset,
    Profile,
    mask_key,
    resolve_connection,
)


def test_save_load_roundtrip_and_permissions(tmp_path):
    path = tmp_path / "models.json"
    config = ModelsConfig()
    config.add(
        Profile("glm-plan", "https://api.z.ai/api/coding/paas/v4", "sk-abcd1234efgh", "glm-4.7")
    )
    config.add(Profile("local", "http://localhost:8000/v1", "sk-local", "qwen3"))
    config.use("local")
    config.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))  # 原子写不留临时文件

    loaded = ModelsConfig.load(path)
    assert [p.name for p in loaded.profiles] == ["glm-plan", "local"]
    assert loaded.active == "local"
    assert loaded.active_profile().model == "qwen3"


def test_load_missing_file_is_empty_config(tmp_path):
    config = ModelsConfig.load(tmp_path / "none.json")
    assert config.profiles == [] and config.active is None
    assert config.active_profile() is None


def test_first_profile_becomes_active_and_remove_reassigns(tmp_path):
    config = ModelsConfig()
    config.add(Profile("a", "https://a.example/v1", "sk-a", "m-a"))
    assert config.active == "a"  # 首个自动 active，开箱即用
    config.add(Profile("b", "https://b.example/v1", "sk-b", "m-b"))
    config.remove("a")
    assert config.active == "b"  # 移除 active 顺延到余下首个
    config.remove("b")
    assert config.active is None and config.profiles == []


@pytest.mark.parametrize(
    "profile",
    [
        Profile("", "https://x.example/v1", "sk-x", "m"),
        Profile("has space", "https://x.example/v1", "sk-x", "m"),
        Profile(" padded ", "https://x.example/v1", "sk-x", "m"),
        Profile("x", "ftp://x.example/v1", "sk-x", "m"),
        Profile("x", "not-a-url", "sk-x", "m"),
        Profile("x", "https://x.example/v1", "", "m"),
        Profile("x", "https://x.example/v1", "sk-x", ""),
    ],
)
def test_validation_rejects_bad_profiles(profile):
    with pytest.raises(ValueError):
        ModelsConfig().add(profile)


def test_duplicate_name_rejected(tmp_path):
    config = ModelsConfig()
    config.add(Profile("dup", "https://x.example/v1", "sk-x", "m"))
    with pytest.raises(ValueError, match="dup"):
        config.add(Profile("dup", "https://y.example/v1", "sk-y", "m"))


def test_load_rejects_corrupt_or_inconsistent_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="无法读取"):
        ModelsConfig.load(bad)

    ghost = tmp_path / "ghost.json"
    ghost.write_text('{"active": "ghost", "profiles": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="ghost"):
        ModelsConfig.load(ghost)

    extra = tmp_path / "extra.json"
    extra.write_text('{"profiles": [{"name": "x", "base_url": "u"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="不合法"):
        ModelsConfig.load(extra)


def test_mask_key_shows_tail_four():
    assert mask_key("sk-abcd1234efgh") == "sk-…efgh"
    assert mask_key("short") == "…"


def test_presets_cover_known_vendors_with_official_endpoints():
    assert set(PRESETS) == {"zai", "zai-cn", "deepseek", "openrouter", "moonshot"}
    assert all(isinstance(p, Preset) for p in PRESETS.values())
    assert all(p.base_url.startswith("https://") for p in PRESETS.values())
    assert all(p.label for p in PRESETS.values())
    # 两条 coding plan 的 OpenAI 兼容端点（国际 docs.z.ai、国内 open.bigmodel.cn，
    # 均官方核实，票 14 Comments）
    assert PRESETS["zai"].base_url == "https://api.z.ai/api/coding/paas/v4"
    assert PRESETS["zai"].model == "glm-5.3"
    assert PRESETS["zai-cn"].base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert PRESETS["deepseek"].model == "deepseek-flash"


def test_resolve_connection_flag_over_active_over_env(tmp_path):
    config = ModelsConfig()
    config.add(Profile("p", "https://x.example/v1", "sk-x", "m-x"))

    # active 整体生效：未给旗标的字段取自 profile，不与 env 逐字段混搭
    assert resolve_connection(None, None, None, config) == (
        "m-x",
        "https://x.example/v1",
        "sk-x",
        "p",
    )
    # 旗标逐字段一次性覆盖（不改写 active）
    assert resolve_connection("m1", None, None, config) == (
        "m1",
        "https://x.example/v1",
        "sk-x",
        "p",
    )
    # 无 active → 原样透传，env 兜底在 LLM 内部
    assert resolve_connection("m", "https://u/v1", "k", ModelsConfig()) == (
        "m",
        "https://u/v1",
        "k",
        None,
    )
