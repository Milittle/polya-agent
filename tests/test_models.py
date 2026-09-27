"""票 03：models.json 新 schema（provider 键位）、迁移、窗口解析、预置表。"""

import json
import stat

import pytest

from polya.models import (
    PROVIDERS,
    ModelEntry,
    ModelsConfig,
    Provider,
    ProviderEntry,
    discover_models,
    format_context_window,
    mask_key,
    resolve_connection,
    resolve_context_window,
)


def _entry(base_url="https://x.example/v1", api_key="sk-x", model="m", models=None):
    return ProviderEntry(
        base_url=base_url,
        api_key=api_key,
        model=model,
        models=models if models is not None else [ModelEntry(model)],
    )


def test_save_load_roundtrip_and_permissions(tmp_path):
    path = tmp_path / "models.json"
    config = ModelsConfig()
    config.add(
        "glm-plan",
        _entry(
            "https://api.z.ai/api/coding/paas/v4",
            "sk-abcd1234efgh",
            "glm-4.7",
            [ModelEntry("glm-4.7", 200_000), ModelEntry("glm-5.3", 1_000_000)],
        ),
    )
    config.add("local", _entry("http://localhost:8000/v1", "sk-local", "qwen3"))
    config.use("local/qwen3")
    config.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))  # 原子写不留临时文件

    loaded = ModelsConfig.load(path)
    assert list(loaded.providers) == ["glm-plan", "local"]
    assert loaded.active == "local/qwen3"
    assert loaded.active_entry()[1].model == "qwen3"
    assert loaded.get("glm-plan").models[1].context_window == 1_000_000


def test_load_missing_file_is_empty_config(tmp_path):
    config = ModelsConfig.load(tmp_path / "none.json")
    assert config.providers == {} and config.active is None
    assert config.active_entry() is None


def test_first_provider_becomes_active_and_remove_reassigns(tmp_path):
    config = ModelsConfig()
    config.add("a", _entry("https://a.example/v1", "sk-a", "m-a"))
    assert config.active == "a/m-a"  # 首个自动 active，开箱即用
    config.add("b", _entry("https://b.example/v1", "sk-b", "m-b"))
    config.remove("a")
    assert config.active == "b/m-b"  # 移除 active 顺延到余下首个
    config.remove("b")
    assert config.active is None and config.providers == {}


@pytest.mark.parametrize(
    "provider_id,entry",
    [
        ("", _entry()),
        ("has space", _entry()),
        (" padded ", _entry()),
        ("x", _entry("ftp://x.example/v1")),
        ("x", _entry("not-a-url")),
        ("x", _entry(api_key="")),
        ("x", _entry(model="")),
        ("x", _entry(models=[ModelEntry("")])),
        ("x", _entry(models=[ModelEntry("m", 0)])),
        ("x", _entry(models=[ModelEntry("m"), ModelEntry("m")])),
    ],
)
def test_validation_rejects_bad_entries(provider_id, entry):
    with pytest.raises(ValueError):
        ModelsConfig().add(provider_id, entry)


def test_duplicate_provider_rejected_but_upsert_updates(tmp_path):
    config = ModelsConfig()
    config.add("dup", _entry(model="m"))
    with pytest.raises(ValueError, match="dup"):
        config.add("dup", _entry(model="m2"))
    config.upsert("dup", _entry(model="m2"))  # 重新登录即覆盖
    assert config.get("dup").model == "m2"


def test_load_rejects_corrupt_or_inconsistent_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="无法读取"):
        ModelsConfig.load(bad)

    ghost = tmp_path / "ghost.json"
    ghost.write_text(
        json.dumps(
            {
                "active": "ghost/m",
                "providers": {"p": {"base_url": "https://x/v1", "api_key": "k", "model": "m"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ghost"):
        ModelsConfig.load(ghost)

    extra = tmp_path / "extra.json"
    extra.write_text('{"providers": {"x": {"base_url": "u"}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="不合法"):
        ModelsConfig.load(extra)

    no_slash = tmp_path / "noslash.json"
    no_slash.write_text(
        json.dumps(
            {
                "active": "p",
                "providers": {"p": {"base_url": "https://x/v1", "api_key": "k", "model": "m"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provider/model"):
        ModelsConfig.load(no_slash)


def test_load_migrates_old_profiles(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "active": "glm-plan",
                "profiles": [
                    {
                        "name": "glm-plan",
                        "base_url": "https://api.z.ai/api/coding/paas/v4",
                        "api_key": "sk-old-key-1234",
                        "model": "glm-4.7",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = ModelsConfig.load(path)
    assert list(config.providers) == ["glm-plan"]
    assert config.active == "glm-plan/glm-4.7"
    assert config.get("glm-plan").base_url == "https://api.z.ai/api/coding/paas/v4"
    # 保存即写回新结构
    config.save(path)
    assert "profiles" not in json.loads(path.read_text(encoding="utf-8"))


def test_resolve_context_window_precedence():
    entry = _entry(models=[ModelEntry("glm-5.3", 1_000_000)])
    assert resolve_context_window("glm-5.3", entry) == 1_000_000  # 条目发现值最优先
    assert resolve_context_window("glm-4.7", entry) == 200_000  # 条目无 → 静态前缀表
    assert resolve_context_window("totally-unknown", None) == 128_000  # → 默认


def test_format_context_window():
    assert format_context_window(128_000) == "128k"
    assert format_context_window(1_000_000) == "1M"
    assert format_context_window(200_000) == "200k"
    assert format_context_window(512) == "512"


def test_mask_key_shows_tail_four():
    assert mask_key("sk-abcd1234efgh") == "sk-…efgh"
    assert mask_key("short") == "…"


def test_presets_cover_first_batch_with_official_endpoints():
    # 首批 8 组（11 条含国内线）：值抄自 pi 的 openai-completions 目录
    assert set(PROVIDERS) == {
        "openrouter",
        "deepseek",
        "zai",
        "zai-coding-cn",
        "moonshotai",
        "moonshotai-cn",
        "groq",
        "together",
        "nvidia",
        "qwen-token-plan",
        "qwen-token-plan-cn",
    }
    assert all(isinstance(p, Provider) for p in PROVIDERS.values())
    assert all(p.base_url.startswith("https://") for p in PROVIDERS.values())
    assert all(p.label for p in PROVIDERS.values())
    assert PROVIDERS["zai"].base_url == "https://api.z.ai/api/coding/paas/v4"
    assert PROVIDERS["zai-coding-cn"].base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert PROVIDERS["deepseek"].model == "deepseek-flash"


def test_resolve_connection_flag_over_active_over_env(tmp_path):
    config = ModelsConfig()
    config.add("p", _entry("https://x.example/v1", "sk-x", "m-x"))

    # active 整体生效：未给旗标的字段取自条目，不与 env 逐字段混搭
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


def test_discover_models_reads_context_fields(monkeypatch):
    class _Model:
        def __init__(self, id, **extra):
            self.id = id
            for key, value in extra.items():
                setattr(self, key, value)

    class _Page:
        data = [
            _Model("a", context_length=200_000),
            _Model("b", max_model_len=1_000_000),
            _Model("c"),
        ]

    class _ModelsResource:
        def list(self):
            return _Page()

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = _ModelsResource()

        def close(self):
            pass

    monkeypatch.setattr("openai.OpenAI", _Client)
    entries = discover_models("https://x.example/v1", "sk-x")
    assert [(e.id, e.context_window) for e in entries] == [
        ("a", 200_000),
        ("b", 1_000_000),
        ("c", None),
    ]
