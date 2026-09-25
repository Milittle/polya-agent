"""LLM 封装层的测试：不发真实请求，用假 client 验证参数透传。"""

from __future__ import annotations

import pytest

from mi_z.llm import LLM


@pytest.fixture
def captured(monkeypatch):
    """把 OpenAI 构造函数换成假的，捕获收到的参数。"""
    box: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            box.update(kwargs)

    monkeypatch.setattr("mi_z.llm.OpenAI", FakeClient)
    return box


def test_timeout_and_retries_are_forwarded(captured):
    LLM(api_key="k", timeout=30, max_retries=5)
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 5


def test_defaults_are_sane(captured):
    LLM(api_key="k")
    assert captured["timeout"] == 120.0
    assert captured["max_retries"] == 2


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        LLM()
