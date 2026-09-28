from types import SimpleNamespace

from polya.llm import LLM
from polya.title import TITLE_MAX_CHARS, parse_title, title_messages


def test_title_prompt_and_parser_keep_chinese_and_bound_output():
    messages = title_messages("设计会话 Banner 标题" * 100)
    assert messages[0]["role"] == "system"
    assert len(messages[1]["content"].encode("utf-8")) <= 960
    assert parse_title('{"title":"  设计会话 Banner 标题。  "}') == "设计会话 Banner 标题"
    assert len(parse_title('{"title":"' + "长" * 100 + '"}')) == TITLE_MAX_CHARS
    assert parse_title("负责处理：设计会话 Banner 标题") is None
    assert parse_title('{"title":""}') is None


def test_llm_generates_title_in_independent_bounded_request():
    llm = LLM(model="test-model", api_key="test-key")
    calls = []

    class Client:
        def with_options(self, **options):
            calls.append(options)
            return self

        @property
        def chat(self):
            return SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"title":"修复登录"}'))]
            )

    llm.client = Client()
    assert llm.generate_title("请帮我修复登录") == "修复登录"
    assert calls[0] == {"timeout": 20.0, "max_retries": 0}
    assert calls[1]["model"] == "test-model"
    assert calls[1]["messages"][1]["content"] == "请帮我修复登录"
