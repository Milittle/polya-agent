"""系统提示词的具名 section 装配。

系统提示词不是一段拼死的字符串，而是一组**有序、具名、可独立替换**的 section。
每个 section 在渲染时用 ``<name>`` 标签包裹（``preamble`` 除外），这样：

- 工具可以贡献 ``snippet`` / ``guidelines``，分别落到 ``tools`` / ``rules`` 段，
  与工具的 API ``description`` 三面分离，不再一处真相变两处；
- 项目记忆、技能目录、工作目录各占一段，可单独增删；
- 需要增量更新时（例如会话中途技能目录变化），只比较变化的 section
  （:func:`diff_sections`），而不是重发整段提示词。

渲染顺序即插入顺序，稳定可预测；同样的输入永远得到同样的字节串——
这是 KV Cache 前缀不变量在提示词侧的前提。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from .tools import ToolRegistry

PREAMBLE = "preamble"  # 唯一不包标签的 section


class SystemPrompt:
    """具名 section 的容器，渲染为带标签的提示词文本。"""

    def __init__(self, preamble: str):
        self._sections: OrderedDict[str, str] = OrderedDict()
        self._preamble = preamble

    def set(self, name: str, content: str) -> SystemPrompt:
        """设置/覆盖一个 section；空内容不产生 section。链式返回自身。"""
        if name == PREAMBLE:
            raise ValueError("preamble 是保留名，请通过构造参数传入")
        if content:
            self._sections[name] = content
        else:
            self._sections.pop(name, None)
        return self

    def sections(self) -> dict[str, str]:
        """渲染后的 section 视图（值已含 ``<name>`` 标签），供 diff 使用。"""
        rendered = {PREAMBLE: self._preamble}
        rendered.update(
            {name: f"<{name}>\n{content}\n</{name}>" for name, content in self._sections.items()}
        )
        return rendered

    def render(self) -> str:
        """完整提示词：preamble + 各 section，用空行连接。"""
        rendered = [f"<{name}>\n{c}\n</{name}>" for name, c in self._sections.items()]
        parts = [self._preamble, *rendered]
        return "\n\n".join(part for part in parts if part)


def tool_snippets(tools: ToolRegistry) -> str:
    """``<tools>`` 段：每个有 snippet 的工具一行 ``- name: snippet``。"""
    lines = [f"- {item.name}: {item.snippet}" for item in tools if item.snippet]
    return "\n".join(lines)


def tool_guidelines(tools: ToolRegistry, extra: Iterable[str] = ()) -> str:
    """``<rules>`` 段：工具指引 + 额外规则，去重、bullet 化、保持顺序。"""
    rules: list[str] = []
    seen: set[str] = set()
    for rule in [*(g for item in tools for g in item.guidelines), *extra]:
        normalized = rule.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            rules.append(normalized)
    return "\n".join(f"- {rule}" for rule in rules)


def diff_sections(previous: dict[str, str], current: dict[str, str]) -> dict[str, str | None]:
    """比较两次渲染的 section，返回变化（值 ``None`` 表示该段被移除）。

    供会话中途增量更新提示词用：只发送变化的 section，保持前缀稳定。
    """
    patch: dict[str, str | None] = {}
    for name, text in current.items():
        if previous.get(name) != text:
            patch[name] = text
    for name in previous:
        if name not in current:
            patch[name] = None
    return patch
