"""工具定义与注册表。

用 ``@tool`` 装饰一个普通函数即可把它变成可供模型调用的工具，
JSON Schema 会根据函数签名和类型注解自动生成。
"""

from __future__ import annotations

import inspect
import json
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _json_type(annotation: Any) -> str:
    """把 Python 类型注解映射成 JSON Schema 的 type 字段。"""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _json_type(inner[0])
    return _JSON_TYPES.get(annotation, "string")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    dangerous: bool = False

    def run(self, arguments: dict) -> str:
        """执行工具并把返回值统一成字符串。"""
        result = self.fn(**arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """按名字索引一组工具，负责生成 schema 和分发调用。"""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for item in tools or []:
            self.add(item)

    def add(self, item: Tool) -> Tool:
        self._tools[item.name] = item
        return item

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [item.schema() for item in self._tools.values()]

    def call(self, name: str, arguments: dict) -> str:
        """调用工具；出错时把错误信息交回模型，让它自己决定下一步。"""
        item = self._tools.get(name)
        if item is None:
            return f"Error: unknown tool '{name}'"
        try:
            return item.run(arguments)
        except Exception as exc:  # noqa: BLE001 - 错误信息是给模型看的，不是给调用方抛的
            return f"Error: {type(exc).__name__}: {exc}"


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    dangerous: bool = False,
):
    """把函数包装成 :class:`Tool`。

    可以作为 ``@tool`` 直接使用，也可以用 ``@tool(name=..., dangerous=True)`` 覆盖元信息。
    未显式提供 description 时，取函数的 docstring。``dangerous`` 标记有副作用的工具
    （写文件、执行命令等），供 Agent 的审批钩子识别。
    """

    def wrap(func: Callable) -> Tool:
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param_name, param in signature.parameters.items():
            properties[param_name] = {"type": _json_type(hints.get(param_name, str))}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        parameters = {"type": "object", "properties": properties, "required": required}
        resolved_description = description or (inspect.getdoc(func) or "").strip() or func.__name__
        return Tool(
            name=name or func.__name__,
            description=resolved_description,
            parameters=parameters,
            fn=func,
            dangerous=dangerous,
        )

    return wrap(fn) if fn is not None else wrap
