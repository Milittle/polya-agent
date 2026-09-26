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


def _json_schema(annotation: Any) -> str | dict:
    """把 Python 类型注解映射成 JSON Schema（type 字符串或完整 schema 对象）。"""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _json_schema(inner[0])
    if origin is list:
        (item,) = get_args(annotation) or (str,)
        item_schema = _json_schema(item)
        if isinstance(item_schema, str):
            item_schema = {"type": item_schema}
        return {"type": "array", "items": item_schema}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    return _JSON_TYPES.get(annotation, "string")


KINDS = ("read", "write", "exec")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    kind: str = "read"  # read=无外部副作用 / write=写文件 / exec=执行命令，权限判定按此分类

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"未知工具类别 kind={self.kind!r}，可选：{KINDS}")

    @property
    def dangerous(self) -> bool:
        """兼容视图：有副作用（write/exec）即危险。审批全面迁到 kind 判定后移除。"""
        return self.kind != "read"

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
    """按名字索引一组工具，负责生成 schema 和分发调用。

    工具定义位于上下文最前部（紧跟系统提示词），中途增删会让 KV Cache
    从首个变动处全部失效（参见《深入理解 AI Agent》2.3 节）。因此
    Agent 首次运行后会调用 :meth:`freeze`，之后修改注册表直接报错。
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        self._frozen = False
        for item in tools or []:
            self.add(item)

    def freeze(self) -> None:
        """锁定注册表：此后任何修改都会抛错。"""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def add(self, item: Tool) -> Tool:
        if self._frozen:
            raise RuntimeError(
                "工具注册表已冻结：对话开始后增删工具会使 KV Cache 失效，"
                "请在构建 Agent 前配置好全部工具。"
            )
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
    kind: str = "read",
):
    """把函数包装成 :class:`Tool`。

    可以作为 ``@tool`` 直接使用，也可以用 ``@tool(name=..., kind=...)`` 覆盖元信息。
    未显式提供 description 时，取函数的 docstring。``kind`` 按副作用分类：
    ``read``（无外部副作用，直接放行）/ ``write``（写文件）/ ``exec``（执行命令），
    供权限判定与审批钩子识别。
    """

    def wrap(func: Callable) -> Tool:
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param_name, param in signature.parameters.items():
            schema = _json_schema(hints.get(param_name, str))
            properties[param_name] = {"type": schema} if isinstance(schema, str) else schema
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        parameters = {"type": "object", "properties": properties, "required": required}
        resolved_description = description or (inspect.getdoc(func) or "").strip() or func.__name__
        return Tool(
            name=name or func.__name__,
            description=resolved_description,
            parameters=parameters,
            fn=func,
            kind=kind,
        )

    return wrap(fn) if fn is not None else wrap
