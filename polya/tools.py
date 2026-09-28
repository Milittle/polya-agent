"""工具定义与注册表。

用 ``@tool`` 装饰一个普通函数即可把它变成可供模型调用的工具，
JSON Schema 会根据函数签名和类型注解自动生成。

工具是「自描述单元」，三面各自独立、互不混淆：

- ``description``：完整用法说明，进请求的 tools 数组（模型逐字看到）。
- ``snippet``：一行摘要，进系统提示词的 ``<tools>`` 段。
- ``guidelines``：零到多条纪律句，进系统提示词的 ``<rules>`` 段。

参数说明写在类型注解里的 ``Annotated[T, "描述"]`` 中，避免与 docstring
挤在一起；``Literal`` / ``Enum`` 会生成 ``enum`` 约束，``dataclass`` 会展开
为嵌套 object，不再把复杂类型静默降级成 ``string``。
"""

from __future__ import annotations

import enum
import inspect
import json
import types
from collections.abc import Callable
from dataclasses import MISSING, dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from .prompts import msg, tool_description

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _unwrap_annotated(annotation: Any) -> tuple[Any, str | None]:
    """剥掉 ``Annotated[T, ...]``，把其中的字符串元数据当作参数描述返回。"""
    if get_origin(annotation) is Annotated:
        base, *meta = get_args(annotation)
        description = next((item for item in meta if isinstance(item, str)), None)
        return base, description
    return annotation, None


def _literal_schema(values: tuple) -> dict:
    """``Literal[...]`` / ``Enum`` 的枚举项 → JSON Schema（带类型与 enum）。"""
    if not values:
        return {}
    python_types = {type(item) for item in values}
    if len(python_types) == 1:
        json_type = _JSON_TYPES.get(next(iter(python_types)))
        schema: dict = {}
        if json_type:
            schema["type"] = json_type
        schema["enum"] = list(values)
        return schema
    return {"enum": list(values)}


def json_schema(annotation: Any) -> dict:
    """把 Python 类型注解映射成 JSON Schema 对象。

    支持：基元、``Optional`` / ``Union`` 单值、``Literal`` / ``Enum``、
    ``list[T]``、``dict[str, T]``、嵌套 ``dataclass``。无法理解的类型返回
    空 schema（不加约束），而不是伪装成 ``string``——错误约束比无约束更糟。
    """
    annotation, _ = _unwrap_annotated(annotation)

    if is_dataclass(annotation) and isinstance(annotation, type):
        try:
            hints = get_type_hints(annotation, include_extras=True)
        except Exception:  # noqa: BLE001 - 无法解析的注解退化为无约束
            hints = {}
        properties: dict[str, dict] = {}
        required: list[str] = []
        for field in dataclass_fields(annotation):
            field_annotation = hints.get(field.name, field.type)
            sub, _ = _unwrap_annotated(field_annotation)
            schema = json_schema(sub)
            _annotate(schema, _field_description(field))
            properties[field.name] = schema
            if field.default is MISSING and field.default_factory is MISSING:
                required.append(field.name)
        result: dict = {"type": "object", "properties": properties}
        if required:
            result["required"] = required
        return result

    origin = get_origin(annotation)
    if origin is Literal:
        return _literal_schema(get_args(annotation))
    if origin in (Union, types.UnionType):
        inner = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(inner) == 1:
            return json_schema(inner[0])
        return {"anyOf": [json_schema(arg) for arg in inner]}
    if origin is list:
        args = get_args(annotation)
        item = args[0] if args else str
        return {"type": "array", "items": json_schema(item)}
    if origin is dict or annotation is dict:
        args = get_args(annotation)
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = json_schema(args[1])
        return schema

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _literal_schema(tuple(member.value for member in annotation))

    json_type = _JSON_TYPES.get(annotation)
    return {"type": json_type} if json_type else {}


def _field_description(field: Any) -> str | None:
    """``dataclasses.field(metadata={"description": ...})`` 的说明。"""
    metadata = getattr(field, "metadata", None) or {}
    value = metadata.get("description")
    return value if isinstance(value, str) else None


def _annotate(schema: dict, description: str | None) -> None:
    if description:
        schema.setdefault("description", description)


KINDS = ("read", "write", "exec", "delegate")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    # read=无副作用 / write=写文件 / exec=执行命令 / delegate=委派，权限判定按此分类
    kind: str = "read"
    snippet: str = ""  # 一行摘要，进系统提示词的 <tools> 段；空则不进
    guidelines: tuple[str, ...] = ()  # 纪律句，进 <rules> 段
    # 等待类工具（bash_output 轮询）：相同参数重复调用是合法等待，无进展熔断跳过它
    poll: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"未知工具类别 kind={self.kind!r}，可选：{KINDS}")

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


def run_tool(tool: Tool, arguments: dict) -> str:
    """执行工具并把异常转成 ``Error:`` 文本——驱动层与注册表共用的单一执行点。

    错误信息是给模型看的（让它自行调整重试），不是给调用方抛的。
    """
    try:
        return tool.run(arguments)
    except Exception as exc:  # noqa: BLE001 - 错误信息是给模型看的，不是给调用方抛的
        return f"Error: {type(exc).__name__}: {exc}"


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
            raise RuntimeError(msg("agent.tools_frozen"))
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
        return run_tool(item, arguments)


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    kind: str = "read",
    snippet: str | None = None,
    guidelines: tuple[str, ...] | list[str] = (),
    poll: bool = False,
):
    """把函数包装成 :class:`Tool`。

    可以作为 ``@tool`` 直接使用，也可以用 ``@tool(name=..., kind=..., snippet=...)``
    覆盖元信息。未显式提供 description 时取函数 docstring。``kind`` 按副作用分类：
    ``read``（无副作用，直接放行）/ ``write``（写文件）/ ``exec``（执行命令）/
    ``delegate``（委派子任务，副作用在子层）。
    参数说明写在 ``Annotated[T, "描述"]`` 里，会进入 schema 的字段 description。
    ``poll=True`` 标记等待类工具（如 ``bash_output``）：同参数重复是合法轮询，
    无进展熔断会跳过它。
    """

    def wrap(func: Callable) -> Tool:
        signature = inspect.signature(func)
        hints = get_type_hints(func, include_extras=True)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param_name, param in signature.parameters.items():
            annotation, annotated_description = _unwrap_annotated(hints.get(param_name, str))
            schema = json_schema(annotation)
            _annotate(schema, annotated_description)
            properties[param_name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        parameters: dict = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        resolved_name = name or func.__name__
        doc = (inspect.getdoc(func) or "").strip() or resolved_name
        resolved_description = description or tool_description(resolved_name, doc)
        return Tool(
            name=resolved_name,
            description=resolved_description,
            parameters=parameters,
            fn=func,
            kind=kind,
            snippet=snippet or "",
            guidelines=tuple(guidelines),
            poll=poll,
        )

    return wrap(fn) if fn is not None else wrap
