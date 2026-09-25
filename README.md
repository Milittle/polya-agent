# mi-z

一个最小可运行的 LLM 工具调用 Agent：模型思考 → 调用工具 → 回填结果 → 直到给出最终答案。
自带一组本地编码工具，可以直接当编码代理用。

## 安装

需要 Python 3.10+，推荐用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
```

或者用 pip：

```bash
pip install -e .
```

## 配置

任何提供 OpenAI 兼容接口（`/v1/chat/completions`）的服务都能接入，示例默认指向 DeepSeek：

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（以及可选的 OPENAI_BASE_URL / OPENAI_MODEL）
```

## 运行示例

```bash
uv run demo.py
```

示例会启动一个本地编码代理，让它列出目录、阅读 `mi_z/agent.py` 并总结。工具调用过程会通过
日志打印出来。有副作用的工具执行前会要求确认；非交互环境（管道、CI）下默认拒绝。

## 用法

定义工具就是一个普通函数加 `@tool` 装饰器，JSON Schema 会从签名和类型注解自动生成：

```python
from mi_z import Agent, LLM, tool


@tool
def get_weather(city: str, unit: str | None = None) -> str:
    """查询某个城市的天气。unit 可选 'celsius' 或 'fahrenheit'。"""
    ...


agent = Agent(llm=LLM(), tools=[get_weather])
print(agent.run("北京今天天气怎么样？"))
```

直接用内置的编码工具：

```python
from mi_z import Agent, LLM, default_tools

agent = Agent(
    llm=LLM(),
    tools=default_tools(root="./my-project"),  # 工具被限制在这个目录内
    approve=lambda tool, args: tool.dangerous is False,  # 拒绝一切副作用工具
)
```

几点说明：

- 工具执行抛出的异常不会中断循环，而是作为错误文本交回模型，让它自行调整——例如参数写错时
  模型有机会重试。
- `approve(tool, arguments) -> bool` 是副作用工具的审批钩子：返回 `False` 时该次调用被跳过，
  模型会收到「用户拒绝」并把结果纳入下一步推理。`Tool.dangerous` 标记了写文件、执行命令这类工具。
- 同一个 `Agent` 实例会保留对话历史，可直接连续调用 `run()` 进行多轮对话；需要重新开始时调用
  `agent.reset()`。
- `max_steps` 限制单次 `run()` 内最多循环多少轮，防止模型陷入反复调用工具的循环。

## 内置工具

`default_tools(root)` 返回以下工具，**所有文件操作都被限制在 `root` 目录内**——解析路径后校验，
`../` 和绝对路径都会被拒绝：

| 工具 | 副作用 | 说明 |
|---|---|---|
| `read_file` | — | 读文件，可用 `start_line` / `end_line` 读片段 |
| `list_dir` | — | 列目录，子目录以 `/` 结尾 |
| `grep` | — | 正则搜索内容，`glob` 限定文件名 |
| `write_file` | ⚠️ | 写/覆盖文件，自动创建父目录 |
| `edit_file` | ⚠️ | 定点替换，默认要求匹配唯一 |
| `run_shell` | ⚠️ | 在 `root` 下执行命令，带超时和输出截断 |

工具结果会进上下文，因此输出统一截断到 8000 字符。

## 项目结构

```
mi_z/
  agent.py    # 核心循环 + 审批钩子
  llm.py      # OpenAI 兼容接口封装
  tools.py    # @tool 装饰器与工具注册表（框架层）
  builtin.py  # 内置编码工具（内容层）
demo.py       # 可运行示例
tests/        # 用假 LLM 验证循环 + 内置工具沙箱测试
```

## 测试

```bash
uv run pytest
```

## Lint

```bash
uv run ruff check .
uv run ruff format --check .
```
