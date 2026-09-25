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
from mi_z.builtin import CODING_SYSTEM_PROMPT

agent = Agent(
    llm=LLM(),
    tools=default_tools(root="./my-project"),  # 工具被限制在这个目录内
    system_prompt=CODING_SYSTEM_PROMPT,  # 编码代理专用提示词（推荐搭配 default_tools）
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
- `status_bar=True` 开启 Agent 状态栏：每轮迭代以 user 消息在上下文**末尾**追加
  `<agent_status>` 元信息（迭代号、各工具累计调用次数、token 用量、时间），工具结果
  也会标注「第 N 次调用」。模型检索强但归纳弱，让它自己从轨迹里数调用次数既慢又容易
  数错——状态栏用代码提前算好。更新采用持久追加（旧状态留在轨迹里，不删改），KV Cache
  前缀始终稳定。也可传入自定义渲染函数 `status_bar=lambda snapshot: ...`。
- `agent.total_usage` / `agent.last_usage` 累计/记录每次请求的 token 用量（响应里没有
  usage 字段时保持为 0 / `None`，不会报错）；只做统计，不进消息历史，不影响缓存前缀。
- 不传 `system_prompt` 时使用内置的通用提示词；`default_tools` 建议搭配
  `mi_z.builtin.CODING_SYSTEM_PROMPT`（围绕六个内置工具的工作流：探查 → 小步修改 →
  验证 → 汇报）。系统提示词应当 100% 静态——动态信息请追加到对话末尾，而不是改写提示词。

## 内置工具

`default_tools(root)` 返回以下工具，**所有文件操作都被限制在 `root` 目录内**——解析路径后校验，
`../` 和绝对路径都会被拒绝：

| 工具 | 副作用 | 说明 |
|---|---|---|
| `read_file` | — | 读文件，**输出带行号**；支持行号片段；拒绝二进制和超大文件 |
| `list_dir` | — | 列目录（单层），子目录以 `/` 结尾 |
| `glob` | — | 按文件名模式递归找文件（自动跳过 `.venv`/`.git` 等） |
| `grep` | — | 正则搜索内容，支持**上下文行**、忽略大小写、`glob` 限定文件名 |
| `write_file` | ⚠️ | 写/覆盖文件，自动创建父目录 |
| `edit_file` | ⚠️ | 定点替换，默认要求匹配唯一 |
| `multi_edit` | ⚠️ | 一次多处替换，**原子生效**（任一处失败全不落盘） |
| `bash` | ⚠️ | **持久会话**执行命令：cwd/环境变量跨调用保持 |
| `bash_output` | — | 非阻塞读取会话新输出（后台/慢速命令） |
| `kill_bash` | ⚠️ | 终止持久会话 |
| `web_fetch` | — | 抓取 URL，HTML 转文本，`<external_content>` 包裹防注入；**拒绝内网/localhost（SSRF 防护）** |

工具结果会进上下文，因此输出统一截断到 8000 字符。`bash` 会话超时会终止并重启
（环境状态丢失）；命令必须非交互。`web_fetch` 用标准库实现，零第三方依赖。

**跨平台**：除 `bash` 会话（需要 bash，Windows 需 WSL/Git Bash）外，全部工具为
纯 Python 实现，Windows 原生可跑——不依赖 grep/rg/find 等系统命令。

## 项目结构

```
mi_z/
  agent.py    # 核心循环 + 审批钩子 + 状态栏注入
  llm.py      # OpenAI 兼容接口封装
  tools.py    # @tool 装饰器与工具注册表（框架层）
  builtin.py  # 内置编码工具 + 编码代理提示词（内容层）
  shell.py    # 持久 bash 会话（读线程 + 哨兵标记协议）
  web.py      # web_fetch：抓取 + HTML 转文本 + 来源标记
  status.py   # 状态栏快照与默认渲染器
demo.py       # 可运行示例
tests/        # 用假 LLM 验证循环 + 内置工具/会话/抓取的沙箱测试
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
