# polya-agent

一个最小可运行的 LLM 工具调用 Agent：模型思考 → 调用工具 → 回填结果 → 直到给出最终答案。
自带一组本地编码工具，可以直接当编码代理用。内部（包名 / CLI 命令 / 品牌）一律叫 **polya**，
取自波利亚（G. Pólya，《怎样解题》）——问题求解代理的祖师爷。

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

示例会启动一个本地编码代理，让它列出目录、阅读 `polya/agent.py` 并总结。工具调用过程会通过
日志打印出来。有副作用的工具执行前会要求确认；非交互环境（管道、CI）下默认拒绝。

## CLI

安装后直接用 `polya`（开发时 `uv run polya`），两种模式：

```bash
uv run polya                    # 交互式 REPL：多轮对话，斜杠命令控制会话
uv run polya -p "修复 pytest 失败的测试" --plan -y   # 单任务模式：执行一次即退出
```

常用参数：`--root DIR` 工作目录（默认 `.`，文件操作被限制在内）、`--plan` 启动进入
规划模式、`--yes` 自动批准一切审批、`--max-steps N`（默认 25）、`--model/--base-url/
--api-key` 覆盖环境变量、`--no-compress` 关闭上下文压缩（默认开启，用量超窗口 80%
时批量压缩旧工具结果）、`--context-window N`（默认 128000）、`--keep-recent N`
（压缩保留区，默认 30）。

REPL 斜杠命令：

| 命令 | 作用 |
|---|---|
| `/help` | 命令列表 |
| `/todos` `/status` | 查看 TODO 清单 / 会话状态（模式、用量、工具计数） |
| `/plan on\|off` | 随时切换规划模式（`exit_plan_mode` 构造时已注册，切换不动工具数组，缓存安全） |
| `/expand [N]` | 展开最近 N 块（默认 5）的工具结果 / 思考全文——滚动区的折叠块在这里看全量 |
| `/reset` | 清空对话历史、TODO 与统计 |
| `/exit` `/quit` | 退出（输入处 Ctrl+D / Ctrl+C 同效） |

**审批交互**：危险工具执行前先展示**变更预览**，再弹出选择列表——写类工具
（`write_file`/`edit_file`/`multi_edit`）给出与磁盘现状比对的行级 diff（红删绿增，
新文件整块标绿，超过 40 行折叠并标注），`bash` 展示完整命令。选项带说明：
`允许`（执行本次调用）、`总是允许`（本会话内该工具不再询问）、`拒绝`（让模型
调整方案）；↑/↓ + Enter 或数字键选择，Esc 拒绝，光标默认停在「拒绝」——Enter
单按绝不放行。`--plan` 模式下计划全文展示后同样以选择列表批准。非交互环境
（管道/CI）默认拒绝一切危险操作，`--yes` 才放行（仅在信任任务时使用）。
**中断**：`run()` 执行中按 Ctrl+C 只终止本次任务（历史保留，未回填的工具结果自动补齐，
对话可继续）；输入提示处按 Ctrl+C/Ctrl+D 直接退出。

### 交互与显示

显示层用 [rich](https://github.com/Textualize/rich)、输入层用
[prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)、`--help`
用 [rich-argparse](https://github.com/Hamatti/rich-argparse) 排版。这些只在
stdin/stdout 是终端时启用；管道/CI 下自动降级为纯文本（补全、流式渲染、颜色都不出现）。

终端交互对标 pi（badlogic/pi-mono）的极简风格：**滚动区永久追加 + 底部小型 live 区**。
完成的内容（思考折叠行、工具块、完整 Markdown 段落）打印进终端原生 scrollback、
永不重绘；只有正在流式输出的尾窗和 spinner 状态行占据底部 live 区，段落完成即提交
进滚动区。`-p` 单任务模式不接渲染器，stdout 只承载最终答案，可安全 `> answer.md`
或接管道。

- **流式输出**：回答逐段流入 live 区（节流重渲染的 Markdown），完成即整体提交滚动区；
  中间轮的 assistant 文字与 thinking 同样可见——思考折叠为一行 dim italic 摘要
  （`✻ 思考 47 字：…`），全文用 `/expand` 查看。
- **工具块**（Claude Code 树形）：头行按工具特化——`⏺ bash  $ pytest -q`、
  `⏺ read_file  polya/ui.py:10-50`、`⏺ write_file  app.py`（陌生工具退回紧凑 JSON）；
  结果首行用 `  ⎿ ` 连接符、续行 4 空格对齐，默认折叠前 8 行 / 600 字符（`… 还有
  N 行未显示（/expand 查看全文）`），耗时以 dim 附在尾行；错误结果整块标红；块间空行分组。
- **实时状态**：底部 spinner 标注阶段（`思考中` / `回复中` / `运行 bash`）、轮次、
  阶段耗时与**上下文占用**（`12.8k/128k（10%）`，随 usage 事件更新，压缩后回落可见）；
  状态由 `Agent.on_event` 事件流驱动（见下）。
- **bash 实时输出**：命令运行期间输出逐行流入 live 区尾窗（默认尾 8 行），不再是
  spinner 干转；全量输出仍由随后的 `⎿` 块承载。
- **输入**（Claude Code 风格）：上下两条横线围出输入区——顶线嵌会话主题
  （`── ✳ count-readme-words ────`），底线是按键提示（`── Enter 发送 · Alt+Enter
  换行 ────`）；`❯` 提示符 + 空输入 dim 占位提示；**Enter 提交**，Alt+Enter（或
  行尾反斜杠 + Enter）换行写多行任务；命令历史持久化到**状态目录** `~/.polya/` 下的
  `history`（上下键翻阅；状态目录亦将承载后续配置），输入 `/` 自动补全斜杠命令，
  并按历史给出灰色建议（`→` 接受）。
- **会话主题**（Claude Code 同款）：首个任务完成后从任务内容**本地**推断一个
  kebab-case slug（如 `count-readme-words`），嵌入输入框顶线并写入终端标签页标题
  （`✳ topic`）。纯本地推断（slug 化 → 输入截断逐级降级）——不为装饰发起任何
  额外 LLM 请求。
- **日志**：走 **stderr**，与渲染共用同一 rich Console，自动排在 live 区上方；SDK 的
  HTTP 明细日志被压到 WARNING，不刷屏。`--no-stream` 可为不支持流式的端点关闭流式
  （分块进度仍在）。

## 用法

定义工具就是一个普通函数加 `@tool` 装饰器，JSON Schema 会从签名和类型注解自动生成：

```python
from polya import Agent, LLM, tool


@tool
def get_weather(city: str, unit: str | None = None) -> str:
    """查询某个城市的天气。unit 可选 'celsius' 或 'fahrenheit'。"""
    ...


agent = Agent(llm=LLM(), tools=[get_weather])
print(agent.run("北京今天天气怎么样？"))
```

直接用内置的编码工具：

```python
from polya import Agent, LLM, default_tools
from polya.builtin import CODING_SYSTEM_PROMPT

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
- `on_event(event, payload)` 是可选观测钩子，CLI 的终端渲染（`polya/ui.py` 的
  `TerminalRenderer`）就建立在它之上；不设钩子时零开销、核心逻辑不受影响，任何前端
  （REPL、全屏 TUI、Web）都能接这条事件流。事件词表（时序：`iteration → [usage] →
  *_delta* → assistant_message → (tool_call → tool_result)*`）：

  | 事件 | 载荷 | 时机 |
  |---|---|---|
  | `iteration` | `{step, max_steps}` | 每轮迭代开头 |
  | `reasoning_delta` / `text_delta` | `{delta}` | 流式片段（设置了钩子且未 `stream=False` 时，LLM 走流式） |
  | `assistant_message` | `{content, reasoning, tool_calls}` | 一轮完整消息落历史后 |
  | `tool_call` | `{name, call_id, arguments}` | 工具分发前（参数已解析） |
  | `tool_result` | `{name, call_id, result, duration_s, error}` | 结果回填历史后（`result` 为原始结果） |
  | `usage` | `{last, total}` | 仅当本次响应带 usage |

  注意它与 `status_bar` 不同：后者是给**模型**看的上下文内容，`on_event` 是给**人**看的进度信号。
- `agent.total_usage` / `agent.last_usage` 累计/记录每次请求的 token 用量（响应里没有
  usage 字段时保持为 0 / `None`，不会报错）；只做统计，不进消息历史，不影响缓存前缀。
- **两阶段模式**：`Agent(plan_mode=True, approve_plan=回调)` 启动时进入规划模式——
  危险工具在分发层被拒（带指引），模型探查后调用 `exit_plan_mode(plan=...)` 提交计划，
  `approve_plan(plan) -> bool` 决定放行（缺省自动批准）；状态栏会显示当前模式。
  工具数组全程不变（中途增删 tools 会破坏 KV Cache 前缀），模式切换只是运行时状态。
- **上下文压缩**：`Agent(compress=True)`（CLI 默认开启，`--no-compress` 关闭）。最近一次
  请求的 prompt tokens 超过 `context_window × compress_threshold`（默认 128K × 80%）时，
  在两次 API 调用之间**批量压缩**保留区（最近 `keep_recent=30` 条）之外的旧 tool 结果：
  一次 LLM 调用（合并式，注入当前任务做任务感知压缩）把它们原地替换为带 `[COMPRESSED]`
  标记的摘要（防重复处理），消息条数与 tool_call_id 配对不变，对话脉络完整；同区的旧
  状态栏消息直接删除（噪声不做摘要）。替换点之后的 KV Cache 会失效——这是有意识的
  权衡，所以阈值高、批量压、低频次。触发判据必须是**最近一次**调用的 prompt_tokens，
  不能用累计用量（每轮重复计入共享前缀，二次增长会过早触发）。连续 3 次压缩失败自动
  熔断（压缩失败不影响主任务，保留完整信息继续跑）。
- **模型能力声明**（`polya.providers.ModelProfile`）：Agent 只问能力、不特判模型名。
  压缩策略按 `supports_inplace_tool_edit` 自动切换——OpenAI 式模型（不回传 reasoning）
  原地替换 tool content；thinking 绑定前缀的模型（`deepseek-reasoner`/`claude`/`gemini`
  档案）改用**摘要重启**：整段旧历史压成一条 `<session_summary>` 消息，模型从摘要冷
  启动（原地替换会使保留的 thinking 全部失效）。`reasoning_passthrough` 控制
  `reasoning_content` 原样保存回传（DeepSeek interleaved thinking）；档案还提供
  `context_window` 与 `temperature` 默认值（o 系列为 `None` = 不传）。`profile_for(model)`
  按前缀匹配，未知模型回落安全默认。CLI 的 `--model` 会自动查档案。
- **前缀不变量断言**（`Agent(prefix_check=True)` / CLI `--prefix-check`）：运行时校验
  每次请求是上一次的严格扩展（深拷贝基线 + 内容比较，有开销故默认关）。破坏前缀对
  纯 KV Cache 只是缓存变贵，对回传 thinking 的模型是**推理连续性断裂**——这条断言把
  「两个压缩点之间严格 append-only」从纪律变成代码。压缩/摘要重启/reset 后基线自动
  重置（压缩点是合法的推理重启点）。
- 不传 `system_prompt` 时使用内置的通用提示词；`default_tools` 建议搭配
  `polya.builtin.CODING_SYSTEM_PROMPT`（围绕内置工具的工作流：任务拆解 → 探查 →
  小步修改 → 验证 → 汇报）。系统提示词应当 100% 静态——动态信息请追加到对话末尾，而不是改写提示词。

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
| `todo_write` | — | 全量重写 TODO 清单（可选；需 `default_tools(todos=store)` + `Agent(todos=store)` 共享同一实例） |

工具结果会进上下文，因此输出统一截断到 8000 字符。`bash` 会话超时会终止并重启
（环境状态丢失）；命令必须非交互。`web_fetch` 用标准库实现，零第三方依赖。

TODO 清单是状态栏的「任务规划」组件：`todo_write` 写入共享的 `TodoStore`，
状态栏每轮把它渲染到上下文末尾——模型不用从历史里回忆还剩什么（外部记忆）：
`agent.reset()` 会连同清单一起清空。

**跨平台**：除 `bash` 会话（需要 bash，Windows 需 WSL/Git Bash）外，全部工具为
纯 Python 实现，Windows 原生可跑——不依赖 grep/rg/find 等系统命令。

## 项目结构

```
polya/
  agent.py    # 核心循环 + 审批钩子 + 状态栏注入 + on_event 进度钩子
  cli.py      # 命令行入口：REPL + 单任务模式 + 终端审批 + rich/prompt_toolkit 显示层
  llm.py      # OpenAI 兼容接口封装
  tools.py    # @tool 装饰器与工具注册表（框架层）
  builtin.py  # 内置编码工具 + 编码代理提示词（内容层）
  shell.py    # 持久 bash 会话（读线程 + 哨兵标记协议）
  web.py      # web_fetch：抓取 + HTML 转文本 + 来源标记
  status.py   # 状态栏快照与默认渲染器
  todos.py    # TODO 清单存储（外部记忆）
  compact.py  # 上下文压缩：原地替换 + 摘要重启（按模型能力选择）
  providers.py # 模型能力声明：只问能力不特判型号
demo.py       # 可运行示例（库用法）
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
