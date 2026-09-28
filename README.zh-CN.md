# polya-agent

一个最小可运行的 LLM 工具调用 Agent：模型思考 → 调用工具 → 回填结果 → 直到给出最终答案。
自带一组本地编码工具，可以直接当编码代理用。内部（包名 / CLI 命令 / 品牌）一律叫 **polya**，
取自波利亚（G. Pólya，《怎样解题》）——问题求解代理的祖师爷。

[English README](./README.md)

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

多家厂商或 coding plan 全程在会话内登录——`/login` 打开 provider 列表（首批：
OpenRouter、DeepSeek、z.ai 国际/国内、Moonshot 国际/国内、Groq、Together、
NVIDIA、Qwen Token Plan 国际/国内，外加「自定义端点」）。base_url 由预置表预填、
可改；录入 key 时输入框让位终端、走 getpass 不回显，不进屏幕与输入历史。
凭据存于 `~/.polya/models.json`（权限 0600）：

```
/login zai
  base_url（回车用 https://api.z.ai/api/coding/paas/v4）:
  api_key（输入不回显）:
已登录 zai：glm-5.3 @ api.z.ai · 窗口 1M，已设为默认启动模型；模型目录后台刷新中。/model 切换，Ctrl+S 设默认。
```

登录后 polya 立即落盘凭据，模型列表在后台线程问 provider 的 `/models` 端点
（15s 超时），向导不等待；端点报了窗口就读（`context_length` / `max_model_len`），
否则回落内置能力表。解析优先级：发现值 > 内置表 > 128k。`/model` 随时切换（无参数跨已登录 provider 展开选项器，`Ctrl+S`
把高亮项存为默认启动模型）；`/logout <provider>` 移除凭据。启动解析优先级：
CLI 旗标 > 默认模型（active）> `OPENAI_*` 环境变量。

## 运行示例

```bash
uv run examples/demo.py
```

示例会启动一个本地编码代理，让它列出目录、阅读 `polya/agent.py` 并总结。工具调用过程会通过
日志打印出来。有副作用的工具执行前会要求确认；非交互环境（管道、CI）下默认拒绝。

## CLI

安装后直接用 `polya`（开发时 `uv run polya`），两种模式：

```bash
uv run polya                    # 交互式 REPL：多轮对话，斜杠命令控制会话
uv run polya -p "修复 pytest 失败的测试" --plan   # 单任务模式：执行一次即退出
```

常用参数：`--root DIR` 工作目录（默认 `.`，文件操作被限制在内）、`--plan` 启动进入
规划模式、`--trust` / `--no-trust` 保存并应用项目信任决定（`--trust` 加载其 `AGENTS.md` / 项目 skills，非交互场景必需）、
`--model/--base-url/
--api-key` 覆盖环境变量、`--no-compress` 关闭上下文压缩（默认开启）、`--no-microcompact`
关闭微压缩、`--context-window N`
（默认 128000）、`--keep-recent N`
（压缩保留区消息数，默认 30）、`--keep-recent-tokens N`（按 token 预算定保留区，
优先于 `--keep-recent`）、`--reserve-tokens N`（全量压缩绝对预留，默认 16384）。

提示词与用户可见文案由 `POLYA_LANG` 选择语言（`zh` 默认，`en` 面向英文受众）；
提示词在导入时求值，会话内稳定，不破 KV Cache 前缀。

REPL 斜杠命令：

| 命令 | 作用 |
|---|---|
| `/help` | 命令列表 |
| `/todos` `/status` | 查看 TODO 清单 / 会话状态（模式、用量、工具计数） |
| `/plan on\|go\|off` | 切换规划模式；`go` 批准当前计划进入执行（`exit_plan_mode` 构造时已注册，切换不动工具数组，缓存安全） |
| `/login [provider\|custom]` | 登录 provider：provider 列表 → base_url（预填可改）→ 隐藏输 key → 立即落盘，模型目录后台刷新；`custom` 自填端点与模型名 |
| `/logout <provider>` | 登出并移除 provider 凭据（若是默认模型则清空 active） |
| `/model [provider/模型]` | 切换模型：对话保留、旧模型 thinking 剥离、能力档案与窗口跟随；无参数跨已登录 provider 展开选项器，`Ctrl+S` 把高亮项存为默认启动模型 |
| `/thinking [off\|low\|medium\|high]` | 设置推理档位（一家一策）：o 系/gpt-5 发 `reasoning_effort`，GLM/DeepSeek 发 `thinking` 开关；档案无档位的模型明确提示不可切 |
| `/details [ID]` | 查看留档块全文：无参数看最近 5 块，带 ID 看指定块——滚动区的折叠块在这里看全量 |
| `/compact [说明]` | 立即压缩上下文（不等阈值）；可选说明聚焦摘要重点 |
| `/rename <主题>` | 重命名当前会话主题与终端标题（单行，最多 120 字） |
| `/resume [名称]` | 恢复已保存会话：无参数在原输入框按主题选择（主题 · 名字 · 更新时间）；自动落盘的会话都在这 |
| `/fork <id>` | 从指定入口分叉出新会话（复制根→该入口的祖先路径，`/tree` 看 id） |
| `/clone` | 复制当前会话为新会话（含投影编辑与主题） |
| `/export [路径]` | 导出当前会话：`.md` 为 Markdown（默认 `~/.polya/exports/<名字>.md`），`.jsonl` 为原始会话 |
| `/import <路径>` | 从任意路径导入会话为新会话：支持 polya JSONL 与 pi 会话格式（按 pi 规则重建活动分支，含 compaction / context_edit） |
| `/trust [决定]` | 查看/保存项目信任决定：`trust` / `trust-parent` / `untrust` / `clear`（true/false/null，父目录继承；下次启动生效） |
| `/new` | 开新会话：清空历史、TODO 与统计，分配新会话名，重置主题、丢弃排队消息并重印启动区；旧会话保留可 `/resume` 找回（别名 `/clear`、`/reset`） |
| `/exit` `/quit` | 退出（输入处 Ctrl+D / 空框双击 Ctrl+C 同效） |

`/plan`、`/login`、`/logout`、`/model` 无参数时在原输入框展开选项，并标记当前值；
`/model` 选项器里 `Ctrl+S` 把高亮项存为默认启动模型。方向键移动，
Tab / Enter 选中，再按 Enter 执行，Esc 关闭菜单。也可直接输入 `/plan on|go|off`、
`/login zai`、`/logout zai`、`/model zai/glm-5.3`，支持参数补全。命令或参数错误时保留草稿并提示；
`/details [ID]` 的 ID 只接受正整数（无参数看最近 5 块）。`/help` 的名称、别名与参数
与补全、执行共用同一平面注册表。命令随到随执行；会改会话树的命令
（`/new`、`/exit`、`/compact`、`/rewind`、`/jump`、`/edit`、`/load`、`/resume`、`/fork`、`/clone`、`/import`、`/model`、`/reload`、`/save`）
需要无运行中的任务，否则提示先按 Esc 中断。管道 REPL 不显示选项菜单，需要显式提供参数。

**会话**：每个会话有稳定名字与元数据（标题 / 创建 / 更新 / cwd），任务收尾自动落盘到
`~/.polya/sessions/<名字>.jsonl`（含 `-p` 单次运行：成功 / 未完成 / 中断 / 异常都会保存），
所以 `/resume` 列出的是真正用过的会话。无参 `/resume`
在原输入框展开选择器（主题 · 名字 · 更新时间），`/resume <名字>` 直切。切会话会清零统计、
TODO 与读改追踪，互不串味。`/fork <id>` 从祖先路径派生新会话，`/clone` 复制当前会话。
`/export [路径]` 按当前分支导出 Markdown；`/save` 仍落原始 JSONL 供 `/load`。

**输入前缀**：`!command` 本地跑 shell、输出进上下文（8000 字符截断）；`#note`
追加一行到项目记忆 `AGENTS.md`（下节）；`@` 触发文件路径补全；`/` 补全命令并带说明列。
大段粘贴自动折叠为 `[Pasted #1 +200 lines]`，提交时展开全文送模型。

**项目记忆**：工作目录**已信任**且有 `AGENTS.md` 时，启动读一次、注入系统提示词尾部
——会话内不变，不违「系统提示词静态」铁律的精神（铁律防的是逐轮变更破缓存）；
未信任目录不加载。`#` 前缀写入的内容下次会话生效。

**Skills**：项目技能放在 `.polya/skills/<名称>/SKILL.md`，用户技能放在
`~/.polya/skills/<名称>/SKILL.md`，目录递归发现——含 `SKILL.md` 的目录即技能根，
不再下钻。Agent Skills 标准位置同样默认加载：`~/.agents/skills/`（用户级）与从
工作目录逐级向上直至 git 仓库根的各级 `.agents/skills/`。同名优先级从低到高：
用户 `.agents` → 用户 `.polya` → 祖先 `.agents`（远→近）→ 项目 `.polya`。文件用
YAML frontmatter 声明 `name` 和 `description`；无效条目警告后跳过。启动只注入目录元数据，模型在任务匹配
或用户点名 `$名称` 时用 `skill_read(name=...)` 加载正文。引用资源通过该工具的
`path` 按技能目录解析，不能越出目录；大文件按行及字符偏移分页。脚本与普通 bash 命令一样
以进程权限运行；项目 skills 仅在已信任目录加载；用户技能不扩大项目文件工具的读取范围。新增技能在下次启动时发现。

仓库自带 `$develop-polya`：读取任务票与约定、实现、检查并修复失败、检查 diff、记录
交付证据。例如在 REPL 输入：`使用 $develop-polya 实现 .scratch/feature/issues/01-task.md`。
技能是流程指导，不能覆盖用户当前指令和审批限制。

**信任与审查器**：工具声明 `kind`（read/write/exec/delegate）；**默认无逐调用审批**，
工具以进程权限运行、被限制在 `--root` 内。每次调用经可插拔 `Reviewer`
（`polya/review.py`）给出 allow / deny：默认 `AllowAllReviewer` 一律放行，仅规划模式下
拒绝非只读工具（delegate 例外，副作用在子调用）；未来的模型审查器（Jev 类）实现同一
协议来拦截命令。deny 只作为工具结果回传模型，不弹窗、不让位终端。

**信任门**：唯一的门，管项目资源是否加载（`AGENTS.md` 与项目/祖先 skills）。决定存入
`~/.polya/trust.json`，格式 `{"<绝对路径>": true | false | null}`，并**沿父目录继承**
（最近的 true/false 胜出；`null` = 显式「无决定」）。只有目录真正存在受保护资源时才询问；
不信任则不加载项目指令（它们是提示注入的载体），工具照常运行。`/trust` 查看已存决定
（含继承来源）并保存 Trust / Trust parent folder / Do not trust / Clear——下次启动生效。
非交互（`-p`/管道）不弹门：用 `--trust` / `--no-trust` 显式做一次性决定。
`/permissions` 与会话授权规则已删除。

**规划模式**：`exit_plan_mode` 把计划打印进滚动区并结束本轮，`plan_mode` 保持开启；
用 `/plan go`（或整条消息精确等于 `批准`/`go` 等白名单）批准执行，其他消息留在只读
计划模式继续修改。

**中断**：交互模式按 Esc 请求中断（补全菜单打开时先关闭菜单）；当前工具或模型
请求可能需要完成后才能停止。已完成步骤保留，未执行工具自动回填中断结果，排队消息
回到输入框。Ctrl+C 清空输入，空框两秒内双击退出；Ctrl+D 退出。`-p` 保留 Ctrl+C 中断。

### 交互与显示

- **启动区**：版本、一句定位与项目级信息（已加载 AGENTS.md，或未信任目录提醒），
  只显示一次随后自然滚走——模型/目录/快捷键留在常驻底栏，不重复。
- **常驻输入**：上下边线紧贴编辑区，续行缩进，最多六行后内部滚动。框外三行：身份行
  左侧是项目（`~` 缩写）、git 分支与会话主题，右侧是 `(provider) 模型 • 思考档`；
  用量行是累计 `↑输入 ↓输出 CR缓存 CH命中% $费用` 与 `ctx 占比/窗口 (auto)`；操作行
  左侧模式与队列，右侧随上下文变化的操作提示。窄终端依次舍 provider、思考档、分支、
  主题与用量细节，保住模型、项目名与模式。运行中仍可编辑草稿。
  交互会话先取首条任务作为临时名称，再异步请求当前模型生成简短名称；生成后更新会话文件，
  `-p` 单次运行在保存前生成主题。
  `/resume` 按主题显示。`/rename <主题>` 优先于自动生成，同步修改底部主题与终端标题，
  `/new`（及别名 `/clear`、`/reset`）重置主题。
- **输入操作**：Enter 发送（steering：下一模型请求前注入）；Alt+Enter 追加
  （follow-up：本任务结束后运行）；Ctrl+J / 行尾 `\` + Enter 换行；Alt+Up 取回排队
  消息；补全菜单打开时 Enter 选择候选。`/` 补命令，`@` 补文件；候选最多六行；
  长粘贴折叠，提交时展开。历史保存在 `~/.polya/history`。
- **忙时排队**：运行中仍可输入。Enter 的 steering 消息在当前批次的所有工具结果回填后、
  下次模型请求前注入；Alt+Enter 的 follow-up 在当前任务结束后作为下一任务运行。
  Alt+Up 把排队消息取回编辑器；Esc 中断也会把排队消息送回编辑器，不静默丢弃。
  `/new`（及别名 `/clear`、`/reset`）、`/exit` 等需先按 Esc 让任务空闲。
- **内容区**：保留原生终端滚动与复制。正文无需等换行，在输入框上方的尾窗持续
  显示 Markdown 尾部（最多八行正文，矮终端自动减少）；消息完成后一次写入滚动区，
  保留表格、列表与代码块排版。中断时保留已生成的正文并标记未完成。
  思考默认折叠，可用 `/details` 查看。
- **工具块**：工具展示名使用 `Bash`、`Read File`，内部标识不变。状态为 `Running` →
  `Ran` / `Failed` / `Denied`，命令高亮；结果预览最多三行、400 字符，bash 流式输出
  在尾窗持续更新尾部，滚动区只留下完成摘要。`+ Show details: /details ID` 按固定编号查看对应工具块
  的完整参数和返回结果
  （工具自身的输出上限仍有效）。
- **任务状态**：输入框上方统一显示当前动作与整轮耗时：`Waiting for model`、`Thinking`、
  `Responding`、`Running …` 或 `Reviewing`；工具切换不重置计时。
  `Stopping` 显示正在等待哪个动作。每轮模型任务留下结束、中断或失败回执及耗时，
  「本轮结束」不代表目标已验证成功。最近保留 20 块，
  过期编号会提示不可用。补全打开时提示 Esc 关闭补全，已请求停止时提示等待当前操作结束。
- **输出协调**：交互模式经同一输出代理，prompt_toolkit 独占输入区刷新，单一渲染路径、
  无 Rich Live。`-p` 的最终答案 stdout / 诊断 stderr 契约保持不变。

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
)  # 默认无逐调用审批：审查器（reviewer）一律放行，见下
```

几点说明：

- 工具执行抛出的异常不会中断循环，而是作为错误文本交回模型，让它自行调整——例如参数写错时
  模型有机会重试。
- `reviewer=Reviewer()` 是工具执行前的可插拔判定：返回 `Review("deny", reason)` 时该次
  调用被跳过，模型会收到该结果并纳入下一步推理（`polya/review.py`）。默认
  `AllowAllReviewer` 一律放行，仅规划模式下拒绝非只读工具。`Tool.kind`
  （read/write/exec/delegate）标记副作用分类。
- 同一个 `Agent` 实例会保留对话历史，可直接连续调用 `run()` 进行多轮对话；需要重新开始时调用
  `agent.reset()`。
- **CLI 无步数上限**（pi 语义）：`steps()` 不设轮数上限，跑到模型给出最终答案；唯一自动
  护栏是无进展熔断，随时可按 Esc 中断。库 API 仍可用
  `Agent(max_steps=..., max_continuations=...)` 启用软检查点（每走满 N 轮发
  `BudgetCheckpoint` 自动续跑，连跳上限后 `BudgetExhausted` 收尾），子代理用它做有界
  子任务。不再抛「超过最大步数」。
- **无进展熔断**（始终开启，无 CLI 开关）：同一工具、同一参数**连续**重复时，第 3 次把
  提醒拼进工具结果，再犯即**可续停止**（历史保留，发消息可继续）。CLI 无步数上限时它是
  唯一自动护栏；`bash_output` 等 `poll` 等待工具豁免（相同参数的轮询是合法等待）。阈值等
  策略在库 API `Agent(loop_guard=..., loop_repeat_limit=...)` 可调。
- `status_bar=True` 开启 Agent 状态栏：每轮迭代以 user 消息在上下文**末尾**追加
  `<agent_status>` 元信息（迭代号、各工具累计调用次数、token 用量、时间），工具结果
  也会标注「第 N 次调用」。模型检索强但归纳弱，让它自己从轨迹里数调用次数既慢又容易
  数错——状态栏用代码提前算好。更新采用持久追加（旧状态留在轨迹里，不删改），KV Cache
  前缀始终稳定。也可传入自定义渲染函数 `status_bar=lambda snapshot: ...`。
- **生成器协议**（ADR 0002，取代旧的 `on_event` 回调）：agent 是一个生成器，
  `yield` 统一事件、`result = yield ToolCall(...)` 把工具的执行权与审查交给消费方
  （驱动层）。`run()` 是**内置驱动**（消费生成器 + 审查器 + 执行器），库用法、
  `-p` 模式与测试复用；需要自定义审查或实时渲染的前端直接消费 `steps()`——CLI 的
  交互驱动（`polya/loop.py`）就这么做。事件词表（`polya/agent.py`，时序：
  `iteration → [usage] → *_delta* → assistant_message → (tool_call)*`，工具结果由
  驱动层执行后回填）：

  | 事件 | 载荷 | 时机 |
  |---|---|---|
  | `iteration` | `{step, max_steps}` | 每轮迭代开头 |
  | `reasoning_delta` / `text_delta` | `{delta}` | 流式片段（LLM 走流式；`stream=False` 关闭） |
  | `assistant_message` | `{content, reasoning, tool_calls}` | 一轮完整消息落历史后 |
  | `tool_call` | `{name, call_id, arguments}` | 请求工具执行，期待 send 回结果字符串 |
  | `usage` | `{last, total}` | 仅当本次响应带 usage |
  | `compaction` | `{before, after}` | 上下文压缩发生时 |
  | `plan_submitted` | `{plan}` | 规划模式提交计划，驱动层打印后结束本轮 |
  | `budget_checkpoint` | `{step, limit, continuation}` | 走满一轮预算，自动续跑（软检查点） |
  | `budget_exhausted` | `{step, limit, continuations}` | 连跳上限后收尾，历史保留可继续 |
  | `no_progress` | `{tool, arguments, count, phase}` | 无进展熔断：`nudged` 提醒 / `stopped` 停止 |

  注意它与 `status_bar` 不同：后者是给**模型**看的上下文内容，事件流是给**驱动层**看的
  控制与进度信号。
- `agent.total_usage` / `agent.last_usage` 累计/记录每次请求的 token 用量（响应里没有
  usage 字段时保持为 0 / `None`，不会报错）；只做统计，不进消息历史，不影响缓存前缀。
- **两阶段模式**：`Agent(plan_mode=True)` 启动时进入规划模式——写工具被审查器拒绝
  （带指引），模型探查后调用 `exit_plan_mode(plan=...)` 提交计划，驱动层打印并结束本轮；
  批准时调 `agent.leave_plan_mode()` 翻转为执行。状态栏会显示当前模式。
  工具数组全程不变（中途增删 tools 会破坏 KV Cache 前缀），模式切换只是运行时状态。
- **上下文压缩**：`Agent(compress=True)`（CLI 默认开启，`--no-compress` 关闭）。最近一次
  请求的 prompt tokens 加新增输入估算超过触发线
  （`min(窗口×阈值, 窗口−reserve_tokens)`，默认 128K×80%、预留 16384——小窗口下绝对
  预留先于百分比触发，保证留出一轮响应的空间，`--reserve-tokens` 可调）时，
  在两次 API 调用之间**批量压缩**保留区（最近 `keep_recent=30` 条）之外的旧 tool 结果：
  一次 LLM 调用（合并式，注入当前任务做任务感知压缩）把它们原地替换为带 `[COMPRESSED]`
  标记的摘要（防重复处理），消息条数与 tool_call_id 配对不变，对话脉络完整；同区的旧
  状态栏消息直接删除（噪声不做摘要）。替换点之后的 KV Cache 会失效——这是有意识的
  权衡，所以阈值高、批量压、低频次。使用最近 usage 校准新增内容；缺失 usage 时按
  UTF-8 大小估算（非精确 tokenizer），不能使用累计用量。空摘要保留原历史；连续三次
  压缩失败熔断。估算达到窗口 95% 时保留历史并停止请求，可切换更大窗口模型继续。
  没有可压工具结果时回退到完整摘要重启；切点保证同批工具结果已全部回填。
  压缩交接保留已加载技能来源与 TODO，摘要要求保留用户修正、验收条件、改动、实际
  验证结果和下一步；摘要重启按固定九节模板（目标与验收/约束与偏好/进展三态/
  关键决策与理由/文件/验证证据/失败路径/技能/下一步）组织，摘要调用的用量计入
  `total_usage`；provider 报上下文超窗错误时压缩后原地重试一次（每任务一次）。
  `history_read(snapshot="1", message=1, offset=0)` 可回查压缩前
  原始消息，每次最多 8000 字符。快照只在当前进程/会话有效，`/new`（`/clear`、`/reset`）清除，
  不是跨进程会话恢复。
- **模型能力声明**（`polya.providers.ModelProfile`）：Agent 只问能力、不特判模型名。
  压缩策略按 `supports_inplace_tool_edit` 自动切换——OpenAI 式模型（不回传 reasoning）
  原地替换 tool content；thinking 绑定前缀的模型（`deepseek-reasoner`/`claude`/`gemini`
  档案）改用**摘要重启**：整段旧历史压成一条 `<session_summary>` 消息，模型从摘要冷
  启动（原地替换会使保留的 thinking 全部失效）。`reasoning_passthrough` 控制
  `reasoning_content` 原样保存回传（DeepSeek interleaved thinking）；档案还提供
  `temperature` 默认值（o 系列为 `None` = 不传）与作为窗口解析链**兑底层**的
  `context_window`（发现值优先，其次档案，最后 128k）。`profile_for(model)`
  按前缀匹配，未知模型回落安全默认。CLI 的 `--model` 会自动查档案。
- **前缀不变量断言**（`Agent(prefix_check=True)` / CLI `--prefix-check`）：运行时校验
  每次请求是上一次的严格扩展（深拷贝基线 + 内容比较，有开销故默认关）。破坏前缀对
  纯 KV Cache 只是缓存变贵，对回传 thinking 的模型是**推理连续性断裂**——这条断言把
  「两个压缩点之间严格 append-only」从纪律变成代码。压缩/摘要重启/reset 后基线自动
  重置（压缩点是合法的推理重启点）。
- 不传 `system_prompt` 时使用内置的通用提示词；`default_tools` 建议搭配
  `polya.builtin.CODING_SYSTEM_PROMPT`（围绕内置工具的工作流：任务拆解 → 探查 →
  小步修改 → 验证 → 汇报）。系统提示词应当 100% 静态——动态信息请追加到对话末尾，而不是
  改写提示词。CLI 启动时将项目记忆 `AGENTS.md` 和 skills 元数据一次性注入系统提示词，
  会话内保持不变。

## 内置工具

`default_tools(root)` 返回以下工具，**所有文件操作都被限制在 `root` 目录内**——解析路径后校验，
`../` 和绝对路径都会被拒绝。`kind` 是副作用分类（审查器与规划只读过滤按它走）：
`read` 无副作用、`write` 写文件、`exec` 执行命令：

| 工具 | kind | 说明 |
|---|---|---|
| `read_file` | read | 读文件，**输出带行号**；支持行号片段；拒绝二进制和超大文件 |
| `list_dir` | read | 列目录（单层），子目录以 `/` 结尾 |
| `glob` | read | 按文件名模式递归找文件（自动跳过 `.venv`/`.git` 等） |
| `grep` | read | 正则搜索内容，支持**上下文行**、忽略大小写、`glob` 限定文件名 |
| `write_file` | write | 写/覆盖文件，自动创建父目录 |
| `edit_file` | write | 定点替换，默认要求匹配唯一 |
| `multi_edit` | write | 一次多处替换，**原子生效**（任一处失败全不落盘） |
| `bash` | exec | **持久会话**执行命令：cwd/环境变量跨调用保持 |
| `bash_output` | read | 读取或等待命令结果；按命令编号、行范围回查完整输出 |
| `kill_bash` | exec | 终止持久会话 |
| `web_fetch` | read | 抓取 URL，HTML 转文本，`<external_content>` 包裹防注入；**拒绝内网/localhost（SSRF 防护）** |
| `todo_write` | read | 全量重写 TODO 清单（可选；需 `default_tools(todos=store)` + `Agent(todos=store)` 共享同一实例） |

驱动层另外装配 `task`（kind `delegate`）：把一个有界子任务交给隔离上下文的子代理
（独立 history 与 shell 会话），把探查链挡在主上下文之外；只有最终报告回到父层，
子代理的工具调用仍逐个经过共享审查器（深度为 1，子代理没有 `task` 工具）。

上下文分两级管理：**微压缩**（不调 LLM）在用量过窗口 60% 后把大块旧工具结果换成
`history_read` 回查指针；**压缩**（LLM 摘要）在 80% 触发；`/compact` 手动触发全量压缩。
用量上报时若端点提供 `cached_tokens` 会显示缓存命中率。

多数工具结果截断到 8000 字符；bash 预览保留头尾及退出码。`bash(timeout=N)` 仅等待
N 秒（0–300），到期返回命令编号，命令继续运行。用 `bash_output(timeout=30)` 再次
等待；完成前不能启动下一条前台命令。`bash_output(command_id=1, start_line=1,
end_line=200)` 回查完整日志，过长页按提示用字符 `offset` 继续。日志在工具集存活期间
有效。命令必须非交互；`kill_bash` 显式终止会话，POSIX 下连同进程组一起终止，下次
调用重启。显式 `&` 后台任务共用 stdout，验证任务宜使用前台命令以准确归属输出。
CLI 另装配只读 `skill_read`，开启压缩时增加只读 `history_read`。
`web_fetch` 用标准库实现，零第三方依赖。

TODO 清单是状态栏的「任务规划」组件：`todo_write` 写入共享的 `TodoStore`，
状态栏每轮把它渲染到上下文末尾——模型不用从历史里回忆还剩什么（外部记忆）：
`agent.reset()` 会连同清单一起清空。

**跨平台**：除 `bash` 会话（需要 bash，Windows 需 WSL/Git Bash）外，全部工具为
纯 Python 实现，Windows 原生可跑——不依赖 grep/rg/find 等系统命令。

## 项目结构

```
polya/
  agent.py       # 生成器协议：事件联合类型 + steps() + run() 内置驱动（ADR 0002）
  loop.py        # 交互驱动：渲染 / 审查 / 执行 / 派发 / steering 队列
  input.py       # InputBox：多行编辑、@ 路径补全、粘贴折叠、steering 按键、状态栏
  render.py      # 滚动区渲染器 + 编辑器上方尾窗
  review.py      # 可插拔工具审查器（allow/deny），默认放行
  trust.py       # 项目信任门（AGENTS.md / 项目 skills）
  executor.py    # 工具执行器（loop 与 run() 共用）
  cli.py         # argparse + Agent 装配 + 单任务模式 + AGENTS.md 启动注入
  llm.py         # OpenAI 兼容接口封装：chat(on_delta) 与 chat_iter 双形态
  models.py      # provider 配置（~/.polya/models.json）：读写校验迁移、预置表、模型发现、启动解析
  tools.py       # @tool 装饰器与工具注册表（kind 分类）
  builtin.py     # 内置编码工具 + 编码代理提示词（内容层）
  shell.py       # 持久 bash 会话（读线程 + 哨兵标记协议）
  web.py         # web_fetch：抓取 + HTML 转文本 + 来源标记
  status.py      # 状态栏快照与默认渲染器
  todos.py       # TODO 清单存储（外部记忆）
  compact.py     # 上下文压缩：原地替换 + 摘要重启（按模型能力选择）
  skills.py      # 技能发现、元数据目录、只读正文与资源加载
  history.py     # 压缩前原始历史的内存只读快照与分页回查
  prompt.py      # 系统提示词具名 section 装配
  i18n.py        # 文案目录：POLYA_LANG 选择语言（zh 默认 / en）
  providers.py   # 模型能力声明：只问能力不特判型号
examples/       # 可运行示例（库用法）与演示工具
tests/          # 用假 LLM 验证循环 + 内置工具/会话/抓取的沙箱测试
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
