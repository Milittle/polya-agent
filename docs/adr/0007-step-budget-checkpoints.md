# 单轮预算改为软检查点，`0` 表示无界

`--max-steps` 曾是单轮的**硬上限**：`steps()` 走满 `max_steps` 轮就
`raise RuntimeError("超过最大步数 N")`。这带来三个问题：长任务被任意阈值腰斩且以
红色失败收场；「任务总预算」与「跑飞熔断」共用一个数字（放宽就不安全、收紧就腰斩）；
没有表达「真正无界」的方式。对照 pi：它的 agent loop **没有任何步数上限**，终止由
模型「不再调用工具」与用户 Esc 决定，安全边界放在工具权限与上下文压缩上。

## Decision

预算从「硬上限」改为「软检查点周期 + 连跳上限」：

1. `max_steps > 0` 时，每走满 `max_steps` 轮且模型仍要继续，`steps()` yield
   `BudgetCheckpoint(step, limit, continuation)` 并**原地自动续跑**，回合不结束。
   连跳 `max_continuations` 次后 yield `BudgetExhausted(step, limit, continuations)`，
   返回一句提示串收尾（`agent.last_run_exhausted = True`）。**删除 `raise`**：
   终止方式从异常改为事件 + 返回值。
2. `max_steps == 0` 表示无界（pi 语义）：不发检查点，循环直到模型给出最终答案。
3. 默认值分层：CLI `--max-steps` = 100、内部续跑上限 = 4（不做 CLI 开关）；
   `Agent()` 默认 `max_steps=10` + `max_continuations=0`（库默认仍是有界硬停，
   不抛异常）；子代理 `max_steps=20` + `max_continuations=0`（有界子任务语义不变，
   首个检查点即报错收尾）。
4. 驱动层按事件展示：交互渲染 dim 检查点提示、预算收尾给「可继续」提示（**不是**
   `[任务失败]`）；`-p` 在收尾时打印答案 + `[未完成]` + 退出码 1，且所有出口
   （成功 / 未完成 / 中断 / 异常）都 `autosave()`。

## Considered Options

- **直接删掉上限（纯 pi 式无界）**：最贴近 pi，但在无进展熔断落地前，模型跑飞会
  无界烧钱。先用 `max_continuations` 兜底，无界作为显式选项 `--max-steps 0`。
- **保留硬上限只调大默认值**：仍会腰斩超长任务，且失败模式不变。
- **用「无进展」替代轮数**：更精准，但检测信号（同工具同参数重复 / 连续无新信息）
  需要单独设计，作为后续票；本 ADR 先解决「不再硬失败」。
- **检查点后暂停等用户确认**：把决策推给用户，交互里多余；pi 是自动续跑 + Esc 中断，
  采用自动续跑。

## Consequences

- `steps()` 不再抛「超过最大步数」；`run()` 返回收尾提示串。子代理的
  `except RuntimeError` 改为识别 `BudgetExhausted` 事件。
- `max_steps == 0` 时状态栏不显示 `/0`（status.py 用「（无上限）」）。
- 事件词表新增 `budget_checkpoint` / `budget_exhausted`，驱动层渲染，引擎不感知 UI。
- 真正跑飞的安全护栏（无进展熔断）尚未实现，自动续跑上限只是数量兜底；
  `--max-steps 0` 的语义在此之前不建议作为默认。
- 交互层 `run_task` 返回值从 `bool`（是否计划）改为字符串收尾原因
  （`"plan"` / `"budget"` / `"final"`）。
- ADR 0002 的生成器协议不变：仍是 yield 事件 + `send` 回填，只是终止语义改为
  「检查点事件 + 返回值」。
