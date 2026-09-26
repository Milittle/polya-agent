# 子代理隔离上下文与压缩工程化

与生产级 agent 的差距分析表明：探查链（grep → read_file × N）全额进主上下文是长任务
腐化的主因，压缩无论多好都是事后、有损的补救。决定：新增 `task` 工具（kind
`delegate`），在父工具调用内**同步**运行一个子 Agent——独立 history、独立 shell
会话、完整读写执行能力——只有最终报告进入父上下文；副作用不发生在 task 工具本身，
而发生在子代理的工具调用上、逐个经过与父会话**共享**的 ApprovalGate（会话规则、
yolo、allow_all 自动适用，审批面板标注「子任务」）。配套把压缩工程化：微压缩
（无 LLM，旧工具结果换快照回查指针，阈值低于全量）、cached_tokens 采集与缓存命中
显示、Compaction 事件渲染、`/compact` 手动压缩。

## Considered Options

- **异步/并行子代理（独立线程或进程池）**：上下文隔离收益相同，但审批的多路借用、
  单 worker 模型、patch_stdout 顺序都要重设计；本期同步已拿走主要收益，推迟。
- **只读探查子代理**：审批递归、渲染嵌套两大难题直接消失，但用户拍板直接做通用
  能力；递归审批经共享 gate + 显式 unrestricted 参数（以父会话审批模式为准，而非
  子 Agent 实例——子 Agent approve 恒为 None，沿用实例取值会把子代理变成事实 yolo）
  化解。
- **新事件类 Microcompaction**：词表膨胀；Compaction 加默认字段 `mode`/`cleared`
  即可区分，既有事件序列锁定测试零扰动。
- **每轮微压缩 vs 阈值触发**：每轮清理破坏前缀缓存的代价分期不可控；阈值（0.6，
  低于全量 0.8）触发 + 保留区外 + 尺寸下限，与「压缩点即缓存重启点」的既有权衡一致。
- **子代理经驱动层一等概念（非工具）**：需要新协议分支；工具 + 闭包 + dispatch
  注入即够，且天然复用 freeze 纪律与 on_shell_output tap 先例。

## Consequences

- 子 Agent 的 `approve` 恒为 None，故 `run_tool_call` 的 unrestricted 必须显式传父
  会话取值（Q6 绕过洞的修法）；`-p` 无 --yes 时子审批同样装闸门版 dispatch，非交互
  默认拒绝不旁路。
- 子审批被拒置共享 rejected_reason，父层停整轮——有意语义，测试锁定，防止将来被
  当 bug「修」掉。
- 渲染器单槽位：子工具执行不进滚动区，进度走父 task 尾窗；`task_progress` 事件须
  重申槽位（gate 的 tool_approval 事件会改写它）。
- 微压缩点与全量压缩点同规：重置前缀基线、last_usage、估算基线，`--prefix-check`
  不误报；thinking 绑定前缀的模型档案不做微压缩（原地改写使保留区 reasoning 失效）。
- 子代理深度 1（子无 task 工具）与同步执行是本期边界，词表与权限层不为更深递归
  预留。
