# Agent 改为生成器协议，审批与工具执行外移至驱动层

Agent 与 UI 的解耦此前靠 `on_event` 回调，但「请求-等待」语义（审批要阻塞等人的决定、
工具结果要回填驱动后续推理）无法用回调表达——bash 实时输出只能靠 cli 层旁路 tap 直喂
渲染器、绕过事件流，成为引擎不感知 UI 的例外路径。决定：agent 是一个生成器，`yield`
统一事件（Text / ReasoningDelta / Usage / ToolCall / Compaction…），`result = yield
ToolCall(...)` 把工具的执行权与审批交给消费方（驱动层）；`run()` 降级为内置驱动
（消费生成器 + 共享执行器 + 审批策略），库用法、`-p` 模式与测试复用之；`on_event`
删除，不做双轨。压缩、状态栏注入等上下文管理仍属 agent；plan 模式判定进入驱动层的
权限判定（decide）。

## Considered Options

- **维持 on_event 回调，把 shell 输出补成事件**：回调仍无法表达审批的阻塞等待，UI 永远
  是被动接收方；tap 修掉了，结构没变。
- **双轨并存（回调 + 生成器）**：两个真相源，事件词表分叉，维护税永久化。
- **asyncio 改造**：纯 Python 同步路线已定，为此引入事件循环不合算。

## Consequences

- 事件词表成为公共契约：需文档 + 测试锁定（词表见 `.scratch/interaction-v2/spec.md`）。
- Ctrl+C 中断走 `gen.close()`：agent 须在 GeneratorExit 路径回填未决 ToolCall（拒绝结果），
  保持历史完整、对话可续。
