# 会话历史改为不可变树 + 确定性投影

对齐 pi 的上下文架构，polya 的原始历史从扁平可变 `list[dict]`（原地被压缩/微压缩改写）
改为**不可变会话树**：每条记录是带稳定 id 的 Entry，按 parentId 连成树，当前路径为
active branch，并 JSONL 持久化。发给模型的上下文改为**确定性投影**——`project(active_branch)`
派生出 provider messages，原始 Entry 永不原地改写。压缩、skills 热加载、task/subagent 全部
落为「树的 append-only 操作」或「投影层操作」。KV Cache 前缀友好从「字节稳定铁律 + prefix_check
的纪律」变成「Entry 不可变 + 投影确定性」的**结构保证**。

## Considered Options

- **维持扁平 append-only + 纪律（现状）**：缓存友好，但「回溯 / 编辑历史 / 分叉」都不可做，
  压缩靠旁挂 `HistoryArchive` 快照 + `history_read(snapshot, message)` 双坐标回查。
- **不可变线性入口 + 投影（无 parentId）**：拿到投影 + entry-id 的主要收益，但分支/回溯/编辑仍不可做，
  后续要树得返工。
- **完整树 + 投影 + JSONL 持久化（选它）**：一次性补齐回溯/编辑/持久化，Entry 天生不可变 +
  id 寻址，加树不返工；成本在会话持久化格式与投影重建引擎，按分期消化。

## Consequences

- **前缀友好仍成立**：投影确定性 + 常见路径 append-only；压缩 / 换模型 / skills reload / 分支切换是
  合法重启点，前缀基线清零、一次有界 miss 后重新稳定。`prefix_check` 检查面从全量历史缩到投影层。
- **reasoning_content（DeepSeek 前缀绑定）作为 assistant Entry 的 payload 冻结回传**，投影按序重放，
  重启点之间 append-only——这是 polya 相对 pi 特有的约束，测试要锁。
- **删除 `HistoryArchive` 快照与 snapshot/message 双坐标**，统一为 entry-id 寻址
  （`history_read(entry_id, offset)`、`/expand`、`/details` 同源）。
- **生成器协议（ADR 0002）不变**：`steps()` 仍 yield 事件、`send` 回填工具结果，只是内部从
  「原地改 `history`」变成「append Entry + 重建投影」。
- **`_size_cache` 按 entry id 而非 list index**，消除「压缩后同长沿用陈旧尺寸」一类失效 bug。
