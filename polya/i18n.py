"""界面文案目录：终端渲染层（render / loop / input）面向用户的文案，支持多语言。

语言分层（``.scratch/language-policy/spec.md``）：本目录只服务「人看什么」；
发给模型的一切——系统提示词、运行时注入消息、工具 schema、状态栏、工具
返回值——固定英文，见 :mod:`polya.prompts`。

语言由环境变量 ``POLYA_LANG`` 选择（``zh`` 默认，``en`` 面向英文受众）；未知值
回落到 ``zh``。文案在**导入时**按当前语言求值，因此运行期间稳定。

新增文案：在 ``_ZH`` 与 ``_EN`` 两个表里都加一条同名键（``ui.*``），用 ``{name}``
占位符传参。en 版对齐 pi 品味：短、符号化、中点分隔。
"""

from __future__ import annotations

import os

_ZH: dict[str, str] = {
    # ---- render.py：事件提示行 ----
    "ui.render.thinking_stream": "✻ 思考中 · {n} 字",
    "ui.render.thinking_done": "✻ 思考 {n} 字",
    "ui.render.thinking_full": "✻ 思考全文（{n} 字）：",
    "ui.render.checkpoint": "↻ 第 {step} 轮检查点，自动继续（第 {continuation} 次）",
    "ui.render.budget_exhausted": (
        "⏸ 已达续跑上限（{step} 轮 / 连跳 {continuations} 次），本轮收尾；发送消息可继续"
    ),
    "ui.render.no_progress_stopped": "⏹ 检测到重复调用 {tool} ×{count}，已停止；发送消息可继续",
    "ui.render.no_progress_nudged": "↺ 检测到重复调用 {tool} ×{count}，已提醒模型",
    "ui.render.plan_submitted": "⏺ 计划已提交",
    "ui.render.plan_hint": "  /plan go 开始执行 · 或直接输入修改意见（仍在计划模式）",
    "ui.render.compact_micro": (
        "⌁ 上下文微压缩：清理旧工具结果约 {cleared} 字符（可 history_read 回查）"
    ),
    "ui.render.compacted": "⌁ 上下文已压缩：{before} → {after} 条消息",
    "ui.render.no_blocks": "（暂无可展开的块——先跑一个任务，或非终端会话不记录）",
    # ---- loop.py：outcome 与提示行 ----
    "ui.loop.new_session": "新会话",
    "ui.loop.truncated_tail": "…（已截断）",
    "ui.loop.noted": "已记入 {path}",
    "ui.loop.tagline": "和你一起理解问题、制定计划、完成验证",
    "ui.loop.untrusted": "⚠ 未信任此目录：AGENTS.md / 项目 skills 未加载（/trust 查看）",
    "ui.loop.loaded_memory": "已加载项目记忆 AGENTS.md",
    "ui.loop.origin_subtask": "子任务",
    "ui.loop.queued_steering": "＋ 已排队（下一轮请求前交给模型）：{text}",
    "ui.loop.queued_followup": "＋ 已排队（本轮结束后交给模型）：{text}",
    "ui.loop.steering_delivered": "＋ 补充已交给模型，将用于下一轮请求。",
    "ui.loop.error": "[错误] {exc}",
    "ui.loop.default_model_set": "已设为默认启动模型：{provider}/{model}",
    "ui.loop.topic_updated": "已更新会话主题：{topic}",
    "ui.loop.queue_dropped": "（已丢弃 {count} 条排队消息）",
    "ui.loop.outcome_final": "本轮结束",
    "ui.loop.outcome_plan_wait": "等待计划确认",
    "ui.loop.outcome_checkpoint": "达检查点收尾",
    "ui.loop.outcome_no_progress": "检测到重复调用，已停止",
    "ui.loop.outcome_interrupted": "本轮已中断",
    "ui.loop.outcome_failed": "本轮失败",
    "ui.loop.plan_approved": "已批准计划，进入执行。",
    "ui.loop.plan_feedback": "计划修改意见已交给模型；仍在计划模式（只读）。",
    "ui.loop.budget_message": "已达单轮预算，历史已保留；继续请直接发送下一条消息。",
    "ui.loop.no_progress_message": (
        "检测到重复调用，本轮已停止；历史已保留，继续请直接发送下一条消息。"
    ),
    "ui.loop.interrupted_message": "已中断本次任务；已完成步骤保留，排队消息回到输入框。",
    "ui.loop.interrupted_short": "已中断本次任务",
    "ui.loop.task_failed": "[任务失败] {exc}",
    "ui.loop.autosave_failed": "[自动保存失败] {exc}",
    "ui.loop.busy_command": "当前任务运行中；先按 Esc 中断再执行 {command}。",
    "ui.loop.usage": "[用量] {usage}",
    # ---- input.py：底栏与提示 ----
    "ui.input.key_hints": "Enter 发送 · /help",
    "ui.input.placeholder": "输入任务，或用 @ 引用文件",
    "ui.input.queued_count": "已排队 {count} 条",
    "ui.input.hint_steer": "Enter 引导 · Alt+Enter 追加",
    "ui.input.hint_select": "Tab / Enter 选择 · Esc 关闭",
    "ui.input.hint_multiline": "Ctrl+J 换行 · Alt+Enter 追加",
    "ui.input.hint_esc_close": "Esc 关闭",
    "ui.input.hint_enter_steer": "Enter 引导",
    "ui.input.hint_selector_run": "选择选项后 Enter 执行 · Esc 关闭",
    "ui.input.hint_selector_model": "Enter 切换 · Ctrl+S 设为默认 · Esc 关闭",
    "ui.input.current": " · 当前",
    "ui.input.aliases": "别名 {aliases}",
    "ui.input.cleared": "已清空（空框双击 Ctrl+C 退出）",
    "ui.input.quit_confirm": "再按一次 Ctrl+C 退出",
}

_EN: dict[str, str] = {
    # ---- render.py: event hint lines ----
    "ui.render.thinking_stream": "✻ Thinking · {n} chars",
    "ui.render.thinking_done": "✻ Thinking · {n} chars",
    "ui.render.thinking_full": "✻ Full thinking ({n} chars):",
    "ui.render.checkpoint": "↻ Checkpoint at turn {step}, auto-continuing (hop {continuation})",
    "ui.render.budget_exhausted": (
        "⏸ Continuation limit reached (turn {step} / {continuations} hops); "
        "send a message to continue"
    ),
    "ui.render.no_progress_stopped": (
        "⏹ No-progress stop: repeated {tool} ×{count}; send a message to continue"
    ),
    "ui.render.no_progress_nudged": "↺ No-progress nudge: repeated {tool} ×{count}",
    "ui.render.plan_submitted": "⏺ Plan submitted",
    "ui.render.plan_hint": "  /plan go to execute · or type feedback (still in plan mode)",
    "ui.render.compact_micro": (
        "⌁ Micro-compaction: cleared ~{cleared} chars of old tool results (history_read to recall)"
    ),
    "ui.render.compacted": "⌁ Context compacted: {before} → {after} messages",
    "ui.render.no_blocks": (
        "(no blocks to expand yet — run a task first; non-terminal sessions record none)"
    ),
    # ---- loop.py: outcomes and hint lines ----
    "ui.loop.new_session": "New session",
    "ui.loop.truncated_tail": "…(truncated)",
    "ui.loop.noted": "Noted to {path}",
    "ui.loop.tagline": "Understand, plan, verify — together",
    "ui.loop.untrusted": "⚠ Directory not trusted: AGENTS.md / project skills not loaded (/trust)",
    "ui.loop.loaded_memory": "Loaded project memory AGENTS.md",
    "ui.loop.origin_subtask": "subtask",
    "ui.loop.queued_steering": "＋ Queued (injected before the next model request): {text}",
    "ui.loop.queued_followup": "＋ Queued (sent after this turn): {text}",
    "ui.loop.steering_delivered": "＋ Steering delivered; it will shape the next model request.",
    "ui.loop.error": "[error] {exc}",
    "ui.loop.default_model_set": "Set as default startup model: {provider}/{model}",
    "ui.loop.topic_updated": "Session topic updated: {topic}",
    "ui.loop.queue_dropped": "({count} queued messages dropped)",
    "ui.loop.outcome_final": "turn complete",
    "ui.loop.outcome_plan_wait": "awaiting plan approval",
    "ui.loop.outcome_checkpoint": "checkpoint reached",
    "ui.loop.outcome_no_progress": "no-progress stop",
    "ui.loop.outcome_interrupted": "interrupted",
    "ui.loop.outcome_failed": "turn failed",
    "ui.loop.plan_approved": "Plan approved; executing.",
    "ui.loop.plan_feedback": "Feedback delivered to the model; still in plan mode (read-only).",
    "ui.loop.budget_message": (
        "Turn budget reached; history preserved. Send another message to continue."
    ),
    "ui.loop.no_progress_message": (
        "Repeated identical tool calls detected; turn stopped with history preserved. "
        "Send another message to continue."
    ),
    "ui.loop.interrupted_message": (
        "Task interrupted; finished steps kept, queued messages returned to the input."
    ),
    "ui.loop.interrupted_short": "Task interrupted",
    "ui.loop.task_failed": "[task failed] {exc}",
    "ui.loop.autosave_failed": "[autosave failed] {exc}",
    "ui.loop.busy_command": "A task is running; press Esc to interrupt before /{command}.",
    "ui.loop.usage": "[usage] {usage}",
    # ---- input.py: bottom bar and hints ----
    "ui.input.key_hints": "Enter to send · /help",
    "ui.input.placeholder": "Type a task, or @ to reference a file",
    "ui.input.queued_count": "queued {count}",
    "ui.input.hint_steer": "Enter steer · Alt+Enter follow-up",
    "ui.input.hint_select": "Tab / Enter select · Esc close",
    "ui.input.hint_multiline": "Ctrl+J newline · Alt+Enter follow-up",
    "ui.input.hint_esc_close": "Esc close",
    "ui.input.hint_enter_steer": "Enter steer",
    "ui.input.hint_selector_run": "Enter to run · Esc close",
    "ui.input.hint_selector_model": "Enter switch · Ctrl+S default · Esc close",
    "ui.input.current": " · current",
    "ui.input.aliases": "aliases {aliases}",
    "ui.input.cleared": "Cleared (double Ctrl+C on empty input to quit)",
    "ui.input.quit_confirm": "Press Ctrl+C again to quit",
}

_CATALOG: dict[str, dict[str, str]] = {"zh": _ZH, "en": _EN}


def current_language() -> str:
    """当前语言：``POLYA_LANG``（zh 默认，en），未知值回落 zh。"""
    value = os.environ.get("POLYA_LANG", "zh").strip().lower()
    return value if value in _CATALOG else "zh"


def t(key: str, **kwargs) -> str:
    """取一条界面文案并按当前语言格式化；缺键回落中文，再缺则原样返回键。"""
    table = _CATALOG[current_language()]
    template = table.get(key) or _ZH.get(key, key)
    return template.format(**kwargs) if kwargs else template
