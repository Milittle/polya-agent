"""界面文案目录：终端渲染层（render / loop / input）面向用户的文案，支持多语言。

语言分层（``.scratch/language-policy/spec.md``）：本目录只服务「人看什么」；
发给模型的一切——系统提示词、运行时注入消息、工具 schema、状态栏、工具
返回值——固定英文，见 :mod:`polya.prompts`。

语言由三个来源解析，优先级从高到低：环境变量 ``POLYA_LANG``（临时覆盖）、
``~/.polya/settings.json`` 的 ``language`` 字段（持久化，进程内读一次）、
默认 ``en``；未知或非法值一路回落到下一级。文案经 ``t()`` 按当前语言格式化。

新增文案：在 ``_ZH`` 与 ``_EN`` 两个表里都加一条同名键（``ui.*``），用 ``{name}``
占位符传参。en 版对齐 pi 品味：短、符号化、中点分隔。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
    "ui.render.plan_submitted": "● 计划已提交",
    "ui.render.plan_hint": "  /plan go 开始执行 · 或直接输入修改意见（仍在计划模式）",
    "ui.render.compact_micro": (
        "⌁ 上下文微压缩：清理旧工具结果约 {cleared} 字符（可 history_read 回查）"
    ),
    "ui.render.compacted": "⌁ 上下文已压缩：{before} → {after} 条消息",
    "ui.render.no_blocks": "（暂无可展开的块——先跑一个任务，或非终端会话不记录）",
    "ui.render.replay_omitted": "… 已省略更早 {count} 条入口（/details <id> 或 history_read 回查）",
    "ui.render.replay_summary": "（压缩摘要）",
    # ---- 状态标签（忙碌行动作：render 产出、loop 赋值、input 显示） ----
    "ui.status.waiting_model": "等待模型",
    "ui.status.thinking": "思考中",
    "ui.status.responding": "回答中",
    "ui.status.reviewing": "审核中",
    "ui.status.running_tool": "执行 {tool}",
    "ui.status.stopping": "停止中",
    "ui.status.stopping_hint": "等待中：{status}",
    "ui.status.current_operation": "当前操作",
    # ---- loop.py：outcome 与提示行 ----
    "ui.loop.new_session": "新会话",
    "ui.loop.truncated_tail": "…（已截断）",
    "ui.loop.noted": "已记入 {path}",
    "ui.loop.tagline": "和你一起理解问题、制定计划、完成验证",
    "ui.loop.untrusted": "⚠ 未信任此目录：AGENTS.md / 项目 skills 未加载（/trust 查看）",
    "ui.loop.loaded_memory": "已加载项目记忆 AGENTS.md",
    "ui.loop.origin_subtask": "子任务",
    "ui.loop.queued_steering": "＋ 已排队（当前工具批次结束后注入下一轮请求）：{text}",
    "ui.loop.queued_followup": "＋ 已排队（本轮结束后交给模型）：{text}",
    "ui.loop.steering_delivered": "＋ 已注入上下文（将随下一轮请求发送）：{text}",
    "ui.loop.error": "[错误] {exc}",
    "ui.loop.default_model_set": "已设为默认启动模型：{provider}/{model}",
    "ui.loop.topic_updated": "已更新会话主题：{topic}",
    "ui.loop.queue_dropped": "（已丢弃 {count} 条排队消息）",
    "ui.loop.queue_empty": "（队列为空）",
    "ui.loop.queue_header": "排队的消息（按送达顺序）：",
    "ui.loop.queue_kind_steering": "引导",
    "ui.loop.queue_kind_followup": "追加",
    "ui.loop.queue_when_steering": "当前工具批次结束后送达",
    "ui.loop.queue_when_followup": "本轮结束后作为下一任务",
    "ui.loop.queue_index_invalid": "队列序号超出范围（当前 {count} 条）",
    "ui.loop.queue_dropped_one": "已丢弃：{text}",
    "ui.loop.queue_recalled_one": "已取回编辑：{text}",
    "ui.loop.queue_usage": "用法：/queue [list | drop N | take N]",
    "ui.loop.paste_empty": "（没有折叠的粘贴块）",
    "ui.loop.paste_header": "折叠的粘贴块：",
    "ui.loop.paste_index_invalid": "粘贴块序号超出范围（当前 {count} 块）",
    "ui.loop.paste_token_missing": "编辑器里已找不到该占位符（可能已被修改或删除）",
    "ui.loop.paste_expanded": "已展开粘贴块 #{index} 到编辑器",
    "ui.loop.paste_dropped": "已删除粘贴块 #{index}",
    "ui.loop.paste_usage": "用法：/paste [list | show N | expand N | drop N]",
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
    "ui.input.indexing": "索引中…（文件联想稍后可用）",
    "ui.input.hint_steer": "Enter 引导 · Alt+Enter 追加",
    "ui.input.hint_select": "Tab / Enter 选择 · Esc 关闭",
    "ui.input.hint_multiline": "Ctrl+J 换行 · Alt+Enter 追加",
    "ui.input.hint_esc_close": "Esc 关闭",
    "ui.input.hint_enter_steer": "Enter 引导",
    "ui.input.hint_interrupt": "esc 中断",
    "ui.input.hint_close_completions": "esc 关闭补全",
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
    "ui.render.plan_submitted": "● Plan submitted",
    "ui.render.plan_hint": "  /plan go to execute · or type feedback (still in plan mode)",
    "ui.render.compact_micro": (
        "⌁ Micro-compaction: cleared ~{cleared} chars of old tool results (history_read to recall)"
    ),
    "ui.render.compacted": "⌁ Context compacted: {before} → {after} messages",
    "ui.render.no_blocks": (
        "(no blocks to expand yet — run a task first; non-terminal sessions record none)"
    ),
    "ui.render.replay_omitted": (
        "… {count} earlier entries omitted (use /details <id> or history_read)"
    ),
    "ui.render.replay_summary": "(compaction summary)",
    # ---- status labels (busy-line action; set by render, shown by input) ----
    "ui.status.waiting_model": "Waiting for model",
    "ui.status.thinking": "Thinking",
    "ui.status.responding": "Responding",
    "ui.status.reviewing": "Reviewing",
    "ui.status.running_tool": "Running {tool}",
    "ui.status.stopping": "Stopping",
    "ui.status.stopping_hint": "waiting: {status}",
    "ui.status.current_operation": "current operation",
    # ---- loop.py: outcomes and hint lines ----
    "ui.loop.new_session": "New session",
    "ui.loop.truncated_tail": "…(truncated)",
    "ui.loop.noted": "Noted to {path}",
    "ui.loop.tagline": "Understand, plan, verify — together",
    "ui.loop.untrusted": "⚠ Directory not trusted: AGENTS.md / project skills not loaded (/trust)",
    "ui.loop.loaded_memory": "Loaded project memory AGENTS.md",
    "ui.loop.origin_subtask": "subtask",
    "ui.loop.queued_steering": (
        "＋ Queued (injected once the current tool batch ends, before the next request): {text}"
    ),
    "ui.loop.queued_followup": "＋ Queued (sent after this turn): {text}",
    "ui.loop.steering_delivered": "＋ Injected into context (sent with the next request): {text}",
    "ui.loop.error": "[error] {exc}",
    "ui.loop.default_model_set": "Set as default startup model: {provider}/{model}",
    "ui.loop.topic_updated": "Session topic updated: {topic}",
    "ui.loop.queue_dropped": "({count} queued messages dropped)",
    "ui.loop.queue_empty": "(queue is empty)",
    "ui.loop.queue_header": "Queued messages (in delivery order):",
    "ui.loop.queue_kind_steering": "steer",
    "ui.loop.queue_kind_followup": "follow-up",
    "ui.loop.queue_when_steering": "after the current tool batch",
    "ui.loop.queue_when_followup": "after this turn, as the next task",
    "ui.loop.queue_index_invalid": "index out of range ({count} queued)",
    "ui.loop.queue_dropped_one": "dropped: {text}",
    "ui.loop.queue_recalled_one": "recalled for editing: {text}",
    "ui.loop.queue_usage": "usage: /queue [list | drop N | take N]",
    "ui.loop.paste_empty": "(no folded paste blocks)",
    "ui.loop.paste_header": "Folded paste blocks:",
    "ui.loop.paste_index_invalid": "paste index out of range ({count} blocks)",
    "ui.loop.paste_token_missing": "placeholder not found in the editor (edited or deleted)",
    "ui.loop.paste_expanded": "expanded paste block #{index} into the editor",
    "ui.loop.paste_dropped": "deleted paste block #{index}",
    "ui.loop.paste_usage": "usage: /paste [list | show N | expand N | drop N]",
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
    "ui.input.indexing": "indexing… (file suggestions appear shortly)",
    "ui.input.hint_steer": "Enter steer · Alt+Enter follow-up",
    "ui.input.hint_select": "Tab / Enter select · Esc close",
    "ui.input.hint_multiline": "Ctrl+J newline · Alt+Enter follow-up",
    "ui.input.hint_esc_close": "Esc close",
    "ui.input.hint_enter_steer": "Enter steer",
    "ui.input.hint_interrupt": "esc to interrupt",
    "ui.input.hint_close_completions": "esc to close completions",
    "ui.input.hint_selector_run": "Enter to run · Esc close",
    "ui.input.hint_selector_model": "Enter switch · Ctrl+S default · Esc close",
    "ui.input.current": " · current",
    "ui.input.aliases": "aliases {aliases}",
    "ui.input.cleared": "Cleared (double Ctrl+C on empty input to quit)",
    "ui.input.quit_confirm": "Press Ctrl+C again to quit",
}

_CATALOG: dict[str, dict[str, str]] = {"zh": _ZH, "en": _EN}

_DEFAULT_LANGUAGE = "en"

_stored_language: str | None = None
_stored_read = False


def settings_path() -> Path:
    """用户设置文件：与 trust.json / models.json 同层的 ``~/.polya/settings.json``。"""
    return Path.home() / ".polya" / "settings.json"


def _settings_language() -> str | None:
    """settings.json 的 ``language`` 字段；进程内首次调用时读一次并缓存。

    读不到（文件不存在 / 非法 JSON / 字段非字符串）一律返回 None，交由
    :func:`current_language` 回落。改文件后需重启进程生效。
    """
    global _stored_language, _stored_read
    if not _stored_read:
        _stored_read = True
        try:
            data = json.loads(settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        value = data.get("language") if isinstance(data, dict) else None
        _stored_language = value.strip().lower() or None if isinstance(value, str) else None
    return _stored_language


def current_language() -> str:
    """当前语言：``POLYA_LANG`` > settings.json 的 ``language`` > 默认 ``en``。"""
    for candidate in (os.environ.get("POLYA_LANG"), _settings_language()):
        if candidate and (candidate := candidate.strip().lower()) in _CATALOG:
            return candidate
    return _DEFAULT_LANGUAGE


def t(key: str, **kwargs) -> str:
    """取一条界面文案并按当前语言格式化；缺键回落中文，再缺则原样返回键。"""
    table = _CATALOG[current_language()]
    template = table.get(key) or _ZH.get(key, key)
    return template.format(**kwargs) if kwargs else template
