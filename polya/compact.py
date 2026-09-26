"""上下文压缩（书 2.7）：把膨胀的 tool 结果换成蒸馏过的知识。

三个动机：控制长度与成本；**总结后的知识比原始记录更利于模型使用**（注意力
检索强、归纳弱——状态栏是给那台"只有一半的检索引擎"补算好的结论，压缩是把
臃肿的原始记录换成结论，同一枚硬币的两面）；缓解上下文焦虑（窗口未满也该压，
否则上下文腐化：装得下但找不到）。

与 KV Cache 的调和（书 2.7 及参考实现 chapter2/context-compression/）：

- System Prompt 与工具定义永不动；
- 压缩只发生在**两次 API 调用之间**，只替换 tool 消息的 content（原地替换：
  消息条数与 tool_call_id 配对不变，对话脉络完整保留）；
- 替换点之后的缓存全部失效——这是有意识的权衡，因此阈值高（80%）、批量压、
  低频次，而不是每轮都压。

注意：若模型把 reasoning/thinking 绑定在前缀上（DeepSeek-R1 类推理模型），
原地替换同样会使保留区的 thinking 失效，书推荐的替代是把整段旧历史压成一条
摘要消息。polya 保存并回传 reasoning_content（agent 侧随消息追加），因此
providers.ModelProfile 按 reasoning_passthrough 能力选择策略：支持原地替换
tool content 的用 compact_messages，thinking 绑定前缀的用 compact_restart。
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from .i18n import t

COMPRESS_MARKER = "[COMPRESSED]"
MICROCLEAR_MARKER = "[CLEARED]"
MICRO_MIN_CHARS = 2000  # 微压缩只清理大于此长度的旧工具结果，避免噪声
MAX_PIECE = 4000  # 单条 tool 内容进压缩请求前的截断阈值（保头尾）


def estimate_tokens(parts) -> int:
    """轻量 token 估算：UTF-8 字节数 / 3，配合服务器 usage 校准实际用量。

    压缩的保留区预算按 token 计（消息条数是伪单位：一条 tool 结果可以是 100
    也可以是 4 万 token），估算函数集中在这里，方便日后替换为真实 tokenizer。
    """
    payload = json.dumps(parts, ensure_ascii=False)
    return len(payload.encode("utf-8")) // 3


def effective_keep(history: list[dict], keep: int, keep_tokens: int | None) -> int:
    """把 token 预算换算成保留区的消息条数（从末尾向前累积到预算为止）。

    ``keep_tokens`` 为 None 时退回消息条数 ``keep``（兼容旧语义）。换算结果
    不超过 ``keep`` 与历史长度，且至少保留 1 条（历史非空时）。
    """
    if keep_tokens is None or keep_tokens <= 0 or not history:
        return keep
    budget = 0
    kept = 0
    for message in reversed(history):
        size = estimate_tokens(message)
        if kept > 0 and budget + size > keep_tokens:
            break
        budget += size
        kept += 1
    return max(1, kept)


def extract_file_operations(messages: list[dict]) -> tuple[set[str], set[str]]:
    """从消息里的工具调用提取 (已读文件, 已改文件)，供压缩摘要累积追踪。

    只认参数的 ``path`` 字段；解析失败或参数不是对象时跳过。read_file 计入
    已读，write_file / edit_file / multi_edit 计入已改。
    """
    read_files: set[str] = set()
    modified_files: set[str] = set()
    writers = {"write_file", "edit_file", "multi_edit"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if name not in (writers | {"read_file"}):
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(arguments, dict):
                continue
            path = arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            (modified_files if name in writers else read_files).add(path)
    return read_files, modified_files


SUMMARY_SYSTEM_PROMPT = t("prompt.summary")


class _Chat(Protocol):
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta=None,
    ): ...


def _is_status_message(message: dict) -> bool:
    content = message.get("content") or ""
    return message.get("role") == "user" and "<agent_status>" in content


def compressible_indices(history: list[dict], keep: int) -> list[int]:
    """可压缩的 tool 消息下标：保留区外、未带压缩标记（防重复处理）。

    微压缩过的指针（``MICROCLEAR_MARKER``）不再进全量摘要——它已经是结论。
    """
    boundary = len(history) - keep
    return [
        i
        for i, message in enumerate(history)
        if message.get("role") == "tool"
        and i < boundary
        and not (message.get("content") or "").startswith((COMPRESS_MARKER, MICROCLEAR_MARKER))
    ]


def stale_status_indices(history: list[dict], keep: int) -> list[int]:
    """可删除的旧状态栏下标：每轮重复的元信息是噪声，对噪声做摘要只是浪费。"""
    boundary = len(history) - keep
    return [i for i, message in enumerate(history) if _is_status_message(message) and i < boundary]


def clearable_indices(
    history: list[dict], keep: int, min_chars: int = MICRO_MIN_CHARS
) -> list[int]:
    """可微压缩的 tool 下标：保留区外、未带任何压缩标记、且内容足够长。

    微压缩不调 LLM：直接把旧工具结果换成回查指针（读原文用 history_read）。
    已在保留区、已压缩或过短的都不动。
    """
    boundary = len(history) - keep
    targets: list[int] = []
    for index, message in enumerate(history):
        if message.get("role") != "tool" or index >= boundary:
            continue
        content = message.get("content") or ""
        if content.startswith((MICROCLEAR_MARKER, COMPRESS_MARKER)):
            continue
        if len(content) >= min_chars:
            targets.append(index)
    return targets


def microcompact(
    history: list[dict],
    keep: int,
    snapshot: str,
    min_chars: int = MICRO_MIN_CHARS,
) -> tuple[list[dict], int] | None:
    """微压缩：把大块旧工具结果换成 ``history_read`` 回查指针（无 LLM 调用）。

    返回 ``(新历史, 清理字符数)``；没有候选时返回 None。指针文本含回查参数
    （快照编号 + 消息编号，从 1 起）；纯替换——消息条数与 tool_call_id 配对不变。
    原列表不动。
    """
    targets = clearable_indices(history, keep, min_chars)
    if not targets:
        return None
    target_set = set(targets)
    cleared_chars = 0
    new_history: list[dict] = []
    for index, message in enumerate(history):
        if index in target_set:
            original = message.get("content") or ""
            cleared_chars += len(original)
            new_history.append(
                {
                    **message,
                    "content": (
                        f"{MICROCLEAR_MARKER} 原始输出 {len(original)} 字符已清理；"
                        f"回查原文：history_read(snapshot={snapshot!r}, message={index + 1})"
                    ),
                }
            )
        else:
            new_history.append(message)
    return new_history, cleared_chars


def _parse_numbered(text: str, targets: list[int]) -> dict[int, str] | None:
    """从压缩回复中解析 ``#i: 摘要`` 分段；一条有效编号都没有时返回 None。"""
    summaries: dict[int, str] = {}
    current_id: int | None = None
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        match = re.match(r"^#(\d+)\s*[:：]\s*(.*)$", raw_line.strip())
        if match:
            if current_id is not None:
                summaries[current_id] = "\n".join(current_lines).strip()
            current_id = int(match.group(1))
            current_lines = [match.group(2)]
        elif current_id is not None:
            current_lines.append(raw_line)
    if current_id is not None:
        summaries[current_id] = "\n".join(current_lines).strip()
    wanted = set(targets)
    result = {i: s for i, s in summaries.items() if i in wanted and s}
    return result or None


def render_conversation(messages: list[dict]) -> str:
    """把一段对话渲染成喂给压缩调用的文本（状态栏跳过，单条超长截断保头尾）。"""
    lines = []
    for message in messages:
        if _is_status_message(message):
            continue
        role = message.get("role", "?")
        content = message.get("content") or ""
        if message.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}({c['function']['arguments']})"
                for c in message["tool_calls"]
            )
            content = f"[调用工具: {calls}]" if not content else f"{content}\n[调用工具: {calls}]"
        if len(content) > MAX_PIECE:
            half = MAX_PIECE // 2
            content = f"{content[:half]}\n...[中段截断]...\n{content[-half:]}"
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def find_restart_split(history: list[dict], keep: int) -> int | None:
    """摘要重启的安全切点：history[:i] 压成一条摘要，history[i:] 原样保留。

    切点之前至少有一条 assistant，且全部工具调用已回填；尾部不以 tool 开始。
    从目标位置向后找完整轮边界，避免在同批工具结果中间切断；没有合适边界则跳过。
    """
    target = len(history) - keep
    if target < 1:
        return None
    pending: set[str] = set()
    seen_assistant = False
    for i, message in enumerate(history):
        if i >= target and seen_assistant and not pending and message.get("role") != "tool":
            return i
        if message.get("role") == "assistant":
            seen_assistant = True
            pending.update(c["id"] for c in message.get("tool_calls", []))
        elif message.get("role") == "tool":
            pending.discard(message.get("tool_call_id", ""))
    if keep == 0 and seen_assistant and not pending:
        return len(history)
    return None


def compact_restart(
    llm: _Chat,
    history: list[dict],
    keep: int,
    query: str,
    previous_summary: str | None = None,
) -> list[dict] | None:
    """摘要重启（书 2.7 对 thinking 绑定模型的推荐方案）。

    thinking/reasoning 与产生它的前缀绑定（签名校验），原地替换旧 tool 内容
    会让保留区的全部 thinking 失效。正解：把切点之前的整段历史压成**一条**
    摘要消息，模型从摘要冷启动重新推理，保留区作为干净的前缀基线。
    """
    split = find_restart_split(history, keep)
    if split is None or split == 0:
        return None
    request = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"当前任务：{query}\n\n"
                + (
                    f"上一版摘要（迭代压缩：请在其基础上合并更新，不要丢失其中仍然有效的信息）：\n"
                    f"{previous_summary}\n\n"
                    if previous_summary
                    else ""
                )
                + f"以下是一段 Agent 对话历史，"
                f"请压缩成一份结构化摘要（保留关键决策、约束及其理由、已排除的"
                f"失败路径、文件路径与结论性输出）。按目标与验收、用户约束、已改文件、"
                f"验证证据、失败路径、技能、未完成事项与下一步组织。\n\n"
                f"{render_conversation(history[:split])}"
            ),
        },
    ]
    response = llm.chat(request, tools=None)
    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        raise RuntimeError("压缩调用返回空摘要")
    summary_message = {
        "role": "user",
        "content": (
            "<session_summary>\n"
            f"[此前对话的压缩摘要（原始 {split} 条消息已折叠）]\n{summary}\n"
            "</session_summary>"
        ),
    }
    # 重启后尾部 reasoning 也不再具有原前缀，保留正文与工具链，丢弃旧推理字段。
    tail = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in history[split:]]
    return [summary_message, *tail]


def compact_messages(
    llm: _Chat,
    history: list[dict],
    keep: int,
    query: str,
    previous_summary: str | None = None,
) -> list[dict] | None:
    """压缩历史并返回新列表（原列表不动）；没有可压消息时返回 None。

    一次 LLM 调用合并压缩全部待压内容（合并式优于逐条：跨源的重复信息一次
    去重）；query 让压缩任务感知（书实验 2-10：40k vs 93k token 的差距来源）。
    LLM 异常向上抛，由调用方计数熔断。
    """
    targets = compressible_indices(history, keep)
    if not targets:
        return None
    stale = set(stale_status_indices(history, keep))

    pieces = []
    for i in targets:
        content = history[i].get("content") or ""
        if len(content) > MAX_PIECE:
            half = MAX_PIECE // 2
            content = f"{content[:half]}\n...[中段截断]...\n{content[-half:]}"
        pieces.append(f"#{i}: {content}")
    request = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"当前任务：{query}\n\n"
                + (
                    f"上一版摘要（迭代压缩：并入仍需保留的信息，不要重复已丢弃的细节）：\n"
                    f"{previous_summary}\n\n"
                    if previous_summary
                    else ""
                )
                + f"以下是 Agent 历史中的 {len(targets)} 条工具结果，请逐条压缩：\n\n"
                + "\n\n".join(pieces)
            ),
        },
    ]
    response = llm.chat(request, tools=None)
    summary_text = response.choices[0].message.content or ""
    if not summary_text.strip():
        raise RuntimeError("压缩调用返回空摘要，保留原始历史")

    summaries = _parse_numbered(summary_text, targets)
    merged_into_earliest: set[int] = set()
    if summaries is None:
        # 完全解析失败：整段输出当作综合摘要放进最早一条待压消息，其余标记
        # 已并入——信息无损，不因输出格式问题而失败
        summaries = {targets[0]: summary_text.strip()}
        merged_into_earliest = set(targets[1:])
    # 部分编号缺失：缺的那几条保留原内容，等下次触发再压（比"并入"更稳）

    new_history: list[dict] = []
    for i, message in enumerate(history):
        if i in stale:
            continue  # 旧状态栏直接删除，不做摘要
        if i in merged_into_earliest:
            message = {**message, "content": f"{COMPRESS_MARKER} (内容已并入前述摘要)"}
        elif i in summaries:
            new_content = summaries[i]
            original = message.get("content") or ""
            message = {
                **message,
                "content": (
                    f"{COMPRESS_MARKER} [原 {len(original)} 字符 → {len(new_content)} 字符]\n"
                    f"{new_content}"
                ),
            }
        new_history.append(message)
    return new_history
