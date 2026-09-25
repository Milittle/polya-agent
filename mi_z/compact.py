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
摘要消息。mi-z 当前不保存 reasoning_content，不受此影响；接入推理模型时需
重新评估这里。
"""

from __future__ import annotations

import re
from typing import Protocol

COMPRESS_MARKER = "[COMPRESSED]"
MAX_PIECE = 4000  # 单条 tool 内容进压缩请求前的截断阈值（保头尾）

SUMMARY_SYSTEM_PROMPT = """\
你负责压缩 Agent 对话历史中的工具结果。压缩原则：

- 信息价值非均匀：关键决策、事实结论、文件路径、命令及其结果的价值高于过程
  细节，高于冗余噪声（导航栏、重复提示语等直接舍弃）。
- 语义完整性：名字、时间、数字、路径等关键信息一个都不能丢——"Sutskever 于
  2024 年 5 月离开 OpenAI"不能压成"Sutskever 离开"。
- 任务相关性：围绕当前任务取舍，对任务推进有用的细节保留，无关的舍去。

输出格式：对每一条输入，输出一段以 `#编号: ` 开头的摘要；多条输入涉及同一
事实时，在最早出现的编号下合并表述。直接输出结果，不要寒暄与解释。"""


class _Chat(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None): ...


def _is_status_message(message: dict) -> bool:
    content = message.get("content") or ""
    return message.get("role") == "user" and "<agent_status>" in content


def compressible_indices(history: list[dict], keep: int) -> list[int]:
    """可压缩的 tool 消息下标：保留区外、未带压缩标记（防重复处理）。"""
    boundary = len(history) - keep
    return [
        i
        for i, message in enumerate(history)
        if message.get("role") == "tool"
        and i < boundary
        and not (message.get("content") or "").startswith(COMPRESS_MARKER)
    ]


def stale_status_indices(history: list[dict], keep: int) -> list[int]:
    """可删除的旧状态栏下标：每轮重复的元信息是噪声，对噪声做摘要只是浪费。"""
    boundary = len(history) - keep
    return [i for i, message in enumerate(history) if _is_status_message(message) and i < boundary]


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

    切点必须是 user 消息（我们的循环总是回填完所有 tool_call 才进下一轮，
    中断也有补齐，故「下一条是 user」保证之前是完整轮结束，tool 链不悬空）。
    从目标位置向后找，保留更多、压得更少（保守侧）。历史太短（保留区盖住
    几乎全部）时返回 None，不做无意义的折叠。
    """
    target = len(history) - keep
    if target < 1:
        return None
    for i in range(target, len(history)):
        if history[i].get("role") == "user":
            return i
    return None


def compact_restart(
    llm: _Chat,
    history: list[dict],
    keep: int,
    query: str,
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
                f"当前任务：{query}\n\n以下是一段 Agent 对话历史，"
                f"请压缩成一份结构化摘要（保留关键决策、约束及其理由、已排除的"
                f"失败路径、文件路径与结论性输出）。\n\n{render_conversation(history[:split])}"
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
    return [summary_message, *history[split:]]


def compact_messages(
    llm: _Chat,
    history: list[dict],
    keep: int,
    query: str,
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
                f"以下是 Agent 历史中的 {len(targets)} 条工具结果，请逐条压缩：\n\n"
                + "\n\n".join(pieces)
            ),
        },
    ]
    response = llm.chat(request, tools=None)
    summary_text = response.choices[0].message.content or ""

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
