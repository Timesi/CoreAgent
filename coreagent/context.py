"""多层上下文压缩。
Claude Code 采用了一种四层策略：
    1. HISTORY_SNIP - 将旧工具输出剪辑为一行摘要
    2. 微型紧凑型 - 基于LLM生成的旧转录摘要（已缓存）
    3. 上下文折叠 - 靠近硬限制时的激进压缩
    4. 自动压缩 - 定期背景压缩
这里通过三层实现相同的理念：  
    第一层（tool_snip）——将详细的工具输出替换为简短版本  
    第二层（summarize）——使用大语言模型生成旧对话的摘要  
    第三层（hard_collapse）——最后手段：仅保留摘要和近期内容
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# LLM只用作类型提示，运行时不导入
if TYPE_CHECKING:
    from .llm import LLM


def _approx_tokens(text: str) -> int:
    """粗略的Token数量，混合英文/中文内容每Token约3个字符"""
    return len(text) // 3


def estimate_tokens(messages: list[dict]) -> int:
    """估算消息列表的Token数量"""
    total = 0
    for m in messages:
        if m.get("content"):
            total += _approx_tokens(m["content"])
        if m.get("tool_calls"):
            total += _approx_tokens(str(m["tool_calls"]))
    return total


class ContextManager():
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        # 每一层级的阈值,最大Token的比例
        self._snip_at = int(max_tokens * 0.5)   # 50% -> 裁剪工具输出
        self._summarize_at = int(max_tokens * 0.7)  # 70% -> 摘要旧对话
        self._collapse_at = int(max_tokens * 0.9)  # 90% -> 上下文折叠

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """检查每一层的消息是否超过阈值，如果超过则压缩。"""
        current = estimate_tokens(messages)
        compressed = False

        # Layer 1: 截取详细工具输出
        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)

        # Layer 2: 使用LLM摘要旧对话
        if current > self._summarize_at and len(messages) > 10:
            if self._summarize_old(messages, llm, keep_recent=8):
                compressed = True
                current = estimate_tokens(messages)

        # Layer 3: 上下文折叠
        if current > self._collapse_at and len(messages) > 4:
            self._hard_collapse(messages, llm)
            compressed = True

        return compressed

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """
        Layer 1：将超过1500个字符的工具结果截断为首行或末行。
        这与Claude Code的HISTORY_SNIP类似，用单行摘要替换旧工具输出，以节省上下文空间。
        """
        changed = False
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if len(content) <= 1500:
                continue
            # 因为要保留前后三行，所以至少需要6行才可以压缩
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            # 保留前三行和后三行，中间内容形成简短摘要
            snipped = (
                "\n".join(lines[:3]) 
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            msg["content"] = snipped
            changed = True
        return changed

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """安全地将消息列表拆分为旧消息和保留尾部应从哪个索引开始。
        将边界回退，以确保“工具”结果永远不会与产生它的消息的工具调用分离,
        一个被遗弃的工具消息没有前序的工具调用，而 OpenAI 兼容的 API 会拒绝它。
        近期消息，确保至少保留keep_recent条消息。
        """
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    def _summarize_old(self, messages: list[dict], llm: LLM | None, keep_recent: int = 8) -> bool:
        """Layer 2：使用LLM生成旧对话的摘要。"""
        if len(messages) <= keep_recent:
            return False

        split = self._safe_split(messages, keep_recent)
        old = messages[:split]
        recent = messages[split:]

        summary = self._get_summary(old, llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(recent)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        """Layer 3：上下文折叠，保留摘要和近期内容。"""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        recent = messages[split:]
        summary = self._get_summary(messages[:split], llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(recent)

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """通过大语言模型生成摘要，或回退到提取方式"""
        flat = self._flatten(messages)

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:
                pass

        # 备用：提取关键行
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "?")
            text = msg.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """回退：在没有LLM的情况下提取文件路径、错误和决策。"""
        import re
        files_seen = set()
        errors = []

        for m in messages:
            text = m.get("content", "") or ""
            # extract file paths
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            # extract error lines
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"

