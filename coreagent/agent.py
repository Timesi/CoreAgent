"""
Agent核心loop：
    user message -> LLM -> tool calls? -> execute -> loop
                        -> text reply? -> return to user
    如果返回的内容中没有工具调用，那么意味着模型的回答结束了。
"""

import concurrent.futures
import inspect
from .llm import LLM
from .tools import ALL_TOOLS
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tools_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)

        # 如果使用自带里工具，需要设置字代码的父Agent引用
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role":"system", "content": self._system}] + self.messages

    def _tool_schema(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _exec_tool(self, tc) -> str:
        # 执行单工具调用，返回工具执行结果
        tool = self._tools_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"

        # 先验证参数，以免工具内部抛出的 TypeError 被调用者误判为参数错误
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"

        try:
            return tool.execute(**tc.arguments)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        # 使用线程并发调用多个工具
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    def _answer_pending_tool_calls(self, tool_calls):
        # 为每次未收到回复的呼叫补充一个工具回复。
        # 与 OpenAI 兼容的 API 会拒绝包含工具调用但未对应工具回复的助手消息，因此当执行中途被中断时，历史记录仍保持有效。
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """处理用户发送的消息"""
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = self.llm.chat(
                messages = self._full_messages(),
                tools = self._tool_schema(),
                on_token = on_token,
            )

            # 如果没有工具调用，表明模型这一轮已经结束了，返回模型的回答
            if not resp.tool_calls:
                self.messages.append(resp.message)
                return resp.content

            # 工具调用 -> 执行（当有多个时并行执行，会同时运行独立的工具）
            self.messages.append(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # 并行执行多工具调用
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip (resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # 执行过程中按 Ctrl+C 会导致助手工具调用消息无响应，从而污染下一次请求
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # 如果输出太大的话，需要压缩
            self.context.maybe_compress(self.messages, self.llm)

        return "(reached maximum tool-call rounds)"

    def reset(self):
        """清理会话历史"""
        self.messages.clear()