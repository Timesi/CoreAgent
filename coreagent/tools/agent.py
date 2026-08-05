"""
子代理创建思路是：对于复杂的子任务，可以创建一个独立的代理，拥有自己的对话历史和工具访问权限。
这样主代理就可以像“去研究这个代码库并汇报结果”一样委派工作，而不会污染其自身的上下文窗口。
子代理运行完毕并返回一段文本摘要。
"""

from .base import Tool

class AgentTool(Tool):
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    # 由 Agent.__init__ 在构造后设置
    _parent_agent = None

    def execute(self, task: str) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        # 此处导入避免循环依赖
        from ..agent import Agent

        parent = self._parent_agent
        sub_agent = Agent(
            llm=parent.llm,
            tools=[t for t in parent.tools if t.name != "agent"],
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
        )

        try:
            result = sub_agent.chat(task)
            # 剪裁子Agent的长结果，避免返回给父Agent的上下文过长
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as e:
            return f"Sub-agent error: {e}"