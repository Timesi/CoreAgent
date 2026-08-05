"""所有tools的基类"""

from abc import ABC, abstractmethod

class Tool(ABC):
    """最小工具接口，子类在此基础上添加新能力"""
    name: str
    description: str
    parameters: dict
    # 仅当工具可与任意其他 True 工具并发时才覆盖为 True。
    # 新工具默认不安全，避免遗漏审计后被错误并发。
    is_concurrency_safe: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """运行工具返回结果"""
        ...

    def schema(self) -> dict:
        """openai函数调用schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            },
        }