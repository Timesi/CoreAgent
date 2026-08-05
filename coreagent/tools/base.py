"""所有tools的基类"""

from abc import ABC, abstractmethod

class Tool(ABC):
    """最小工具接口，子类在此基础上添加新能力"""
    name: str
    description: str
    parameters: dict

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