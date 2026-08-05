"""配置环境变量和默认值。"""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """从当前工作目录加载 .env 文件，向上遍历至主目录。如果缺少 python-dotenv，则不执行任何操作。"""
    try:
        from dotenv import load_dotenv
        # 先搜索当前目录，然后向上查找父目录直到 ~
        env_path = Path(".env")
        if not env_path.exists():
            cur = Path.cwd()
            home = Path.home()
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
                cur = cur.parent
        load_dotenv(env_path, override=False)
    except ImportError:
        pass  # 未安装 python-dotenv，静默跳过


@dataclass
class Config:
    model: str = "gpt-5.6-luna"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"

    @classmethod
    def from_env(cls) -> "Config":
        # 如果存在 .env 文件，则加载（不会覆盖已有的环境变量）
        _load_dotenv()
        # 自动获取常见的环境变量
        api_key = (
            os.getenv("COREAGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        return cls(
            model=os.getenv("COREAGENT_MODEL", "gpt-5.6-luna"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("COREAGENT_BASE_URL"),
            max_tokens=int(os.getenv("COREAGENT_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("COREAGENT_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("COREAGENT_MAX_CONTEXT", "128000")),
            provider=os.getenv("COREAGENT_PROVIDER", "openai"),
        )
