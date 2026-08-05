"""会话持久化 - 保存和恢复对话。
精简为：消息的 JSON 序列化 + 模型配置来保存会话
"""

import json
import re
import time
import uuid
from pathlib import Path

SESSIONS_DIR = Path.home() / ".coreagent" / "sessions"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SESSION_ID_LEN = 100  # 将文件名长度控制在系统限制之内


def _normalize_session_id(session_id: str | None) -> str:
    """规范化会话ID，确保安全的文件名"""
    # 没有传入session_id，说明是保存会话，生成一个新的UUID
    if not session_id:
        return _new_session_id()

    # 传入了session_id，规范化为安全的文件名
    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    if len(name) > _MAX_SESSION_ID_LEN:
        name = name[:_MAX_SESSION_ID_LEN].strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    """生成一个新的会话ID"""
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _session_path(session_id: str) -> Path:
    """返回会话文件的路径"""
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """将会话保存到磁盘，并保存session ID"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    data = {
        "id": session_id,
        "model": model,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id

def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """加载保存的会话，返回 (messages, model) 或者 None."""
    path = _session_path(session_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["messages"], data["model"]
    except (json.JSONDecodeError, KeyError, OSError):
        # 截断的会话文件不应导致恢复崩溃
        return None


def list_sessions() -> list[dict]:
    """列出可获得的session，最新的在前"""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # 将第一个用户消息作为预览
            preview = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "user" and msg.get("content"):
                    preview = msg["content"][:80]
                    break
            sessions.append({
                "id": data.get("id", f.stem),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sessions[:20]