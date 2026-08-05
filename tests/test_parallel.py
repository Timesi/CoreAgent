import threading
from typing import ClassVar

import pytest

from coreagent.agent import Agent
from coreagent.llm import LLMResponse, ToolCall
from coreagent.tools import ALL_TOOLS
from coreagent.tools.base import Tool


class StubLLM:
    """只返回预置响应，不访问模型服务。"""

    def __init__(self, responses=()):
        self._responses = iter(responses)

    def chat(self, messages, tools=None, on_token=None):
        return next(self._responses)


class FakeTool(Tool):
    description = "Test tool"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self, name, is_concurrency_safe, run):
        self.name = name
        self.is_concurrency_safe = is_concurrency_safe
        self._run = run

    def execute(self, **kwargs) -> str:
        return self._run(**kwargs)


class UnmarkedTool(Tool):
    name = "unmarked"
    description = "Tool that inherits the default concurrency policy"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self, run):
        self._run = run

    def execute(self, **kwargs) -> str:
        return self._run(**kwargs)


def _call(name, index):
    return ToolCall(id=f"call-{index}", name=name, arguments={})


def _agent(*tools):
    return Agent(llm=StubLLM(), tools=list(tools))


def _start_execution(agent, calls, on_tool=None):
    state = {}

    def target():
        try:
            state["results"] = agent._exec_tool_calls(calls, on_tool)
        except Exception as exc:  # noqa: BLE001 - re-raise worker failures in the test thread
            state["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    return thread, state


def _finish_execution(thread, state):
    thread.join(timeout=2)
    assert not thread.is_alive(), "tool execution did not finish"
    if "error" in state:
        raise state["error"]
    return state["results"]


def test_safe_tools_run_concurrently():
    first_started = threading.Event()
    second_started = threading.Event()

    def run_first():
        first_started.set()
        if not second_started.wait(timeout=1):
            return "first timed out"
        return "first"

    def run_second():
        second_started.set()
        if not first_started.wait(timeout=1):
            return "second timed out"
        return "second"

    agent = _agent(
        FakeTool("first", True, run_first),
        FakeTool("second", True, run_second),
    )

    results = agent._exec_tool_calls([
        _call("first", 1),
        _call("second", 2),
    ])

    assert results == ["first", "second"]


def test_builtin_tools_have_expected_concurrency_flags():
    flags = {tool.name: tool.is_concurrency_safe for tool in ALL_TOOLS}

    assert flags["read_file"] is True
    assert flags["grep"] is True
    assert flags["glob"] is True
    assert flags["write_file"] is False
    assert flags["edit_file"] is False
    assert flags["bash"] is False
    assert flags["agent"] is False


def test_single_safe_tool_runs_inline():
    caller_thread = threading.get_ident()

    def record_thread():
        return str(threading.get_ident())

    agent = _agent(FakeTool("safe", True, record_thread))
    results = agent._exec_tool_calls([_call("safe", 1)])

    assert results == [str(caller_thread)]


def test_unsafe_tool_waits_for_safe_batch():
    first_started = threading.Event()
    second_started = threading.Event()
    release_safe_tools = threading.Event()
    unsafe_started = threading.Event()

    def run_safe(started, result):
        started.set()
        release_safe_tools.wait(timeout=1)
        return result

    def run_unsafe():
        unsafe_started.set()
        return "unsafe"

    agent = _agent(
        FakeTool("first", True, lambda: run_safe(first_started, "first")),
        FakeTool("second", True, lambda: run_safe(second_started, "second")),
        FakeTool("unsafe", False, run_unsafe),
    )
    calls = [
        _call("first", 1),
        _call("second", 2),
        _call("unsafe", 3),
    ]

    thread, state = _start_execution(agent, calls)
    both_safe_started = (
        first_started.wait(timeout=1)
        and second_started.wait(timeout=1)
    )
    unsafe_started_early = unsafe_started.is_set()
    release_safe_tools.set()
    results = _finish_execution(thread, state)

    assert both_safe_started
    assert not unsafe_started_early
    assert unsafe_started.is_set()
    assert results == ["first", "second", "unsafe"]


def test_safe_tool_after_unsafe_tool_starts_later():
    unsafe_started = threading.Event()
    release_unsafe = threading.Event()
    safe_started = threading.Event()

    def run_unsafe():
        unsafe_started.set()
        release_unsafe.wait(timeout=1)
        return "unsafe"

    def run_safe():
        safe_started.set()
        return "safe"

    agent = _agent(
        FakeTool("unsafe", False, run_unsafe),
        FakeTool("safe", True, run_safe),
    )

    thread, state = _start_execution(agent, [
        _call("unsafe", 1),
        _call("safe", 2),
    ])
    unsafe_did_start = unsafe_started.wait(timeout=1)
    safe_started_early = safe_started.is_set()
    release_unsafe.set()
    results = _finish_execution(thread, state)

    assert unsafe_did_start
    assert not safe_started_early
    assert safe_started.is_set()
    assert results == ["unsafe", "safe"]


def test_multiple_unsafe_tools_keep_model_order():
    execution_order = []

    def record(name):
        execution_order.append(name)
        return name

    agent = _agent(
        FakeTool("first", False, lambda: record("first")),
        FakeTool("second", False, lambda: record("second")),
        FakeTool("third", False, lambda: record("third")),
    )

    results = agent._exec_tool_calls([
        _call("first", 1),
        _call("second", 2),
        _call("third", 3),
    ])

    assert execution_order == ["first", "second", "third"]
    assert results == ["first", "second", "third"]


def test_unmarked_tool_is_a_serial_barrier():
    unmarked_started = threading.Event()
    release_unmarked = threading.Event()
    safe_started = threading.Event()

    def run_unmarked():
        unmarked_started.set()
        release_unmarked.wait(timeout=1)
        return "unmarked"

    def run_safe():
        safe_started.set()
        return "safe"

    unmarked = UnmarkedTool(run_unmarked)
    agent = _agent(
        unmarked,
        FakeTool("safe", True, run_safe),
    )

    thread, state = _start_execution(agent, [
        _call("unmarked", 1),
        _call("safe", 2),
    ])
    unmarked_did_start = unmarked_started.wait(timeout=1)
    safe_started_early = safe_started.is_set()
    release_unmarked.set()
    results = _finish_execution(thread, state)

    assert unmarked.is_concurrency_safe is False
    assert unmarked_did_start
    assert not safe_started_early
    assert safe_started.is_set()
    assert results == ["unmarked", "safe"]


def test_unknown_tool_is_serial_and_returns_existing_error():
    execution_order = []

    def run(name):
        execution_order.append(name)
        return name

    agent = _agent(
        FakeTool("before", True, lambda: run("before")),
        FakeTool("after", True, lambda: run("after")),
    )

    results = agent._exec_tool_calls([
        _call("before", 1),
        _call("missing", 2),
        _call("after", 3),
    ])

    assert execution_order == ["before", "after"]
    assert results == [
        "before",
        "Error: unknown tool 'missing'",
        "after",
    ]


def test_results_keep_model_tool_call_order():
    first_started = threading.Event()
    second_finished = threading.Event()
    completion_order = []

    def run_first():
        first_started.set()
        second_finished.wait(timeout=1)
        completion_order.append("first")
        return "first result"

    def run_second():
        first_started.wait(timeout=1)
        completion_order.append("second")
        second_finished.set()
        return "second result"

    agent = _agent(
        FakeTool("first", True, run_first),
        FakeTool("second", True, run_second),
    )

    results = agent._exec_tool_calls([
        _call("first", 1),
        _call("second", 2),
    ])

    assert completion_order == ["second", "first"]
    assert results == ["first result", "second result"]


def test_on_tool_fires_when_each_batch_starts():
    unsafe_started = threading.Event()
    release_unsafe = threading.Event()
    notifications = []

    def run_unsafe():
        unsafe_started.set()
        release_unsafe.wait(timeout=1)
        return "unsafe"

    agent = _agent(
        FakeTool("first", True, lambda: "first"),
        FakeTool("second", True, lambda: "second"),
        FakeTool("unsafe", False, run_unsafe),
        FakeTool("after", True, lambda: "after"),
    )
    calls = [
        _call("first", 1),
        _call("second", 2),
        _call("unsafe", 3),
        _call("after", 4),
    ]

    def on_tool(name, arguments):
        notifications.append(name)

    thread, state = _start_execution(agent, calls, on_tool)
    unsafe_did_start = unsafe_started.wait(timeout=1)
    notifications_while_unsafe_runs = list(notifications)
    release_unsafe.set()
    results = _finish_execution(thread, state)

    assert unsafe_did_start
    assert notifications_while_unsafe_runs == ["first", "second", "unsafe"]
    assert notifications == ["first", "second", "unsafe", "after"]
    assert results == ["first", "second", "unsafe", "after"]


def test_keyboard_interrupt_answers_pending_tool_calls():
    def interrupt():
        raise KeyboardInterrupt

    calls = [
        _call("interrupt", 1),
        _call("interrupt", 2),
    ]
    llm = StubLLM([
        LLMResponse(content="", tool_calls=calls),
    ])
    agent = Agent(
        llm=llm,
        tools=[FakeTool("interrupt", False, interrupt)],
    )

    with pytest.raises(KeyboardInterrupt):
        agent.chat("run tools")

    tool_messages = [
        message for message in agent.messages
        if message.get("role") == "tool"
    ]
    assert tool_messages == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "[interrupted]",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "[interrupted]",
        },
    ]
