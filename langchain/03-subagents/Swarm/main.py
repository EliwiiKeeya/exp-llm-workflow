from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_swarm import create_swarm

from agent import coder, researcher, reviewer, architect

config = RunnableConfig(configurable={"thread_id": "1"})
checkpointer = InMemorySaver()
store = InMemoryStore()
workflow = create_swarm(
    agents=[coder, researcher, reviewer, architect],
    default_active_agent="ResearchAgent",
)
app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)

query = """
我需要一个简单的命令行代办程序应用，包含基本功能说明，
请通过研究、设计、实现和审阅完成这个任务。
"""

stream = app.stream_events(
    {"messages": [{"role": "user", "content": query}]},
    config,
    version="v3",
)

for event in stream:
    method = event.get("method")
    data = event.get("params", {}).get("data", {})

    if method == "messages":
        # 文本流
        if (data[0].get("event")) == "content-block-delta":
            delta = data[0]["delta"]
            if delta["type"] == "text-delta":
                print(delta["text"], end="", flush=True)
        # 工具调用
        elif (data[0].get("event")) == "content-block-finish":
            content = data[0]["content"]
            if content["type"] == "tool_call":
                tool_name = content["name"]
                tool_args = content["args"]
                print(f"\n[Tool Call]\n{tool_name}({tool_args})")
