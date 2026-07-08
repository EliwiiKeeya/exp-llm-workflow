import dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from model import CustomChatDeepSeekV3

dotenv.load_dotenv()

model = CustomChatDeepSeekV3()

ARCHITECT_AGENT_PROMPT = (
    "你是一个架构师。"
    "根据研究结果设计系统架构。"
)

agent = create_agent(
    model,
    name="ArchitectAgent",
    system_prompt=ARCHITECT_AGENT_PROMPT,
)

def build_agent_with_tools(tools):
    return create_agent(
        model,
        name="ArchitectAgent",
        system_prompt=ARCHITECT_AGENT_PROMPT,
        tools=tools
    )

if __name__ == "__main__":
    query = """
请根据以下研究结果设计一个简单的Hello World应用的系统架构：
研究结果：
1. 需要在终端显示Hello, World!消息。
2. 应用程序需要有一个主入口点来启动。
3. 需要一个函数来处理Hello, World!消息的生成和显示。
    """

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": query}]},
        version="v3",
    )
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name}({item.input})")
