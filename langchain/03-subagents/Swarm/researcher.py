import dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from model import CustomChatDeepSeekV3

dotenv.load_dotenv()

model = CustomChatDeepSeekV3()

RESEARCH_AGENT_PROMPT = (
    "你是一个研究助手。"
    "根据需求收集和分析信息。"
    "可以和其他子代理协作以获取所需信息。"
    "包括架构助手,代码助手和审阅助手。"
)

agent = create_agent(
    model,
    name="ResearchAgent",
    system_prompt=RESEARCH_AGENT_PROMPT,
)

def build_agent_with_tools(tools):
    return create_agent(
        model,
        name="ResearchAgent",
        system_prompt=RESEARCH_AGENT_PROMPT,
        tools=tools
    )

if __name__ == "__main__":
    query = """请根据以下需求收集和分析信息：
需求：
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
