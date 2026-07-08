import dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from model import CustomChatDeepSeekV3

dotenv.load_dotenv()

model = CustomChatDeepSeekV3()

CODER_AGENT_PROMPT = (
    "你是一个代码助手。"
    "根据架构设计实现代码。"
)

agent = create_agent(
    model,
    name="CodeAgent",
    system_prompt=CODER_AGENT_PROMPT,
)


def build_agent_with_tools(tools):
    return create_agent(
        model,
        name="CodeAgent",
        system_prompt=CODER_AGENT_PROMPT,
        tools=tools
    )


if __name__ == "__main__":
    query = """请根据以下架构设计实现一个简单的Hello World应用：
架构设计：
1. 使用Python编程语言。
2. 创建一个名为hello_world.py的文件。
3. 在hello_world.py中定义一个函数hello()，该函数打印"Hello, World!"。
4. 在hello_world.py的主程序中调用hello()函数。
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
