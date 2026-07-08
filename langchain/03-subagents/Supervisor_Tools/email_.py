import dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from model import CustomChatDeepSeekV3

dotenv.load_dotenv()

model = CustomChatDeepSeekV3()


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """发送电子邮件。需要正确格式的地址。"""
    return f"邮件已发送至 {', '.join(to)} - 主题: {subject}"


EMAIL_AGENT_PROMPT = (
    "你是一个邮件助手。"
    "根据自然语言请求撰写专业的邮件。"
    "提取收件人信息并撰写适当的主题行和正文。"
    "使用 send_email 发送消息。"
    "在最终响应中始终确认已发送的内容。"
)

agent = create_agent(
    model,
    name="EmailAgent",
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)

if __name__ == "__main__":
    query = "发送一封邮件给设计团队: design-team@example.com，提醒他们查看新的模型。"

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
