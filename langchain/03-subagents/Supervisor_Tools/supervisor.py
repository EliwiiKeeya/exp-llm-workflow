import dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from model import CustomChatDeepSeekV3
from email_ import agent as email_agent
from calendar_ import agent as calendar_agent

dotenv.load_dotenv()

model = CustomChatDeepSeekV3()


@tool
def schedule_event(request: str) -> str:
    """使用自然语言安排日历事件。

    当用户想要创建、修改或检查日历约会时使用此工具。
    处理日期/时间解析、可用性检查和事件创建。

    输入：自然语言调度请求（例如，“下周二下午2点与设计团队开会”）
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """使用自然语言管理电子邮件。

    当用户想要发送通知、提醒或任何电子邮件通信时使用此工具。
    处理收件人提取、主题生成和电子邮件撰写。

    输入：自然语言电子邮件请求（例如，“发送他们关于会议的提醒”）
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def get_contact_info() -> str:
    """获取联系人信息。
    当用户想要获取某人的联系信息时使用此工具。
    """
    # Stub: In practice, this would query a contacts database or API
    return f"""
    设计团队: design-team@example.com
    """

SUPERVISOR_PROMPT = (
    "你是一个乐于助人的个人助理。"
    "你可以安排日历事件并发送电子邮件。"
    "将用户请求分解为适当的工具调用并协调结果。"
    "当请求涉及多个操作时，根据需要顺序或并行使用多个工具。"
)

agent = create_agent(
    model,
    name="SupervisorAgent",
    tools=[schedule_event, manage_email, get_contact_info],
    system_prompt=SUPERVISOR_PROMPT,
)

querys = [
    # "安排一个团队会议，明天上午9点",
    (
        "安排一个与设计团队的会议，下周二下午2点，持续1小时，"
        "并发送一封邮件提醒他们查看新的模型。"
    )
]

if __name__ == "__main__":
    for query in querys:
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
