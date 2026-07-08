import dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from model import CustomChatDeepSeekV3

dotenv.load_dotenv()

model = CustomChatDeepSeekV3()

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
) -> str:
    """创建一个日历事件。需要精确的 ISO 日期时间格式。"""
    return f"Event created: {title} from {start_time} to {end_time}"


@tool
def get_available_time_slots() -> list[str]:
    """检查给定与会者在特定日期的日历可用性."""
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "你是一个日历调度助手。"
    "将自然语言调度请求（例如 '下周二下午2点的会议'）"
    "解析为正确的 ISO 日期时间格式。"
    "在需要时使用 get_available_time_slots 检查可用性。"
    "如果没有合适的时间段，请停止并在响应中确认不可用性。"
    "使用 create_calendar_event 安排事件。"
    "在最终响应中始终确认已安排的内容。"
)

agent = create_agent(
    model,
    name="CalendarAgent",
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

if __name__ == "__main__":
    query = "安排一个团队会议，下周二下午2点，持续1小时"

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
