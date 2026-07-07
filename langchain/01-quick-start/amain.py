# -*- encoding: utf-8 -*-
# @File			: amain.py
# @Date			: 2026/07/04 21:09:22
# @Author		: Eliwii_Keeya
import asyncio

from langchain.agents import create_agent

from model import CustomChatDeepSeekV4Flash

model = CustomChatDeepSeekV4Flash()


def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

# 运行代理
async def main():
    tasks = [
        agent.ainvoke({"messages": [{"role": "user", "content": "北京的天气怎么样"}]})
        for _ in range(5)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

results = asyncio.run(main())
print(results)
