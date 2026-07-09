# -*- encoding: utf-8 -*-
# @File			: main.py
# @Date			: 2026/07/04 21:09:22
# @Author		: Eliwii_Keeya

from langchain.agents import create_agent

from wiki import WikiRAG
from model import CustomChatDeepSeekV3

model = CustomChatDeepSeekV3()

wiki_rag = WikiRAG()

def call_wiki_rag(keyword: str, rag_query: str) -> str:
    """从Minecraft 中文 Wiki检索信息。

    Args:
        keyword: 检索关键词
        rag_query: RAG查询语句 使用余弦相似度获取相关性最高的内容
    Returns:
        ans: 一个字符串列表，包含与查询最相关的内容
    """
    document = wiki_rag.retrieve(keyword)
    embeddings = wiki_rag.split_and_embed(document)
    chunks = embeddings.similarity_search(rag_query, k=99)
    ans = [
        chunk.page_content for chunk in chunks
    ]

    return "\n\n".join(ans)

SYSTEM_PROMPT = (
    "你是一个Minecraft游戏中的AI助手,"
    "你可以从Minecraft 中文 Wiki中检索信息并回答玩家的问题。"
    "当玩家提出问题时, 你可以使用call_wiki_rag函数来获取相关信息。"
)


agent = create_agent(
    model=model,
    tools=[call_wiki_rag],
    system_prompt=SYSTEM_PROMPT,
)

# 运行代理
response = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查询一下草方块的获取方法"}]}
)
print(response)
