import re
import logging
from typing import Dict, Any, List, Optional

import dotenv
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from model import CustomBGEM3Embeddings

# 设置日志
logger = logging.getLogger(__name__)
dotenv.load_dotenv()  # 加载 .env 文件中的环境变量


class RegexTextSplitter(TextSplitter):
    def __init__(
        self,
        pattern: str,
        filter: Optional[str] = None,
        length_floor: Optional[int] = None,
        keep_separator: bool = False
    ):
        super().__init__()
        self.pattern = pattern
        self.filter = filter
        self.length_floor = length_floor
        self.keep_separator = keep_separator

    def split_text(self, text: str):
        # 用正则切分
        parts = re.split(self.pattern, text)

        filtered = []
        for p in parts:
            chunk = p.strip()
            if not chunk:
                continue
            if self.filter and re.match(self.filter, chunk):
                continue
            if self.length_floor and len(chunk) < self.length_floor:
                continue
            filtered.append(chunk)

        logger.info(f"切分得到 {len(filtered)} 个块")
        return filtered


class WikiRAG():
    """
    Wiki RAG 实现（仅支持简体中文）

    功能：
    1. 检索 Minecraft Wiki 中文版
    2. 返回检索结果作为 Resource
    """

    def __init__(self):
        self.text_splitter = RegexTextSplitter(
            pattern=r"(=+.*=+[\s\S]*?)(?=\n=)",
            filter=r"^=+.*=+$",
            length_floor=50
        )
        self.embeddings = CustomBGEM3Embeddings()
        self.vectorstores = {}

    # 仅简体中文
    WIKI_API = "https://zh.minecraft.wiki/api.php"

    def retrieve(self, query: str, **kwargs) -> Document:
        """
        执行 Wiki 检索

        Args:
            query: 检索查询

        Returns:
            检索结果内容
        """
        logger.info(f"WikiRAG 收到检索请求: query='{query}', kwargs={kwargs}")

        try:
            import httpx

            with httpx.Client(timeout=20.0) as client:
                metadata = None
                search_url = f"{self.WIKI_API}?action=query&list=search&srsearch={query}&format=json"
                logger.info(f"搜索 URL: {search_url}")
                search_response = client.get(search_url)
                search_data = search_response.json()
                logger.info(f"搜索响应状态码: {search_response.status_code}")

                titles = self._parse_search_results(search_data)
                logger.info(f"解析到标题: {titles}")
                if not titles:
                    return Document(page_content="未找到相关 Wiki 条目", metadata=metadata)

                title = titles[0]
                logger.info(f"获取页面: {title}")
                extract_url = (
                    f"{self.WIKI_API}?action=query&titles={title}"
                    f"&prop=extracts&explaintext=true&format=json"
                )
                logger.info(f"页面 URL: {extract_url}")
                extract_response = client.get(extract_url)
                extract_data = extract_response.json()
                logger.info(f"页面响应状态码: {extract_response.status_code}")

                page_content = self._parse_page_extract(extract_data, title)
                return Document(
                    page_content=page_content,
                    metadata={"source": title}
                )
        except ImportError:
            logger.warning("httpx 不可用")
            return Document(page_content=f"RAG 查询失败: httpx 不可用", metadata=metadata)
        except Exception as e:
            logger.error(f"Wiki API 请求失败: {e}")
            return Document(page_content=f"Wiki 查询失败: {e}", metadata=metadata)

    def _parse_search_results(self, data: Dict[str, Any]) -> List[str]:
        """解析搜索结果，返回标题列表"""
        logger.info(f"解析搜索数据: {data}")
        query = data.get("query", {})
        search = query.get("search", [])
        logger.info(f"搜索结果: {search}")
        return [item["title"] for item in search[:3]]

    def _parse_page_extract(self, data: Dict[str, Any], fallback_title: str) -> str:
        """解析页面摘要"""
        logger.info(f"解析页面数据: {data}")
        query = data.get("query", {})
        pages = query.get("pages", {})
        logger.info(f"页面: {pages}")

        for page_id, page_data in pages.items():
            if page_id != "-1":
                extract = page_data.get("extract", "")
                if extract:
                    logger.info(f"获取到摘要: {len(extract)} 字符")
                    return extract

        logger.warning(f"无法获取 '{fallback_title}' 的详细信息")
        return f"无法获取 '{fallback_title}' 的详细信息"

    def split_and_embed(self, document: Document) -> InMemoryVectorStore:
        key = document.metadata["source"]
        vectorstore = self.vectorstores.get(key)
        if not vectorstore:
            vectorstore = InMemoryVectorStore(embedding=self.embeddings)
            self.vectorstores[key] = vectorstore
            chunks = self.text_splitter.split_documents([document])
            vectorstore.add_documents(
                chunks,
                ids=[
                    f"{chunk.metadata['source']}_{i}"
                    for i, chunk
                    in enumerate(chunks)
                ]
            )
        return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wiki_rag = WikiRAG()
    query = "红石"
    document = wiki_rag.retrieve(query)
    print(f"检索结果:\n{document.page_content[:200]}...")
    embeddings = wiki_rag.split_and_embed(document)
    # print(f"分块结果: {len(chunks)} 块")

    # query = (
    #     '=== 自然生成 ===\n丛林神庙中自然生成15个红石线。\n'
    #     '林地府邸中的一种监狱房间内可以找到5个红石线。\n'
    #     '远古城市中心的地下室内可以找到多个红石线。\n'
    #     '在Java版中，试炼密室的encounter_4结构中可以找到一个红石线。'
    # )
    query = "试炼密室的encounter_4结构中可以找到一个红石线。"
    results = embeddings.similarity_search_with_score(query, k=12)
    for i, (chunk, score) in enumerate(results):
        print(
            f"结果 {i + 1}: {chunk.metadata['source']}, 内容长度: {len(chunk.page_content)}, 相似度: {score}")
        print(f"内容预览: {chunk.page_content}\n")
