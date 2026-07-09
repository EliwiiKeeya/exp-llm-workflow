## RAG
### Embedding
1. Embedding管理不参与对话状态如Checkpointer
2. 多轮对话可能需要相同的信息 无需重新检索 因为对话History中包含了过去返回的信息
3. 多轮对话可能会检索同一个文档 若想避免重新向量化 需要向量库 并根据元数据查询
4. 同一向量空间和权重情况下 Embedding映射是确定的 不会因输入chunk数量的不同而改变
5. chunk之间相互独立 互不影响

### VectorStore
1. add_document 会重复插入相同文档
2. 文档在metadata中维护唯一标识 并传入ids 能够避免重复插入
3. 可以考虑避免不同文件解析出的帧混淆影响检索结果
```python
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
```

### Note
分块结果中很多帧只有标题, 影响了召回质量: 考虑增加过滤条件  
RAG可能没有获取到有价值的信息, 未来可以考虑利用生成器尝试获取更多信息  
余弦相似度不太好用 实际场景或许需要更复杂Retriever或者关键词匹配等搜索算法  
未来可以尝试使用重排序模型解决上下文语义缺失问题的效果
