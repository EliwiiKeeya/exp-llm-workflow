## 快速开始
### MVP组件
1. Model: 模型 定义LLM的调用方式
2. Agent：智能体 集成Model, Tool, Memory等
3. Tool: 工具

### ChatModel
LangChain为多个供应商提供了客制化的ChatModel, 其中包括OpenAI. 考虑目前大部分大模型供应商均支持OpenAI接口, 推测`ChatOpenAI`可以用于调用各种其他支持OpenAI接口的模型.
经调查, [官方文档](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI)确实指出`ChatOpenAI`指向OpenAI官方API, 警告不推荐用于处理不标准的响应.
最终接入MaaS运行成功.
MaaS API给出了详细的速率参数限制, 需要考虑接口调用限速, 优先查找是否具有官方提供的接口.

### 使用 RateLimiter 初始化 ChatModel:
`RateLimiter` 能够限制Model调用LLM API的速率
官方提供了一个抽象类`BaseRateLimiter`. 规定同步和异步需要分别设计方法
同时提供了一个类`InMemoryRateLimiter`. 基于令牌桶设计的速率限制器，提供了一个确保线程安全的示例, 仅限于内存, 不支持多进程.

**在`InMemoryRateLimiter`中注意到的问题**:

1. 没有协程锁, 经测试发现使用`ansycio.gather()`进行多协程并发时限速器无法正常工作
2. 漏桶的策略简单, 不适配复杂策略
3. 不支持基于token的限额

### Experiment
参考`InMemoryRateLimiter`实现, 继承`BaseRateLimiter`开发了`CustomRateLimiter`以自定义复杂策略.
1. 在异步方法`acquire`中加入协程锁, 保证协程安全
2. 定义并使用元类`CustomRateLimiterMeta`, 单例模式实现`CustomRateLimiter`, 确保使用相同限速器的多个实例不产生冲突
3. 采用[双速双桶算法](https://support.huawei.com/enterprise/zh/doc/EDOC1100033973/4e5900ad), 实现RPS与RPM双策略限速
4. 继承`ChatOpenAI`, 根据调用的不同模型派生对应的类, 并配置对应的限制器策略, 便于直接创建实例

最终体验了LangSmith框架 创建了体验实例 存储并查看了调用日志及数据
