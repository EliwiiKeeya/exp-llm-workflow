## LangGraph

### Agent图示例

[官方示例]()

1. Model: 模型 定义LLM的调用方式
2. Tool: 工具
3. State: 聊天信息状态
4. LLM Node: LLM调用节点 定义调用LLM的节点 返回调用后的State
5. Tool Node: Tool调用节点 定义调用Tool的节点 返回调用后的State
6. Logic: 定义Agent路由逻辑 定义处理LLM响应的相关行为

```Mermaid
graph LR
    START --> llm_call
    llm_call --> should_continue
    should_continue --> tool_node
    should_continue --> END
    tool_node --> START

```

### 思考了一个问题: `create_agent`接口是怎样构建Agent的 与示例相同么?

审阅源代码得知, 依据传入的各种参数, 包含了更多的自动构建逻辑, 例如上游是否被其他模型或智能体调用, 下游是否调用其他模型或智能体, 是否还有其他的中间件

相同点:

- 默认entry_node -> model
- 默认model (-> tools) -> exit
- 构造节点最后编译为图(Graph)
- 注册tool为model的条件边

猜想:  
后续编排图可以直接用`create_agent`创建的agent实例
