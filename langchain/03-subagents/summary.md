## Subagents
### Supervisor(Tools) 模式
参考了[官方示例](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)  
示例给出的提示词与工具模板不匹配, 进行了修改.

```Mermaid
graph TD
    SupervisorAgent --> CalendarAgent
    SupervisorAgent --> EmailAgent                    
```

### Swarm模式
参考了[Strands-Agents示例](https://strandsagents.com/docs/user-guide/concepts/multi-agent/swarm/)

```Mermaid
graph TD
    Researcher <--> Reviewer
    Researcher <--> Architect
    Reviewer <--> Architect
    Coder <--> Researcher
    Coder <--> Reviewer
    Coder <--> Architect
```

**开发中遇到的问题及解决过程:**  

 `interleave`方法报错 字典无"tool_calls"键值对  

查询API `stream_events`方法返回一个迭代器`Iter[StreamEvent]`  
最终指向数据模型`StandardStreamEvent`以及`CustomStreamEvent`  
实际对应流式传输的帧 无法定位
进入调试 分别使用Swarm和单Agent已编译的图调用`stream_events`  
发现帧数据的组织方式不同 考虑重写delta的解析方式  
在Reference检索`interleave`方法API 查询到类`GraphRunStream`  
根据说明 `interleave` 方法传入的字符串字段应该与`extensions`属性对应  
进入调试 Swarm和单Agent返回的对象确实为`GraphRunStream`  
查询二者`extensions`属性并比较 结果不同  
Swarm图返回的`GraphRunStream.extensions`确实不包含"tool_calls"  
dump一轮调用所有帧到文件 自定义了从帧中提取delta信息的逻辑

```python
{
    'type': 'event',
    'method': 'messages',
        'params': {
    'namespace': ['ResearchAgent:e0f1ad06-a56e-e41e-1ad8-a8118648f5fa'],
        'timestamp': 1783513847196,
            'data': (
                {
                    'event': 'content-block-delta',
                    'index': 0, 'delta': { 'type': 'text-delta', 'text': '我将' }
                }, 
                {
                    'thread_id': '1',
                    'ls_integration': 'langchain_chat_model',
                    'langgraph_step': 1,
                    'langgraph_node': 'model',
                    'langgraph_triggers': ('branch:to:model',),
                    'langgraph_path': ('__pregel_pull', 'model'), 
                    'langgraph_checkpoint_ns': 'ResearchAgent:e0f1ad06-a56e-e41e-1ad8-a8118648f5fa|model:8ef85277-0b9e-93d1-7332-c0e664ba5c00',
                    'lc_agent_name': 'ResearchAgent',
                    'checkpoint_ns': 'ResearchAgent:e0f1ad06-a56e-e41e-1ad8-a8118648f5fa',
                    'ls_provider': 'openai',
                    'ls_model_name': 'deepseek-v3.2',
                    'ls_model_type': 'chat',
                    'ls_temperature': None,
                    'lc_versions': {
                        'langchain-core': '1.4.8',
                        'langchain': '1.3.11',
                        'langchain-openai': '1.3.3' 
                    },
                    'run_id': '019f41b5-8e68-7550-ab35-83fe6c2cd7bc' 
                }
            )
    },
    'seq': 6
}
```

```python
for event in stream:
    method = event.get("method")
    data = event.get("params", {}).get("data", {})

    if method == "messages":
        # 文本流
        if (data[0].get("event")) == "content-block-delta":
            delta = data[0]["delta"]
            if delta["type"] == "text-delta":
                print(delta["text"], end="", flush=True)
        # 工具调用
        elif (data[0].get("event")) == "content-block-finish":
            content = data[0]["content"]
            if content["type"] == "tool_call":
                tool_name = content["name"]
                tool_args = content["args"]
                print(f"\n[Tool Call]\n{tool_name}({tool_args})")

```

得到了正确的流式响应  
判断产生这一问题的原因是 Swarm图比Agent图多封装了一级  
构建了Agent图作为Swarm的子图  
最终流式响应的帧结构不同  
