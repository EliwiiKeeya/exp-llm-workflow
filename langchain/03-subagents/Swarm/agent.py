from langgraph_swarm import create_handoff_tool

from coder import build_agent_with_tools as build_coder_agent
from researcher import build_agent_with_tools as build_researcher_agent
from reviewer import build_agent_with_tools as build_reviewer_agent 
from architect import build_agent_with_tools as build_architect_agent

coder = build_coder_agent(
    tools=[
        create_handoff_tool(
            agent_name="ResearchAgent",
            description="移交至研究助手"
        ),
        create_handoff_tool(
            agent_name="ReviewAgent",
            description="移交至审阅助手"
        ),
        create_handoff_tool(
            agent_name="ArchitectAgent",
            description="移交至架构助手"
        )
    ]
)

researcher = build_researcher_agent(
    tools=[
        create_handoff_tool(
            agent_name="CodeAgent",
            description="移交至代码助手"
        ),
        create_handoff_tool(
            agent_name="ReviewAgent",
            description="移交至审阅助手"
        ),
        create_handoff_tool(
            agent_name="ArchitectAgent",
            description="移交至架构助手"
        )
    ]
)

reviewer = build_reviewer_agent(
    tools=[
        create_handoff_tool(
            agent_name="CodeAgent",
            description="移交至代码助手"
        ),
        create_handoff_tool(
            agent_name="ResearchAgent",
            description="移交至研究助手"
        ),
        create_handoff_tool(
            agent_name="ArchitectAgent",
            description="移交至架构助手"
        )
    ]
)

architect = build_architect_agent(
    tools=[
        create_handoff_tool(
            agent_name="CodeAgent",
            description="移交至代码助手"
        ),
        create_handoff_tool(
            agent_name="ResearchAgent",
            description="移交至研究助手"
        ),
        create_handoff_tool(
            agent_name="ReviewAgent",
            description="移交至审阅助手"
        )
    ]
)
