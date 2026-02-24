"""
supervisor_agent/agent.py

Unified Supervisor — FULL LangGraph Tool-Calling version.
A2A server on port 8888.
"""

import os
import uvicorn
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from travel_agent.agent import (
    travel_graph,
    get_weather,
    get_current_time,
    OCIChatModel,
)

load_dotenv()


# ═══════════════════════════════════════════════
# LLM (Tool Calling)
# ═══════════════════════════════════════════════

supervisor_llm = OCIChatModel()


# ═══════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════

@tool
def langgraph_travel_tool(query: str) -> str:
    """Search flights or hotels using the travel graph."""
    final_state = travel_graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        {"recursion_limit": 6},
    )

    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "No travel results."


TOOLS = [
    langgraph_travel_tool,
    get_weather,
    get_current_time,
]

TOOLS_MAP = {t.name: t for t in TOOLS}

supervisor_llm = supervisor_llm.bind_tools(TOOLS)


# ═══════════════════════════════════════════════
# LangGraph Supervisor Graph
# ═══════════════════════════════════════════════

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]


def llm_node(state: SupervisorState):
    return {"messages": [supervisor_llm.invoke(state["messages"])]}

def tool_node(state: SupervisorState):
    last = state["messages"][-1]
    results = []

    print("\n🛠 TOOL CALL DETECTED")

    for call in last.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        print(f"   → Tool Name: {tool_name}")
        print(f"   → Tool Args: {tool_args}")

        output = TOOLS_MAP[tool_name].invoke(tool_args)

        print(f"   → Tool Output: {output}\n")

        results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=call["id"]
            )
        )

    return {"messages": results}


def should_use_tools(state: SupervisorState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


builder = StateGraph(SupervisorState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("llm")
builder.add_conditional_edges("llm", should_use_tools)
builder.add_edge("tools", "llm")

supervisor_graph = builder.compile()


# ═══════════════════════════════════════════════
# A2A Executor
# ═══════════════════════════════════════════════

class SupervisorAgentExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()

        print(f"\n📩 USER QUERY: {query}")

        final_state = supervisor_graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 6},
        )

        answer = next(
            (m.content for m in reversed(final_state["messages"])
             if isinstance(m, AIMessage) and m.content),
            "No response."
        )

        print(f"\n✅ FINAL ANSWER:\n{answer}\n")

        await event_queue.enqueue_event(
            new_agent_text_message(answer)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


# ═══════════════════════════════════════════════
# A2A Server Bootstrap
# ═══════════════════════════════════════════════

def main():
    PORT = int(os.environ.get("SUPERVISOR_AGENT_PORT", 8888))
    HOST = os.environ.get("AGENT_HOST", "localhost")
    BIND = os.environ.get("AGENT_BIND_HOST", "0.0.0.0")

    agent_card = AgentCard(
        name="UnifiedSupervisorAgent",
        description=(
            "Unified supervisor: flights and hotels via LangGraph, "
            "weather and current time via LangGraph tools."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="3.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="travel",
                name="Travel Search",
                description="Flights and hotels search.",
                tags=["travel", "flights", "hotels"],
                examples=["Find flights to London."],
            ),
            AgentSkill(
                id="weather",
                name="Weather",
                description="Weather lookup.",
                tags=["weather"],
                examples=["Weather in New York"],
            ),
            AgentSkill(
                id="time",
                name="Time",
                description="Current time lookup.",
                tags=["time"],
                examples=["Current time in New York"],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=SupervisorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    print(f"🚀 Supervisor Agent → http://{HOST}:{PORT}")
    uvicorn.run(server.build(), host=BIND, port=PORT)


if __name__ == "__main__":
    main()