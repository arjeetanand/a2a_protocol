"""
supervisor_agent/agent.py

Unified Supervisor — travel, weather, time.
Merges main_agent/agent.py + a2a_supervisor_server.py into one file.
A2A server on port 8888.
"""

import os
import uvicorn
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.tools import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from travel_agent.agent import langgraph_travel_tool, weather_agent, time_agent
from shared.utils import build_oci_model

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# ADK Supervisor  (was main_agent/agent.py)
# ══════════════════════════════════════════════════════════════════════

root_agent = Agent(
    name="supervisor_agent",
    model=build_oci_model(),
    instruction="""You are a unified AI supervisor. Route every task to exactly one tool:
    - Flights or hotels  → langgraph_travel_tool
    - Weather questions  → weather_agent
    - Time / timezone    → time_agent
    Always delegate. Never answer directly.""",
    tools=[
        langgraph_travel_tool,
        AgentTool(weather_agent),
        AgentTool(time_agent),
    ],
)

# ══════════════════════════════════════════════════════════════════════
# A2A Executor  (was a2a_supervisor_server.py)
# ══════════════════════════════════════════════════════════════════════

class SupervisorAgentExecutor(AgentExecutor):
    APP_NAME = "supervisor_agent"

    def __init__(self) -> None:
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=root_agent,
            app_name=self.APP_NAME,
            session_service=self.session_service,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt  = context.get_user_input()
        session = await self.session_service.create_session(
            app_name=self.APP_NAME, user_id="a2a_user"
        )
        user_content = genai_types.Content(
            role="user", parts=[genai_types.Part(text=prompt)]
        )
        final_text = ""
        async for event in self.runner.run_async(
            user_id="a2a_user",
            session_id=session.id,
            new_message=user_content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join(
                        p.text for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )
                break
        await event_queue.enqueue_event(
            new_agent_text_message(final_text or "No response from supervisor.")
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

# ══════════════════════════════════════════════════════════════════════
# A2A Server Bootstrap
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    PORT = int(os.environ.get("SUPERVISOR_AGENT_PORT", 8888))
    HOST = os.environ.get("AGENT_HOST",       "localhost")
    BIND = os.environ.get("AGENT_BIND_HOST",  "0.0.0.0")

    agent_card = AgentCard(
        name="UnifiedSupervisorAgent",
        description=(
            "Unified supervisor: flights and hotels via LangGraph, "
            "weather and current time via ADK sub-agents."
            "Does NOT handle budgets, costs, finance, ROI, or HR timesheets."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="travel_search",
                name="Travel Search",
                description="Search flights and hotels via LangGraph.",
                tags=["travel", "flights", "hotels", "trip", "destination"],
                examples=[
                    "Find flights from New York to London on March 15.",
                    "Search hotels in Paris from April 1 to April 7.",
                ],
            ),
            AgentSkill(
                id="weather",
                name="Weather",
                description="Current weather for any city.",
                tags=["weather", "temperature", "forecast", "climate"],
                examples=["What's the weather in New York?"],
            ),
            AgentSkill(
                id="time",
                name="Current Time",
                description="Current local time for any city.",
                tags=["time", "timezone", "clock", "current time"],
                examples=["What time is it in New York?"],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=SupervisorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    print(f"🚀 Supervisor Agent → http://{HOST}:{PORT}")
    uvicorn.run(server.build(), host=BIND, port=PORT)

if __name__ == "__main__":
    main()
