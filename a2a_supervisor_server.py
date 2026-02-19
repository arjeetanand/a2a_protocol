"""
a2a_supervisor_server.py
Wraps the unified supervisor agent (AutoGen + LangGraph + ADK sub-agents)
into an A2A-compliant server.

Run with:
    python a2a_supervisor_server.py
"""

import os

import uvicorn
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.utils import new_agent_text_message

# ── Import your existing supervisor ──────────────────────────────────────────
# Adjust the import path to wherever main_agent/agent.py lives relative to
# this file.  If you run from the project root (adk/), use:
from main_agent.agent import root_agent          # ADK Agent object


# ── A2A Executor ─────────────────────────────────────────────────────────────
class SupervisorAgentExecutor(AgentExecutor):
    """
    Bridges the ADK root_agent into the A2A execution model.

    ADK agents are run via a Runner + SessionService; we create a fresh
    session per request so conversations don't bleed between calls.
    """

    APP_NAME = "a2a_supervisor"

    def __init__(self) -> None:
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=root_agent,
            app_name=self.APP_NAME,
            session_service=self.session_service,
        )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        from google.genai import types as genai_types   # ADK uses google-genai types

        prompt = context.get_user_input()

        # Create a unique session for this request
        session = await self.session_service.create_session(
            app_name=self.APP_NAME,
            user_id="a2a_user",
        )

        # Build the ADK content object
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )

        # Collect the final text reply from the agent's event stream
        final_text = ""
        async for event in self.runner.run_async(
            user_id="a2a_user",
            session_id=session.id,
            new_message=user_content,
        ):
            # ADK fires many event types; we only care about the final reply
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join(
                        part.text
                        for part in event.content.parts
                        if hasattr(part, "text") and part.text
                    )
                break

        if not final_text:
            final_text = "The supervisor agent did not return a response."

        await event_queue.enqueue_event(new_agent_text_message(final_text))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        pass   # cancellation not implemented


# ── Server entry-point ────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()

    PORT = int(os.environ.get("SUPERVISOR_AGENT_PORT", 8888))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    # ── Declare the skills this server advertises ─────────────────────────────
    skills = [
        # AgentSkill(
        #     id="budget_analysis",
        #     name="Budget Analysis",
        #     description=(
        #         "Itemizes costs, flags expensive items, and suggests cheaper "
        #         "alternatives using an AutoGen multi-agent pipeline."
        #     ),
        #     tags=["budget", "cost", "finance"],
        #     examples=[
        #         "I need 3 developers at $150/hr for 2 weeks and an AWS server at $500/month.",
        #         "Is $10,000 a reasonable budget for a small office renovation?",
        #     ],
        # ),
        AgentSkill(
            id="travel_search",
            name="Travel Search",
            description=(
                "Searches for flights and hotels using a LangGraph agent."
            ),
            tags=["travel", "flights", "hotels"],
            examples=[
                "Find flights from New York to London on March 15.",
                "Search hotels in Paris from April 1 to April 7.",
            ],
        ),
        AgentSkill(
            id="weather",
            name="Weather",
            description="Returns current weather for a given city.",
            tags=["weather"],
            examples=["What's the weather in New York?"],
        ),
        AgentSkill(
            id="time",
            name="Current Time",
            description="Returns the current local time for a given city.",
            tags=["time", "timezone"],
            examples=["What time is it in New York?"],
        ),
    ]

    agent_card = AgentCard(
        name="UnifiedSupervisorAgent",
        description=(
            "A unified AI supervisor that routes tasks to specialised agents: "
            "AutoGen (budget), LangGraph (travel), and ADK (weather/time)."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SupervisorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"🚀  Unified Supervisor A2A server running on http://{HOST}:{PORT}")
    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()