"""
budget_agent_server.py

Standalone AutoGen Budget Agent
A2A-compliant FastAPI server on port 8892
NO ADK
"""

import os
import uvicorn
from dotenv import load_dotenv

# A2A
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from a2a.server.events import EventQueue
# AutoGen
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

load_dotenv()

# ─────────────────────────────────────────────────────────────
# AutoGen Model Client
# ─────────────────────────────────────────────────────────────

model_client = OllamaChatCompletionClient(
    model="llama3.2",
    host="http://localhost:11434",
)

# ─────────────────────────────────────────────────────────────
# AutoGen Agents
# ─────────────────────────────────────────────────────────────

cost_estimator = AssistantAgent(
    name="Cost_Estimator",
    model_client=model_client,
    system_message="""You are a Cost Estimator agent.
Extract all items and costs.
Compute TOTAL COST clearly.
Return clean itemized output only.""",
)

budget_analyst = AssistantAgent(
    name="Budget_Analyst",
    model_client=model_client,
    system_message="""You are a Budget Analyst agent.
Review the itemized list.
Flag expensive items.
Highlight largest % contributors.
Give a short financial verdict.""",
)

def build_team():
    return RoundRobinGroupChat(
        participants=[cost_estimator, budget_analyst],
        max_turns=2,   # deterministic execution
    )

async def run_team(task: str) -> str:
    team = build_team()
    result = await team.run(task=task)

    return "\n\n".join(
        f"[{msg.source}]: {msg.content}"
        for msg in result.messages
        if hasattr(msg, "content") and msg.content
    )

# ─────────────────────────────────────────────────────────────
# A2A Executor
# ─────────────────────────────────────────────────────────────

class BudgetAgentExecutor(AgentExecutor):
    APP_NAME = "budget_agent"

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_input = context.get_user_input()

        try:
            report = await run_team(user_input)

            await event_queue.enqueue_event(
                new_agent_text_message(report)
            )

        except Exception as e:
            await event_queue.enqueue_event(
                new_agent_text_message(f"❌ Error: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        await event_queue.enqueue_event(
            new_agent_text_message("Budget request cancelled.")
        )

# ─────────────────────────────────────────────────────────────
# Server Bootstrap
# ─────────────────────────────────────────────────────────────

def main():
    PORT = int(os.environ.get("BUDGET_AGENT_PORT", 8892))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    agent_card = AgentCard(
        name="BudgetAgent",
        description="AutoGen multi-agent budget estimation and evaluation service.",
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="budget_analysis",
                name="Budget Analysis",
                description="Extracts costs and evaluates financial reasonableness.",
                tags=["budget", "cost", "finance"],
                examples=[
                    "Flight 5000, Hotel 8000, Food 2000"
                ],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=BudgetAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    print(f"💰 Budget Agent running at http://{HOST}:{PORT}")
    uvicorn.run(app.build(), host=HOST, port=PORT)

if __name__ == "__main__":
    main()
