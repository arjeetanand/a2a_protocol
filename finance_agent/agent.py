"""
finance_agent/agent.py

Finance Agent — handles ROI and project cost estimation.
Runs as an A2A server on port 8890.
"""

import os
import uvicorn
from dotenv import load_dotenv

# ── A2A ─────────────────────────────────────────
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

# ── ADK ─────────────────────────────────────────
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from google.adk.models.lite_llm import LiteLlm

# ────────────────────────────────────────────────

load_dotenv()

# ── OCI Model Setup ────────────────────────────

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val.strip()


def build_oci_model() -> LiteLlm:
    return LiteLlm(
        model="oci/xai.grok-4",
        max_tokens=1500,
        oci_region=require_env("OCI_REGION"),
        oci_user=require_env("OCI_USER"),
        oci_fingerprint=require_env("OCI_FINGERPRINT"),
        oci_tenancy=require_env("OCI_TENANCY"),
        oci_compartment_id=require_env("OCI_COMPARTMENT_ID"),
        oci_key_file=require_env("OCI_KEY_FILE"),
        oci_serving_mode="ON_DEMAND",
    )


oci_model = build_oci_model()

# ── Finance tools ─────────────────────────────────────────────────────────────

def calculate_roi(investment: float, returns: float) -> dict:
    """
    Calculate Return on Investment (ROI).

    Args:
        investment: Total amount invested.
        returns: Total returns received.

    Returns:
        dict with roi_percent and interpretation.
    """
    if investment <= 0:
        return {"status": "error", "error_message": "Investment must be greater than 0."}
    roi = ((returns - investment) / investment) * 100
    interpretation = (
        "Profitable" if roi > 0
        else "Break-even" if roi == 0
        else "Loss-making"
    )
    return {
        "status": "success",
        "roi_percent": round(roi, 2),
        "interpretation": interpretation,
    }


def estimate_project_cost(
    num_developers: int,
    hourly_rate: float,
    duration_weeks: int,
    infra_monthly: float = 0.0,
) -> dict:
    """
    Estimate total project cost based on team size, rate, and duration.

    Args:
        num_developers: Number of developers.
        hourly_rate: Hourly rate per developer in USD.
        duration_weeks: Project duration in weeks.
        infra_monthly: Monthly infrastructure cost (optional).

    Returns:
        dict with total cost breakdown.
    """
    hours_per_week = 40
    dev_cost = num_developers * hourly_rate * hours_per_week * duration_weeks
    infra_cost = infra_monthly * (duration_weeks / 4.33)
    total = dev_cost + infra_cost
    return {
        "status": "success",
        "developer_cost": round(dev_cost, 2),
        "infrastructure_cost": round(infra_cost, 2),
        "total_estimated_cost": round(total, 2),
        "duration_weeks": duration_weeks,
        "team_size": num_developers,
    }


# ── ADK Agent ─────────────────────────────────────────────────────────────────

finance_agent = Agent(
    name="finance_agent",
    model=oci_model,
    instruction="""You are a sharp Finance Analyst assistant.

Your capabilities:
- Calculate ROI using calculate_roi(investment, returns)
- Estimate project costs using estimate_project_cost(...)

Rules:
- Always use tools for calculations — never guess numbers.
- Present results in a clean, structured way with currency formatting.
- Highlight the largest cost driver and suggest one actionable saving.
- Keep answers concise but complete.""",
    tools=[calculate_roi, estimate_project_cost],
)


# ── A2A Executor ──────────────────────────────────────────────────────────────
class FinanceAgentExecutor(AgentExecutor):

    APP_NAME = "finance_agent"

    def __init__(self):
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=finance_agent,
            app_name=self.APP_NAME,
            session_service=self.session_service,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        prompt = context.get_user_input()

        session = await self.session_service.create_session(
            app_name=self.APP_NAME,
            user_id="a2a_user",
        )

        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
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
            new_agent_text_message(
                final_text or "Finance agent returned no response."
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


# ── Server Bootstrap ───────────────────────────

def main():

    PORT = int(os.environ.get("FINANCE_AGENT_PORT", 8890))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    agent_card = AgentCard(
        name="FinanceAgent",
        description=(
            "Finance analysis agent for ROI calculation "
            "and project cost estimation."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="roi",
                name="ROI Calculation",
                description="Calculate return on investment percentage.",
                tags=["roi", "investment", "returns"],
                examples=["I invested 10000 and got back 15000. What is ROI?"],
            ),
            AgentSkill(
                id="project_cost",
                name="Project Cost Estimation",
                description="Estimate project cost based on team and duration.",
                tags=["project cost", "estimation", "developers"],
                examples=["3 developers at 120 per hour for 6 weeks with 400 infra"],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=FinanceAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    print(f"💰 Finance Agent → http://{HOST}:{PORT}")
    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()