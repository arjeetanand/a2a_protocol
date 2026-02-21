"""
finance_agent/agent.py

Finance Agent — handles budget analysis, cost estimation, ROI, expenses.
Runs as an A2A server on port 8890.
"""

import os
from dotenv import load_dotenv
import uvicorn

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from shared.utils import build_oci_model

load_dotenv()

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


def analyze_budget(items: str) -> dict:
    """
    Parse a plain-text budget breakdown and return a structured analysis.

    Args:
        items: A comma-separated list of 'item:cost' pairs.
               Example: "developer:15000, AWS:500, design:3000"

    Returns:
        dict with total, item breakdown, and largest expense.
    """
    try:
        parsed = {}
        for entry in items.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            name, cost_str = entry.rsplit(":", 1)
            parsed[name.strip()] = float(cost_str.strip())

        if not parsed:
            return {"status": "error", "error_message": "No valid 'item:cost' pairs found."}

        total = sum(parsed.values())
        largest = max(parsed, key=parsed.get)
        return {
            "status": "success",
            "items": parsed,
            "total_cost": total,
            "largest_expense": f"{largest} (${parsed[largest]:,.2f})",
            "suggestion": f"Consider reducing {largest} to lower overall spend.",
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


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
- Analyze a budget breakdown using analyze_budget(items)
- Estimate project costs using estimate_project_cost(...)

Rules:
- Always use tools for calculations — never guess numbers.
- Present results in a clean, structured way with currency formatting.
- Highlight the largest cost driver and suggest one actionable saving.
- Keep answers concise but complete.""",
    tools=[calculate_roi, analyze_budget, estimate_project_cost],
)


# ── A2A Executor ──────────────────────────────────────────────────────────────

# class FinanceAgentExecutor(AgentExecutor):
#     APP_NAME = "finance_agent"

#     def __init__(self) -> None:
#         self.session_service = InMemorySessionService()
#         self.runner = Runner(
#             agent=finance_agent,
#             app_name=self.APP_NAME,
#             session_service=self.session_service,
#         )

#     async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
#         from google.genai import types as genai_types

#         prompt = context.get_user_input()
#         session = await self.session_service.create_session(
#             app_name=self.APP_NAME,
#             user_id="a2a_user",
#         )
#         user_content = genai_types.Content(
#             role="user",
#             parts=[genai_types.Part(text=prompt)],
#         )

#         final_text = ""
#         async for event in self.runner.run_async(
#             user_id="a2a_user",
#             session_id=session.id,
#             new_message=user_content,
#         ):
#             if event.is_final_response():
#                 if event.content and event.content.parts:
#                     final_text = "".join(
#                         part.text
#                         for part in event.content.parts
#                         if hasattr(part, "text") and part.text
#                     )
#                 break

#         await event_queue.enqueue_event(
#             new_agent_text_message(final_text or "Finance agent returned no response.")
#         )

#     async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
#         pass

# # class FinanceAgentExecutor(BaseADKExecutor):
# #     APP_NAME = "finance_agent"

# #     def __init__(self):
# #         super().__init__(finance_agent)

# # ── Server entry-point ────────────────────────────────────────────────────────

# def main() -> None:
#     PORT = int(os.environ.get("FINANCE_AGENT_PORT", 8890))
#     HOST = os.environ.get("AGENT_HOST", "localhost")

#     agent_card = AgentCard(
#         name="FinanceAgent",
#         description="Handles budget analysis, ROI calculation, cost estimation, and expense tracking.",
#         url=f"http://{HOST}:{PORT}/",
#         version="1.0.0",
#         default_input_modes=["text"],
#         default_output_modes=["text"],
#         capabilities=AgentCapabilities(streaming=False),
#         skills=[
#             AgentSkill(
#                 id="roi_calculation",
#                 name="ROI Calculation",
#                 description="Calculate return on investment.",
#                 tags=["roi", "invest", "returns"],
#                 examples=["I invested $10,000 and got back $14,000. What's my ROI?"],
#             ),
#             AgentSkill(
#                 id="budget_analysis",
#                 name="Budget Analysis",
#                 description="Analyze a cost breakdown and identify savings.",
#                 tags=["budget", "cost", "expense"],
#                 examples=["My budget: developer:15000, AWS:500, design:3000"],
#             ),
#             AgentSkill(
#                 id="project_cost",
#                 name="Project Cost Estimator",
#                 description="Estimate total project cost from team size and duration.",
#                 tags=["cost", "price", "money", "finance", "revenue"],
#                 examples=["3 developers at $120/hr for 6 weeks, $400/month infra"],
#             ),
#         ],
#     )

#     request_handler = DefaultRequestHandler(
#         agent_executor=FinanceAgentExecutor(),
#         task_store=InMemoryTaskStore(),
#     )
#     server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

#     print(f"💰 Finance Agent running at http://{HOST}:{PORT}")
#     uvicorn.run(server.build(), host=HOST, port=PORT)


# if __name__ == "__main__":
#     main()


# ── A2A Bootstrap ───────────────────────────────────────────
def main() -> None:
    import uvicorn
    
    PORT = int(os.environ.get("FINANCE_AGENT_PORT", 8890))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    a2a_app = to_a2a(finance_agent, port=PORT)

    uvicorn.run(a2a_app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
