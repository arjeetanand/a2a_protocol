# leave_agent/agent.py

import os
import uvicorn
import json
from dotenv import load_dotenv

from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

from leave_agent.leave_policy import evaluate_leave

load_dotenv()


from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

# ─────────────────────────────────────────────
# CrewAI Agents
# ─────────────────────────────────────────────

hr_data_agent = Agent(
    role="HR Data Analyst",
    goal="Interpret structured leave evaluation data.",
    backstory="Expert in HR compliance and workforce analytics.",
    llm=llm,
)

policy_agent = Agent(
    role="Leave Policy Officer",
    goal="Generate final leave decision summary.",
    backstory="Responsible for enforcing corporate leave policy.",
    llm=llm,
)



def run_leave_crew(employee_name: str) -> str:
    evaluation = evaluate_leave(employee_name)

    analysis_task = Task(
        description=f"""
        The following structured leave evaluation data is provided:
        {json.dumps(evaluation, indent=2)}

        Summarize the checks clearly.
        """,
        expected_output="A clear summary of all leave validation checks.",
        agent=hr_data_agent,
    )

    decision_task = Task(
        description="""
        Based on the analysis above:
        - Clearly state Leave Status (Approved / Rejected / Conditional)
        - Provide bullet-point reasons
        - Keep it professional and management-friendly
        """,
        expected_output="Final leave decision with status and bullet-point reasons.",
        agent=policy_agent,
    )

    crew = Crew(
        agents=[hr_data_agent, policy_agent],
        tasks=[analysis_task, decision_task],
        verbose=False,
    )
    output = crew.kickoff()
    return output.raw

    # return crew.kickoff()
# ─────────────────────────────────────────────
# A2A Executor
# ─────────────────────────────────────────────

class LeaveAgentExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()

        # simple name extraction
        name = None
        for candidate in ["arjeet", "rahul", "priya"]:
            if candidate in query.lower():
                name = candidate
                break

        if not name:
            await event_queue.enqueue_event(
                new_agent_text_message("Please specify employee name.")
            )
            return

        try:
            result = run_leave_crew(name)
            await event_queue.enqueue_event(
                new_agent_text_message(result)
            )
        except Exception as e:
            await event_queue.enqueue_event(
                new_agent_text_message(f"❌ Leave Agent error: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


# ─────────────────────────────────────────────
# A2A Bootstrap
# ─────────────────────────────────────────────

def main():
    PORT = int(os.environ.get("LEAVE_AGENT_PORT", 8896))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    agent_card = AgentCard(
        name="LeaveApprovalAgent",
        description=(
            "Evaluates leave requests using timesheet data (MCP), "
            "project allocation, deliverables, leave balance, "
            "and team coverage. Returns Approved, Rejected, or Conditional."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="leave_evaluation",
                name="Leave Evaluation",
                description="Evaluate leave eligibility for an employee.",
                tags=["leave", "vacation", "pto", "approval"],
                examples=[
                    "Can Arjeet take leave next Friday?",
                    "Approve leave for Rahul.",
                ],
            )
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=LeaveAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    print(f"🟢 Leave Approval Agent → http://{HOST}:{PORT}")
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()