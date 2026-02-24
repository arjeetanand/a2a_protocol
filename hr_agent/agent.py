"""
hr_agent/agent.py

HR Timesheet Agent — A2A server on port 8893.

Architecture (mirrors budget_agent pattern):
  Step 1  Data Lookup     → pure Python    (0 LLM calls)
  Step 2  LLM Synthesis   → 1 Ollama call  (1 LLM call)
"""

import json
import os
import re
# from collections import defaultdict

import httpx
import uvicorn
from dotenv import load_dotenv

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

# from hr_agent.timesheet_data import TIMESHEET_DB, get_current_week_range

from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio
import concurrent.futures

load_dotenv()


MCP_URL = os.environ.get("TIMESHEET_MCP_URL", "http://localhost:8895/sse")


async def _mcp_get_hours(employee_name: str) -> dict:
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_employee_hours",
                {"employee_name": employee_name},
            )
            text = result.content[0].text if result.content else "{}"
            return json.loads(text)

async def _mcp_list_employees() -> dict:
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_employees", {})
            text = result.content[0].text if result.content else "{}"
            return json.loads(text)


async def _mcp_team_summary() -> dict:
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("team_summary", {})
            text = result.content[0].text if result.content else "{}"
            return json.loads(text)

       
def get_employee_hours_via_mcp(employee_name: str) -> dict:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_mcp_get_hours(employee_name))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=30)
    

def list_employees_via_mcp() -> dict:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_mcp_list_employees())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=30)


def team_summary_via_mcp() -> dict:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_mcp_team_summary())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=30)
    
# ══════════════════════════════════════════════════════════════════════
# Step 1 — Pure-Python HR Tools  (0 LLM calls)
# ══════════════════════════════════════════════════════════════════════

def get_timesheet_status(employee_name: str) -> dict:
    data = get_employee_hours_via_mcp(employee_name)

    if data.get("status") == "not_found":
        return data

    if data.get("status") == "no_timesheet":
        return {
            "status": "not_filled",
            "employee": employee_name,
            "week": data.get("week"),
        }

    if data.get("status") == "ok":
        return {
            "status": "filled",
            "employee": employee_name,
            "week": data.get("week"),
            "total_hours": data.get("total_hours"),
            "days_logged": data.get("days_logged"),
            "project_breakdown": data.get("project_breakdown"),
        }

    return {"status": "error", "raw": data}


# ══════════════════════════════════════════════════════════════════════
# Step 2 — Intent Detection + Name Extraction  (pure Python)
# ══════════════════════════════════════════════════════════════════════

_INTENTS = {
    "status":  r"\b(status|filled|submit|log|check|has|did|hours|worked)\b",
    "list":    r"\b(list|all|everyone|employees)\b",
    "summary": r"\b(summary|overview|completion|report|team|missing|who hasn)\b",
}


def detect_intent(query: str) -> set[str]:
    q = query.lower()
    return {k for k, pat in _INTENTS.items() if re.search(pat, q)}


def get_all_employee_names_from_mcp() -> list:
    data = list_employees_via_mcp()
    return list(data.get("employees", {}).keys())

def extract_employee_name(query: str) -> str | None:
    names = get_all_employee_names_from_mcp()
    q = query.lower()

    for name in names:
        if name in q:
            return name

    return None

# ══════════════════════════════════════════════════════════════════════
# Step 3 — Action Dispatcher  (pure Python, 0 LLM calls)
# ══════════════════════════════════════════════════════════════════════

def run_hr_actions(query: str) -> dict:
    """
    Detect intent → run matching tools → return structured data.
    Zero LLM calls. The LLM only sees the result of this function.
    """
    intent   = detect_intent(query)
    emp_name = extract_employee_name(query)
    computed: dict = {"query": query, "detected_intent": list(intent)}

    # Specific employee check (highest priority)
    if emp_name:
        computed["timesheet_status"] = get_timesheet_status(emp_name)

    # "status" intent but no name found → flag for clarification
    elif "status" in intent:
        computed["clarification_needed"] = True
        computed["available_employees"]  = get_all_employee_names_from_mcp()

    # List employees
    if "list" in intent:
        computed["employee_list"] = list_employees_via_mcp()

    # Team summary / completion report
    if "summary" in intent:
        computed["team_summary"] = team_summary_via_mcp()

    # Fallback: nothing matched → full team summary
    if not any(k in computed for k in
               ["timesheet_status", "employee_list", "team_summary", "clarification_needed"]):
        computed["team_summary"] = team_summary_via_mcp()

    return computed


# ══════════════════════════════════════════════════════════════════════
# Step 4 — Single Ollama Synthesis  (exactly 1 LLM call)
# ══════════════════════════════════════════════════════════════════════

_HR_PROMPT = """
    You are a professional HR assistant. Pre-computed timesheet data is below.
    Write a concise, professional HR response using ONLY these exact facts.

    Rules:
    - First sentence = direct answer to the manager's question.
    - Cite exact numbers (hours, projects, dates) from the data.
    - Structure: Direct Answer → Key Details → Action / Follow-up (if needed).
    - If a timesheet is missing, suggest a polite follow-up.
    - Stay under 120 words. Never say "based on the data provided".

    Manager Query: {query}

    Pre-Computed HR Data:
    {data}

    Write the HR response now:"""


async def synthesize_hr_response(computed: dict) -> str:
    """Exactly ONE Ollama call — converts pre-computed data into a narrative."""
    prompt = _HR_PROMPT.format(
        query=computed.get("query", ""),
        data=json.dumps(computed, indent=2),
    )
    host  = os.getenv("OLLAMA_HOST",     "http://localhost:11434")
    model = os.getenv("OLLAMA_HR_MODEL", "llama3.2")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{host}/api/chat",
            json={
                "model":    model,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ══════════════════════════════════════════════════════════════════════
# A2A Executor
# ══════════════════════════════════════════════════════════════════════

class HRAgentExecutor(AgentExecutor):
    """
    Pipeline:
      run_hr_actions()         ← pure Python, 0 LLM calls
      synthesize_hr_response() ← exactly 1 Ollama call
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        try:

            print("\n📩 HR QUERY:", query)

            computed = run_hr_actions(query)

            print("\n🧠 INTENT DETECTED:", computed.get("detected_intent"))
            if "employee_list" in computed:
                print("🔎 TOOL CALLED: list_employees (via MCP)")
            if "team_summary" in computed:
                print("🔎 TOOL CALLED: team_summary (via MCP)")
            if "timesheet_status" in computed:
                print("🔎 TOOL CALLED: get_employee_hours (via MCP)")
                
            print("\n📊 COMPUTED DATA:")
            print(json.dumps(computed, indent=2))

            narrative = await synthesize_hr_response(computed)

            print("\n🤖 LLM RAW RESPONSE:")
            print(narrative)
            print("────────────────────────────────────\n")

            await event_queue.enqueue_event(new_agent_text_message(narrative))
        except Exception as e:
            await event_queue.enqueue_event(
                new_agent_text_message(f"❌ HR Agent error: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════
# A2A Server Bootstrap
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    PORT = int(os.environ.get("HR_AGENT_PORT", 8893))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    agent_card = AgentCard(
        name="hr_agent",
        description=(
            "HR Timesheet Agent. Checks whether individual employees have submitted "
            "their weekly timesheets, lists all registered employees, and reports "
            "team-wide timesheet completion rates. "
            "NOT for salary, payslip, net pay, or payroll calculations — "
            "use PayrollAgent for those."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="timesheet_status",
                name="Timesheet Status Check",
                description=(
                    "Check whether a named employee has submitted their timesheet "
                    "this week. Returns hours logged and project breakdown if filled, "
                    "or a missing-submission flag if not."
                ),
                tags=[
                    "timesheet",
                    "timesheet submission",
                    "weekly hours",
                    "attendance log",
                    "team completion report",
                ],
                # tags=[
                #     "timesheet", "timesheet status", "submitted timesheet",
                #     "filled timesheet", "has X submitted", "did X fill",
                #     "hours logged", "attendance", "LOP", "absent",
                #     "who submitted", "who has not submitted",
                # ],
                examples=[
                    "Has Arjeet filled his timesheet this week?",
                    "Check timesheet status for Priya.",
                    "Did Rahul submit his timesheet?",
                    "How many hours did Arjeet log this week?",
                ],
            ),
            AgentSkill(
                id="employee_list",
                name="Employee List",
                description="List all registered employees with their timesheet fill status.",
                tags=["employees", "list employees", "all employees", "who is registered"],
                examples=[
                    "List all employees.",
                    "Who is in the system?",
                ],
            ),
            AgentSkill(
                id="team_summary",
                name="Team Completion Summary",
                description=(
                    "Team-wide timesheet completion report: how many submitted, "
                    "who is missing, completion rate percentage."
                ),
                tags=[
                    "team summary", "completion report", "who hasn't submitted",
                    "missing timesheets", "attendance report", "team overview",
                ],
                examples=[
                    "Who hasn't filled their timesheet this week?",
                    "Give me the team timesheet completion report.",
                    "How many employees have missing timesheets?",
                ],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=HRAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    print(f"👥 HR Agent (Single LLM Call) → http://{HOST}:{PORT}")
    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
