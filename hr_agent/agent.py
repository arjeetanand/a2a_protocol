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
from collections import defaultdict

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

from hr_agent.timesheet_data import TIMESHEET_DB, get_current_week_range

load_dotenv()


# ══════════════════════════════════════════════════════════════════════
# Step 1 — Pure-Python HR Tools  (0 LLM calls)
# ══════════════════════════════════════════════════════════════════════

def get_timesheet_status(employee_name: str) -> dict:
    """Lookup a single employee's timesheet — pure Python, no LLM."""
    name = employee_name.strip().lower()
    week_start, week_end = get_current_week_range()

    if name not in TIMESHEET_DB:
        return {
            "status": "not_found",
            "employee": employee_name,
            "available_employees": list(TIMESHEET_DB.keys()),
        }

    entries = TIMESHEET_DB[name]
    if not entries:
        return {
            "status": "not_filled",
            "employee": employee_name,
            "week": f"{week_start} to {week_end}",
        }

    project_hours: dict[str, float] = defaultdict(float)
    for entry in entries:
        project_hours[entry["project_id"]] += entry["hours"]

    return {
        "status": "filled",
        "employee": employee_name,
        "week": f"{week_start} to {week_end}",
        "total_hours": sum(project_hours.values()),
        "days_logged": len(entries),
        "project_breakdown": dict(project_hours),
    }


def list_all_employees() -> dict:
    """Return every employee with their fill status — pure Python, no LLM."""
    week_start, week_end = get_current_week_range()
    summary = {
        name: {
            "filled": bool(entries),
            "total_hours": sum(e["hours"] for e in entries),
            "projects": list({e["project_id"] for e in entries}),
        }
        for name, entries in TIMESHEET_DB.items()
    }
    return {
        "status": "success",
        "week": f"{week_start} to {week_end}",
        "employee_count": len(summary),
        "employees": summary,
    }


def get_team_summary() -> dict:
    """Team-wide completion overview with % rate — pure Python, no LLM."""
    data = list_all_employees()
    emps = data["employees"]
    filled  = [n for n, d in emps.items() if d["filled"]]
    missing = [n for n, d in emps.items() if not d["filled"]]
    rate = round(len(filled) / len(emps) * 100, 1) if emps else 0.0
    return {
        "status": "success",
        "week": data["week"],
        "total_employees": len(emps),
        "filled_count": len(filled),
        "missing_count": len(missing),
        "completion_rate_percent": rate,
        "filled": filled,
        "missing": missing,
    }


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


def extract_employee_name(query: str) -> str | None:
    """Match any known employee name inside the query (case-insensitive)."""
    q = query.lower()
    for name in TIMESHEET_DB:
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
        computed["available_employees"]  = list(TIMESHEET_DB.keys())

    # List employees
    if "list" in intent:
        computed["employee_list"] = list_all_employees()

    # Team summary / completion report
    if "summary" in intent:
        computed["team_summary"] = get_team_summary()

    # Fallback: nothing matched → full team summary
    if not any(k in computed for k in
               ["timesheet_status", "employee_list", "team_summary", "clarification_needed"]):
        computed["team_summary"] = get_team_summary()

    return computed


# ══════════════════════════════════════════════════════════════════════
# Step 4 — Single Ollama Synthesis  (exactly 1 LLM call)
# ══════════════════════════════════════════════════════════════════════

_HR_PROMPT = """\
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
            computed  = run_hr_actions(query)                    # 0 LLM
            narrative = await synthesize_hr_response(computed)   # 1 LLM
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
            "HR Timesheet Agent. Checks individual employee timesheet status, "
            "lists all employees, and generates team completion summaries. "
            "Pure-Python data lookups + single Ollama call for narration."
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
                description="Check if a named employee has filled their timesheet this week.",
                tags=["timesheet", "employee", "hours", "status", "filled", "submitted", "log"],
                examples=[
                    "Has Arjeet filled his timesheet this week?",
                    "Check timesheet for Priya.",
                    "How many hours did Rahul log?",
                ],
            ),
            AgentSkill(
                id="employee_list",
                name="Employee List",
                description="List all registered employees with their fill status.",
                tags=["employees", "list", "all", "hr", "registered"],
                examples=["List all employees", "Who is in the system?"],
            ),
            AgentSkill(
                id="team_summary",
                name="Team Completion Summary",
                description="Team-wide timesheet completion rate with filled/missing breakdown.",
                tags=["team", "summary", "completion", "overview", "report", "missing", "attendance"],
                examples=[
                    "Who hasn't filled their timesheet?",
                    "Give me the team timesheet completion report.",
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
