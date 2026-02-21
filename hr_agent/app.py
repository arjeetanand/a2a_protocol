
# hr_agent/app.py

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from agent_framework import tool
from agent_framework.ollama import OllamaChatClient
from timesheet_data import TIMESHEET_DB, get_current_week_range
from collections import defaultdict

load_dotenv()

app = FastAPI(title="HR Timesheet Agent API")

# ---------- Tools ----------

@tool(approval_mode="never_require")
def get_timesheet_status(employee_name: str) -> str:
    name = employee_name.strip().lower()
    week_start, week_end = get_current_week_range()

    if name not in TIMESHEET_DB:
        return f"No employee named '{employee_name}' found."

    entries = TIMESHEET_DB[name]
    if not entries:
        return f"{employee_name} has NOT filled their timesheet this week."

    project_hours = defaultdict(float)
    for entry in entries:
        project_hours[entry["project_id"]] += entry["hours"]

    total_hours = sum(project_hours.values())

    project_breakdown = ", ".join(
        [f"{pid} ({hrs} hrs)" for pid, hrs in project_hours.items()]
    )

    return (
        f"{employee_name} HAS filled their timesheet.\n"
        f"Total: {total_hours} hrs\n"
        f"Projects: {project_breakdown}"
    )


@tool(approval_mode="never_require")
def list_all_employees() -> str:
    return f"Employees: {', '.join(TIMESHEET_DB.keys())}"


# ---------- Agent ----------

agent = OllamaChatClient().as_agent(
    name="HRTimesheetAgent",
    instructions="""
    You are an HR assistant.
    Always use tools for checking timesheets.
    Be concise and professional.
    """,
    tools=[get_timesheet_status, list_all_employees],
)


# ---------- API Schema ----------

class Query(BaseModel):
    message: str


@app.post("/query")
async def query_agent(query: Query):
    response = await agent.run(query.message)
    return {"response": response}