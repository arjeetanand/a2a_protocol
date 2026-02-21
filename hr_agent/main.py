# hr_agent/main.py
import asyncio
from dotenv import load_dotenv
from collections import defaultdict

from agent_framework import tool
from agent_framework.ollama import OllamaChatClient
from timesheet_data import TIMESHEET_DB, get_current_week_range

load_dotenv()


@tool(approval_mode="never_require")
def get_timesheet_status(employee_name: str) -> str:
    """
    Check if an employee has filled their timesheet this week.
    Returns total hours worked and project IDs they logged time against.

    Args:
        employee_name: Full or partial name of the employee (case-insensitive)
    """
    name = employee_name.strip().lower()
    week_start, week_end = get_current_week_range()

    # Employee not found
    if name not in TIMESHEET_DB:
        return (
            f"No employee named '{employee_name}' found in the system. "
            f"Available employees: {', '.join(TIMESHEET_DB.keys())}."
        )

    entries = TIMESHEET_DB[name]

    # No entries this week
    if not entries:
        return (
            f"Employee '{employee_name}' has NOT filled their timesheet "
            f"for the current week ({week_start} to {week_end})."
        )

    # Aggregate hours per project
    project_hours: dict[str, float] = defaultdict(float)
    for entry in entries:
        project_hours[entry["project_id"]] += entry["hours"]

    total_hours = sum(project_hours.values())
    days_logged = len(entries)

    # Build a readable breakdown
    project_breakdown = ", ".join(
        [f"{pid} ({hrs} hrs)" for pid, hrs in project_hours.items()]
    )

    return (
        f"Employee '{employee_name}' HAS filled their timesheet this week "
        f"({week_start} to {week_end}).\n"
        f"- Total hours logged: {total_hours} hrs across {days_logged} day(s)\n"
        f"- Project breakdown: {project_breakdown}"
    )


@tool(approval_mode="never_require")
def list_all_employees() -> str:
    """
    List all employees registered in the timesheet system.
    Use this if the HR manager wants to see all employee names.
    """
    names = list(TIMESHEET_DB.keys())
    return f"Registered employees: {', '.join(names)}."


agent = OllamaChatClient().as_agent(
        name="HRTimesheetAgent",
        instructions="""
            You are an HR assistant that helps managers track employee timesheet submissions.
            When asked about an employee:
            1. ALWAYS use the get_timesheet_status tool with their name.
            2. Report clearly whether they have or haven't filled it.
            3. If filled, mention total hours and which project IDs.
            4. Be professional and concise.
            If asked to list all employees, use list_all_employees tool.
        """,
        tools=[get_timesheet_status, list_all_employees],
    )

async def main():
    print("=== HR Timesheet Agent ===\n")

    agent = OllamaChatClient().as_agent(
        name="HRTimesheetAgent",
        instructions="""
            You are an HR assistant that helps managers track employee timesheet submissions.
            When asked about an employee:
            1. ALWAYS use the get_timesheet_status tool with their name.
            2. Report clearly whether they have or haven't filled it.
            3. If filled, mention total hours and which project IDs.
            4. Be professional and concise.
            If asked to list all employees, use list_all_employees tool.
        """,
        tools=[get_timesheet_status, list_all_employees],
    )

    # --- Test queries ---
    queries = [
        "Has Arjeet filled his timesheet this week?",
        "Check timesheet status for Priya.",
        "What about Rahul? How many hours and which project?",
        "List all employees in the system.",
    ]

    for query in queries:
        print(f"HR Manager: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}\n{'-'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
