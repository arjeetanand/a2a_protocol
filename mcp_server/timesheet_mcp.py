# mcp_server/timesheet_mcp.py
"""
mcp_server/timesheet_mcp.py

Shared MCP (Model Context Protocol) server — port 8895.
Exposes ONE canonical tool:

    get_employee_hours(employee_name)

Why MCP here?
  - HR agent and Payroll agent both need hours data.
  - MCP makes the tool discoverable by ANY MCP-compatible client.
  - Single source of truth: change timesheet logic once, every agent sees it.

Transport: SSE  (Server-Sent Events over HTTP — works with FastMCP defaults)

Install:
    pip install mcp

Run:
    python -m mcp_server.timesheet_mcp
    # or
    python mcp_server/timesheet_mcp.py
"""

import json
from collections import defaultdict
from mcp.server.fastmcp import FastMCP
import uvicorn
# ── Re-use the canonical timesheet DB (same data HR agent reads) ─────────────
from hr_agent.timesheet_data import TIMESHEET_DB, get_current_week_range

# ── FastMCP server instance ───────────────────────────────────────────────────
mcp = FastMCP(
    name="TimesheetMCP",
    instructions=(
        "Provides real-time employee timesheet data. "
        "Call get_employee_hours to fetch hours logged, "
        "project breakdown, and week range for any employee."
    ),
)


def log_mcp_call(tool_name: str, args: dict, response: dict):
    print("\n──────────────── MCP EVENT ────────────────")
    print(f"📥 TOOL CALLED: {tool_name}")
    print(f"🔎 ARGS: {json.dumps(args, indent=2)}")
    print(f"📤 RESPONSE:")
    print(json.dumps(response, indent=2))
    print("───────────────────────────────────────────\n")

# ── MCP Tool ─────────────────────────────────────────────────────────────────

# @mcp.tool(
#     description=(
#         "Fetch total hours logged by an employee this week. "
#         "Returns total_hours, project breakdown, and week range. "
#         "Returns status='no_timesheet' if employee registered but hasn't logged. "
#         "Returns status='not_found' with available names if employee unknown."
#     )
# )
# def get_employee_hours(employee_name: str) -> str:
#     """
#     Canonical timesheet lookup — shared by HR Agent and Payroll Agent.

#     Args:
#         employee_name: Employee name (case-insensitive). E.g. "arjeet", "Rahul".

#     Returns:
#         JSON string with keys: status, employee, week, total_hours, project_breakdown.
#     """
#     name = employee_name.strip().lower()
#     week_start, week_end = get_current_week_range()

#     if name not in TIMESHEET_DB:
#         return json.dumps({
#             "status":              "not_found",
#             "employee":            employee_name,
#             "week":                f"{week_start} to {week_end}",
#             "available_employees": list(TIMESHEET_DB.keys()),
#         })

#     entries = TIMESHEET_DB[name]

#     if not entries:
#         return json.dumps({
#             "status":      "no_timesheet",
#             "employee":    employee_name,
#             "week":        f"{week_start} to {week_end}",
#             "total_hours": 0.0,
#         })

#     project_hours: dict[str, float] = defaultdict(float)
#     for entry in entries:
#         project_hours[entry["project_id"]] += entry["hours"]

#     return json.dumps({
#         "status":            "ok",
#         "employee":          employee_name,
#         "week":              f"{week_start} to {week_end}",
#         "total_hours":       sum(project_hours.values()),
#         "days_logged":       len(entries),
#         "project_breakdown": dict(project_hours),
#     })

@mcp.tool(
    description=(
        "Fetch total hours logged by an employee this week..."
    )
)
def get_employee_hours(employee_name: str) -> str:
    name = employee_name.strip().lower()
    week_start, week_end = get_current_week_range()

    args = {"employee_name": employee_name}

    if name not in TIMESHEET_DB:
        response = {
            "status": "not_found",
            "employee": employee_name,
            "week": f"{week_start} to {week_end}",
            "available_employees": list(TIMESHEET_DB.keys()),
        }
        log_mcp_call("get_employee_hours", args, response)
        return json.dumps(response)

    entries = TIMESHEET_DB[name]

    if not entries:
        response = {
            "status": "no_timesheet",
            "employee": employee_name,
            "week": f"{week_start} to {week_end}",
            "total_hours": 0.0,
        }
        log_mcp_call("get_employee_hours", args, response)
        return json.dumps(response)

    project_hours = defaultdict(float)
    for entry in entries:
        project_hours[entry["project_id"]] += entry["hours"]

    response = {
        "status": "ok",
        "employee": employee_name,
        "week": f"{week_start} to {week_end}",
        "total_hours": sum(project_hours.values()),
        "days_logged": len(entries),
        "project_breakdown": dict(project_hours),
    }

    log_mcp_call("get_employee_hours", args, response)
    return json.dumps(response)

# @mcp.tool(
#     description=(
#         "List all registered employees with their timesheet status "
#         "for the current week."
#     )
# )
# def list_employees() -> str:
#     week_start, week_end = get_current_week_range()

#     summary = {}
#     for name, entries in TIMESHEET_DB.items():
#         summary[name] = {
#             "filled": bool(entries),
#             "total_hours": sum(e["hours"] for e in entries),
#             "projects": list({e["project_id"] for e in entries}),
#         }

#     return json.dumps({
#         "status": "ok",
#         "week": f"{week_start} to {week_end}",
#         "employee_count": len(summary),
#         "employees": summary,
#     })

@mcp.tool(
    description="List all registered employees..."
)
def list_employees() -> str:
    week_start, week_end = get_current_week_range()

    summary = {}
    for name, entries in TIMESHEET_DB.items():
        summary[name] = {
            "filled": bool(entries),
            "total_hours": sum(e["hours"] for e in entries),
            "projects": list({e["project_id"] for e in entries}),
        }

    response = {
        "status": "ok",
        "week": f"{week_start} to {week_end}",
        "employee_count": len(summary),
        "employees": summary,
    }

    log_mcp_call("list_employees", {}, response)
    return json.dumps(response)

@mcp.tool(
    description=(
        "Return team-wide timesheet completion summary: "
        "how many submitted, how many missing, completion rate percentage."
    )
)
def team_summary() -> str:
    week_start, week_end = get_current_week_range()

    filled = []
    missing = []

    for name, entries in TIMESHEET_DB.items():
        if entries:
            filled.append(name)
        else:
            missing.append(name)

    total = len(TIMESHEET_DB)
    rate = round(len(filled) / total * 100, 1) if total else 0.0

    response = {
        "status": "ok",
        "week": f"{week_start} to {week_end}",
        "total_employees": total,
        "filled_count": len(filled),
        "missing_count": len(missing),
        "completion_rate_percent": rate,
        "filled": filled,
        "missing": missing,
    }

    log_mcp_call("team_summary", {}, response)
    return json.dumps(response)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # SSE transport so any HTTP MCP client can connect
    # MCP Inspector:  npx @modelcontextprotocol/inspector http://localhost:8895/sse
    print("🔌 Timesheet MCP server → http://localhost:8895/sse")
    uvicorn.run(mcp.sse_app(), host="localhost", port=8895)
    # mcp.run(transport="sse", port=8895)