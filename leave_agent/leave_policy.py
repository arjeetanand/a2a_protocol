# leave_agent/leave_policy.py

from hr_agent.leave_data import LEAVE_BALANCE_DB
from hr_agent.project_data import PROJECT_ALLOCATION_DB
from hr_agent.deliverables_data import DELIVERABLES_DB
from hr_agent.agent import (
    get_employee_hours_via_mcp,
    team_summary_via_mcp,
)

import json


def evaluate_leave(employee_name: str) -> dict:
    name = employee_name.strip().lower()

    result = {
        "employee": employee_name,
        "checks": {},
        "final_status": None,
    }

    # ── 1. Timesheet Check ─────────────────────────────────────────
    timesheet_data = get_employee_hours_via_mcp(name)
    if timesheet_data.get("status") != "ok":
        result["checks"]["timesheet"] = "Incomplete"
        result["final_status"] = "Rejected"
        return result

    total_hours = timesheet_data.get("total_hours", 0)
    result["checks"]["timesheet"] = "Complete"
    result["checks"]["hours_logged"] = total_hours

    if total_hours < 35:
        result["checks"]["low_hours_flag"] = True

    # ── 2. Leave Balance ───────────────────────────────────────────
    leave_data = LEAVE_BALANCE_DB.get(name)
    if not leave_data or leave_data["annual_leave_remaining"] <= 0:
        result["checks"]["leave_balance"] = "Insufficient"
        result["final_status"] = "Rejected"
        return result

    result["checks"]["leave_balance"] = leave_data["annual_leave_remaining"]

    # ── 3. Deliverables ────────────────────────────────────────────
    deliverables = DELIVERABLES_DB.get(name, {})
    if deliverables.get("high_priority_tasks", 0) > 0:
        result["checks"]["high_priority_tasks"] = True
        result["final_status"] = "Conditional"

    # ── 4. Project Allocation ──────────────────────────────────────
    allocation = PROJECT_ALLOCATION_DB.get(name, {})
    if allocation.get("critical_resource"):
        result["checks"]["critical_resource"] = True
        result["final_status"] = "Conditional"

    # ── 5. Team Coverage ───────────────────────────────────────────
    team_summary = team_summary_via_mcp()
    if team_summary.get("missing_count", 0) > 1:
        result["checks"]["team_coverage_risk"] = True
        result["final_status"] = "Conditional"

    # ── Final Status Logic ─────────────────────────────────────────
    if not result["final_status"]:
        result["final_status"] = "Approved"

    return result