# hr_agent/timesheet_data.py
from datetime import date, timedelta

# Simulated weekly timesheet records.
# Key: employee_name (lowercase)
# Value: list of daily entry dicts for the current week
TIMESHEET_DB: dict[str, list[dict]] = {
    "arjeet": [
        {"date": "2026-02-16", "project_id": "PROJ-101", "hours": 8.0},
        {"date": "2026-02-17", "project_id": "PROJ-101", "hours": 7.5},
        {"date": "2026-02-18", "project_id": "PROJ-202", "hours": 6.0},
        {"date": "2026-02-19", "project_id": "PROJ-202", "hours": 8.0},
    ],
    "rahul": [
        {"date": "2026-02-16", "project_id": "PROJ-303", "hours": 8.0},
    ],
    "priya": [],  # registered but has NOT filled this week
}


def get_current_week_range() -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=today.weekday())   # Monday
    end   = start + timedelta(days=6)                 # Sunday
    return start.isoformat(), end.isoformat()
