# payroll_agent/payroll_data.py
"""
Indian payroll configuration.
All amounts in INR (₹).
Mirrors the Oracle payslip structure from the reference document.
"""

from datetime import date, timedelta

# ── Employee master — fixed CTC components (monthly) ─────────────────────────
# Earnings: Basic | HRA | Special Allowance | Billable Hours Bonus (variable)
# Deductions: PF (12% Basic) | Profession Tax | Employee Club

EMPLOYEE_MASTER: dict[str, dict] = {
    "arjeet": {
        "emp_code":         "1938127",
        "full_name":        "Arjeet Anand",
        "designation":      "Associate Consultant, CSS GSC TechCloud",
        "department":       "GSC-Integration - PaaS",
        "location":         "BENGALURU",
        "pan_no":           "DVLPAXXXXX",
        "bank_account":     "50100737XXXXX",
        "uan":              "10211xxxxx",
        "date_of_joining":  "06-08-2024",
        "date_of_birth":    "27-mm-2002",
        "gender":           "M",
        "pf_number":        "PYBOM00XXXX",
        # Fixed monthly components (INR)
        "basic":             30_375,
        "hra":               13_000,
        "special_allowance": 24_125,
        # Bonus per billable hour logged this week (INR/hr)
        "bonus_per_hour":    500,
        "currency":          "INR",
    },
    "rahul": {
        "emp_code":         "1938200",
        "full_name":        "Rahul Sharma",
        "designation":      "Junior Engineer, GSC TechCloud",
        "department":       "GSC-Integration - PaaS",
        "location":         "BENGALURU",
        "pan_no":           "ABCDE1234F",
        "bank_account":     "50100000000001",
        "uan":              "102117516050",
        "date_of_joining":  "01-01-2025",
        "date_of_birth":    "15-05-1998",
        "gender":           "M",
        "pf_number":        "PYBOM0023430000005",
        "basic":             20_000,
        "hra":                8_000,
        "special_allowance": 12_000,
        "bonus_per_hour":    300,
        "currency":          "INR",
    },
    "priya": {
        "emp_code":         "1938201",
        "full_name":        "Priya Nair",
        "designation":      "UI/UX Designer, GSC TechCloud",
        "department":       "GSC-Design",
        "location":         "BENGALURU",
        "pan_no":           "PQRST5678G",
        "bank_account":     "50100000000002",
        "uan":              "102117516051",
        "date_of_joining":  "15-03-2025",
        "date_of_birth":    "22-11-1999",
        "gender":           "F",
        "pf_number":        "PYBOM0023430000006",
        "basic":             22_000,
        "hra":                9_000,
        "special_allowance": 14_000,
        "bonus_per_hour":    350,
        "currency":          "INR",
    },
}

# ── Statutory deductions ──────────────────────────────────────────────────────
PF_RATE = 0.12  # 12% of Basic

FLAT_DEDUCTIONS: dict[str, int] = {
    "Profession Tax": 200,
    "Employee Club":   50,
}

# ── Company info (for payslip header) ────────────────────────────────────────
COMPANY = {
    "name":      "Kaiser Permanete",
    "address":   [
        "Tech Hub, Block B, Level 4",
        "No.1, Koramangala Inner Ring Road,",
        "Bengaluru - 560034",
    ],
    "logo_text": "KP",
}


def get_pay_period() -> dict:
    """Return current month's pay period label and day count."""
    today  = date.today()
    first  = today.replace(day=1)
    if today.month == 12:
        last = today.replace(day=31)
    else:
        last = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    return {
        "month_label":   today.strftime("%B %Y").upper(),
        "from_date":     first.strftime("%d-%m-%Y"),
        "to_date":       last.strftime("%d-%m-%Y"),
        "days_in_month": last.day,
    }