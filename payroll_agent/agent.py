# payroll_agent/agent.py
"""
payroll_agent/agent.py

Payroll Agent — MAF (Microsoft Agent Framework) + MCP client.
A2A server on port 8894.

Tool chain orchestrated by MAF:
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. get_employee_hours_via_mcp(employee_name)                   │
  │        └─► calls TimesheetMCP server (port 8895) via MCP client │
  │            shared tool — same data HR agent reads               │
  │                                                                 │
  │  2. calculate_gross_pay(employee_name, total_hours)             │
  │        └─► fixed: Basic + HRA + Special Allowance              │
  │            variable: billable_hours × bonus_per_hour           │
  │            deductions: PF (12% Basic) + PT + Club               │
  │                                                                 │
  │  3. generate_payslip_pdf(employee_name, gross_data, output_dir) │
  │        └─► renders Jinja2 HTML template → weasyprint → PDF     │
  │            returns local file path                              │
  └─────────────────────────────────────────────────────────────────┘

Install:
    pip install mcp weasyprint jinja2 agent-framework

Run:
    # 1. Start MCP server first
    python -m mcp_server.timesheet_mcp

    # 2. Start Payroll Agent
    python -m payroll_agent
"""

import asyncio
import json
import os
import re
import math
import uvicorn
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# ── Microsoft Agent Framework ─────────────────────────────────────────────────
from agent_framework import tool
from agent_framework.ollama import OllamaChatClient

# ── MCP Client ────────────────────────────────────────────────────────────────
from mcp import ClientSession
from mcp.client.sse import sse_client

# ── PDF + templating ──────────────────────────────────────────────────────────
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML as WeasyprintHTML

# ── A2A ───────────────────────────────────────────────────────────────────────
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

# ── Payroll config ────────────────────────────────────────────────────────────
from payroll_agent.payroll_data import (
    EMPLOYEE_MASTER,
    PF_RATE,
    FLAT_DEDUCTIONS,
    COMPANY,
    get_pay_period,
)

load_dotenv()

os.environ["WEASYPRINT_DLL_DIRECTORIES"] = r"C:\msys64\mingw64\bin"

MCP_URL      = os.environ.get("TIMESHEET_MCP_URL", "http://localhost:8895/sse")
PDF_OUT_DIR = os.environ.get("PAYSLIP_OUTPUT_DIR", r"payslips")

# PDF_OUT_DIR  = os.environ.get("PAYSLIP_OUTPUT_DIR", "payroll_agent\payslips")
TEMPLATE_DIR = Path(__file__).parent  # payroll_agent/ folder


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — get_employee_hours_via_mcp
# Calls the shared TimesheetMCP server. Both HR agent and Payroll agent
# call the SAME MCP tool — single source of truth.
# ══════════════════════════════════════════════════════════════════════════════

async def _mcp_get_hours(employee_name: str) -> str:
    """Async MCP client call — connects to TimesheetMCP SSE endpoint."""
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_employee_hours",
                {"employee_name": employee_name},
            )
            # MCP returns a list of content blocks
            return result.content[0].text if result.content else "{}"

import concurrent.futures

@tool(approval_mode="never_require")
def get_employee_hours_via_mcp(employee_name: str) -> str:
    """Fetch billable hours via shared Timesheet MCP server."""

    def _run_in_thread():                       # ← isolated thread = isolated loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_mcp_get_hours(employee_name))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_in_thread).result(timeout=30)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — calculate_gross_pay
# Indian payslip structure:
#   Earnings  = Basic + HRA + Special Allowance + Billable Hours Bonus
#   Deductions = PF (12% of Basic) + Profession Tax (₹200) + Employee Club (₹50)
# ══════════════════════════════════════════════════════════════════════════════

@tool(approval_mode="never_require")
def calculate_gross_pay(employee_name: str, total_hours: float) -> str:
    """
    Compute full Indian payslip figures for an employee given their
    billable hours logged this week.

    Earnings:
      - Basic, HRA, Special Allowance (fixed, from rate card)
      - Billable Hours Bonus = total_hours × bonus_per_hour (variable)

    Deductions:
      - Provident Fund = 12% of Basic
      - Profession Tax = ₹200 flat
      - Employee Club  = ₹50 flat

    Args:
        employee_name: Employee name (must exist in master).
        total_hours:   Total billable hours logged (from get_employee_hours_via_mcp).

    Returns:
        JSON with all earnings, deductions, gross pay, net pay, and PF YTD.
    """
    name = employee_name.strip().lower()

    if name not in EMPLOYEE_MASTER:
        available = ", ".join(EMPLOYEE_MASTER.keys())
        return json.dumps({
            "status": "error",
            "error":  f"No rate card for '{employee_name}'. Known: {available}",
        })

    emp = EMPLOYEE_MASTER[name]

    # ── Fixed earnings ────────────────────────────────────────────────────────
    basic             = emp["basic"]
    hra               = emp["hra"]
    special_allowance = emp["special_allowance"]

    # ── Variable: billable hours bonus ────────────────────────────────────────
    billable_bonus = round(total_hours * emp["bonus_per_hour"])

    total_earnings = basic + hra + special_allowance + billable_bonus

    # ── Deductions ────────────────────────────────────────────────────────────
    pf_amount        = round(basic * PF_RATE)
    profession_tax   = FLAT_DEDUCTIONS["Profession Tax"]
    employee_club    = FLAT_DEDUCTIONS["Employee Club"]

    total_deductions = pf_amount + profession_tax + employee_club

    net_pay = total_earnings - total_deductions

    # ── YTD estimates (month 1 of fiscal for POC — multiply × months_worked) ──
    # date_of_joining → months worked
    try:
        doj = date(*reversed([int(x) for x in emp["date_of_joining"].split("-")]))
        today = date.today()
        months_worked = max(1, (today.year - doj.year) * 12 + (today.month - doj.month) + 1)
    except Exception:
        months_worked = 6  # fallback

    return json.dumps({
        "status":           "ok",
        "employee":         emp["full_name"],
        "emp_code":         emp["emp_code"],
        "currency":         emp["currency"],
        "total_hours":      total_hours,
        "bonus_per_hour":   emp["bonus_per_hour"],
        # Earnings
        "basic":            basic,
        "hra":              hra,
        "special_allowance":special_allowance,
        "billable_bonus":   billable_bonus,
        "total_earnings":   total_earnings,
        # Deductions
        "pf_amount":        pf_amount,
        "profession_tax":   profession_tax,
        "employee_club":    employee_club,
        "total_deductions": total_deductions,
        "net_pay":          net_pay,
        # YTD (for payslip display)
        "months_worked":    months_worked,
        "basic_ytd":        basic * months_worked,
        "hra_ytd":          hra * months_worked,
        "sa_ytd":           special_allowance * months_worked,
        "bonus_ytd":        billable_bonus,       # only this period
        "pf_ytd":           pf_amount * months_worked,
        "pt_ytd":           profession_tax * months_worked,
        "club_ytd":         employee_club * months_worked,
    })



@tool(approval_mode="never_require")
def compute_payroll_data(employee_name: str) -> str:
    """
    Deterministically compute payroll details:
      - Fetch hours via MCP
      - Calculate gross & net pay
    Returns structured payroll JSON.
    """

    try:
        # STEP 1 — Hours
        hours_json_str = get_employee_hours_via_mcp(employee_name)
        hours_data = json.loads(hours_json_str)

        if hours_data.get("status") != "ok":
            return json.dumps(hours_data)

        total_hours = hours_data["total_hours"]

        # STEP 2 — Salary Calculation
        gross_json_str = calculate_gross_pay(employee_name, total_hours)
        gross_data = json.loads(gross_json_str)

        if gross_data.get("status") != "ok":
            return json.dumps(gross_data)

        return json.dumps({
            "status": "ok",
            "employee": gross_data["employee"],
            "total_hours": total_hours,
            "bonus": gross_data["billable_bonus"],
            "pf": gross_data["pf_amount"],
            "net_pay": gross_data["net_pay"],
            "gross_data": json.loads(gross_json_str),  # ← NOT STRING
            # "gross_data": gross_json_str,
            "project_breakdown": hours_data.get("project_breakdown", {})
        })

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — generate_payslip_pdf
# Jinja2 → HTML → weasyprint → PDF file
# Called LAST by MAF after the previous two tools have run.
# ══════════════════════════════════════════════════════════════════════════════

def _number_to_words(n: int) -> str:
    """Convert integer rupee amount to English words (simplified)."""
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
            "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
            "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty",
            "Sixty", "Seventy", "Eighty", "Ninety"]

    def _two_digits(num: int) -> str:
        if num < 20:
            return ones[num]
        return tens[num // 10] + (" " + ones[num % 10] if num % 10 else "")

    def _three_digits(num: int) -> str:
        if num >= 100:
            return ones[num // 100] + " Hundred" + (" " + _two_digits(num % 100) if num % 100 else "")
        return _two_digits(num)

    if n == 0:
        return "Zero"
    result = ""
    if n >= 100_000:
        result += _three_digits(n // 100_000) + " Lakh "
        n %= 100_000
    if n >= 1_000:
        result += _three_digits(n // 1_000) + " Thousand "
        n %= 1_000
    if n:
        result += _three_digits(n)
    return result.strip()


def _fmt_inr(amount) -> str:
    """Format integer as INR with comma separators: 80,709.00"""
    return f"{int(amount):,.0f}.00"

@tool(approval_mode="never_require")
def generate_payslip_pdf(
    employee_name: str,
    gross_data: dict,
    project_breakdown: dict,
) -> str:

    print(">>> PDF TOOL EXECUTED")
    try:
        # ── Deserialize if LLM passed JSON strings ────────────────────────
        if isinstance(gross_data, str):
            gross_data = json.loads(gross_data)
        if isinstance(project_breakdown, str):
            project_breakdown = json.loads(project_breakdown) if project_breakdown else {}
        if not isinstance(project_breakdown, dict):
            project_breakdown = {}

        name = employee_name.strip().lower()
        emp  = EMPLOYEE_MASTER.get(name, {})
        pay_period = get_pay_period()

        projects_summary = (
            ", ".join(f"{pid} ({hrs} hrs)" for pid, hrs in project_breakdown.items())
            if project_breakdown else "N/A"
        )

        # ── Build template context ────────────────────────────────────────────
        ctx = {
            # Company
            "logo_text":       COMPANY["logo_text"],
            "company_name":    COMPANY["name"],
            "company_address": COMPANY["address"],
            # Pay period
            "month_label":     pay_period["month_label"],
            "days_in_month":   pay_period["days_in_month"],
            # Employee info
            "emp_code":        emp.get("emp_code",        "N/A"),
            "full_name":       emp.get("full_name",       employee_name.title()),
            "designation":     emp.get("designation",     ""),
            "department":      emp.get("department",      ""),
            "location":        emp.get("location",        ""),
            "pan_no":          emp.get("pan_no",          ""),
            "bank_account":    emp.get("bank_account",    ""),
            "uan":             emp.get("uan",             ""),
            "date_of_joining": emp.get("date_of_joining", ""),
            "date_of_birth":   emp.get("date_of_birth",  ""),
            "gender":          emp.get("gender",          ""),
            "pf_number":       emp.get("pf_number",      ""),
            # Bonus breakdown panel
            "total_hours":     gross_data.get("total_hours",     0),
            "bonus_per_hour":  _fmt_inr(gross_data.get("bonus_per_hour", 0)),
            "projects_summary":projects_summary,
            "billable_bonus":  _fmt_inr(gross_data.get("billable_bonus", 0)),
            # Earnings
            "basic":           _fmt_inr(gross_data.get("basic",            0)),
            "hra":             _fmt_inr(gross_data.get("hra",              0)),
            "special_allowance":_fmt_inr(gross_data.get("special_allowance",0)),
            "total_earnings":  _fmt_inr(gross_data.get("total_earnings",   0)),
            # Earnings YTD
            "basic_ytd":       _fmt_inr(gross_data.get("basic_ytd",        0)),
            "hra_ytd":         _fmt_inr(gross_data.get("hra_ytd",          0)),
            "sa_ytd":          _fmt_inr(gross_data.get("sa_ytd",           0)),
            "bonus_ytd":       _fmt_inr(gross_data.get("bonus_ytd",        0)),
            # Deductions
            "pf_amount":       _fmt_inr(gross_data.get("pf_amount",        0)),
            "profession_tax":  _fmt_inr(gross_data.get("profession_tax",   0)),
            "employee_club":   _fmt_inr(gross_data.get("employee_club",    0)),
            "total_deductions":_fmt_inr(gross_data.get("total_deductions", 0)),
            # Deductions YTD
            "pf_ytd":          _fmt_inr(gross_data.get("pf_ytd",          0)),
            "profession_tax_ytd": _fmt_inr(gross_data.get("pt_ytd",       0)),
            "employee_club_ytd":  _fmt_inr(gross_data.get("club_ytd",     0)),
            # Net pay
            "net_pay":         _fmt_inr(gross_data.get("net_pay",          0)),
            "net_pay_words":   f"Rupees {_number_to_words(int(gross_data.get('net_pay', 0)))} Only.",
        }

        # ── Render HTML ───────────────────────────────────────────────────────
        env       = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
        template  = env.get_template("payslip_template.html")
        html_str  = template.render(**ctx)

        # ── Write PDF ─────────────────────────────────────────────────────────
        os.makedirs(PDF_OUT_DIR, exist_ok=True)
        safe_name = re.sub(r"[^a-z0-9]", "_", name)
        month_tag = date.today().strftime("%b_%Y").lower()
        pdf_path  = os.path.join(PDF_OUT_DIR, f"payslip_{safe_name}_{month_tag}.pdf")
        import os
        print("CWD:", os.getcwd())
        print("Saving to:", pdf_path)

        WeasyprintHTML(string=html_str, base_url=str(TEMPLATE_DIR)).write_pdf(pdf_path)

        return json.dumps({
            "status":   "ok",
            "pdf_path": pdf_path,
            "message":  f"Payslip PDF generated for {emp.get('full_name', employee_name.title())}.",
        })

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# MAF Agent
# ══════════════════════════════════════════════════════════════════════════════

def build_payroll_agent():
    return OllamaChatClient().as_agent(
        name="PayrollAgent",
        instructions="""
            You are a Payroll Assistant.

            RULES:
            1. ALWAYS call compute_payroll_data(employee_name) first.
            2. ALWAYS call generate_payslip_pdf using the values returned:
            - employee_name (string)
            - gross_data    (the gross_data dict from compute_payroll_data output)
            - project_breakdown (the project_breakdown dict from compute_payroll_data output)
            Do NOT skip this step. Do NOT wait to be asked.
            3. After both tools complete, summarize: net pay, hours, bonus, PF deducted,
            and tell the user the PDF file path.

            Never fabricate numbers. Always execute both tools.
            """,

        tools=[
            compute_payroll_data,
            generate_payslip_pdf,
        ],
    )
# ══════════════════════════════════════════════════════════════════════════════
# A2A Executor
# ══════════════════════════════════════════════════════════════════════════════

class PayrollAgentExecutor(AgentExecutor):
    """
    Thin A2A bridge — MAF does all orchestration internally.
    The executor just invokes agent.run() and pushes the result.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        try:
            agent  = build_payroll_agent()
            result = await agent.run(query)         # returns AgentRunResponse
            text   = result.text                    # ← .text extracts final string
            await event_queue.enqueue_event(new_agent_text_message(text or "No response."))
        except Exception as e:
            await event_queue.enqueue_event(
                new_agent_text_message(f"❌ Payroll Agent error: {str(e)}")
            )


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass



# ══════════════════════════════════════════════════════════════════════════════
# A2A Server Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    PORT = int(os.environ.get("PAYROLL_AGENT_PORT", 8894))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    agent_card = AgentCard(
        name="PayrollAgent",
        description=(
            "Indian Payroll Calculator. Generates PDF payslips and computes salary "
            "figures: gross pay (Basic + HRA + Special Allowance + Billable Hours Bonus), "
            "statutory deductions (PF 12%, Profession Tax, Employee Club), and net take-home. "
            "NOT for timesheet submission status — only for pay and salary calculations."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="payslip_generation",
                name="Payslip Generation",
                description=(
                    "Generates a PDF payslip for a named employee. "
                    "Computes gross pay from billable hours and fixed CTC components, "
                    "deducts PF and taxes, and returns net salary with an itemised breakdown."
                ),
                tags=[
                    "payslip", "generate payslip", "pay slip", "salary slip",
                    "salary", "net pay", "net salary", "gross pay", "take home",
                    "CTC", "compensation", "wages", "earnings",
                    "PF", "provident fund", "profession tax",
                    "billable hours bonus", "bonus calculation",
                    "how much does X earn", "how much will X take home",
                    "calculate salary", "calculate pay", "payroll",
                ],
                examples=[
                    "Generate a payslip for Arjeet.",
                    "What is Rahul's net pay this week?",
                    "How much does Arjeet take home after PF deductions?",
                    "Calculate Priya's salary for this month.",
                    "Show me Arjeet's payslip with bonus breakdown.",
                    "What will Rahul's gross pay be?",
                ],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=PayrollAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    print(f"💸 Payroll Agent (MAF + MCP) → http://{HOST}:{PORT}")
    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()