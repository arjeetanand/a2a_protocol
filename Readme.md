# 🤖 Multi-Agent A2A Learning System (Advanced Version)

> A production-style multi-framework, multi-protocol AI architecture demonstrating **A2A (Agent-to-Agent) interoperability** across:
>
> * Google ADK
> * LangGraph
> * AutoGen
> * Microsoft Agent Framework (MAF)
> * CrewAI
> * MCP (Model Context Protocol)

This system demonstrates real agent orchestration, cross-framework interoperability, shared tools, and dynamic semantic routing.

---

# 🧠 What This System Demonstrates

| Capability                  | Demonstrated Via                         |
| --------------------------- | ---------------------------------------- |
| Framework Interoperability  | ADK + LangGraph + AutoGen + MAF + CrewAI |
| Cross-Agent Communication   | A2A Protocol                             |
| Shared Tool Infrastructure  | MCP Server (TimesheetMCP)                |
| LLM-Based Semantic Routing  | Smart Gateway (LangGraph Router)         |
| Deterministic Tool Chains   | Payroll (MAF)                            |
| Single-LLM Hybrid Pattern   | HR Agent                                 |
| Multi-Agent Collaboration   | Budget Agent (AutoGen)                   |
| Policy + Reasoning Workflow | Leave Approval Agent (CrewAI)            |

---

# 🏗️ Full System Architecture

```
User
 │
 ▼
Holiday Agent (ADK)
 │
 ▼
Smart Gateway (LangGraph Router)
 │
 ├── Finance Agent (ADK)
 ├── Budget Agent (AutoGen + Ollama)
 ├── Supervisor Agent (LangGraph)
 ├── HR Agent (MCP + Ollama)
 ├── Payroll Agent (MAF + MCP + PDF)
 └── Leave Approval Agent (CrewAI)

Shared Infrastructure:
    └── Timesheet MCP Server (Port 8895)
```

---

# 🧩 Agents Overview

## 🌴 Holiday Agent (Entry Point)

**Framework:** Google ADK
**Role:** User-facing entry agent

Modes:

* Direct answers (holidays, PTO, general questions)
* Delegates domain queries via `ask_agent_gateway`

### Test Case

```
What are public holidays in India in 2025?
```

### Delegation Test

```
Generate a payslip for Arjeet.
```

---

## 🔀 Smart A2A Gateway (Port 9000)

**Framework:** LangGraph
**Role:** LLM-based semantic router

* Discovers agents dynamically via AgentCard
* Builds routing prompt
* Uses OCI model for semantic selection
* Calls matched agents in parallel

No keyword routing. Fully semantic.

---

## 💰 Finance Agent (Port 8890)

**Framework:** Google ADK
**Model:** OCI via LiteLLM

### Tools

* `calculate_roi`
* `analyze_budget`
* `estimate_project_cost`

### Test Case

```
I invested 200000 and got back 280000. What is my ROI?
```

---

## 💸 Budget Agent (Port 8892)

**Framework:** AutoGen
**Model:** Ollama llama3.2
**Pattern:** Multi-agent collaboration

Agents:

* Cost_Estimator
* Budget_Analyst

### Test Case

```
Flight 5000, Hotel 8000, Food 2000 — is this reasonable?
```

---

## 👥 HR Agent (Port 8893)

**Pattern:**

* Step 1 → Deterministic Python logic (0 LLM calls)
* Step 2 → Single Ollama synthesis call

**Data Source:** MCP Timesheet Server

### Capabilities

* Timesheet status
* Employee list
* Team summary

### Test Case

```
Has Arjeet filled his timesheet this week?
```

---

## 🔌 Timesheet MCP Server (Port 8895)

**Framework:** FastMCP
**Role:** Shared tool provider

Exposes:

* `get_employee_hours`
* `list_employees`
* `team_summary`

Used by:

* HR Agent
* Payroll Agent
* Leave Agent

Single source of truth for hours data.

### Test via MCP Inspector

```
npx @modelcontextprotocol/inspector http://localhost:8895/sse
```

---

## 💸 Payroll Agent (Port 8894)

**Framework:** Microsoft Agent Framework (MAF)
**Model:** Ollama

### Deterministic Tool Chain

1. `compute_payroll_data`

   * Fetch hours from MCP
   * Calculate gross & net pay
2. `generate_payslip_pdf`

   * Jinja2 → HTML → WeasyPrint → PDF

Always executes both tools.

### Test Case

```
Generate a payslip for Arjeet.
```

Output includes:

* Net pay
* Bonus
* PF deducted
* PDF file path

---

## 🟢 Leave Approval Agent (Port 8896)

**Framework:** CrewAI
**Pattern:** Multi-agent reasoning

Checks:

* Timesheet completion (MCP)
* Leave balance
* Project allocation
* Deliverables
* Team coverage

Final Status:

* Approved
* Rejected
* Conditional

### Test Case

```
Can Arjeet take leave next Friday?
```

---

## 🧠 Unified Supervisor Agent (Port 8888)

**Framework:** LangGraph
**Pattern:** Tool-calling loop

Tools:

* Travel search
* Weather
* Time

### Test Case

```
Find flights to London and show weather.
```

---

# 🚀 Startup Order (Important)

## 1️⃣ Start MCP Server FIRST

```
python -m mcp_server.timesheet_mcp
```

## 2️⃣ Start All Downstream Agents

```
python -m finance_agent
python -m budget_agent
python -m supervisor_agent
python -m hr_agent
python -m payroll_agent
python -m leave_agent
```

## 3️⃣ Start Gateway

```
python gateway/a2a_gateway_server.py
```

## 4️⃣ Start Holiday Agent (Entry Point)

```
python holiday_agent/agent.py
```

---

# 🧪 End-to-End Test Scenarios

### Finance

```
What is ROI if I invest 100000 and get 130000?
```

### Budget

```
Trip budget: flight 6000, hotel 10000, food 3000
```

### HR

```
Who hasn't filled their timesheet?
```

### Payroll

```
Generate payslip for Rahul.
```

### Leave

```
Approve leave for Priya.
```

### Multi-Agent Parallel Routing

```
Generate payslip for Arjeet and check his timesheet status.
```

Gateway routes to:

* PayrollAgent
* HRAgent

---

# ⚙️ Environment Setup

## Install Dependencies

```
pip install -r requirements.txt
```

## Ollama (Required for Budget, HR, Payroll, Leave)

```
ollama pull llama3.2
ollama serve
```

## OCI Required (Finance, Gateway Router, Travel)

```
OCI_REGION=
OCI_USER=
OCI_FINGERPRINT=
OCI_TENANCY=
OCI_COMPARTMENT_ID=
OCI_KEY_FILE=
```

---

# 🗂️ Project Structure

```
.
├── gateway/
├── holiday_agent/
├── finance_agent/
├── budget_agent/
├── supervisor_agent/
├── hr_agent/
├── payroll_agent/
├── leave_agent/
├── mcp_server/
├── shared/
```

---

# 🏆 Architectural Patterns Demonstrated

| Pattern                              | Agent      |
| ------------------------------------ | ---------- |
| Deterministic tool orchestration     | Payroll    |
| Hybrid deterministic + LLM synthesis | HR         |
| Multi-agent collaboration            | Budget     |
| Policy reasoning workflow            | Leave      |
| Tool-calling state graph             | Supervisor |
| Semantic LLM router                  | Gateway    |
| Shared tool infrastructure           | MCP        |

---

# 🎯 Why This Architecture Matters

* 6 different frameworks
* 1 shared agent protocol (A2A)
* 1 shared tool layer (MCP)
* Parallel routing
* Deterministic payroll engine
* PDF generation
* Policy automation
* Multi-agent collaboration

All interoperating cleanly under a unified architecture.

---

Built to learn. Designed to scale.
