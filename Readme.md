# 🤖 Multi-Agent A2A Learning System

> **A hands-on learning project** that demonstrates how to build a real multi-agent system using the **A2A (Agent-to-Agent) protocol** — combining three different AI frameworks (**Google ADK**, **AutoGen**, **LangGraph**) into one unified, production-style architecture.

---

## 🎯 Why This Project Exists

Learning A2A in isolation is straightforward. But the real challenge is:

> *"How do I connect agents built with different frameworks so they can talk to each other?"*

This project solves that. It answers:

| Question | What this project teaches |
|----------|---------------------------|
| What is A2A? | How agents expose themselves as HTTP servers with standard `AgentCard` + `AgentSkill` metadata |
| How do agents communicate? | Via the A2A client/server protocol over HTTP — framework-agnostic |
| Can I mix frameworks? | Yes — ADK, AutoGen, and LangGraph agents all coexist and interoperate |
| How do I route queries? | A central Gateway reads a JSON registry and fans out to matching agents in parallel |
| What is the entry point? | The **Holiday Agent** — it talks to users and delegates complex tasks to the gateway |

---

## 🧠 What is the A2A Protocol?

**A2A (Agent-to-Agent)** is an open protocol by Google that lets AI agents communicate with each other regardless of the framework they were built with.

Each A2A agent:
- Runs as an **HTTP server** (via FastAPI/Starlette + Uvicorn)
- Exposes a `/.well-known/agent.json` endpoint with its **AgentCard** (name, description, skills)
- Accepts tasks via `POST /` and responds with structured messages
- Is completely **framework-agnostic** — the caller doesn't need to know if it's ADK, AutoGen, or LangGraph

```
Client → POST http://agent-host:PORT/
       ← AgentCard metadata, streamed or batched text response
```

---

## 🏗️ Full System Architecture

```
User / Test Client
       │
       ▼
┌─────────────────────────────────┐
│        Holiday Agent            │  ← Entry Point (ADK + OCI)
│  - Answers holiday/leave Qs     │
│  - Delegates everything else    │
│    via ask_agent_gateway tool   │
└────────────────┬────────────────┘
                 │  A2A call to Gateway
                 ▼
┌─────────────────────────────────┐
│         A2A Gateway             │  Port: 9000
│  - Reads agent_registry.json    │
│  - Keyword matches (whole-word) │
│  - Fan-out: calls ALL matches   │
│    IN PARALLEL via asyncio      │
│  - Merges responses             │
└──┬──────────┬──────────┬────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
✈️ Travel    💰 Finance  📊 Analytics 💸 Budget
  Agent       Agent       Agent        Agent
 :8889        :8890       :8891        :8892
(LangGraph)  (ADK+OCI)  (ADK+OCI)  (AutoGen+Ollama)
```

---

## 🔧 Frameworks Used & Why

### 🟦 Google ADK (Agent Development Kit)
Used for: **Finance Agent**, **Analytics Agent**, **Holiday Agent**, **Weather/Time sub-agents**

ADK is Google's framework for building structured tool-using agents. It handles:
- Tool registration and auto-invocation
- Session management per request
- Clean integration with OCI, Gemini, and LiteLLM models

```python
# How an ADK agent is built
finance_agent = Agent(
    name="finance_agent",
    model=oci_model,         # Any LLM via LiteLLM
    instruction="...",       # System prompt
    tools=[calculate_roi, analyze_budget, estimate_project_cost],  # Python functions
)

# How it becomes an A2A server in one line
a2a_app = to_a2a(finance_agent, port=8890)
```

### 🟨 AutoGen (Microsoft)
Used for: **Budget Agent**

AutoGen is a multi-agent collaboration framework. Instead of one agent doing everything, multiple specialized agents take turns on a shared task. Great for multi-step reasoning workflows.

```python
# Two agents collaborate in turns
cost_estimator = AssistantAgent("Cost_Estimator", ...)  # Itemizes costs
budget_analyst  = AssistantAgent("Budget_Analyst", ...)  # Evaluates and flags expensive items

team = RoundRobinGroupChat(
    participants=[cost_estimator, budget_analyst],
    max_turns=2,   # Each agent speaks once, in order
)
result = await team.run(task=user_input)
```

### 🟩 LangGraph (LangChain)
Used for: **Travel Agent**

LangGraph models agent logic as a **state machine graph** — nodes are actions (call LLM, call tools), edges are transitions. Ideal for agents that loop until they decide they're done.

```
[START] → [LLM Node] ──has tool calls──→ [Tool Node] ──loops back──→ [LLM Node]
                    └──no tool calls──→ [END]
```

```python
# The travel graph: LLM calls tools, tools return results, LLM synthesizes answer
builder = StateGraph(TravelState)
builder.add_node("llm",   travel_llm_node)   # calls OCIChatModel
builder.add_node("tools", travel_tool_node)  # runs search_flights / search_hotels
builder.add_conditional_edges("llm", should_use_tools)  # loop or stop
travel_graph = builder.compile()
```

---

## 🤖 All Agents — What They Do

### 🌴 Holiday Agent *(Entry Point)*
The user-facing agent. Powered by **ADK + OCI**. Has a single special tool: `ask_agent_gateway`.

- **Answers directly**: public holidays, leave/PTO calculations, general knowledge
- **Delegates to gateway**: anything involving travel, finance, analytics, weather, or live data

```
You: "Plan a trip to Paris and estimate the cost"
→ Holiday Agent: "This needs travel + finance data"
→ Calls ask_agent_gateway("Plan a trip to Paris and estimate the cost")
→ Gateway fans out to Travel Agent AND Finance Agent in parallel
→ Holiday Agent gets merged response and presents it to you
```

### 🔀 A2A Gateway *(Router — Port 9000)*
The brain of the routing system. Reads `gateway/agent_registry.json` on **every request** (live reload — no restart needed to add agents).

**Routing logic:**
1. Lowercases the query
2. Matches whole-word keywords using `\b` regex against each agent's keyword list
3. If multiple agents match → calls them all **in parallel** using `asyncio.gather`
4. Merges responses with section headers

```json
// gateway/agent_registry.json
{
  "travel":    { "url": "http://localhost:8889", "keywords": ["flight","hotel","travel","trip","destination"] },
  "finance":   { "url": "http://localhost:8890", "keywords": ["roi","budget","cost","expense","invest"] },
  "analytics": { "url": "http://localhost:8891", "keywords": ["analytics","metrics","trend","growth","kpi"] },
  "budget":    { "url": "http://localhost:8892", "keywords": ["budget","estimate","price"] }
}
```

### ✈️ Travel Agent *(LangGraph — Port 8889)*
Searches for flights and hotels using a LangGraph state machine. Also wraps two ADK sub-agents as tools.

| Tool | Returns |
|------|---------|
| `search_flights(origin, destination, date)` | Available flights with airlines, times, prices |
| `search_hotels(city, checkin, checkout)` | Hotels with stars, price/night, rating |
| `weather_agent` (ADK sub-agent) | Current weather for a city |
| `time_agent` (ADK sub-agent) | Current local time for a city |

### 💰 Finance Agent *(ADK + OCI — Port 8890)*
Sharp financial analyst. Never guesses — always uses tools.

| Tool | What it does |
|------|--------------|
| `calculate_roi(investment, returns)` | ROI % + Profitable / Break-even / Loss-making verdict |
| `analyze_budget("item:cost, item:cost")` | Parses budget string, finds total, flags largest expense |
| `estimate_project_cost(devs, rate, weeks, infra)` | Full project cost breakdown |

**Example:** `"3 developers at $120/hr for 6 weeks with $400/month AWS"` → `$87,240.00 total`

### 📊 Analytics Agent *(ADK + OCI — Port 8891)*
Data-driven metrics expert. Uses emojis for trend direction (📈📉➡️).

| Tool | What it does |
|------|--------------|
| `calculate_growth_rate(current, previous)` | Period-over-period % change + trend direction |
| `generate_metrics_report(json_string)` | Summary: avg, top, bottom metrics from JSON |
| `analyze_trend(csv_values)` | Time-series analysis: direction, min/max, overall change |
| `build_dashboard_summary(title, metrics_json)` | Formatted text dashboard of KPIs |

### 💸 Budget Agent *(AutoGen + Ollama llama3.2 — Port 8892)*
A two-agent AutoGen team that collaborates to analyze any budget input.

```
Turn 1 → Cost_Estimator:  "Flight: $5,000 | Hotel: $8,000 | Food: $2,000 | TOTAL: $15,000"
Turn 2 → Budget_Analyst:  "Hotel = 53% of budget — excessive. Flight reasonable. Verdict: High spend."
```

No ADK, no OCI — runs **fully locally** via Ollama. Demonstrates that A2A wraps ANY Python logic.

---

## 🚀 Startup Order

> ⚠️ The gateway must start **last**. All downstream agents must be reachable before the gateway handles any request.

```
Step 1 ─ Start Downstream Agents (any order):

  Terminal 1:  python -m finance_agent       # port 8890 — ADK + OCI
  Terminal 2:  python -m analytics_agent     # port 8891 — ADK + OCI
  Terminal 3:  python budget_agent/server.py # port 8892 — AutoGen + Ollama
  Terminal 4:  python travel_agent/server.py # port 8889 — LangGraph + OCI

Step 2 ─ Start the Gateway:

  Terminal 5:  python gateway/a2a_gateway_server.py   # port 9000

Step 3 ─ Start the Entry Point Agent:

  Terminal 6:  python holiday_agent/agent.py  # talks to gateway at 9000
```

**Verify agents are up** by checking their AgentCard:
```bash
curl http://localhost:8890/.well-known/agent.json   # Finance
curl http://localhost:8891/.well-known/agent.json   # Analytics
curl http://localhost:8892/.well-known/agent.json   # Budget
curl http://localhost:9000/.well-known/agent.json   # Gateway
```

---

## 🧪 Test Use Cases

### 1. ✈️ Travel Only
```
"Find flights from Delhi to London on March 20"
→ Gateway routes to: travel
→ LangGraph runs: search_flights("Delhi", "London", "March 20")
```

### 2. 💰 Finance Only
```
"I invested ₹2,00,000 and got back ₹2,80,000. What is my ROI?"
→ Gateway routes to: finance
→ ADK calls: calculate_roi(200000, 280000) → 40% Profitable
```

### 3. 📊 Analytics Only
```
"Analyze this trend: 100, 120, 115, 140, 160, 180"
→ Gateway routes to: analytics
→ ADK calls: analyze_trend("100,120,115,140,160,180") → 📈 Upward, +80%
```

### 4. 💸 Budget Only
```
"Flight 5000, Hotel 8000, Food 2000, Visa 3000"
→ Gateway routes to: budget
→ AutoGen team: Cost_Estimator itemizes → Budget_Analyst flags Hotel at 44%
```

### 5. 🌍 Multi-Agent Fan-out
```
"Plan a trip to Paris and analyze if the budget of $15,000 is reasonable"
→ Gateway routes to: travel AND finance AND budget — ALL IN PARALLEL
→ Merged response with ✈️ Travel + 💰 Finance + 💸 Budget sections
```

### 6. 🌴 Via Holiday Agent (Full E2E)
```
"What are the public holidays in India in 2025?"
→ Holiday Agent: answers DIRECTLY (no gateway call needed)

"Find me flights to New York and calculate my ROI for last quarter"
→ Holiday Agent: calls ask_agent_gateway(query)
→ Gateway: fans out to travel + finance in parallel
→ Holiday Agent: presents merged result to user
```

---

## 🔧 Adding a New Agent

No code changes to the gateway ever needed:

1. **Build your agent** (any framework)
2. **Wrap it as an A2A server** and start it on a port
3. **Register it** in `gateway/agent_registry.json`:

```json
{
  "hr_agent": {
    "url": "http://localhost:8895",
    "keywords": ["salary", "leave", "employee", "payroll", "hr"]
  }
}
```

The gateway hot-reloads the registry on every request — zero downtime.

---

## ⚙️ Environment Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Pull Local Model (Budget Agent only)
```bash
ollama pull llama3.2
ollama serve
```

### `.env` Configuration
```env
# OCI (required for ADK, Finance, Analytics, Travel agents)
OCI_REGION=us-chicago-1
OCI_USER=ocid1.user.oc1...
OCI_FINGERPRINT=<fingerprint>
OCI_TENANCY=ocid1.tenancy.oc1...
OCI_COMPARTMENT_ID=ocid1.compartment.oc1...
OCI_KEY_FILE=~/.oci/oci_api_key.pem

# Ports (optional — these are the defaults)
FINANCE_AGENT_PORT=8890
ANALYTICS_AGENT_PORT=8891
BUDGET_AGENT_PORT=8892
GATEWAY_PORT=9000
AGENT_HOST=localhost
AGENT_TIMEOUT_SECS=600
```

---

## 🗂️ Project Structure

```
.
├── gateway/
│   ├── a2a_gateway_server.py   # Central fan-out gateway
│   └── agent_registry.json     # ← only file you ever edit to add agents
│
├── holiday_agent/
│   └── agent.py                # Entry-point agent; delegates via ask_agent_gateway
│
├── travel_agent/
│   └── agent.py                # LangGraph state machine + ADK weather/time sub-agents
│
├── finance_agent/
│   ├── agent.py                # ADK finance agent (ROI, budget, project cost)
│   └── __main__.py
│
├── analytics_agent/
│   ├── agent.py                # ADK analytics agent (growth, trends, KPIs, dashboard)
│   └── __main__.py
│
├── budget_agent/
│   ├── agent.py                # AutoGen budget team logic
│   └── server.py               # A2A server wrapper (no ADK)
│
├── main_agent/
│   └── agent.py                # Unified ADK supervisor (direct wiring, no gateway)
│
├── a2a_supervisor_server.py    # A2A wrapper for the unified supervisor
│
└── shared/
    ├── utils.py                # OCI model builder + env helpers
    ├── executor.py             # Reusable BaseADKExecutor for ADK agents
    └── auto_agent_card.py      # Auto-generates AgentCard from ADK agent tools
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|------------|------|
| Agent Protocol | [A2A](https://google.github.io/A2A/) | Inter-agent HTTP communication standard |
| ADK Framework | [Google ADK](https://google.github.io/adk-docs/) | Finance, Analytics, Holiday, Weather, Time agents |
| Multi-agent Team | [AutoGen](https://microsoft.github.io/autogen/) | Budget Agent (Cost Estimator + Budget Analyst) |
| Graph Agent | [LangGraph](https://langchain-ai.github.io/langgraph/) | Travel Agent (state machine with tools) |
| LLM Backend | [OCI Generative AI](https://www.oracle.com/ai) | Primary LLM for ADK + LangGraph agents |
| Local LLM | [Ollama (llama3.2)](https://ollama.com/) | Budget Agent — fully offline |
| Model Bridge | [LiteLLM](https://litellm.ai/) | Connects OCI models to ADK/LangGraph |
| ASGI Server | [Uvicorn](https://www.uvicorn.org/) | Runs every A2A agent as HTTP server |
| HTTP Client | [httpx](https://www.python-httpx.org/) | Gateway → agent async HTTP calls |
| Console Logs | [Rich](https://rich.readthedocs.io/) | Beautiful terminal routing tables + panels |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) | `.env` file loading |

---

## 📚 Key A2A Concepts Learned

1. **AgentCard** — the "business card" of an agent: name, URL, skills, capabilities
2. **AgentSkill** — a declared capability with id, description, tags, and examples
3. **AgentExecutor** — the class you implement to handle incoming A2A requests
4. **EventQueue** — how you stream or send back responses to the caller
5. **ClientFactory** — how one A2A agent calls another agent's server
6. **InMemoryTaskStore** — stores in-flight task state per request

---

*Built to learn. Built to extend. Zero magic — every routing decision is in plain JSON.*
