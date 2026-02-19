"""
adk/main_agent/agent.py
Unified supervisor that orchestrates:
  - AutoGen  → budget analysis
  - LangGraph → flight / hotel search
  - ADK       → weather and time sub-agents
"""

import sys
import os

# Make sibling packages (budget_agent, travel_agent) importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool

from google.adk.runners import Runner

#    (folder was renamed to avoid shadowing the autogen-agentchat package)
from budget_agent.agent import autogen_budget_tool

# Import everything needed from travel_agent
from travel_agent.agent import (
    langgraph_travel_tool,
    weather_agent,
    time_agent,
    OCI_REGION,
    OCI_USER,
    OCI_FINGERPRINT,
    OCI_TENANCY,
    OCI_COMPARTMENT_ID,
    OCI_KEY_FILE,
)


from shared.utils import require_env, build_oci_model
oci_model = build_oci_model()


# ── Unified Supervisor ────────────────────────────────────────────────────────
root_agent = Agent(
    name="supervisor_agent",
    model=oci_model,
    instruction="""You are a unified AI supervisor. Route every task to exactly one tool:

    - Budget / cost estimation  → autogen_budget_tool
      (sends task to AutoGen: Cost_Estimator → Budget_Analyst)

    - Flight or hotel search                   → langgraph_travel_tool
      (sends query to LangGraph travel graph)

    - Weather questions                        → weather_agent
    - Time / timezone questions                → time_agent

    Always delegate. Never answer directly.""",
    tools=[
        # autogen_budget_tool,        # AutoGen runtime  (sync bridge)
        langgraph_travel_tool,      # LangGraph runtime (sync bridge)
        AgentTool(weather_agent),   # ADK sub-agent
        AgentTool(time_agent),      # ADK sub-agent
    ],
)