"""
adk/budget_agent/agent.py
AutoGen multi-agent budget analysis team.
NOTE: folder is named 'budget_agent', NOT 'autogen', to avoid
      shadowing the 'autogen-agentchat' package on sys.path.
"""

import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat


# ── Shared Ollama client ──────────────────────────────────────────────────────
model_client = OllamaChatCompletionClient(
    model="llama3.2",
    host="http://localhost:11434",
)


# ── Agent 1: Cost Estimator ───────────────────────────────────────────────────
cost_estimator = AssistantAgent(
    name="Cost_Estimator",
    model_client=model_client,
    description="Extracts and itemizes all costs from user input",
    system_message="""You are a Cost Estimator agent.
Your job is to:
1. Extract every item/service mentioned by the user.
2. List each with its individual cost and quantity.
3. Compute and display the TOTAL COST clearly.
4. Format output as a clean itemized list.
Do NOT suggest alternatives. Just enumerate costs precisely.""",
)


# ── Agent 2: Budget Analyst ───────────────────────────────────────────────────
budget_analyst = AssistantAgent(
    name="Budget_Analyst",
    model_client=model_client,
    description="Evaluates the cost breakdown and flags expensive items",
    system_message="""You are a Budget Analyst agent.
You will receive an itemized cost list from the Cost_Estimator.
Your job is to:
1. Review the total cost and each line item.
2. Identify which items are overpriced or unnecessary.
3. Highlight items that consume the largest % of the budget.
4. Give a short financial verdict: is this budget reasonable or excessive?
Do NOT suggest alternatives yet. Focus on evaluation only.""",
)


# ── Agent 3: Alternatives Advisor ─────────────────────────────────────────────
# alternatives_advisor = AssistantAgent(
#     name="Alternatives_Advisor",
#     model_client=model_client,
#     description="Suggests cheaper alternatives for overpriced items",
#     system_message="""You are an Alternatives Advisor agent.
# You will receive a budget analysis from the Budget_Analyst.
# Your job is to:
# 1. For each flagged expensive or unnecessary item, suggest 1-2 cheaper alternatives.
# 2. Estimate the potential savings per item and in total.
# 3. Keep your suggestions practical and realistic.
# 4. End your response with the word TERMINATE on its own line to signal completion.""",
# )

def _build_team() -> RoundRobinGroupChat:
    return RoundRobinGroupChat(
        participants=[cost_estimator, budget_analyst],
        max_turns=2,
    )


async def _run_autogen_team(task: str) -> str:
    team = _build_team()
    result = await team.run(task=task)
    return "\n\n".join(
        f"[{msg.source}]: {msg.content}"
        for msg in result.messages
        if hasattr(msg, "content") and msg.content
    )


async def autogen_budget_tool(items_description: str) -> dict:
    try:
        result = await _run_autogen_team(items_description)
        return {"status": "success", "report": result}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

