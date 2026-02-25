"""
budget_agent_server.py

Standalone AutoGen Budget Agent
A2A-compliant FastAPI server on port 8892
NO ADK
"""

import os
import uvicorn
from dotenv import load_dotenv
import litellm

# A2A
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from a2a.server.events import EventQueue
# AutoGen
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    RequestUsage,
    ModelCapabilities,
)

from autogen_core import CancellationToken
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# AutoGen Model Client
# ─────────────────────────────────────────────────────────────

# model_client = OllamaChatCompletionClient(
#     model="llama3.2",
#     host="http://localhost:11434",
# )

import os
# from autogen_ext.models.litellm import LiteLLMChatCompletionClient


def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val.strip()


class OCIGenAIClient(ChatCompletionClient):
    """
    LiteLLM-backed OCI GenAI client for AutoGen 0.7+
    Works without autogen-ext[litellm] extra.
    """

    def __init__(self, model: str = "oci/meta.llama-3.1-70b-instruct", max_tokens: int = 1500):
        self._model = model
        self._max_tokens = max_tokens
        self._total_prompt = 0
        self._total_completion = 0

        # OCI credentials — litellm reads these automatically
        self._oci_kwargs = dict(
            oci_region=require_env("OCI_REGION"),
            oci_user=require_env("OCI_USER"),
            oci_fingerprint=require_env("OCI_FINGERPRINT"),
            oci_tenancy=require_env("OCI_TENANCY"),
            oci_compartment_id=require_env("OCI_COMPARTMENT_ID"),
            oci_key_file=require_env("OCI_KEY_FILE"),
            oci_serving_mode="ON_DEMAND",
        )

    def _format_messages(self, messages: list[LLMMessage]) -> list[dict]:
        role_map = {
            "SystemMessage": "system",
            "UserMessage": "user",
            "AssistantMessage": "assistant",
        }
        result = []
        for m in messages:
            role = role_map.get(type(m).__name__, "user")
            content = m.content if isinstance(m.content, str) else str(m.content)
            result.append({"role": role, "content": content})
        return result

    async def create(
        self,
        messages: list[LLMMessage],
        *,
        tools=None,
        json_output=None,
        extra_create_args: dict = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:

        print("\n================ LLM CALL ================")

        formatted_messages = self._format_messages(messages)

        print("\n🧠 PROMPT SENT TO OCI:")
        for m in formatted_messages:
            print(f"{m['role'].upper()}: {m['content']}")

        response = await litellm.acompletion(
            model=self._model,
            messages=self._format_messages(messages),
            max_tokens=self._max_tokens,
            **self._oci_kwargs,
            **extra_create_args,
        )

        content = response.choices[0].message.content or ""

        print("\n🤖 RAW LLM RESPONSE:")
        print(content)
        print("==========================================\n")
        
        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0

        self._total_prompt += prompt_tokens
        self._total_completion += completion_tokens

        return CreateResult(
            content=content,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=response.choices[0].finish_reason or "stop",
            cached=False,
        )

    async def create_stream(self, messages, **kwargs):
        raise NotImplementedError("Streaming not supported.")
    
    async def close(self) -> None:
        """Clean up any resources (nothing to close for litellm)."""
        pass

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_prompt,
            completion_tokens=self._total_completion,
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(self, messages, **kwargs) -> int:
        return sum(len(str(getattr(m, "content", ""))) // 4 for m in messages)

    def remaining_tokens(self, messages, **kwargs) -> int:
        return self._max_tokens - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=False,
            function_calling=False,
            json_output=False,
        )

    @property
    def model_info(self):
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        }

load_dotenv()

# Drop-in replacement for OllamaChatCompletionClient
model_client = OCIGenAIClient(
    model="oci/xai.grok-4",   # or "oci/meta.llama-3.1-70b-instruct"
    max_tokens=1500,
)
# ─────────────────────────────────────────────────────────────
# AutoGen Agents
# ─────────────────────────────────────────────────────────────

cost_estimator = AssistantAgent(
    name="Cost_Estimator",
    model_client=model_client,
    system_message="""You are a Cost Estimator agent.
Extract all items and costs.
Compute TOTAL COST clearly.
Return clean itemized output only.""",
)

budget_analyst = AssistantAgent(
    name="Budget_Analyst",
    model_client=model_client,
    system_message="""You are a Budget Analyst agent.
Review the itemized list.
Flag expensive items.
Highlight largest % contributors.
Give a short financial verdict.""",
)

def build_team():
    return RoundRobinGroupChat(
        participants=[cost_estimator, budget_analyst],
        max_turns=2,   # deterministic execution
    )

async def run_team(task: str) -> str:
    team = build_team()
    result = await team.run(task=task)

    return "\n\n".join(
        f"[{msg.source}]: {msg.content}"
        for msg in result.messages
        if hasattr(msg, "content") and msg.content
    )

# ─────────────────────────────────────────────────────────────
# A2A Executor
# ─────────────────────────────────────────────────────────────

class BudgetAgentExecutor(AgentExecutor):
    APP_NAME = "budget_agent"

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_input = context.get_user_input()

        try:
            report = await run_team(user_input)

            await event_queue.enqueue_event(
                new_agent_text_message(report)
            )

        except Exception as e:
            await event_queue.enqueue_event(
                new_agent_text_message(f"❌ Error: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        await event_queue.enqueue_event(
            new_agent_text_message("Budget request cancelled.")
        )

# ─────────────────────────────────────────────────────────────
# Server Bootstrap
# ─────────────────────────────────────────────────────────────

def main():
    PORT = int(os.environ.get("BUDGET_AGENT_PORT", 8892))
    HOST = os.environ.get("AGENT_HOST", "localhost")

    agent_card = AgentCard(
        name="BudgetAgent",
        description=(
                    "Estimates and evaluates budgets for any purpose: travel trips, "
                    "projects, events. Give it a list of items and costs — it itemizes, "
                    "totals, and flags expensive items. Use this whenever the user asks "
                    "for a budget, cost breakdown, or spend estimation."
                ),
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="budget_analysis",
                name="Budget Analysis",
                description=(
                    "ONLY analyze explicitly structured item:cost lists."
                "Takes a list of items and costs, computes total spend, "
                "flags expensive items, and gives a financial verdict."
                ),
                tags=[
                    "budget", "cost", "spend", "estimate", "price",
                    "trip budget", "travel cost", "breakdown", "itemize",
                    "how much", "total cost"
                ],
                examples=[
                    "Budget for my New York trip: flights 800, hotel 1200, food 300",
                    "Flight 5000, Hotel 8000, Food 2000 — is this reasonable?",
                    "What would a 5-day Paris trip cost me?",
                ],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=BudgetAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    print(f"💰 Budget Agent running at http://{HOST}:{PORT}")
    uvicorn.run(app.build(), host=HOST, port=PORT)

if __name__ == "__main__":
    main()
