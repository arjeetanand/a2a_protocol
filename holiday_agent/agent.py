"""
holiday_agent/agent.py

"""

import asyncio
import concurrent.futures
import os

import httpx
from dotenv import load_dotenv

from a2a.client import Client, ClientConfig, ClientFactory, create_text_message_object
from a2a.types import Artifact, Message, Task
from a2a.utils.message import get_message_text

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from shared.utils import require_env, build_oci_model


load_dotenv()

oci_model = build_oci_model()


# ── Gateway connection (single endpoint for ALL downstream agents) ─────────────
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "localhost")
GATEWAY_PORT = os.environ.get("GATEWAY_PORT", "9000")
GATEWAY_URL  = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
TIMEOUT_SECS = float(os.environ.get("AGENT_TIMEOUT_SECS", 600))


# ── A2A call (async) ──────────────────────────────────────────────────────────
async def _call_gateway_async(query: str) -> str:
    httpx_timeout = httpx.Timeout(connect=10.0, read=TIMEOUT_SECS, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=httpx_timeout) as httpx_client:
        client: Client = await ClientFactory.connect(
            GATEWAY_URL,
            client_config=ClientConfig(httpx_client=httpx_client),
        )
        message   = create_text_message_object(content=query)
        responses = client.send_message(message)

        text_content = ""
        async for response in responses:
            if isinstance(response, Message):
                text_content = get_message_text(response)
            elif isinstance(response, tuple):
                task: Task = response[0]
                if task.artifacts:
                    artifact: Artifact = task.artifacts[0]
                    text_content = get_message_text(artifact)

        return text_content or "The gateway returned no response."


# ── Thread bridge (avoids asyncio.run clash with ADK's event loop) ────────────
def _run_in_thread(query: str) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_call_gateway_async(query))
    finally:
        loop.close()


# ── ADK Tool ──────────────────────────────────────────────────────────────────
def ask_agent_gateway(query: str) -> dict:
    """
    Send a query to the A2A Gateway, which automatically routes it to the
    correct downstream agent (travel, finance, analytics, etc.).

    Use this tool for ANY query involving:
    - Travel: flights, hotels, destinations, weather, trip time
    - Finance: budgets, costs, expenses, ROI, investments
    - Analytics: reports, metrics, data analysis, trends, growth

    The gateway handles routing — you never need to know which agent handles what.

    Args:
        query: The user's question in natural language.

    Returns:
        dict with 'status' and 'report' keys.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_thread, query)
            result = future.result(timeout=TIMEOUT_SECS)
        return {"status": "success", "report": result}
    except concurrent.futures.TimeoutError:
        return {"status": "error", "error_message": "Gateway timed out. Please try again."}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


# ── Holiday Checker Agent ─────────────────────────────────────────────────────
root_agent = Agent(
    name="holiday_checker_agent",
    model=oci_model,
    instruction="""You are a friendly Holiday Checker assistant.

You have two modes:

1. DIRECT ANSWER — answer yourself for:
   - Public holidays and dates
   - Leave / PTO calculations
   - General knowledge questions
   - Casual advice or non-real-time queries

2. CALL ask_agent_gateway — delegate for ANY query that:
   - Requires live data
   - Requires numeric computation
   - Involves cost, finance, analytics, travel, weather, or specialized services
   - Seems domain-specific or system-based

Rules:
- When unsure, delegate.
- Never fabricate live data (prices, metrics, analysis).
- The gateway decides which downstream agent handles it.
- You do NOT need to know agent names.
- Present responses clearly and warmly.
""",
    tools=[ask_agent_gateway],
)
