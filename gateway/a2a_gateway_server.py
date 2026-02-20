"""
gateway/a2a_gateway_server.py

SMART A2A Gateway — LLM-based semantic routing.

- Discovers agents via AgentCard (GET /)
- Uses LLM to decide which agent(s) should handle a query
- Calls selected agents in parallel
- Merges responses

Run:
    python gateway/a2a_gateway_server.py
"""

import asyncio
import json
import os
import time
from typing import List, Dict

import httpx
import uvicorn
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from a2a.client import Client, ClientConfig, ClientFactory, create_text_message_object
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from a2a.types import Artifact, Message, Task
from a2a.utils.message import get_message_text

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types


from shared.utils import build_oci_model

load_dotenv()

oci_model = build_oci_model()

console = Console()

TIMEOUT_SECS = float(os.environ.get("AGENT_TIMEOUT_SECS", 600))

# 👉 Add agents here (no keywords needed)
AGENT_URLS = [
    "http://localhost:8890",  # finance
    "http://localhost:8891",  # analytics
    "http://localhost:8892",  # budget
    "http://localhost:8888",  # supervisor/travel
]

_request_counter = 0


# ─────────────────────────────────────────────────────────────
# 1️⃣ Agent Discovery
# ─────────────────────────────────────────────────────────────

# async def fetch_agent_card(url: str) -> Dict:
#     async with httpx.AsyncClient() as client:
#         res = await client.get(url)
#         return res.json()

async def fetch_agent_card(url: str) -> Dict:
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{url}/.well-known/agent-card.json")
        res.raise_for_status()
        return res.json()


async def discover_agents() -> List[Dict]:
    agents = []
    for url in AGENT_URLS:
        try:
            card = await fetch_agent_card(url)
            agents.append({
                "name": card["name"],
                "url": url,
                "description": card.get("description", ""),
                "skills": card.get("skills", []),
            })
        except Exception as e:
            console.print(f"[red]Failed to fetch AgentCard from {url}: {e}[/red]")
    return agents


# ─────────────────────────────────────────────────────────────
# 2️⃣ LLM Routing Decision
# ─────────────────────────────────────────────────────────────

def build_routing_prompt(query: str, agents: List[Dict]) -> str:
        agent_descriptions = "\n\n".join([
            f"Agent: {a['name']}\n"
            f"Description: {a['description']}\n"
            f"Skills: {', '.join(s.get('name', '') for s in a['skills'])}"
            for a in agents
        ])

        return f"""
    You are a routing AI.

    Available agents:

    {agent_descriptions}

    User query:
    "{query}"

    Return a JSON array of agent names that should handle this query.
    If multiple agents are relevant, return multiple.
    Return ONLY valid JSON.
    """


async def decide_agents(query: str, agents: List[Dict]) -> List[str]:
    """
    Uses ADK Runner to execute router_agent properly.
    """

    prompt = build_routing_prompt(query, agents)

    # Create session
    session = await router_session_service.create_session(
        app_name="router_app",
        user_id="router_user",
    )

    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)],
    )

    final_text = ""

    async for event in router_runner.run_async(
        user_id="router_user",
        session_id=session.id,
        new_message=user_content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = "".join(
                    part.text
                    for part in event.content.parts
                    if hasattr(part, "text") and part.text
                )
            break

    try:
        selected = json.loads(final_text.strip())
        if isinstance(selected, list):
            return selected
    except Exception:
        console.print("[red]Router returned invalid JSON.[/red]")

    return []

# ─────────────────────────────────────────────
# Router Agent (ADK Native)
# ─────────────────────────────────────────────

router_agent = Agent(
    name="router_agent",
    model=oci_model,
    instruction="""
You are a routing AI.

Given:
- A list of available agents (name + description + skills)
- A user query

Return ONLY a JSON array of agent names that should handle the query.

Rules:
- Return valid JSON only.
- No explanations.
- If multiple agents are relevant, include all.
- If none match, return an empty JSON array [].
"""
)

router_session_service = InMemorySessionService()

router_runner = Runner(
    agent=router_agent,
    app_name="router_app",
    session_service=router_session_service,
)


# ─────────────────────────────────────────────────────────────
# 3️⃣ Downstream A2A Call
# ─────────────────────────────────────────────────────────────

async def _call_one_agent(
    agent_name: str,
    url: str,
    query: str,
    httpx_client: httpx.AsyncClient,
):
    t0 = time.monotonic()

    try:
        client: Client = await ClientFactory.connect(
            url,
            client_config=ClientConfig(httpx_client=httpx_client),
        )

        message = create_text_message_object(content=query)
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

        elapsed = time.monotonic() - t0
        return agent_name, text_content, elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        return agent_name, f"[ERROR] {str(e)}", elapsed


# ─────────────────────────────────────────────────────────────
# 4️⃣ Gateway Executor
# ─────────────────────────────────────────────────────────────

class SmartGatewayExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        global _request_counter
        _request_counter += 1
        req_id = _request_counter

        query = context.get_user_input()

        console.print()
        console.rule(f"[bold blue]⚡ REQUEST #{req_id}[/bold blue]")
        console.print(Panel(query, title="Incoming Query"))

        agents = await discover_agents()

        if not agents:
            await event_queue.enqueue_event(
                new_agent_text_message("No agents available.")
            )
            return

        selected_names = await decide_agents(query, agents)

        selected_agents = [
            a for a in agents if a["name"] in selected_names
        ]

        if not selected_agents:
            await event_queue.enqueue_event(
                new_agent_text_message("LLM could not determine a suitable agent.")
            )
            return

        console.print(
            f"[green]LLM selected:[/green] {', '.join(selected_names)}"
        )

        httpx_timeout = httpx.Timeout(
            connect=10.0,
            read=TIMEOUT_SECS,
            write=30.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=httpx_timeout) as httpx_client:
            tasks = [
                _call_one_agent(a["name"], a["url"], query, httpx_client)
                for a in selected_agents
            ]
            raw_results = await asyncio.gather(*tasks)

            console.print("\n[bold cyan]🔎 Downstream Agent Responses[/bold cyan]\n")

            for agent_name, text, elapsed in raw_results:
                console.print(Panel(
                    text,
                    title=f"{agent_name} ({elapsed:.2f}s)",
                    border_style="cyan"
                ))

        # Merge responses
        sections = []
        for agent_name, text, elapsed in raw_results:
            sections.append(
                f"## {agent_name}\n\n{text}\n\n⏱ {elapsed:.2f}s"
            )

        final_text = "\n\n---\n\n".join(sections)

        await event_queue.enqueue_event(
            new_agent_text_message(final_text)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


# ─────────────────────────────────────────────────────────────
# 5️⃣ Server Bootstrap
# ─────────────────────────────────────────────────────────────

def main():
    HOST = os.environ.get("GATEWAY_HOST", "localhost")
    PORT = int(os.environ.get("GATEWAY_PORT", 9000))

    agent_card = AgentCard(
        name="SmartA2AGateway",
        description=(
            "LLM-powered semantic A2A router. "
            "Discovers agents via AgentCard and routes dynamically."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="3.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SmartGatewayExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    console.print(
        Panel(
            f"🚀 Smart A2A Gateway running at http://{HOST}:{PORT}",
            title="SMART LLM ROUTER",
            border_style="green",
        )
    )

    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
