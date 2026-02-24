"""
gateway/a2a_gateway_server.py

SMART A2A Gateway — LLM-based semantic routing (LangGraph Version)

Design principle:
  - The router prompt is GENERIC and PERMANENT — it never names agents.
  - Routing intelligence lives entirely in each agent's AgentCard.
  - To add a new agent: append its URL to AGENT_URLS.

Run:
    python gateway/a2a_gateway_server.py
"""

import asyncio
import json
import os
import time
from typing import List, Dict, TypedDict

import httpx
import uvicorn
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

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

# LangGraph
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage


from travel_agent.agent import OCIGenAILLM


load_dotenv()

oci_model = OCIGenAILLM()

console = Console()
TIMEOUT_SECS = float(os.environ.get("AGENT_TIMEOUT_SECS", 600))


# ─────────────────────────────────────────────────────────────
# AGENT REGISTRY
# ─────────────────────────────────────────────────────────────

AGENT_URLS = [
    "http://localhost:8890",  # finance google adk
    "http://localhost:8892",  # budget autogen
    "http://localhost:8888",  # supervisor langgraph
    "http://localhost:8893",  # hr MCP
    "http://localhost:8894",  # payroll MAF
    "http://localhost:8896", # leave approval crewai
]

_request_counter = 0


# ─────────────────────────────────────────────────────────────
# 1️⃣  Agent Discovery
# ─────────────────────────────────────────────────────────────

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
                "name":        card["name"],
                "url":         url,
                "description": card.get("description", ""),
                "skills":      card.get("skills", []),
            })
        except Exception as e:
            console.print(f"[red]✗ AgentCard unavailable at {url}: {e}[/red]")
    return agents


# ─────────────────────────────────────────────────────────────
# 2️⃣  Routing Prompt Builder (UNCHANGED LOGIC)
# ─────────────────────────────────────────────────────────────

def build_routing_prompt(query: str, agents: List[Dict]) -> str:
    agent_blocks = []

    for a in agents:
        skill_lines = []
        for s in a["skills"]:
            tags     = ", ".join(s.get("tags", []))
            examples = " | ".join(s.get("examples", []))
            skill_lines.append(
                f"    • {s.get('name', '')}: {s.get('description', '')}\n"
                f"      Tags: {tags}\n"
                f"      Examples: {examples}"
            )

        skills_block = "\n".join(skill_lines) if skill_lines else "    (no skills listed)"

        agent_blocks.append(
            f"Agent: {a['name']}\n"
            f"  Description: {a['description']}\n"
            f"  Skills:\n{skills_block}"
        )

    agents_section = "\n\n".join(agent_blocks)

    return f"""You are a routing engine for a multi-agent system.

Each agent below has published its own capabilities via AgentCard.
Use their descriptions, skill tags, and example queries to decide routing.

{agents_section}

User query: "{query}"

Return a JSON array of agent names that should handle this query.
Match on descriptions, tags, and examples — not on your own assumptions.
If the query spans multiple domains, include ALL relevant agents.
Return ONLY valid JSON. No explanation. No markdown.
If nothing matches: []
"""


# ─────────────────────────────────────────────────────────────
# 3️⃣  LangGraph Router (Replaces Google ADK)
# ─────────────────────────────────────────────────────────────

class RouterState(TypedDict):
    query: str
    agents: List[Dict]
    prompt: str
    raw_output: str
    selected: List[str]


# Node 1 — Build Prompt
def build_prompt_node(state: RouterState):
    prompt = build_routing_prompt(state["query"], state["agents"])
    return {"prompt": prompt}


# Node 2 — Call LLM
async def llm_node(state: RouterState):
    response = await asyncio.to_thread(
        oci_model.invoke,
        state["prompt"]
    )

    return {"raw_output": response}

# Node 3 — Parse JSON
def parse_node(state: RouterState):
    try:
        parsed = json.loads(state["raw_output"].strip())
        if isinstance(parsed, list):
            return {"selected": parsed}
    except Exception:
        console.print(
            f"[red]Router returned invalid JSON: {state['raw_output']!r}[/red]"
        )
    return {"selected": []}


# Build Graph
graph_builder = StateGraph(RouterState)

graph_builder.add_node("build_prompt", build_prompt_node)
graph_builder.add_node("call_llm", llm_node)
graph_builder.add_node("parse", parse_node)

graph_builder.set_entry_point("build_prompt")

graph_builder.add_edge("build_prompt", "call_llm")
graph_builder.add_edge("call_llm", "parse")
graph_builder.add_edge("parse", END)

router_graph = graph_builder.compile()


async def decide_agents(query: str, agents: List[Dict]) -> List[str]:
    result = await router_graph.ainvoke({
        "query": query,
        "agents": agents,
    })
    return result.get("selected", [])


# ─────────────────────────────────────────────────────────────
# 4️⃣  Downstream A2A Call (UNCHANGED)
# ─────────────────────────────────────────────────────────────

async def _call_one_agent(agent_name: str, url: str, query: str, httpx_client: httpx.AsyncClient):
    t0 = time.monotonic()
    try:
        client: Client = await ClientFactory.connect(
            url,
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

        elapsed = time.monotonic() - t0
        return agent_name, text_content, elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        return agent_name, f"[ERROR] {str(e)}", elapsed


# ─────────────────────────────────────────────────────────────
# 5️⃣  Gateway Executor (UNCHANGED)
# ─────────────────────────────────────────────────────────────

class SmartGatewayExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        global _request_counter
        _request_counter += 1
        req_id = _request_counter

        query = context.get_user_input()

        console.print()
        console.rule(f"[bold blue]⚡ REQUEST #{req_id}")
        console.print(Panel(query, title="Incoming Query"))

        agents = await discover_agents()

        if not agents:
            await event_queue.enqueue_event(
                new_agent_text_message("No agents available.")
            )
            return

        console.print(
            f"[dim]Discovered: {', '.join(a['name'] for a in agents)}[/dim]"
        )

        selected_names = await decide_agents(query, agents)
        selected_agents = [a for a in agents if a["name"] in selected_names]

        if not selected_agents:
            await event_queue.enqueue_event(
                new_agent_text_message("No agent matched this query.")
            )
            return

        console.print(f"[green]Routed to:[/green] {', '.join(selected_names)}")

        httpx_timeout = httpx.Timeout(
            connect=10.0, read=TIMEOUT_SECS, write=30.0, pool=10.0,
        )

        async with httpx.AsyncClient(timeout=httpx_timeout) as httpx_client:
            tasks = [
                _call_one_agent(a["name"], a["url"], query, httpx_client)
                for a in selected_agents
            ]

            raw_results = await asyncio.gather(*tasks)

            console.print("\n[bold cyan]🔎 Agent Responses[/bold cyan]\n")

            for agent_name, text, elapsed in raw_results:
                console.print(Panel(
                    text,
                    title=f"{agent_name} ({elapsed:.2f}s)",
                    border_style="cyan",
                ))

        sections = [
            f"## {name}\n\n{text}\n\n⏱ {elapsed:.2f}s"
            for name, text, elapsed in raw_results
        ]

        await event_queue.enqueue_event(
            new_agent_text_message("\n\n---\n\n".join(sections))
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


# ─────────────────────────────────────────────────────────────
# 6️⃣  Server Bootstrap (UNCHANGED)
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

    console.print(Panel(
        f"🚀 Smart A2A Gateway → http://{HOST}:{PORT}",
        title="SMART LLM ROUTER (LangGraph)",
        border_style="green",
    ))

    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()