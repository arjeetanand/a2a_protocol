
"""
gateway/a2a_gateway_server.py

A2A Gateway — written ONCE, never touched again.

To add a new agent:
  1. Start your new agent on some port (e.g. 8892)
  2. Add an entry to gateway/agent_registry.json
  3. That's it. No code changes needed here.

Key feature: MULTI-AGENT FAN-OUT
  If a query matches multiple agents (e.g. travel + finance),
  all matching agents are called IN PARALLEL and their responses
  are merged into one structured reply.

Run with:
    python gateway/a2a_gateway_server.py
"""

import asyncio
import json
import os
import time

import httpx
import uvicorn
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule

from a2a.client import Client, ClientConfig, ClientFactory, create_text_message_object
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Artifact, Message, Task
from a2a.utils import new_agent_text_message
from a2a.utils.message import get_message_text

load_dotenv()

console = Console()

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "agent_registry.json")
TIMEOUT_SECS  = float(os.environ.get("AGENT_TIMEOUT_SECS", 600))

AGENT_ICONS = {
    "travel":    "✈️ ",
    "finance":   "💰",
    "analytics": "📊",
    "budget":    "💸",
    "weather":   "🌤️ ",
}

AGENT_COLORS = {
    "travel":    "cyan",
    "finance":   "green",
    "analytics": "magenta",
    "budget":    "yellow",
    "weather":   "blue",
}

SECTION_ICONS = {
    "travel":    "✈️  Travel",
    "finance":   "💰 Finance",
    "analytics": "📊 Analytics",
    "budget":    "💰 Budget",
    "weather":   "🌤️  Weather",
}

_request_counter = 0  # global request counter


# ── Registry helpers ───────────────────────────────────────────────────────────
def _load_registry() -> dict:
    """Reload from disk every request so edits are picked up live."""
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


# def _match_agents(query: str) -> list[tuple[str, str, list[str]]]:
#     """
#     Return ALL agents whose keywords appear in the query.
#     Returns list of (agent_name, url, matched_keywords) tuples — deduped by URL.
#     """
#     query_lower = query.lower()
#     matched: list[tuple[str, str, list[str]]] = []
#     seen_urls: set[str] = set()

#     for agent_name, config in _load_registry().items():
#         hit_keywords = [kw for kw in config["keywords"] if kw in query_lower]
#         if hit_keywords and config["url"] not in seen_urls:
#             matched.append((agent_name, config["url"], hit_keywords))
#             seen_urls.add(config["url"])

#     return matched

import re

def _match_agents(query: str) -> list[tuple[str, str, list[str]]]:
    query_lower = query.lower()
    matched = []
    seen_urls = set()

    for agent_name, config in _load_registry().items():
        hit_keywords = []

        for kw in config["keywords"]:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, query_lower):
                hit_keywords.append(kw)

        if hit_keywords and config["url"] not in seen_urls:
            matched.append((agent_name, config["url"], hit_keywords))
            seen_urls.add(config["url"])

    return matched


# ── Logging helpers ────────────────────────────────────────────────────────────
def _log_incoming_request(request_id: int, query: str) -> None:
    console.print()
    console.rule(f"[bold white]⚡ REQUEST #{request_id}[/bold white]", style="bold blue")
    console.print(
        Panel(
            f"[bold yellow]{query}[/bold yellow]",
            title="[bold white]📥 Incoming Query[/bold white]",
            border_style="blue",
            padding=(0, 2),
        )
    )


def _log_routing_table(
    request_id: int,
    query: str,
    matched: list[tuple[str, str, list[str]]],
) -> None:
    table = Table(
        title=f"[bold white]🔀 Routing Decision — Request #{request_id}[/bold white]",
        box=box.ROUNDED,
        border_style="bright_blue",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("Agent",           style="bold", width=14)
    table.add_column("URL",             style="dim cyan", width=30)
    table.add_column("Matched Keywords", style="bold yellow")

    for agent_name, url, kws in matched:
        color = AGENT_COLORS.get(agent_name, "white")
        icon  = AGENT_ICONS.get(agent_name, "🤖")
        kw_highlights = ", ".join(
            f"[bold green underline]{kw}[/bold green underline]" for kw in kws
        )
        table.add_row(
            f"[{color}]{icon} {agent_name.upper()}[/{color}]",
            url,
            kw_highlights,
        )

    console.print(table)


def _log_no_match(query: str) -> None:
    console.print(
        Panel(
            f"[red]No agent matched for:[/red]\n[yellow]{query}[/yellow]\n\n"
            "[dim]Tip: Add keywords to gateway/agent_registry.json[/dim]",
            title="[bold red]❌ No Match[/bold red]",
            border_style="red",
        )
    )


def _log_agent_start(agent_name: str, url: str) -> None:
    color = AGENT_COLORS.get(agent_name, "white")
    icon  = AGENT_ICONS.get(agent_name, "🤖")
    console.print(
        f"  [dim]⏳ Calling[/dim] [{color}]{icon} {agent_name.upper()}[/{color}]"
        f" [dim]→[/dim] [cyan]{url}[/cyan] [dim]...[/dim]"
    )


def _log_agent_result(
    agent_name: str,
    url: str,
    response_text: str,
    elapsed: float,
    success: bool = True,
) -> None:
    color  = AGENT_COLORS.get(agent_name, "white")
    icon   = AGENT_ICONS.get(agent_name, "🤖")
    status = "[bold green]✅ SUCCESS[/bold green]" if success else "[bold red]❌ ERROR[/bold red]"

    # Truncate long responses for console readability
    preview = response_text.strip().replace("\n", " ")
    if len(preview) > 200:
        preview = preview[:200] + "…"

    console.print(
        Panel(
            f"{status}  [dim]({elapsed:.2f}s)[/dim]\n\n"
            f"[bold]Agent:[/bold] [{color}]{icon} {agent_name.upper()}[/{color}]\n"
            f"[bold]URL:  [/bold] [cyan]{url}[/cyan]\n\n"
            f"[bold]Response Preview:[/bold]\n[italic dim]{preview}[/italic dim]",
            title=f"[{color}]{icon} {agent_name.capitalize()} Response[/{color}]",
            border_style=color,
            padding=(0, 2),
        )
    )


def _log_fanout_summary(
    request_id: int,
    results: list[tuple[str, str]],
    total_elapsed: float,
) -> None:
    table = Table(
        title=f"[bold white]📦 Fan-Out Summary — Request #{request_id}[/bold white]",
        box=box.SIMPLE_HEAD,
        border_style="bright_green",
        padding=(0, 1),
    )
    table.add_column("Agent",   style="bold", width=14)
    table.add_column("Status",  width=12)
    table.add_column("Response Length", justify="right")

    for agent_name, response_text in results:
        color   = AGENT_COLORS.get(agent_name, "white")
        icon    = AGENT_ICONS.get(agent_name, "🤖")
        is_err  = response_text.startswith(f"[{agent_name}] error")
        status  = "[red]❌ Error[/red]" if is_err else "[green]✅ OK[/green]"
        length  = f"{len(response_text):,} chars"
        table.add_row(
            f"[{color}]{icon} {agent_name.upper()}[/{color}]",
            status,
            f"[dim]{length}[/dim]",
        )

    console.print(table)
    console.print(
        f"  [bold green]⚡ Total time:[/bold green] [yellow]{total_elapsed:.2f}s[/yellow]"
        f"  [dim]|[/dim]  [bold]Agents called:[/bold] [cyan]{len(results)}[/cyan]"
    )
    console.print()


# ── Single downstream call ─────────────────────────────────────────────────────
async def _call_one_agent(
    agent_name: str,
    url: str,
    query: str,
    httpx_client: httpx.AsyncClient,
) -> tuple[str, str, float]:
    """Call one downstream A2A agent and return (agent_name, response_text, elapsed_secs)."""
    _log_agent_start(agent_name, url)
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
        text_content = text_content or f"[{agent_name}] returned no response."
        _log_agent_result(agent_name, url, text_content, elapsed, success=True)
        return agent_name, text_content, elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        err_text = f"[{agent_name}] error: {e}"
        _log_agent_result(agent_name, url, err_text, elapsed, success=False)
        return agent_name, err_text, elapsed


# ── Gateway Executor ───────────────────────────────────────────────────────────
class GatewayExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        global _request_counter
        _request_counter += 1
        req_id = _request_counter

        query   = context.get_user_input()
        matched = _match_agents(query)  # now returns (name, url, [keywords])

        _log_incoming_request(req_id, query)

        # ── No agent matched ────────────────────────────────────────────────
        if not matched:
            _log_no_match(query)
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "No registered agent matched this request. "
                    "Add the relevant keywords to gateway/agent_registry.json."
                )
            )
            return

        _log_routing_table(req_id, query, matched)

        httpx_timeout = httpx.Timeout(
            connect=10.0,
            read=TIMEOUT_SECS,
            write=30.0,
            pool=10.0,
        )

        # ── Fan-out: call ALL matched agents in parallel ─────────────────
        t_start = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx_timeout) as httpx_client:
            tasks = [
                _call_one_agent(name, url, query, httpx_client)
                for name, url, _ in matched
            ]
            raw_results: list[tuple[str, str, float]] = await asyncio.gather(*tasks , return_exceptions=True)

        total_elapsed = time.monotonic() - t_start
        results: list[tuple[str, str]] = [(n, t) for n, t, _ in raw_results]

        _log_fanout_summary(req_id, results, total_elapsed)

        # ── Merge responses ────────────────────────────────────────────────
        routing_header = (
            "> 🔀 **Gateway routed to:** "
            + ", ".join(f"`{n}` → {u}" for n, u, _ in matched)
        )

        if len(results) == 1:
            agent_name, text = results[0]
            url = matched[0][1]
            final_text = f"> 🔀 **Gateway routed to:** `{agent_name}` → {url}\n\n{text}"
        else:
            sections = [routing_header]
            for agent_name, text in results:
                icon = SECTION_ICONS.get(agent_name, f"🤖 {agent_name.capitalize()}")
                sections.append(f"## {icon}\n\n{text}")
            final_text = "\n\n---\n\n".join(sections)

        await event_queue.enqueue_event(new_agent_text_message(final_text))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


# ── Server setup ───────────────────────────────────────────────────────────────
def main() -> None:
    HOST = os.environ.get("GATEWAY_HOST", "localhost")
    PORT = int(os.environ.get("GATEWAY_PORT", 9000))

    skills = []
    try:
        for agent_name, config in _load_registry().items():
            skills.append(
                AgentSkill(
                    id=agent_name,
                    name=agent_name.capitalize(),
                    description=f"Routes to {config['url']}",
                    tags=config["keywords"],
                    examples=[f"Example: {config['keywords'][0]} related query"],
                )
            )
    except Exception:
        pass

    agent_card = AgentCard(
        name="A2AGateway",
        description=(
            "Central A2A gateway with multi-agent fan-out. "
            "Routes queries to ALL matching downstream agents in parallel "
            "and merges their responses. Edit agent_registry.json to add agents."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=GatewayExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # ── Startup Banner ─────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]🚀 A2A Gateway (fan-out) running at http://{HOST}:{PORT}[/bold green]\n"
        f"[dim]📋 Registry: {REGISTRY_PATH}[/dim]",
        title="[bold white]A2A GATEWAY[/bold white]",
        border_style="bright_green",
        padding=(1, 4),
    ))

    # ── Agent Registry Table ───────────────────────────────────────────────
    try:
        reg_table = Table(
            title="[bold white]📋 Registered Agents[/bold white]",
            box=box.ROUNDED,
            border_style="bright_blue",
            show_lines=True,
            padding=(0, 1),
        )
        reg_table.add_column("Agent",    style="bold", width=14)
        reg_table.add_column("URL",      style="cyan", width=30)
        reg_table.add_column("Keywords", style="yellow")

        for name, cfg in _load_registry().items():
            color = AGENT_COLORS.get(name, "white")
            icon  = AGENT_ICONS.get(name, "🤖")
            reg_table.add_row(
                f"[{color}]{icon} {name.upper()}[/{color}]",
                cfg["url"],
                ", ".join(cfg["keywords"]),
            )
        console.print(reg_table)
    except Exception as e:
        console.print(f"[red]Could not load registry: {e}[/red]")

    console.print()
    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
