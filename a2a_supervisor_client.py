"""
a2a_supervisor_client.py
Calls the Unified Supervisor A2A server and displays the response.

Usage:
    python a2a_supervisor_client.py
    python a2a_supervisor_client.py "Find flights from NYC to Paris on June 10"
"""

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

from a2a.client import Client, ClientConfig, ClientFactory, create_text_message_object
from a2a.types import AgentCard, Artifact, Message, Task
from a2a.utils.message import get_message_text


# ── Pretty-print the agent card ───────────────────────────────────────────────
def display_agent_card(agent_card: AgentCard) -> None:
    print("\n" + "=" * 60)
    print("  AGENT CARD")
    print("=" * 60)
    print(f"  Name        : {agent_card.name}")
    print(f"  Description : {agent_card.description}")
    print(f"  Version     : {agent_card.version}")
    print(f"  URL         : {agent_card.url}")
    print(f"  Protocol    : {agent_card.protocol_version}")

    if agent_card.skills:
        print("\n  Skills:")
        for skill in agent_card.skills:
            print(f"    • [{skill.id}] {skill.name}")
            print(f"        {skill.description}")
            if skill.examples:
                for ex in skill.examples:
                    print(f"        Example: {ex}")
    print("=" * 60 + "\n")


# ── Main async runner ─────────────────────────────────────────────────────────
async def run(prompt: str) -> None:
    load_dotenv()

    host = os.environ.get("AGENT_HOST", "localhost")
    port = os.environ.get("SUPERVISOR_AGENT_PORT", "8888")
    base_url = f"http://{host}:{port}"

    print(f"Connecting to Supervisor A2A server at {base_url} ...")

    timeout_secs = float(os.environ.get("AGENT_TIMEOUT_SECS", 600))

    # httpx timeout: controls the raw HTTP read/write window
    httpx_timeout = httpx.Timeout(
        connect=10.0,       # TCP connection
        read=timeout_secs,  # waiting for the server to send the response body
        write=30.0,         # uploading the request body
        pool=10.0,          # waiting for a free connection from the pool
    )

    async with httpx.AsyncClient(timeout=httpx_timeout) as httpx_client:

        # ── Step 1: Connect & discover ────────────────────────────────────────
        client: Client = await ClientFactory.connect(
            base_url,
            client_config=ClientConfig(
                httpx_client=httpx_client,
            ),
        )

        agent_card = await client.get_card()
        display_agent_card(agent_card)

        # ── Step 2: Build and send the message ────────────────────────────────
        message = create_text_message_object(content=prompt)
        print(f"Prompt  : {prompt}")
        print("Waiting for agent response...\n")

        responses = client.send_message(message)

        text_content = ""

        # ── Step 3: Collect the response ──────────────────────────────────────
        async for response in responses:
            if isinstance(response, Message):
                # Agent replied directly with a final message
                print(f"[Message ID] {response.message_id}")
                text_content = get_message_text(response)

            elif isinstance(response, tuple):
                # Response came back as a completed Task with artifacts
                task: Task = response[0]
                print(f"[Task ID] {task.id}")
                if task.artifacts:
                    artifact: Artifact = task.artifacts[0]
                    print(f"[Artifact ID] {artifact.artifact_id}")
                    text_content = get_message_text(artifact)

        # ── Step 4: Display the final answer ──────────────────────────────────
        print("=" * 60)
        print("  AGENT RESPONSE")
        print("=" * 60)
        if text_content:
            print(text_content)
        else:
            print("No response received or task did not complete successfully.")
        print("=" * 60 + "\n")


# ── Predefined demo prompts ───────────────────────────────────────────────────
DEMO_PROMPTS = [
    # Budget
    "I need 2 senior developers at $120/hr for 3 weeks, cloud hosting at $800/month, "
    "and design tools at $200 one-time. Is this reasonable?",
    # Travel
    "Find flights from New York to Los Angeles on July 20.",
    # Weather
    "What is the weather in New York right now?",
    # Time
    "What time is it in New York?",
]


if __name__ == "__main__":
    # Accept a custom prompt from the command line, otherwise cycle through demos
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        asyncio.run(run(user_prompt))
    else:
        print("No prompt supplied – running all demo prompts.\n")
        for p in DEMO_PROMPTS:
            asyncio.run(run(p))
            print()