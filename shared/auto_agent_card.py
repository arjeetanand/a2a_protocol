import inspect
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


def build_auto_agent_card(
    agent,
    host: str,
    port: int,
    description: str,
    version: str = "1.0.0",
):
    """
    Auto-generate an A2A AgentCard from an ADK Agent instance.
    - Extracts tools as skills
    - Uses tool docstrings for descriptions
    """

    skills = []

    for tool in agent.tools:
        tool_name = tool.__name__

        # Extract docstring safely
        doc = inspect.getdoc(tool) or "No description provided."

        # Extract first line as short description
        short_desc = doc.split("\n")[0]

        skills.append(
            AgentSkill(
                id=tool_name,
                name=tool_name.replace("_", " ").title(),
                description=short_desc,
                tags=tool_name.split("_"),
                examples=[],  # optional: can auto-parse later
            )
        )

    return AgentCard(
        name=agent.name,
        description=description,
        url=f"http://{host}:{port}/",
        version=version,
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )
