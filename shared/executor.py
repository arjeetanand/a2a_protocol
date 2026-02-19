"""
shared/base_executor.py
Reusable A2A AgentExecutor base for all ADK-backed agents.
"""
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

FALLBACK_MSG = "The agent did not return a response."


class BaseADKExecutor(AgentExecutor):
    """Base executor — subclasses only need to set APP_NAME and agent."""

    APP_NAME: str = ""

    def __init__(self, agent) -> None:
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=agent,
            app_name=self.APP_NAME,
            session_service=self.session_service,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from google.genai import types as genai_types

        prompt = context.get_user_input()
        session = await self.session_service.create_session(
            app_name=self.APP_NAME, user_id="a2a_user"
        )
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )
        final_text = ""
        async for event in self.runner.run_async(
            user_id="a2a_user",
            session_id=session.id,
            new_message=user_content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join(
                        p.text for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )
                break

        await event_queue.enqueue_event(
            new_agent_text_message(final_text or FALLBACK_MSG)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
