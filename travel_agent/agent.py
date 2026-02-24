"""
adk/travel_agent/agent.py
LangGraph travel search graph + ADK weather/time sub-agents.
Exports everything that main_agent/agent.py imports.
"""

import os
import json
import re
from typing import Any, List, Optional, Annotated, TypedDict
from dotenv import load_dotenv

load_dotenv()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required env var: {key}  →  add it to your .env file"
        )
    return val.strip()


# ── OCI credentials (exported so main_agent can reuse them) ──────────────────
OCI_REGION         = _require_env("OCI_REGION")
OCI_USER           = _require_env("OCI_USER")
OCI_FINGERPRINT    = _require_env("OCI_FINGERPRINT")
OCI_TENANCY        = _require_env("OCI_TENANCY")
OCI_COMPARTMENT_ID = _require_env("OCI_COMPARTMENT_ID")
OCI_KEY_FILE       = _require_env("OCI_KEY_FILE")


# ── OCI raw client ────────────────────────────────────────────────────────────
import oci

OCI_CONFIG = {
    "user":        OCI_USER,
    "fingerprint": OCI_FINGERPRINT,
    "tenancy":     OCI_TENANCY,
    "region":      OCI_REGION,
    "key_file":    OCI_KEY_FILE,
}

OCI_ENDPOINT   = f"https://inference.generativeai.{OCI_REGION}.oci.oraclecloud.com"
COMPARTMENT_ID = OCI_COMPARTMENT_ID
MODEL_OCID     = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyajqi26fkxly6qje5ysvezzrypapl7ujdnqfjq6hzo2loq"

oci_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=OCI_CONFIG,
    service_endpoint=OCI_ENDPOINT,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240),
)


# ─────────────────────────────────────────────
# 1. Raw OCI LLM wrapper
# ─────────────────────────────────────────────
from langchain_core.language_models.llms import LLM


class OCIGenAILLM(LLM):
    @property
    def _llm_type(self):
        return "oci_genai"

    def _call(self, prompt: str, stop=None) -> str:
        content = oci.generative_ai_inference.models.TextContent()
        content.text = prompt

        message = oci.generative_ai_inference.models.Message()
        message.role = "USER"
        message.content = [content]

        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.api_format = (
            oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        )
        chat_request.messages          = [message]
        chat_request.max_tokens        = 2048
        chat_request.temperature       = 1
        chat_request.frequency_penalty = 0
        chat_request.presence_penalty  = 0
        chat_request.top_p             = 0.95
        chat_request.top_k             = 1

        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=MODEL_OCID
        )
        chat_detail.chat_request   = chat_request
        chat_detail.compartment_id = COMPARTMENT_ID

        response = oci_client.chat(chat_detail)
        return response.data.chat_response.choices[0].message.content[0].text


_raw_llm = OCIGenAILLM()


# ─────────────────────────────────────────────
# 2. Chat wrapper (prompt-injected tool calling)
# ─────────────────────────────────────────────
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration


class OCIChatModel(BaseChatModel):
    tools_map: dict = {}
    tools_schema: list = []

    @property
    def _llm_type(self):
        return "oci_chat"

    def bind_tools(self, tools: list) -> "OCIChatModel":
        schema, tmap = [], {}
        for t in tools:
            schema.append({
                "name":        t.name,
                "description": t.description,
                "parameters":  t.args_schema.model_json_schema() if t.args_schema else {},
            })
            tmap[t.name] = t
        return OCIChatModel(tools_map=tmap, tools_schema=schema)

    def _build_prompt(self, messages: List[BaseMessage]) -> str:
        parts = []
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

        if self.tools_schema and not has_tool_results:
            parts.append("You have access to the following tools (call them as JSON):")
            parts.append(json.dumps(self.tools_schema, indent=2))
            parts.append(
                "\nTo call a tool respond ONLY with this JSON (no extra text):\n"
                '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}\n'
                "If no tool is needed, respond normally in plain text."
            )
        elif has_tool_results:
            parts.append(
                "You have received tool results below. "
                "DO NOT call any more tools. "
                "Summarize the results in a clear, friendly plain-text response."
            )

        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"[SYSTEM]: {msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"[USER]: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"[ASSISTANT]: {msg.content}")
            elif isinstance(msg, ToolMessage):
                parts.append(f"[TOOL RESULT]: {msg.content}")

        return "\n".join(parts)

    def _parse_tool_call(self, text: str) -> Optional[dict]:
        try:
            clean = re.sub(r"```(?:json)?|```", "", text).strip()
            data  = json.loads(clean)
            if "tool_call" in data:
                return data["tool_call"]
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        print("➡ Calling OCI model (LangGraph)...")
        raw_text  = _raw_llm.invoke(self._build_prompt(messages))
        print("⬅ OCI responded")
        tool_call = self._parse_tool_call(raw_text)

        if tool_call:
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "id":   f"call_{tool_call['name']}",
                    "name": tool_call["name"],
                    "args": tool_call.get("arguments", {}),
                }],
            )
        else:
            ai_msg = AIMessage(content=raw_text)

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


# ─────────────────────────────────────────────
# 3. LangGraph tools & graph
# ─────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date."""
    return json.dumps({
        "flights": [
            {"airline": "Delta",   "departs": "08:00", "arrives": "11:30", "price": "$320"},
            {"airline": "United",  "departs": "13:00", "arrives": "16:30", "price": "$289"},
            {"airline": "JetBlue", "departs": "18:00", "arrives": "21:30", "price": "$265"},
        ],
        "route": f"{origin} → {destination}",
        "date":  date,
    })


@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """Search for hotels in a city for given check-in and check-out dates."""
    return json.dumps({
        "hotels": [
            {"name": "Marriott Times Square", "stars": 4, "price_per_night": "$240", "rating": 4.5},
            {"name": "Pod Hotel NYC",          "stars": 3, "price_per_night": "$129", "rating": 4.1},
            {"name": "The Plaza",              "stars": 5, "price_per_night": "$850", "rating": 4.8},
        ],
        "city":     city,
        "checkin":  checkin,
        "checkout": checkout,
    })


_LANGGRAPH_TOOLS     = [search_flights, search_hotels]
_LANGGRAPH_TOOLS_MAP = {t.name: t for t in _LANGGRAPH_TOOLS}
_langgraph_llm       = OCIChatModel().bind_tools(_LANGGRAPH_TOOLS)


class TravelState(TypedDict):
    messages: Annotated[list, add_messages]


def _travel_llm_node(state: TravelState):
    return {"messages": [_langgraph_llm.invoke(state["messages"])]}


def _travel_tool_node(state: TravelState):
    last_msg = state["messages"][-1]
    results  = []
    for call in last_msg.tool_calls:
        output = _LANGGRAPH_TOOLS_MAP[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


def _should_use_tools(state: TravelState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


_travel_builder = StateGraph(TravelState)
_travel_builder.add_node("llm",   _travel_llm_node)
_travel_builder.add_node("tools", _travel_tool_node)
_travel_builder.set_entry_point("llm")
_travel_builder.add_conditional_edges("llm", _should_use_tools)
_travel_builder.add_edge("tools", "llm")
travel_graph = _travel_builder.compile()


# ── ADK-compatible bridge ─────────────────────────────────────────────────────
def langgraph_travel_tool(query: str) -> dict:
    """
    Search for flights or hotels using the LangGraph travel agent.

    Args:
        query: Natural language travel question (e.g. 'Find flights from NYC to LA on March 10').

    Returns:
        dict with 'status' and 'report' keys.
    """
    try:
        final_state = travel_graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 6},
        )
        answer = next(
            m.content for m in reversed(final_state["messages"])
            if isinstance(m, AIMessage) and m.content
        )
        return {"status": "success", "report": answer}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


# ─────────────────────────────────────────────
# Weather & Time Tools (Pure LangGraph)
# ─────────────────────────────────────────────
import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Retrieves the current weather report for a specified city."""
    if city.lower() == "new york":
        return "Sunny, 25°C (77°F) in New York."
    return f"Weather info for '{city}' is unavailable."


@tool
def get_current_time(city: str) -> str:
    """Returns the current time in a specified city."""
    if city.lower() == "new york":
        now = datetime.datetime.now(ZoneInfo("America/New_York"))
        return f"Current time in {city}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    return f"No timezone info for {city}."
