from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class TicketState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    ticket_text: str
    customer_id: str

    # ── LLM conversation (tool calls + results accumulate here) ────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Populated by classify_node ─────────────────────────────────
    urgency: Literal["critical", "high", "medium", "low"] | None
    product: str | None
    issue_type: str | None
    sentiment: Literal["frustrated", "neutral", "satisfied"] | None
    detected_language: str | None
    customer_data : str | None

    # ── Populated by decide_node ───────────────────────────────────
    action: Literal["auto_respond", "escalate"] | None
    specialist_queue: str | None
    draft_response: str | None
    reasoning: str | None

    # ── Populated by specialist_node ───────────────────────────────
    specialist_notes: str | None  # technical findings (logs + technical KB summary)


# ── Structured output schemas (used by LLM calls only) ─────────────────────────

class Classification(BaseModel):
    urgency: Literal["critical", "high", "medium", "low"]
    product: str
    issue_type: str
    sentiment: Literal["frustrated", "neutral", "satisfied"]
    detected_language: str


class Decision(BaseModel):
    action: Literal["auto_respond", "escalate"]
    specialist_queue: str | None = None
    draft_response: str | None = None
    reasoning: str
