"""
LangGraph state graph assembly.

Flow:
  START → classify → generalist ⇄ general_tools → decide
                                                      │
                                          route_specialist ──→ specialist ⇄ specialist_tools → END
                                          auto_respond / escalate ──────────────────────────→ END

The generalist ↔ general_tools loop and specialist ↔ specialist_tools loop each
continue until the LLM produces a message with no tool_calls.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import classify_node, decide_node, generalist_node, specialist_node
from .state import TicketState
from .tools import GENERAL_TOOLS, SPECIALIST_TOOLS


def _after_generalist(state: TicketState) -> Literal["general_tools", "decide"]:
    last = state["messages"][-1]
    return "general_tools" if getattr(last, "tool_calls", None) else "decide"


def _after_decide(state: TicketState) -> Literal["specialist", "__end__"]:
    return "specialist" if state.get("action") == "escalate" else "__end__"


def _after_specialist(state: TicketState) -> Literal["specialist_tools", "__end__"]:
    last = state["messages"][-1]
    return "specialist_tools" if getattr(last, "tool_calls", None) else "__end__"


def build_graph():
    g = StateGraph(TicketState)

    g.add_node("classify", classify_node)
    g.add_node("generalist", generalist_node)
    g.add_node("general_tools", ToolNode(GENERAL_TOOLS, handle_tool_errors=True))
    g.add_node("decide", decide_node)
    g.add_node("specialist", specialist_node)
    g.add_node("specialist_tools", ToolNode(SPECIALIST_TOOLS, handle_tool_errors=True))

    g.add_edge(START, "classify")
    g.add_edge("classify", "generalist")
    g.add_conditional_edges("generalist", _after_generalist, {"general_tools": "general_tools", "decide": "decide"})
    g.add_edge("general_tools", "generalist")
    g.add_conditional_edges("decide", _after_decide, {"specialist": "specialist", "__end__": END})
    g.add_conditional_edges("specialist", _after_specialist, {"specialist_tools": "specialist_tools", "__end__": END})
    g.add_edge("specialist_tools", "specialist")

    return g.compile()
