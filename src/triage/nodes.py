"""
LangGraph node functions.

Each node has a single responsibility:
  classify_node   — structured classification (urgency, product, issue type, sentiment, language)
  generalist_node — ReAct loop: check_customer_profile + lookup_general_kb
  decide_node     — final triage decision (action + optional draft response)
  specialist_node — ReAct loop: get_recent_logs + lookup_technical_kb
                    always runs when decide_node sets action = "escalate"

classify_node and decide_node use `with_structured_output` (JSON schema mode).
generalist_node and specialist_node use `bind_tools` for tool calling.
These two LLM modes cannot coexist on one invocation, so they run in separate nodes.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console

from .llm import get_llm
from .state import Classification, Decision, TicketState
from .data_store import get_customer_profile
from .tools import GENERAL_TOOLS, SPECIALIST_TOOLS

_console = Console()

# LLMs are instantiated lazily (first call) so the module can be imported
# without API keys present — useful for tests and graph compilation.
_classify_llm = None
_generalist_llm = None
_specialist_llm = None
_decide_llm = None


def _get_llms():
    global _classify_llm, _generalist_llm, _specialist_llm, _decide_llm
    if _classify_llm is None:
        base = get_llm()
        _classify_llm = base.with_structured_output(Classification)
        _generalist_llm = base.bind_tools(GENERAL_TOOLS)
        _specialist_llm = base.bind_tools(SPECIALIST_TOOLS)
        _decide_llm = base.with_structured_output(Decision)
    return _classify_llm, _generalist_llm, _specialist_llm, _decide_llm


# ── Node 1: Classify ───────────────────────────────────────────────────────────

def classify_node(state: TicketState) -> dict:
    """Extract urgency, product, issue type, sentiment, and language from the ticket."""
    classify_llm, _, _, _ = _get_llms()

    result: Classification = classify_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a support ticket classifier. "
                    "Analyse the ticket text and extract:\n"
                    "- urgency: critical / high / medium / low\n"
                    "  (critical = production down or financial harm imminent; "
                    "high = major blocker; medium = functional issue; low = question/cosmetic)\n"
                    "- product: the product or feature area mentioned\n"
                    "- issue_type: short category, e.g. 'billing', 'login', 'outage', 'feature-request'\n"
                    "- sentiment: frustrated / neutral / satisfied\n"
                    "- detected_language: primary language of the ticket (e.g. 'English', 'Thai')\n\n"
                    "Be precise. A customer mentioning multiple charges and a deadline is 'critical'."
                )
            ),
            HumanMessage(content=state["ticket_text"]),
        ]
    )
    customer_data = get_customer_profile(state["customer_id"])

    _console.log(
        f"[bold cyan]\\[classify][/] customer=[cyan]{state['customer_id']}[/]"
        f"  urgency=[yellow]{result.urgency}[/]"
        f"  sentiment=[magenta]{result.sentiment}[/]"
        f"  lang=[green]{result.detected_language}[/]"
        f"  issue=[dim]{result.issue_type}[/]"
        # f"  plan=[dim]{customer_data.get('plan') if customer_data else 'unknown'}[/]"
    )

    return {
        "urgency": result.urgency,
        "product": result.product,
        "issue_type": result.issue_type,
        "sentiment": result.sentiment,
        "detected_language": result.detected_language,
        # "customer_data": customer_data,
    }

# ── Node 2: Generalist (ReAct loop) ───────────────────────────────────────────

def generalist_node(state: TicketState) -> dict:
    """
    Gather high-level context using general tools:
      - check_customer_profile: account plan, tenure, churn risk
      - lookup_general_kb: user-facing articles

    LangGraph re-invokes this node after ToolNode executes tool calls,
    continuing until the LLM produces a reply with no further tool calls.
    """
    has_tool_results = any(isinstance(m, ToolMessage) for m in state["messages"])

    if not state["messages"]:
        context = (
            f"Support ticket (language: {state['detected_language']}):\n"
            f"{state['ticket_text']}\n\n"
            f"Customer ID: {state['customer_id']}\n"
            f"Urgency: {state['urgency']} | Issue type: {state['issue_type']}\n\n"
            "Call check_customer_profile and lookup_general_kb once each, then stop."
        )
        init_messages = [HumanMessage(content=context)]
    elif has_tool_results:
        # Tool results are already in the conversation — force the LLM to conclude.
        init_messages = [HumanMessage(content="All tool results are now available. Do NOT call any more tools. Acknowledge the findings and stop.")]
    else:
        init_messages = []

    _, generalist_llm, _, _ = _get_llms()
    response = generalist_llm.invoke(state["messages"] + init_messages)

    if isinstance(response, AIMessage) and response.tool_calls:
        tool_names = ", ".join(tc["name"] for tc in response.tool_calls)
        _console.log(
            f"[bold cyan]\\[generalist][/] customer=[cyan]{state['customer_id']}[/]"
            f"  calling tools: [yellow]{tool_names}[/]"
        )
    else:
        _console.log(
            f"[bold cyan]\\[generalist][/] customer=[cyan]{state['customer_id']}[/]"
            f"  [green]tool calls complete[/]"
        )

    return {"messages": init_messages + [response]}


# ── Node 3: Decide ────────────────────────────────────────────────────────────

def decide_node(state: TicketState) -> dict:
    """Produce the final triage decision based on classification + generalist findings."""
    tool_results = _extract_tool_results(state["messages"])

    prompt = (
        f"You are a senior support triage agent. Based on the analysis below, "
        f"decide the correct action.\n\n"
        f"Ticket classification:\n"
        f"  Urgency:  {state['urgency']}\n"
        f"  Product:  {state['product']}\n"
        f"  Issue:    {state['issue_type']}\n"
        f"  Sentiment:{state['sentiment']}\n"
        f"  Language: {state['detected_language']}\n\n"
        f"Tool findings:\n{tool_results}\n\n"
        f"Choose one action:\n"
        f"  auto_respond — issue is self-service; provide a helpful draft_response "
        f"in the ticket's detected language\n"
        f"  escalate     — a human must handle this (urgent, complex, VIP, or needs "
        f"technical investigation); set specialist_queue to one of: "
        f"billing | technical | enterprise | general; leave draft_response null\n\n"
        f"Note: all escalated tickets are automatically enriched with technical logs and "
        f"knowledge base notes before reaching the human — no need to distinguish urgency level.\n\n"
        f"Provide clear reasoning. "
        f"For auto_respond: write draft_response in {state['detected_language']}."
    )

    _, _, _, decide_llm = _get_llms()
    result: Decision = decide_llm.invoke([HumanMessage(content=prompt)])

    _console.log(
        f"[bold cyan]\\[decide][/] customer=[cyan]{state['customer_id']}[/]"
        f"  action=[bold]{'[red]' if result.action == 'escalate' else '[green]'}{result.action}[/][/]"
        + (f"  queue=[dim]{result.specialist_queue}[/]" if result.specialist_queue else "")
    )

    return {
        "action": result.action,
        "specialist_queue": result.specialist_queue,
        "draft_response": result.draft_response,
        "reasoning": result.reasoning,
    }


# ── Node 4: Specialist (ReAct loop) ───────────────────────────────────────────

def specialist_node(state: TicketState) -> dict:
    """
    Deep technical investigation that runs whenever decide_node sets action = "escalate".

    Steps the LLM is instructed to follow:
      1. get_recent_logs    — pull raw error logs for the customer
      2. lookup_technical_kb — search engineering/ops KB using issue context + log details

    When the loop ends (no more tool_calls), the LLM's summary is stored as
    specialist_notes for the human specialist who picks up the ticket.
    """
    specialist_started = any(
        isinstance(m, HumanMessage) and "[SPECIALIST]" in m.content
        for m in state["messages"]
    )
    specialist_start_idx = next(
        (i for i, m in enumerate(state["messages"])
         if isinstance(m, HumanMessage) and "[SPECIALIST]" in m.content),
        None,
    )
    has_specialist_results = specialist_start_idx is not None and any(
        isinstance(m, ToolMessage)
        for m in state["messages"][specialist_start_idx:]
    )

    if not specialist_started:
        context = (
            f"[SPECIALIST] Technical investigation for customer {state['customer_id']}.\n\n"
            f"Ticket summary:\n"
            f"  Issue type: {state['issue_type']} | Urgency: {state['urgency']}\n"
            f"  Queue: {state['specialist_queue']}\n"
            f"  Generalist reasoning: {state['reasoning']}\n\n"
            "Call get_recent_logs then lookup_technical_kb once each (use error codes "
            "from the logs in your KB query), then write a concise technical summary: "
            "root cause hypothesis, affected services, and recommended next action."
        )
        init_messages = [HumanMessage(content=context)]
    elif has_specialist_results:
        init_messages = [HumanMessage(content="All tool results are now available. Do NOT call any more tools. Write your technical summary now.")]
    else:
        init_messages = []

    _, _, specialist_llm, _ = _get_llms()
    response = specialist_llm.invoke(state["messages"] + init_messages)

    update: dict = {"messages": init_messages + [response]}

    if isinstance(response, AIMessage) and response.tool_calls:
        tool_names = ", ".join(tc["name"] for tc in response.tool_calls)
        _console.log(
            f"[bold magenta]\\[specialist][/] customer=[cyan]{state['customer_id']}[/]"
            f"  calling tools: [yellow]{tool_names}[/]"
        )
    else:
        # Loop complete — persist the LLM's technical summary
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        update["specialist_notes"] = content
        _console.log(
            f"[bold magenta]\\[specialist][/] customer=[cyan]{state['customer_id']}[/]"
            f"  [green]investigation complete[/] — notes stored"
        )

    return update


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_tool_results(messages: list) -> str:
    parts = [m.content for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(parts) if parts else "No tool results available."
