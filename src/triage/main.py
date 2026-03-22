"""
CLI entry-point.  Run with:
    python -m triage.main
or after `pip install -e .`:
    triage
"""

from __future__ import annotations

import os
import sys
import textwrap

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .graph import build_graph
from .state import TicketState

load_dotenv(override=True)
console = Console()

# ── Sample tickets (from assignment) ──────────────────────────────────────────

SAMPLE_TICKETS: list[dict] = [
    {
        "customer_id": "C001",
        "label": "Payment failure / duplicate charges",
        "ticket_text": textwrap.dedent("""
            [Message 1 - 3 hours ago]
            My payment failed when I tried to upgrade to Pro. Can you check what's wrong?

            [Message 2 - 2 hours ago]
            I tried again with a different card. Now I see TWO pending charges but my
            account still shows Free plan??

            [Message 3 - 1 hour ago]
            Okay this is getting ridiculous. Just checked my bank app - I have THREE
            charges of $29.99 now. None of them refunded. And I STILL don't have Pro access.

            [Message 4 - just now]
            HELLO?? Is anyone there??? I need this fixed NOW. I have a presentation in 2
            hours and I need the Pro export features. If these charges aren't reversed by
            end of day I'm disputing all of them with my bank.
        """).strip(),
    },
    {
        "customer_id": "C002",
        "label": "Enterprise outage (Thai) / Error 500",
        "ticket_text": textwrap.dedent("""
            [Message 1 - 2 hours ago]
            ระบบเข้าไม่ได้ครับ ขึ้น error 500
            (Can't access the system, showing error 500)

            [Message 2 - 1.5 hours ago]
            ลองหลายเครื่องแล้ว ทั้ง Chrome, Safari, Firefox ผลเหมือนกันหมด
            เพื่อนร่วมงานก็เข้าไม่ได้เหมือนกัน
            (Tried multiple machines - Chrome, Safari, Firefox - same result.
            Coworkers also can't access)

            [Message 3 - 45 mins ago]
            ตอนนี้ลูกค้าโวยเข้ามาเยอะมาก เรามี demo กับลูกค้ารายใหญ่บ่ายนี้
            ถ้าระบบไม่กลับมา deal นี้อาจจะหลุด
            (Customers are flooding in with complaints now. We have a demo with a
            major client this afternoon. If the system doesn't come back, we might
            lose this deal)

            [Message 4 - just now]
            เช็ค status.company.com แล้ว บอกว่า all systems operational
            แต่เราใช้งานไม่ได้จริงๆ ช่วยเช็คให้หน่อยได้ไหมครับ region Asia มีปัญหาหรือเปล่า?
            (Checked status.company.com - it says all systems operational, but we
            really can't use it. Can you please check? Is there an issue with the
            Asia region?)
        """).strip(),
    },
    {
        "customer_id": "C003",
        "label": "Dark mode bug / feature request",
        "ticket_text": textwrap.dedent("""
            [Message 1 - 2 days ago]
            Hey, just wondering if you support dark mode? No rush 😊\

            [Message 2 - 1 day ago]
            Thanks for the reply! Oh nice, so it's in Settings > Appearance.
            Found it! But hmm I'm on Pro plan and I only see 'Light' and 'System
            Default' options. No dark mode toggle?

            [Message 3 - 1 day ago, 3 hours later]
            Okay so I switched to 'System Default' and my Mac is set to dark mode,
            but your app still shows light theme. Is this a bug or am I missing
            something?

            [Message 4 - today]
            Also random question while I have you - is there a way to schedule dark
            mode? Like auto-switch at 6pm? Some apps have that. Would be cool if
            you guys added it 🌙
        """).strip(),
    },
    {
        "customer_id": "C004",
        "label": "How to set dark mode",
        "ticket_text": textwrap.dedent("""
            [Message 1 - just now]
            Tell me how to set dark mode
        """).strip(),
    },
    {
        "customer_id": "C005",
        "label": "CSV export keeps failing with error",
        "ticket_text": textwrap.dedent("""
            [Message 1 - 5 hours ago]
            Hey team, the CSV export feature is broken for me. Every time I click
            "Export to CSV" it just shows a spinner and then fails with a generic
            error message. No download starts.

            [Message 2 - 3 hours ago]
            Tried exporting smaller date ranges — same issue. Also tried Excel format,
            that fails too. Seems like anything in the export menu is broken. I've
            been using this feature daily for almost a year, never had this before.

            [Message 3 - 1 hour ago]
            I need this data for a report I'm presenting tomorrow morning. This is
            really blocking me. My dataset is about 142k rows — could that be causing
            it? Happy to provide more details if it helps diagnose.

            [Message 4 - just now]
            Still broken. Is this a known issue? Any ETA on a fix or a workaround?
            Even a partial export would help at this point.
        """).strip(),
    },
]

# ── Styling helpers ────────────────────────────────────────────────────────────

_URGENCY_STYLE = {
    "critical": "bold red",
    "high": "bold yellow",
    "medium": "yellow",
    "low": "green",
}
_ACTION_STYLE = {
    "auto_respond": "bold green",
    "escalate": "bold red",
}


def _styled(value: str | None, style_map: dict) -> Text:
    if not value:
        return Text("-", style="dim")
    return Text(value, style=style_map.get(value, "white"))


def _wrap(text: str | None, width: int = 80) -> str:
    if not text:
        return "-"
    return "\n".join(textwrap.wrap(text, width=width))


def print_result(state: dict, ticket_num: int, label: str) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    table.add_column("Field", style="bold cyan", min_width=18, no_wrap=True)
    table.add_column("Value", overflow="fold")

    table.add_row("Language", state.get("detected_language") or "-")
    table.add_row("Product", state.get("product") or "-")
    table.add_row("Issue Type", state.get("issue_type") or "-")
    table.add_row("Sentiment", state.get("sentiment") or "-")
    table.add_row("Urgency", _styled(state.get("urgency"), _URGENCY_STYLE))
    table.add_row("Action", _styled(state.get("action"), _ACTION_STYLE))

    if state.get("specialist_queue"):
        table.add_row("Queue", state["specialist_queue"])

    if state.get("draft_response"):
        table.add_row("Draft Response", _wrap(state["draft_response"]))

    table.add_row("Reasoning", _wrap(state.get("reasoning") or "-"))

    if state.get("specialist_notes"):
        table.add_row("Specialist Notes", _wrap(state["specialist_notes"]))

    console.print(
        Panel(
            table,
            title=f"[bold]Ticket #{ticket_num}: {label}[/]",
            border_style="blue",
        )
    )


# ── Initial state factory ──────────────────────────────────────────────────────

def _initial_state(ticket: dict) -> TicketState:
    return {
        "ticket_text": ticket["ticket_text"],
        "customer_id": ticket["customer_id"],
        "messages": [],
        "urgency": None,
        "product": None,
        "issue_type": None,
        "sentiment": None,
        "detected_language": None,
        "customer_data": None,
        "action": None,
        "specialist_queue": None,
        "draft_response": None,
        "reasoning": None,
        "specialist_notes": None,
    }


# ── Entry-point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        console.print(
            "[bold red]Error:[/] OPENAI_API_KEY or GOOGLE_API_KEY is not set.\n"
            "Copy [dim].env.example[/] to [dim].env[/] and add your key."
        )
        sys.exit(1)

    else:
        load_dotenv(override=True)

    graph = build_graph()
    total = len(SAMPLE_TICKETS)

    for i, ticket in enumerate(SAMPLE_TICKETS, start=1):
        console.rule(f"[bold blue]Ticket {i}/{total}: {ticket['label']}")
        console.print(
            Panel(
                ticket["ticket_text"],
                title="[dim]Raw ticket[/]",
                border_style="dim",
            )
        )
        console.print("[dim]Processing...[/]")
        console.log(
            f"[bold]\\[ticket {i}/{total}][/] id=[cyan]{ticket['customer_id']}[/]"
            f"  label=[dim]{ticket['label']}[/]"
        )

        result = graph.invoke(_initial_state(ticket))
        print_result(result, i, ticket["label"])
        console.print()


if __name__ == "__main__":
    main()
