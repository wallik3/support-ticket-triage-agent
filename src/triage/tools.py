"""
Tools available to generalist and specialist agents.

Knowledge base search uses hybrid RAG:
  - BM25 (sparse) for exact keyword/term matching
  - OpenAI embeddings (dense) for semantic similarity
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists

Two separate KBs:
  _GENERAL_KB    — user-facing articles; used by generalist (research_node)
  _TECHNICAL_KB  — engineering/ops articles; used by specialist (specialist_node)

The index is built lazily per KB on first query so module import is free.
"""

import json
import os
from functools import cache

import numpy as np
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from rich.console import Console

from .data_store import get_customer_profile as _get_customer_profile
from .data_store import get_recent_logs as _get_recent_logs

_console = Console()

# ── General knowledge base (user-facing) ──────────────────────────────────────

_GENERAL_KB: list[dict] = [
    {
        "title": "Billing & Duplicate Charges",
        "keywords": ["billing", "charge", "payment", "refund", "duplicate", "invoice", "upgrade"],
        "snippet": (
            "Duplicate charges occur when payment processing retries before the "
            "first transaction settles. Pending charges that do not complete are "
            "released within 3–5 business days. For immediate relief, email "
            "billing@company.com with your transaction IDs. Refunds are processed "
            "within 5–7 business days."
        ),
    },
    {
        "title": "Plan Upgrade Issues",
        "keywords": ["upgrade", "pro", "plan", "access", "feature", "export"],
        "snippet": (
            "If a payment succeeds but your plan shows no change, try logging out "
            "and back in. If the issue persists, contact support with your payment "
            "confirmation number. Pro export features are enabled within minutes of "
            "a confirmed upgrade."
        ),
    },
    {
        "title": "Error 500 / Service Outage",
        "keywords": ["500", "error", "outage", "down", "operational", "status", "region", "asia"],
        "snippet": (
            "Error 500 affecting multiple users usually indicates a regional infrastructure "
            "issue. Check status.company.com for live updates. If the status page shows "
            "'Operational' but you still see errors, report to support immediately — status "
            "pages can lag behind actual incidents."
        ),
    },
    {
        "title": "Dark Mode & Appearance Settings",
        "keywords": ["dark", "mode", "theme", "appearance", "light", "system", "schedule"],
        "snippet": (
            "Dark mode is available under Settings > Appearance. The 'System Default' "
            "option mirrors your OS setting; it may take up to 30 seconds to sync after "
            "an OS-level change. Scheduled dark mode (auto-switch by time) is on the "
            "product roadmap for Q3. Pro users can upvote the feature at feedback.company.com."
        ),
    },
    {
        "title": "Reset Password / Login Issues",
        "keywords": ["password", "login", "access", "sign in", "forgot"],
        "snippet": (
            "Go to the login page and click 'Forgot password'. A reset link will be "
            "sent to your registered email within 2 minutes. Check your spam folder "
            "if it doesn't arrive."
        ),
    },
    {
        "title": "Enterprise SLA & Support Commitments",
        "keywords": ["enterprise", "sla", "sre", "escalate", "critical", "response time"],
        "snippet": (
            "Enterprise plan SLA: critical issues (P1) — 15-minute initial response, "
            "1-hour resolution target. A dedicated account manager is assigned for "
            "business-impact escalations. Contact enterprise-support@company.com for "
            "direct SRE access."
        ),
    },
]

# ── Technical knowledge base (engineering/ops) ────────────────────────────────

_TECHNICAL_KB: list[dict] = [
    {
        "title": "Payment Gateway Error Code Reference",
        "keywords": [
            "CHARGE_CREATION_FAILED", "PAYMENT_METHOD_DECLINED", "idempotency",
            "stripe", "duplicate", "pending charge", "payment-service",
        ],
        "snippet": (
            "CHARGE_CREATION_FAILED: Idempotency key collision — occurs when the same "
            "customer token triggers multiple charge attempts within the settlement window. "
            "Root cause: client-side retry without exponential back-off. Action: void all "
            "pending charges via Stripe dashboard > Payments > filter by customer token, "
            "then manually activate the plan via admin panel (/admin/customers/{id}/plan). "
            "PAYMENT_METHOD_DECLINED: Card declined by issuer. Check Stripe decline code "
            "for specific reason (insufficient_funds, card_not_supported, etc.)."
        ),
    },
    {
        "title": "Error 500 / HTTP_503 Infrastructure Playbook",
        "keywords": [
            "HTTP_500", "HTTP_503", "RegionRouter", "NullPointerException",
            "asia", "shard", "canary", "deployment", "rollout", "load-balancer",
        ],
        "snippet": (
            "NullPointerException in RegionRouter.getRegion(): asia shard config missing. "
            "This occurs when a new deployment rolls out without populating the asia region "
            "config map. Immediate fix: kubectl rollout undo deployment/app-server-asia. "
            "HTTP_503 with 'canary rollout in progress': load balancer is shifting traffic; "
            "rollback canary with: kubectl set image deployment/app-server-asia. "
            "Status page lag >1h: trigger manual status update via ops-tools/status-page-cli."
        ),
    },
    {
        "title": "Plan Activation Failure (Charge Success, Plan Not Updated)",
        "keywords": [
            "PENDING_CHARGE_DETECTED", "plan activation", "billing-db",
            "billing-service", "plan not updated", "stripe webhook",
        ],
        "snippet": (
            "PENDING_CHARGE_DETECTED with no plan activation: Stripe webhook delivery "
            "may have failed (check Stripe webhook logs for 5xx responses from our endpoint). "
            "Manual recovery: confirm charge ID in Stripe, then POST to "
            "/internal/billing/activate with {customer_id, charge_id, plan}. "
            "Multiple pending charges without activation: issue refunds for all but the "
            "most recent charge, then activate plan manually."
        ),
    },
    {
        "title": "Export Service Failures (EXPORT_FAILED / EXPORT_TIMEOUT)",
        "keywords": [
            "EXPORT_FAILED", "EXPORT_TIMEOUT", "toCSV", "export-worker",
            "export", "csv", "download", "OOM", "dataset", "large",
        ],
        "snippet": (
            "EXPORT_FAILED (toCSV undefined): null dataset reference in ExportJob.run — "
            "usually caused by a query that returns no schema when dataset is empty or filter "
            "returns 0 rows. Check export-worker logs for the job_id and query params. "
            "EXPORT_TIMEOUT / OOM for large datasets (>100k rows): export-worker memory limit "
            "is 512MB; datasets >100k rows may exceed it. Workaround: POST to "
            "/internal/export/chunked with {customer_id, chunk_size: 25000} to split the job. "
            "Permanent fix: increase worker memory limit in export-worker/k8s/deployment.yaml."
        ),
    },
    {
        "title": "Frontend Feature Flag: Dark Mode for Pro Plan",
        "keywords": [
            "DARK_MODE_OPTION_MISSING", "dark_mode_pro", "feature-flag",
            "frontend-config", "ui-service", "appearance", "Pro plan",
        ],
        "snippet": (
            "DARK_MODE_OPTION_MISSING on Pro plan: feature flag 'dark_mode_pro' is disabled "
            "in the frontend config service. To enable for a specific customer: "
            "POST /internal/feature-flags/override {customer_id, flag: 'dark_mode_pro', enabled: true}. "
            "To enable globally for Pro tier: update frontend-config/flags.yaml and redeploy. "
            "Root cause: flag was inadvertently set to false during the Q1 config migration."
        ),
    },
]


# ── Hybrid RAG index ──────────────────────────────────────────────────────────
#
# Built lazily on first query, cached per KB type.
# BM25 (sparse) + OpenAI embeddings (dense) fused via Reciprocal Rank Fusion.

_RRF_K = 60


def _tokenise(text: str) -> list[str]:
    return text.lower().split()


def _get_embeddings():
    """Return the appropriate embedder based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


@cache
def _build_index(kb_type: str):
    """Build BM25 + embedding index for the given KB. Cached per kb_type."""
    kb = _GENERAL_KB if kb_type == "general" else _TECHNICAL_KB
    corpus = [f"{a['title']} {a['snippet']}" for a in kb]
    bm25 = BM25Okapi([_tokenise(doc) for doc in corpus])
    embedder = _get_embeddings()
    try:
        vecs = np.array(embedder.embed_documents(corpus), dtype=np.float32)
    except Exception as e:
        _console.log(f"[bold red]\\[_build_index][/] embed_documents failed for kb=[yellow]{kb_type}[/]: {e}")
        raise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return bm25, vecs, embedder


def _rrf(rankings: list[list[int]], k: int = _RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion — returns {doc_idx: rrf_score} sorted desc."""
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def _hybrid_search(query: str, kb_type: str, top_k: int = 3) -> list[dict]:
    """Search a KB with hybrid RAG. Returns articles augmented with _scores."""
    kb = _GENERAL_KB if kb_type == "general" else _TECHNICAL_KB

    bm25, vecs, embedder = _build_index(kb_type)

    bm25_scores = bm25.get_scores(_tokenise(query))
    sparse_ranking = list(np.argsort(bm25_scores)[::-1])

    q_vec = np.array(embedder.embed_query(query), dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)
    dense_scores = vecs @ q_vec
    dense_ranking = list(np.argsort(dense_scores)[::-1])

    rrf_scores = _rrf([sparse_ranking, dense_ranking])
    results = []
    for idx in list(rrf_scores)[:top_k]:
        article = dict(kb[idx])
        article["_scores"] = {
            "bm25": round(float(bm25_scores[idx]), 4),
            "dense": round(float(dense_scores[idx]), 4),
            "rrf": round(rrf_scores[idx], 6),
        }
        results.append(article)
    return results


# ── Tool definitions ───────────────────────────────────────────────────────────

def check_customer_profile(customer_id: str) -> str:
    """Retrieve account plan, tenure, region, seats, and churn risk for a customer.
    Use this to understand account context before deciding on an action."""
    profile = _get_customer_profile(customer_id)
    if not profile:
        _console.log(f"[bold yellow]\\[tool:check_customer_profile][/] customer=[cyan]{customer_id}[/]  [red]not found[/]")
        return f"No record found for customer '{customer_id}'."

    _console.log(
        f"[bold yellow]\\[tool:check_customer_profile][/] customer=[cyan]{customer_id}[/]"
        f"  plan=[magenta]{profile.get('plan')}[/]"
        f"  region=[dim]{profile.get('region')}[/]"
        f"  tenure=[dim]{profile.get('tenure_months')}mo[/]"
        f"  churn_risk=[{'red' if profile.get('churn_risk') == 'high' else 'green'}]{profile.get('churn_risk')}[/]"
    )
    return json.dumps(profile, indent=2, default=str)


@tool
def lookup_general_kb(query: str) -> str:
    """Search the general knowledge base for user-facing articles (billing FAQs,
    appearance settings, login help, upgrade steps, SLA commitments).
    Uses hybrid RAG (BM25 + embeddings fused via RRF). Returns top 3 articles."""
    _console.log(f"[bold yellow]\\[tool:general_kb][/] query=[dim]{query!r}[/]")
    hits = _hybrid_search(query, kb_type="general", top_k=3)
    if not hits:
        _console.log(f"[bold yellow]\\[tool:general_kb][/] query=[dim]{query!r}[/]  [red]no hits[/]")
        return "No relevant articles found."

    for rank, h in enumerate(hits, start=1):
        s = h["_scores"]
        _console.log(
            f"[bold yellow]\\[tool:general_kb][/] #{rank} [green]{h['title']}[/]"
            f"  bm25=[dim]{s['bm25']}[/]  dense=[dim]{s['dense']}[/]  rrf=[dim]{s['rrf']}[/]"
        )
    return "\n\n---\n\n".join(f"**{h['title']}**\n{h['snippet']}" for h in hits)


@tool
def get_recent_logs(customer_id: str) -> str:
    """Retrieve recent technical error logs for a customer from the error log store.
    Returns log entries with timestamp, level, service, error_code, message, and
    affected_component. Use this to diagnose the root cause before searching the
    technical knowledge base."""
    logs = _get_recent_logs(customer_id)
    if not logs:
        _console.log(f"[bold yellow]\\[tool:recent_logs][/] customer=[cyan]{customer_id}[/]  [dim]no logs[/]")
        return f"No error logs found for customer '{customer_id}'."

    error_codes = [l["error_code"] for l in logs if l.get("error_code")]
    services = list({l["service"] for l in logs if l.get("service")})
    _console.log(
        f"[bold yellow]\\[tool:recent_logs][/] customer=[cyan]{customer_id}[/]"
        f"  entries=[dim]{len(logs)}[/]"
        f"  services=[yellow]{', '.join(services)}[/]"
        f"  codes=[red]{', '.join(error_codes)}[/]"
    )
    return json.dumps(logs, indent=2, default=str)


@tool
def lookup_technical_kb(query: str) -> str:
    """Search the technical knowledge base for engineering/ops articles
    (error code references, infrastructure playbooks, feature flag procedures).
    Pass query enriched with error codes and log details from get_recent_logs
    for best results. Uses hybrid RAG (BM25 + embeddings fused via RRF)."""
    _console.log(f"[bold yellow]\\[tool:technical_kb][/] query=[dim]{query!r}[/]")
    hits = _hybrid_search(query, kb_type="technical", top_k=3)
    if not hits:
        _console.log(f"[bold yellow]\\[tool:technical_kb][/] query=[dim]{query!r}[/]  [red]no hits[/]")
        return "No relevant technical articles found."

    for rank, h in enumerate(hits, start=1):
        s = h["_scores"]
        _console.log(
            f"[bold yellow]\\[tool:technical_kb][/] #{rank} [green]{h['title']}[/]"
            f"  bm25=[dim]{s['bm25']}[/]  dense=[dim]{s['dense']}[/]  rrf=[dim]{s['rrf']}[/]"
        )
    return "\n\n---\n\n".join(f"**{h['title']}**\n{h['snippet']}" for h in hits)


# ── Tool lists exported to nodes ───────────────────────────────────────────────

GENERAL_TOOLS = [check_customer_profile, lookup_general_kb]
SPECIALIST_TOOLS = [get_recent_logs, lookup_technical_kb]
