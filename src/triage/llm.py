"""
LLM factory.

Reads two environment variables:
  LLM_PROVIDER  — "openai" (default) | "gemini"

Provider defaults:
  openai  → gpt-4o-mini       (requires OPENAI_API_KEY)
  gemini  → gemini-2.5-flash  (requires GOOGLE_API_KEY)

Any LangChain-compatible chat model that supports tool-calling and
structured output can be added here by extending the match block.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


def get_llm() -> BaseChatModel:
    """Return a configured chat model based on LLM_PROVIDER / LLM_MODEL env vars."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", _PROVIDER_DEFAULTS.get(provider, ""))

    if not model:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: openai, gemini")

    match provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=0)

        case "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as e:
                raise ImportError(
                    "Gemini support requires: pip install 'support-ticket-triage-agent[gemini]'"
                ) from e
            return ChatGoogleGenerativeAI(model=model, temperature=0)

        case _:
            raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: openai, gemini")
