"""Integrations around the Tez engine.

  tez.integrations.RemoteTez   the engine's decide / schema API over HTTP, for any server that speaks the
                               /v1/systemone wire format (standard library + requests; always available)
  tez.integrations.langchain   LangChain / LangGraph components: TezRouter, TezGuard, TezTriage, TezEvaluator
                               (needs langchain-core: pip install "tez-decisions[langchain]")

Importing this package never imports LangChain.
"""
from .remote import DEFAULT_URL, RemoteTez

__all__ = ["DEFAULT_URL", "RemoteTez"]
