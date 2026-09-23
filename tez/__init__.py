"""Tez: an open, local "System One" decision engine.

One frozen language model reads a state and answers many typed questions (yes/no, choice, score) from a
single next-token distribution each, with calibrated probabilities, optional per-question probes and a
conformal act/escalate gate. The server speaks TypeSafe's /v1/systemone wire format (a drop-in for Jev
clients); see docs/API.md.

    from tez import Tez
    tez = Tez(backend="http://127.0.0.1:8091", template="gemma4")
    tez.decide("Help! My payouts have been failing for 3 days.",
               questions={"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}})
"""
from ._version import RELEASE_DATE, __version__
from .backends import Backend, FakeBackend, LlamaCppBackend
from .engine import Tez
from .errors import BackendRequestError, BackendUnavailable, InvalidRequest, TezError
from .schema import Question, Schema, load_schema, load_schemas, parse_question

__all__ = [
    "__version__", "RELEASE_DATE", "Tez", "Backend", "LlamaCppBackend", "FakeBackend", "Question", "Schema",
    "load_schema", "load_schemas", "parse_question", "TezError", "InvalidRequest", "BackendUnavailable",
    "BackendRequestError",
]
