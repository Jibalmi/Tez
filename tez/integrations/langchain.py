"""LangChain / LangGraph components on top of Tez. Needs langchain-core: pip install "tez-decisions[langchain]".

  TezRouter     a LangGraph conditional-edge function (and a Runnable): returns the chosen label, or `fallback`
                when the gate escalates, when abstain picks "none of these fits", or below `min_confidence`
  TezGuard      screens text with a schema's yes/no questions (default: the prompt-injection-guard use case) and
                raises TezGuardError, filters flagged items out, or annotates
  TezTriage     answers a schema's questions about the state: {"triage": {question id: answer}}, a LangGraph update
  TezEvaluator  a LangChain-style string evaluator: evaluate_strings(prediction, input=...)

Every component runs in-process with tez=Tez(...), or over HTTP against any server that speaks the /v1/systemone
wire format with base_url=... (default http://127.0.0.1:8787), api_key=... and timeout=... (seconds).

    from langgraph.graph import START, MessagesState, StateGraph
    from tez.integrations.langchain import TezRouter

    route = TezRouter("Which team should handle this message?",
                      criteria={"billing": "Payments, invoices", "technical": "Something is broken"},
                      min_confidence=0.5, fallback="human")
    graph = StateGraph(MessagesState)
    ...
    graph.add_conditional_edges(START, route, ["billing", "technical", "human"])

What a component decides on: `state_key` when given (state[state_key]); otherwise, for a LangGraph state with
`messages`, the latest message; otherwise the input itself (a string, a message, a document, or an object that is
sent as a JSON state).
"""
from __future__ import annotations

from typing import Any

try:
    from langchain_core.messages import RemoveMessage
    from langchain_core.runnables import Runnable
except ImportError as exc:  # pragma: no cover - only without the optional dependency
    raise ImportError('tez.integrations.langchain needs langchain-core: pip install "tez-decisions[langchain]" '
                      '(or pip install "langchain-core>=0.3")') from exc

from ._logic import Evaluator, Guard, GuardReport, Route, Router, TezGuardError, Triage, make_decider
from .remote import RemoteTez

__all__ = ["TezRouter", "TezGuard", "TezGuardError", "TezTriage", "TezEvaluator", "Route", "GuardReport", "RemoteTez"]


class _Invoke:
    """invoke() runs `_run` inside LangChain's callback and tracing context; calling the object does the same.
    `name` (the class name) is what traces and LangGraph's branch names show."""

    name: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._call_with_config(self._run, input, config)  # type: ignore[attr-defined]

    def __call__(self, input: Any) -> Any:
        return self.invoke(input)

    def _run(self, input: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


def _remove_messages(messages: list) -> list | None:
    """LangGraph's add_messages reducer deletes a message when it receives RemoveMessage(id); None (fall back to the
    filtered list) when a flagged message has no id."""
    ids = [getattr(m, "id", None) for m in messages]
    if not ids or not all(isinstance(i, str) and i for i in ids):
        return None
    return [RemoveMessage(id=i) for i in ids]


class TezRouter(_Invoke, Runnable[Any, str]):
    """Route a state to a label with one Tez decision.

    TezRouter("Which team?", criteria={"billing": "...", "technical": "..."})   an inline choice question
    TezRouter({"type": "choice", "instructions": "...", "criteria": {...}})     a wire-format question
    TezRouter("Is this urgent?")                                                 yes/no: routes "true" / "false"
    TezRouter(schema="support-triage", question_id="topic")                     a schema's question, with its
                                                                                 fitted probe, calibration and gate
    The label is the choice (noul: "true"/"false"; score: the most likely level, as a string). `fallback` is returned
    instead when the answer's gate decision is "escalate" (a gate applies when alpha is set or the schema has a
    default one; alpha=False turns a schema's default off), when abstain=True and no option fits, or when the
    confidence is below min_confidence. decide(state) returns the whole Route (label, confidence, reason, answer).
    """

    def __init__(self, question: Any = None, *, criteria: Any = None, instructions: Any = None, schema: str | None = None,
                 question_id: str | None = None, alpha: Any = None, min_confidence: float | None = None,
                 abstain: bool = False, readout: str = "auto", fallback: str = "escalate", state_key: str | None = None,
                 tez: Any = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 30):
        self.router = Router(question, criteria=criteria, instructions=instructions, schema=schema,
                             question_id=question_id, alpha=alpha, min_confidence=min_confidence, abstain=abstain,
                             readout=readout, fallback=fallback, state_key=state_key,
                             decider=make_decider(tez, base_url, api_key, timeout))

    def __repr__(self) -> str:
        r = self.router
        return f"TezRouter(question_id={r.question_id!r}, schema={r.schema!r}, fallback={r.fallback!r})"

    def _run(self, state: Any) -> str:
        return self.router.route(state)

    def decide(self, state: Any) -> Route:
        """The full routing decision: label, the answer's own label, confidence, gate decision, fallback reason."""
        return self.router.decide(state)


class TezGuard(_Invoke, Runnable[Any, Any]):
    """Screen inputs with a schema's yes/no questions; an item is flagged when P(yes) >= threshold for any of them.

    schema       a loaded schema (default "prompt-injection-guard", examples/usecases/prompt-injection-guard)
    question_id  one question id or a list of them (default: every yes/no question of the schema)
    action       "raise"     return the input unchanged, or raise TezGuardError(violations, response)
                 "filter"    drop flagged items from a list (a flagged single input becomes None)
                 "annotate"  never block: {"input": input, out_key: report}
    Inputs: a string, a message, a document, or a list of them (each item is screened), or a LangGraph state: the
    value at state_key, else the latest of its `messages`. A state gets a partial update back, as a node returns:
    {} (raise), {key: filtered} (filter; flagged messages with ids become RemoveMessage) or {out_key: report}.
    check(input) returns the GuardReport without acting on it.
    """

    def __init__(self, schema: str = "prompt-injection-guard", *, question_id: Any = None, action: str = "raise",
                 threshold: float = 0.5, state_key: str | None = None, out_key: str = "tez_guard", tez: Any = None,
                 base_url: str | None = None, api_key: str | None = None, timeout: float = 30):
        self.guard = Guard(schema, question_id=question_id, action=action, threshold=threshold, state_key=state_key,
                           out_key=out_key, decider=make_decider(tez, base_url, api_key, timeout),
                           remover=_remove_messages)

    def __repr__(self) -> str:
        g = self.guard
        return f"TezGuard(schema={g.schema!r}, action={g.action!r}, threshold={g.threshold:g})"

    def _run(self, value: Any) -> Any:
        return self.guard.apply(value)

    def check(self, value: Any) -> GuardReport:
        return self.guard.check_value(value)


class TezTriage(_Invoke, Runnable[Any, dict]):
    """Answer a schema's questions (or inline ones) about the state in one request.

    Returns {out_key: {question id: answer}}, a LangGraph partial update (out_key=None returns the answers alone).
    Each answer is the wire answer (noul / choice / score), plus the question's tez block under "tez" (readout,
    decision, p_correct, calibration_id) when the server sent one. questions: a mapping of wire-format questions, or a
    list of the schema's question ids to ask only those.
    """

    def __init__(self, schema: str | None = None, *, state_key: str | None = None, out_key: str | None = "triage",
                 questions: Any = None, readout: str = "auto", abstain: bool = False, alpha: Any = None, tez: Any = None,
                 base_url: str | None = None, api_key: str | None = None, timeout: float = 30):
        self.triage = Triage(schema, state_key=state_key, out_key=out_key, questions=questions, readout=readout,
                             abstain=abstain, alpha=alpha, decider=make_decider(tez, base_url, api_key, timeout))

    def __repr__(self) -> str:
        return f"TezTriage(schema={self.triage.schema!r}, out_key={self.triage.out_key!r})"

    def _run(self, value: Any) -> dict:
        return self.triage.run(value)


class TezEvaluator(Evaluator):
    """A LangChain-style string evaluator (the StringEvaluator interface, without depending on `langchain`).

    TezEvaluator({"correct": {"type": "noul", "instructions": "Does the output answer the input correctly?"}})
    evaluate_strings(prediction, input=None, reference=None) decides the state {"input", "output", "reference"} and
    returns {"results": [{"key", "score", "value", "answer"}, ...]}, plus "score" and "value" at the top level when
    there is one question. score: noul P(yes); score questions the expected level scaled to 0..1; choice None.
    """

    def __init__(self, questions: Any, *, schema: str | None = None, readout: str = "auto", tez: Any = None,
                 base_url: str | None = None, api_key: str | None = None, timeout: float = 30):
        super().__init__(questions, schema=schema, readout=readout, decider=make_decider(tez, base_url, api_key, timeout))

    def __repr__(self) -> str:
        return f"TezEvaluator(questions={list(self._questions or self._ids or [])!r})"

    async def aevaluate_strings(self, prediction: Any = None, *, reference: Any = None, input: Any = None,
                                **kwargs: Any) -> dict:
        import asyncio
        return await asyncio.to_thread(self.evaluate_strings, prediction, reference=reference, input=input, **kwargs)
