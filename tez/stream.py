"""Streaming decisions: one session per utterance, fed word by word (the voice loop's hot path).

    tez = Tez(backend="inproc:gemma-4-12b-it-Q8_0.gguf", schemas="data/voice/schema.yaml")
    session = tez.stream(schema="voice-desktop", on_action=run)
    for word in words:                 # complete words
        u = session.feed(word)         # u.decision, u.confidence, u.committed, u.action
    u = session.update("open no")      # or the recogniser's text as it grows: the last word may still change
    u = session.end()                  # end of utterance: deferred actions, approval requests, `none` hand-off
    session.reset()                    # next utterance (the constant prefix stays evaluated)

A session answers one choice question of a schema (its question_first prompt: instructions and options first, the
transcript last) after every word. Each step:
  1. decision  the transcript so far is the state. With the in-process backend (tez.inproc) the question's constant
               prefix is evaluated once; per word a copy of the committed sequence receives the new word plus the answer
               tail, and only those tokens are evaluated (InprocStreamReader). The answer is read by
               Tez.decide_question itself, so letters, probes, blends, `__none__`, the fitted temperature and, for
               unfitted questions, Tez.unfitted_temperature apply exactly as in a /v1/systemone request.
  2. policy    the class-aware early-commit rules measured in experiments/stream_policy.py, read from the schema's
               `actions:` block (Policy): what fires now, if anything. `on_action` is called at once.
  3. commit    the word joins the committed sequence, after the action went out and before the next word arrives.
Any other backend (llama-server over HTTP, FakeBackend) gets the whole prompt per word; llama-server's prompt cache
then reuses the prefix.

Action classes (the schema's `actions:` block, one entry per option; see data/voice/schema.yaml):
  R0 instant, reversible   fires at the first step with p >= tau held for `stable` consecutive steps (default 1)
  R1 refinable             after `stable` consecutive steps (default 2) its ancestor (`refines`) fires early; the action
                           itself fires at the end of the utterance, when its slots are complete
  R2 deferred              end of utterance only (the default for options the block does not list)
  R3 gated                 never executed: at the end it becomes an approval request (Action.gated)
`none` on a prefix means keep listening. `none` at the end means nothing to do here: Update.handoff tells the caller
to pass the transcript to its conversational path. `tau` is on the scale of the probabilities the session reports,
which follow Tez's temperature rules, so a threshold chosen at temperature 1 (stream_policy.py's 0.9 on Gemma 4 12B
Q8_0 letters) must be chosen again when a default temperature applies (BENCHMARKS.md section 5c).

Undo: session.undo() retracts the last action that fired (on_undo is called with it); Policy(retract=True) also
retracts, at the end, every early action the final decision does not confirm (neither the same action nor an ancestor
of it). Policy(residual=True) reads what follows an early action as a new decision with a line naming the action
already taken (experiments/residual_check.py: 22/22 compound commands with that line, 20/22 without).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .engine import ReadAhead, Tez
from .errors import InvalidRequest
from .prompt import QUESTION_FIRST, build_prompt, needs_tournament, render_state
from .schema import NONE_KEY, Question, Schema

CLASSES = ("R0", "R1", "R2", "R3")
CLASS_NAMES = {"instant": "R0", "refinable": "R1", "deferred": "R2", "gated": "R3"}
DEFAULT_STABLE = {"R0": 1, "R1": 2, "R2": 1, "R3": 1}
DEFAULT_TAU = 0.9
NONE_KEYS = frozenset({"none", NONE_KEY})
PRIOR_LINE = "Already done earlier in this utterance: {prior}. The transcript below is what followed.\n"
_SENTINEL = "\u0000TEZ-STREAM-STATE\u0000"


# ---------------------------------------------------------------------------------------------- policy
@dataclass(frozen=True)
class ActionSpec:
    """How one option may fire while the user is still speaking."""

    name: str
    cls: str = "R2"
    stable: int = 1              # consecutive steps with this answer before it fires (R0) or opens its ancestor (R1)
    tau: float | None = None     # this option's threshold (None: the policy's)
    refines: str | None = None   # the ancestor it refines: opened early for R1; for scoring, consistent with this one


def parse_action(name: str, raw: Any) -> ActionSpec:
    """One entry of a schema's `actions:` block: {class: R0|R1|R2|R3 (or instant|refinable|deferred|gated),
    stable: n, tau: p, refines: option}, or just the class as a string."""
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        raw = {"class": raw}
    if not isinstance(raw, Mapping):
        raise InvalidRequest(f"actions.{name} must be a mapping like {{class: R0, stable: 2}} or a class name")
    unknown = sorted(set(raw) - {"class", "stable", "tau", "refines", "slots", "run", "say"})
    if unknown:
        raise InvalidRequest(f"actions.{name}: unknown settings {unknown}")
    cls = str(raw.get("class", "R2"))
    cls = CLASS_NAMES.get(cls.lower(), cls.upper())
    if cls not in CLASSES:
        raise InvalidRequest(f"actions.{name}.class must be one of R0, R1, R2, R3 (instant, refinable, deferred, gated)")
    stable = raw.get("stable", DEFAULT_STABLE[cls])
    if isinstance(stable, bool) or not isinstance(stable, int) or stable < 1:
        raise InvalidRequest(f"actions.{name}.stable must be a whole number of at least 1")
    tau = raw.get("tau")
    if tau is not None and (isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0 <= tau <= 1):
        raise InvalidRequest(f"actions.{name}.tau must be a probability between 0 and 1")
    refines = raw.get("refines")
    if refines is not None and not isinstance(refines, str):
        raise InvalidRequest(f"actions.{name}.refines must be the id of another option")
    return ActionSpec(name=name, cls=cls, stable=int(stable), tau=None if tau is None else float(tau), refines=refines)


@dataclass
class Policy:
    """The class-aware early-commit policy (see the module docstring). tau: the threshold on the decision's probability
    for early actions. default_class: the class of options without an entry (R2: nothing fires early unless declared).
    end_tau: the probability the final decision needs to act at the end (0: always, as measured). retract: undo, at
    the end, early actions the final decision does not confirm. residual: read what follows an early action as a new
    decision with a prior line."""

    tau: float = DEFAULT_TAU
    actions: dict[str, ActionSpec] = field(default_factory=dict)
    default_class: str = "R2"
    end_tau: float = 0.0
    retract: bool = False
    residual: bool = False

    @classmethod
    def from_schema(cls, schema: Schema | None, question: Question | None = None, **kw: Any) -> Policy:
        raw = (schema.actions if schema is not None else None) or {}
        actions = {name: parse_action(name, spec) for name, spec in raw.items()}
        if question is not None and question.type == "choice":
            known = set(question.criteria) | set(NONE_KEYS)
            for spec in actions.values():
                if spec.name not in known:
                    raise InvalidRequest(f"actions.{spec.name} is not an option of question '{question.id}'")
                if spec.refines is not None and spec.refines not in known:
                    raise InvalidRequest(f"actions.{spec.name}.refines: '{spec.refines}' is not an option of "
                                         f"question '{question.id}'")
        return cls(actions=actions, **kw)

    def spec(self, name: str) -> ActionSpec:
        s = self.actions.get(name)
        if s is None:
            s = ActionSpec(name=name, cls=self.default_class, stable=DEFAULT_STABLE[self.default_class])
            self.actions[name] = s
        return s

    def threshold(self, name: str) -> float:
        s = self.spec(name)
        return self.tau if s.tau is None else s.tau

    def ancestors(self, name: str) -> list[str]:
        """The chain of options `name` refines (nearest first; cycles stop)."""
        out, seen = [], {name}
        cur = self.spec(name).refines if name in self.actions else None
        while cur is not None and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = self.actions[cur].refines if cur in self.actions else None
        return out

    def consistent(self, done: str, final: str) -> bool:
        """An action that fired is consistent with a final decision when it is that decision or one of its ancestors."""
        return done == final or done in self.ancestors(final)

    def early(self, decision: str, p: float, stable: int, live: list[str], none_keys=NONE_KEYS) -> tuple[str, str] | None:
        """What fires while the user is still speaking, given the decision after a word, its probability, its stability
        in words and the actions already live (in order): (kind, option) or None. experiments/stream_policy.py's
        executed_sequence, one step at a time."""
        if decision in none_keys or p < self.threshold(decision):
            return None
        spec = self.spec(decision)
        if spec.cls in ("R2", "R3"):
            return None
        if spec.cls == "R1":
            anc = spec.refines
            if anc is None or stable < spec.stable or anc in live:
                return None
            return ("ancestor", anc)
        if stable < spec.stable or (live and live[-1] == decision):
            return None
        return ("early", decision)

    def at_end(self, decision: str, p: float, live: list[str], none_keys=NONE_KEYS) -> tuple[str, str] | None:
        """What the final decision does at the end of the utterance: ("end" | "approval", option), ("handoff", None)
        for none (or below end_tau), or None when it is already the last live action."""
        if decision in none_keys or p < self.end_tau:
            return ("handoff", None)
        if live and live[-1] == decision:
            return None
        return ("approval" if self.spec(decision).cls == "R3" else "end", decision)

    def replay(self, steps: list[tuple[str, float, int]], none_keys=NONE_KEYS) -> list[tuple[int, int, str, str]]:
        """The actions a sequence of steps (decision, p, word count) fires, as (step index, word, kind, option): the
        policy alone, on logged decisions (the replay harness re-scores thresholds with it); an end action carries
        the last step's index. Stability counts words, the latest reading of each word standing for it. No
        retraction or residual segments (those change what is read)."""
        by_k: dict[int, tuple[str, int]] = {}
        live: list[str] = []
        out: list[tuple[int, int, str, str]] = []
        for i, (decision, p, k) in enumerate(steps):
            for j in [j for j in by_k if j >= k]:
                del by_k[j]
            prev = by_k.get(k - 1)
            stable = prev[1] + 1 if prev is not None and prev[0] == decision else 1
            by_k[k] = (decision, stable)
            got = self.early(decision, p, stable, live, none_keys)
            if got is not None:
                live.append(got[1])
                out.append((i, k, got[0], got[1]))
        if steps:
            decision, p, k = steps[-1]
            got = self.at_end(decision, p, live, none_keys)
            if got is not None and got[0] != "handoff":
                out.append((len(steps) - 1, k, got[0], got[1]))
        return out


@dataclass
class Action:
    """An action the session decided to take (or to retract).

    kind   early     fired on a prefix (R0)
           ancestor  the early action an R1 decision implies (its `refines`)
           end       fired at the end of the utterance (R1, R2, or anything the final decision names)
           approval  an R3 decision at the end: not executed, an approval request (gated is True)
           undo      the retraction of an earlier action (`undoes`)"""

    name: str
    kind: str
    cls: str
    word: int                    # the word count at which it was decided (0 = before any word)
    p: float                     # the probability of the decision that fired it
    transcript: str
    gated: bool = False
    refines: str | None = None
    undoes: Action | None = None
    undone: bool = False


@dataclass
class Update:
    """What one word (or the end of the utterance) changed.

    decision     the most probable option after this word (its key, `none` / `__none__` included)
    confidence   its probability, as Tez reports it (temperature, probe, blend: see the module docstring)
    committed    True when an action fired at this step (action is the last of them, actions all of them)
    stable       consecutive words whose latest reading gave this decision
    partial      the last word may still grow (update() without final): it was read, not committed
    handoff      at the end: the final decision is none (or below the policy's end_tau), so nothing acts at the end
                 and the caller's conversational path should take the transcript
    timings      decide_ms (word in -> decision), policy_ms, dispatch_ms (on_action callbacks), commit_ms (the word
                 joining the committed sequence, after the dispatch), tokens (evaluated for the decision)"""

    k: int
    word: str | None
    transcript: str
    state: str
    decision: str
    confidence: float
    probabilities: dict[str, float]
    none: bool
    committed: bool
    action: Action | None
    actions: list[Action]
    stable: int
    final: bool = False
    handoff: bool = False
    partial: bool = False
    readout: str = "letters"
    meta: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------- the session
def _split(prompt: str) -> tuple[str, str]:
    if prompt.count(_SENTINEL) != 1:
        raise InvalidRequest("the question's prompt does not end with the state (streaming needs question_first)")
    head, tail = prompt.split(_SENTINEL)
    return head, tail


class StreamSession:
    """A streaming decision over one choice question (see the module docstring). Made by Tez.stream()."""

    def __init__(self, engine: Tez, schema: Schema | str | None = None, question: str | Question | None = None, *,
                 policy: Policy | None = None, readout: str = "auto", alpha: float | None = None,
                 on_action: Callable[[Action], Any] | None = None, on_undo: Callable[[Action], Any] | None = None,
                 incremental: bool = True, clock: Callable[[], float] = time.perf_counter):
        self.engine = engine
        if isinstance(schema, str):
            if schema not in engine.schemas:
                raise InvalidRequest(f"unknown schema '{schema}'")
            schema = engine.schemas[schema]
        self.schema: Schema | None = schema
        self.question = self._pick_question(schema, question)
        q = self.question
        if q.type != "choice":
            raise InvalidRequest(f"a stream answers a choice question; '{q.id}' is {q.type}")
        self.none_keys = NONE_KEYS & (set(q.criteria) | {NONE_KEY})
        self.abstain = not any(k in NONE_KEYS for k in q.criteria)
        self.policy = policy if policy is not None else Policy.from_schema(schema, q)
        self.readout = readout
        self.alpha = alpha
        self.on_action = on_action
        self.on_undo = on_undo
        self.clock = clock
        self.setup = engine.question_setup(q, schema, readout, self.abstain, QUESTION_FIRST, 1)
        if self.setup.layout != QUESTION_FIRST:
            raise InvalidRequest("streaming reads question_first prompts")
        self._parts: list[tuple[str, str, str, int, bool]] = []     # (kind, head, tail, k, embed)
        s = self.setup
        if s.use_probe and engine.embedder is engine.backend:
            head, tail = _split(engine.probe_prompt(q, _SENTINEL, s.shots_probe, layout=QUESTION_FIRST))
            self._parts.append(("embed", head, tail, 0, True))
        if s.needs_letters and not needs_tournament(len(s.options)):
            head, tail = _split(build_prompt(q, _SENTINEL, engine.template, s.options, s.shots_letters, QUESTION_FIRST))
            same = [i for i, (_, h, t, _, _) in enumerate(self._parts) if (h, t) == (head, tail)]
            if same:                          # the probe reads the letters prompt itself: one sequence serves both
                i = same[0]
                self._parts[i] = ("both", head, tail, len(s.options), True)
            else:
                self._parts.append(("letters", head, tail, len(s.options), False))
        self._reader = None
        if incremental and self._parts and hasattr(engine.backend, "stream_reader"):
            self._reader = engine.backend.stream_reader([(h, t, k, e) for _, h, t, k, e in self._parts])
        self.reset()

    @staticmethod
    def _pick_question(schema: Schema | None, question: str | Question | None) -> Question:
        if isinstance(question, Question):
            return question
        if schema is None:
            raise InvalidRequest("a stream needs a schema (or a Question)")
        if question is None:
            choices = [q for q in schema.questions.values() if q.type == "choice"]
            if len(choices) != 1:
                raise InvalidRequest(f"schema '{schema.name}' has {len(choices)} choice questions: name the question")
            return choices[0]
        if question not in schema.questions:
            raise InvalidRequest(f"schema '{schema.name}' has no question '{question}'")
        return schema.questions[question]

    # ------------------------------------------------------------------ lifecycle
    @property
    def incremental(self) -> bool:
        return self._reader is not None

    def load(self) -> StreamSession:
        """Load the model and evaluate the constant prefix now, not on the first word."""
        if self._reader is not None:
            self._reader.load()
        return self

    def reset(self) -> None:
        """Start a new utterance with the same question, policy and callbacks."""
        self.words: list[str] = []
        self.actions: list[Action] = []
        self.updates: list[Update] = []
        self._cursor = 0              # the first word of the current segment (residual mode)
        self._prior: str | None = None
        self._by_k: dict[int, tuple[str, int]] = {}   # word count -> (latest decision, stability in words)
        self._final = False
        self._ended = False
        if self._reader is not None and self._reader._seqs:
            self._reader.reset()

    def close(self) -> None:
        """Free the backend sequences this session holds."""
        if self._reader is not None:
            self._reader.close()

    @property
    def transcript(self) -> str:
        return " ".join(self.words)

    def state(self) -> str:
        """The state the question reads now: the current segment, after the prior line in residual mode."""
        seg = " ".join(self.words[self._cursor:])
        return seg if self._prior is None else PRIOR_LINE.format(prior=self._prior) + seg

    # ------------------------------------------------------------------ steps
    def feed(self, word: str) -> Update:
        """One more complete word: decide, apply the policy (on_action fires here), then commit the word."""
        word = str(word).strip()
        if not word or len(word.split()) != 1:
            raise InvalidRequest("feed() takes one non-empty word")
        u = self.update(" ".join(self.words + [word]), final=True)
        assert u is not None
        return u

    def update(self, text: str, final: bool = False) -> Update | None:
        """The transcript so far, as a streaming recogniser emits it: every word but the last is complete, the last may
        still grow ("note" -> "notes") unless final. Decides at once on the text as it stands (a decisive word is acted
        on as soon as it is heard, not when the next word shows it complete), then commits the complete words only, so
        a growing last word is re-read on the scratch copy instead of forcing a rebuild. Stability counts words: a
        decision is stable over the words whose latest reading gave it. None when the text did not change."""
        if self._ended:
            raise InvalidRequest("this utterance has ended: call reset() first")
        words = str(text).split()
        if not words:
            raise InvalidRequest("update() needs at least one word")
        if words == self.words and final == self._final:
            return None
        t0 = self.clock()
        self.words, self._final = words, bool(final)
        k = len(words)
        state = self.state()
        r, reads_meta = self._decide(state)
        t1 = self.clock()
        decision, p, probs = self._answer(r)
        for j in [j for j in self._by_k if j >= k]:
            del self._by_k[j]
        prev = self._by_k.get(k - 1)
        stable = prev[1] + 1 if prev is not None and prev[0] == decision and k - 1 > self._cursor else 1
        self._by_k[k] = (decision, stable)
        fired = self._policy_step(k, decision, p, stable)
        t2 = self.clock()
        for a in fired:
            self._dispatch(a)
        t3 = self.clock()
        if fired and self.policy.residual and fired[-1].kind in ("early", "ancestor"):
            self._cursor, self._prior = k, fired[-1].name
        commit_n = self._commit()
        t4 = self.clock()
        u = Update(k=k, word=words[-1], transcript=self.transcript, state=state, decision=decision, confidence=p,
                   probabilities=probs, none=decision in self.none_keys, committed=bool(fired),
                   action=fired[-1] if fired else None, actions=list(self.actions), stable=stable, readout=r.readout,
                   meta=dict(r.meta), partial=not final,
                   timings={"decide_ms": (t1 - t0) * 1000.0, "policy_ms": (t2 - t1) * 1000.0,
                            "dispatch_ms": (t3 - t2) * 1000.0, "commit_ms": (t4 - t3) * 1000.0,
                            "commit_tokens": commit_n, **reads_meta})
        self.updates.append(u)
        return u

    def end(self) -> Update:
        """The utterance ended: the final decision acts (R1, R2), asks (R3), or hands off (none)."""
        if self._ended:
            raise InvalidRequest("end() was already called for this utterance")
        self._ended = True
        t0 = self.clock()
        last = self.updates[-1] if self.updates else None
        decision = last.decision if last else (NONE_KEY if self.abstain else "none")
        p = last.confidence if last else 1.0
        fired: list[Action] = []
        handoff = False
        if self.policy.retract and decision not in self.none_keys:
            for a in list(self.actions):
                if a.kind in ("early", "ancestor") and not a.undone and not self.policy.consistent(a.name, decision):
                    fired.append(self._undo_action(a))
        live = [a for a in self.actions if a.kind != "undo" and not a.undone]
        got = self.policy.at_end(decision, p, [a.name for a in live], self.none_keys)
        if got is not None and got[0] == "handoff":
            handoff = True
        elif got is not None:
            spec = self.policy.spec(decision)
            a = Action(name=decision, kind=got[0], cls=spec.cls, word=len(self.words), p=p, transcript=self.transcript,
                       gated=got[0] == "approval", refines=spec.refines)
            self.actions.append(a)
            fired.append(a)
        t1 = self.clock()
        for a in fired:
            if a.kind != "undo":
                self._dispatch(a)
        t2 = self.clock()
        u = Update(k=len(self.words), word=None, transcript=self.transcript, state=self.state(), decision=decision,
                   confidence=p, probabilities=dict(last.probabilities) if last else {},
                   none=decision in self.none_keys, committed=any(a.kind != "undo" for a in fired),
                   action=fired[-1] if fired else None, actions=list(self.actions), stable=last.stable if last else 0,
                   final=True, handoff=handoff, readout=last.readout if last else "letters",
                   timings={"policy_ms": (t1 - t0) * 1000.0, "dispatch_ms": (t2 - t1) * 1000.0})
        self.updates.append(u)
        return u

    def undo(self, action: Action | None = None) -> Action | None:
        """Retract an action (default: the last one that fired and is not undone). Returns the undo, or None."""
        if action is None:
            live = [a for a in self.actions if a.kind != "undo" and not a.undone and not a.gated]
            if not live:
                return None
            action = live[-1]
        if action.undone or action.kind == "undo":
            return None
        return self._undo_action(action)

    # ------------------------------------------------------------------ internals
    def _undo_action(self, a: Action) -> Action:
        a.undone = True
        u = Action(name=a.name, kind="undo", cls=a.cls, word=len(self.words), p=a.p, transcript=self.transcript,
                   undoes=a)
        self.actions.append(u)
        if self.on_undo is not None:
            self.on_undo(u)
        return u

    def _dispatch(self, a: Action) -> None:
        if self.on_action is not None:
            self.on_action(a)

    def _decide(self, state: str) -> tuple[Any, dict]:
        reads = None
        meta: dict[str, Any] = {}
        if self._reader is not None:
            text = render_state(state)
            got = self._reader.read(text)
            reads = ReadAhead()
            for (_, head, tail, _, _), (ls, emb) in zip(self._parts, got):
                prompt = head + text + tail
                if ls is not None:
                    reads.letters[prompt] = (ls, 0)
                if emb is not None:
                    reads.embeds[prompt] = (emb, 0)
                timings = (ls or emb).timings or {}
                meta["tokens"] = timings.get("batch_prompt_n")
                meta["read_ms"] = timings.get("prompt_ms")
                meta["restored"] = timings.get("restored")
            reads.batches.append({"kind": "stream", "prompts": len(got)})
        r = self.engine.decide_question(self.question, state, self.schema, self.readout, self.abstain, self.alpha,
                                        QUESTION_FIRST, 1, reads=reads)
        return r, meta

    def _answer(self, r: Any) -> tuple[str, float, dict[str, float]]:
        p = np.asarray(r.probabilities, dtype=float)
        j = int(np.argmax(p))
        return r.keys[j], float(p[j]), {k: float(v) for k, v in zip(r.keys, p)}

    def _commit(self) -> int:
        """Commit the complete words of the current state (all of them after a final word)."""
        if self._reader is None:
            return 0
        if self._final:
            return self._reader.commit(render_state(self.state()))
        keep = self.words[:-1]
        seg = " ".join(keep[self._cursor:])
        state = seg if self._prior is None else PRIOR_LINE.format(prior=self._prior) + seg
        if not state:
            return 0
        return self._reader.commit(render_state(state))

    def _policy_step(self, k: int, decision: str, p: float, stable: int) -> list[Action]:
        """Policy.early on this step; the action it names, if any."""
        live = [a.name for a in self.actions if a.kind != "undo" and not a.undone]
        got = self.policy.early(decision, p, stable, live, self.none_keys)
        if got is None:
            return []
        kind, name = got
        spec = self.policy.spec(name)
        a = Action(name=name, kind=kind, cls=spec.cls, word=k, p=p, transcript=self.transcript,
                   refines=spec.refines if kind == "early" else None)
        self.actions.append(a)
        return [a]


def open_session(engine: Tez, schema: Schema | str | None = None, question: str | Question | None = None,
                 **kw: Any) -> StreamSession:
    """StreamSession(engine, schema, question, ...): the same as engine.stream(...)."""
    return StreamSession(engine, schema, question, **kw)
