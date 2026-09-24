"""The decision engine behind the server, the CLI and the Python API.

One request = one state and many typed questions. Each question is read independently:
  letters  one /completion call (a chunked tournament above 26 options), per-question temperature
  probe    one /embedding call through the question's logistic probe, blended with the letters prior
           (weight n / (n + 10) on the probe) unless the schema was fitted with --no-blend
and gets Jev's answer shape, plus Tez's per-question block (readout, gate decision, p_correct).

Prompt layout (tez/prompt.py): a request asks for `auto`, `question_first` or `state_first` (request `tez.layout`,
else the schema's `layout:`, else the engine's default). `auto` resolves per question: a question with a usable fit
keeps the layout it was fitted under (so a fit never goes stale because a request had more questions), any other
question is read state_first when the request has two or more questions and question_first otherwise. Questions read
state_first run back to back, so the backend's prompt cache reuses the state between them.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._version import RELEASE_DATE, __version__
from .artifacts import Fitted, FittedQuestion, load_artifacts
from .backends import DEFAULT_N_PROBS, Backend, make_backend
from .errors import (BackendRequestError, BackendUnavailable, InternalError, InvalidRequest, NotFound, PayloadTooLarge,
                     TezError)
from .gate import decide as gate_decide
from .gate import lookup
from .hooks import DecisionContext, HookSet, default_hooks, new_run_id, normalise_hooks
from .prompt import (QUESTION_FIRST, STATE_FIRST, TOURNAMENT_NONE, build_prompt, fingerprint, needs_tournament,
                     pick_finalists, resolve_layout, tournament_plan)
from .readout import assemble, blend, softmax, temper
from .schema import LAYOUTS, NONE_KEY, Question, Schema, load_schemas, parse_alpha, parse_layout, parse_questions

log = logging.getLogger("tez")
READOUTS = ("auto", "letters", "probe")
DEFAULT_MODEL_ALIAS = "tez-latest"


@dataclass(frozen=True)
class Limits:
    """Request limits (413 payload_too_large past any of them). The defaults are tez serve's; the Python API applies
    none unless given limits=. None switches one off."""

    max_body_bytes: int | None = 2 * 1024 * 1024
    max_questions: int | None = 64
    max_state_chars: int | None = 50_000
    max_batch: int | None = 64

    @classmethod
    def unlimited(cls) -> Limits:
        return cls(None, None, None, None)

    def check_questions(self, n: int) -> None:
        if self.max_questions is not None and n > self.max_questions:
            raise PayloadTooLarge(f"too many questions: {n} (the limit is {self.max_questions}; tez serve --max-questions)")

    def check_state(self, state: Any, where: str = "state") -> None:
        if self.max_state_chars is None:
            return
        n = len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
        if n > self.max_state_chars:
            raise PayloadTooLarge(f"{where} is too large: {n:,} characters (the limit is {self.max_state_chars:,}; "
                                  f"tez serve --max-state-chars)")

    def check_batch(self, n: int) -> None:
        if self.max_batch is not None and n > self.max_batch:
            raise PayloadTooLarge(f"too many states in one batch: {n} (the limit is {self.max_batch}; tez serve --max-batch)")


@dataclass
class DecideRequest:
    state: Any
    questions: dict[str, Question]
    schema: Schema | None
    readout: str
    abstain: bool
    alpha: float | None
    model: str
    layout: str = "auto"          # as requested (auto | question_first | state_first), before per-question resolution


@dataclass
class QuestionResult:
    answer: dict
    meta: dict
    tokens: int
    readout: str
    probabilities: np.ndarray
    keys: list[str]
    layout: str = QUESTION_FIRST


class Tez:
    """A local decision engine.

        tez = Tez(backend="http://127.0.0.1:8091", template="gemma4", schemas="schemas/")
        tez.decide("Help! My payouts have been failing for 3 days.", schema="support-triage")

    backend        llama-server URL, "fake" (offline demo backend) or a Backend instance
    template       prompt template of the letters model: gemma4 | qwen3
    schemas        a directory of *.yaml schemas, a schema file, Schema objects, or a list of these
    embed_backend  optional separate server for probe features (default: the letters backend)
    layout         default prompt layout: auto (default) | question_first | state_first
    n_probs        next-token log-probabilities a letter readout asks llama-server for (default 200)
    hooks          hooks run on every decision (tez.hooks; docs/HOOKS.md); hooks_raise=False logs a failing hook
                   instead of failing the decision
    """

    def __init__(self, backend: Any = None, template: str = "gemma4", schemas: Any = None, embed_backend: Any = None,
                 embed_template: str | None = None, cache_prompt: bool = True, data_dir: str | Path | None = None,
                 model_name: str | None = None, embed_model_name: str | None = None, *, layout: str = "auto",
                 n_probs: int = DEFAULT_N_PROBS, hooks: Any = None, hooks_raise: bool = True):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}, got {layout!r}")
        self.layout = layout
        self.hooks: list = normalise_hooks(hooks)
        self.hooks_raise = bool(hooks_raise)
        self._hooks_lock = threading.Lock()
        self.backend: Backend = make_backend(backend, template, cache_prompt, model_name, n_probs)
        self.template = self.backend.template
        if embed_backend is None:
            self.embedder: Backend = self.backend
            self.embed_backend_url: str | None = None
        else:
            self.embedder = make_backend(embed_backend, embed_template or template, cache_prompt, embed_model_name, n_probs)
            self.embed_backend_url = self.embedder.url
        self.data_dir = Path(data_dir) if data_dir else None
        self.schemas: dict[str, Schema] = {}
        self.fitted: dict[str, Fitted] = {}
        self._status: dict[tuple[str, str], dict] = {}
        self._warned: set = set()
        self._feedback_lock = threading.Lock()
        if schemas is not None:
            self.add_schemas(schemas)

    def __repr__(self) -> str:
        return f"Tez(backend={self.backend!r}, schemas={sorted(self.schemas)})"

    # ---------------------------------------------------------------------------------- hooks
    def add_hook(self, hook: Any) -> Tez:
        """Install one hook (or a list) after the engine's current ones. Returns the engine."""
        items = normalise_hooks(hook)
        with self._hooks_lock:
            self.hooks = self.hooks + items
        return self

    def remove_hook(self, hook: Any) -> bool:
        """Remove a hook by identity; True when it was installed."""
        with self._hooks_lock:
            kept = [h for h in self.hooks if h is not hook]
            removed = len(kept) != len(self.hooks)
            self.hooks = kept
        return removed

    def hookset(self, hooks: Any = None) -> HookSet:
        """The hooks of one call: process-wide defaults, the engine's, then the call's own."""
        return HookSet(default_hooks() + list(self.hooks) + normalise_hooks(hooks), self.hooks_raise)

    # ---------------------------------------------------------------------------------- schemas
    def add_schemas(self, source: Any) -> list[str]:
        """Load schemas (and their fitted artefacts, if any). Returns the names added."""
        if isinstance(source, Schema):
            loaded = {source.name: source}
        elif isinstance(source, Mapping):
            loaded = load_schemas(list(source.values()))
        else:
            loaded = load_schemas(source)
        for name, schema in loaded.items():
            if name in self.schemas:
                raise InvalidRequest(f"schema '{name}' is already loaded")
            self.schemas[name] = schema
            self.reload_fitted(name)
        return list(loaded)

    def add_presets(self, names: Any = None) -> list[str]:
        """Load the built-in presets (tez.presets; all of them by default). A preset whose name is already loaded is
        skipped: your own schema of that name wins. Returns the names added."""
        from . import presets
        wanted = presets.names() if names is None else list(names)
        added = []
        for name in wanted:
            if name in self.schemas:
                log.info("preset %s skipped: a schema of that name is already loaded", name)
                continue
            added += self.add_schemas(presets.load(name))
        return added

    def reload_fitted(self, name: str) -> Fitted | None:
        schema = self.schemas[name]
        self.fitted.pop(name, None)
        for key in [k for k in self._status if k[0] == name]:
            del self._status[key]
        if schema.builtin:              # presets are zero-shot: copy one into a schema directory to fit it
            return None
        fitted = load_artifacts(schema.artifact_dir)
        if fitted is None:
            return None
        self.fitted[name] = fitted
        for qid, fq in fitted.questions.items():
            self._status[(name, qid)] = self._static_check(schema, qid, fq)
        return fitted

    def _static_check(self, schema: Schema, qid: str, fq: FittedQuestion) -> dict:
        """Is a fitted calibration / probe still valid for the schema as it is now, under the layout it was fitted
        with? (Model names are checked later; so is the layout a request asks for.)"""
        out = {"letters": False, "probe": False, "reason": None, "layout": fq.layout}
        sq = schema.questions.get(qid)
        if sq is None:
            out["reason"] = "question is no longer in the schema"
            return out
        if sq.type != fq.type or sq.keys() != fq.options:
            out["reason"] = "question type or options changed since tez fit"
            return out
        shots = schema.shots(qid)
        reasons = []
        if fq.letters is not None:
            if fq.letters.template != self.backend.template:
                reasons.append(f"letters were calibrated with the {fq.letters.template} template")
            elif fq.letters.prompt_sha != fingerprint(sq, self.backend.template, sq.options(), shots, fq.letters.layout):
                reasons.append("the prompt changed since tez fit (instructions, options or examples)")
            else:
                out["letters"] = True
        if fq.probe is not None and fq.probe_cal is not None:
            if fq.probe_cal.template != self.embedder.template:
                reasons.append(f"the probe was fitted with the {fq.probe_cal.template} template")
            elif fq.probe_cal.prompt_sha != fingerprint(sq, self.embedder.template, sq.options(), shots, fq.probe_cal.layout):
                reasons.append("the probe prompt changed since tez fit (instructions, options or examples)")
            else:
                out["probe"] = True
        out["reason"] = "; ".join(dict.fromkeys(reasons)) or fq.note
        return out

    def configured_layout(self, schema: Schema | None) -> str:
        """The layout a request gets when it names none: the schema's `layout:`, else the engine's default."""
        if schema is not None and schema.layout is not None:
            return schema.layout
        return self.layout

    def _layout_mismatch(self, fit_layout: str, layout: str) -> str | None:
        if layout in (QUESTION_FIRST, STATE_FIRST) and layout != fit_layout:
            return f"fitted under the {fit_layout} layout, read with {layout}"
        return None

    def _usable(self, schema_name: str, qid: str, layout: str | None = None,
                check_model: bool = True) -> tuple[FittedQuestion | None, bool, bool]:
        """(fitted question, letters calibration usable, probe usable), including the model-name check and, when a
        concrete layout is given, that the fit was made under it. check_model=False skips the model names when they
        are not known without asking the backend (tez plan never calls it)."""
        fitted = self.fitted.get(schema_name)
        fq = fitted.questions.get(qid) if fitted else None
        st = self._status.get((schema_name, qid))
        if fq is None or st is None:
            return None, False, False
        if layout is not None and self._layout_mismatch(fq.layout, layout):
            return fq, False, False

        def model_ok(cal: Any, backend: Backend) -> bool:
            if cal is None:
                return False
            name = backend.model_name() if check_model else backend.known_model_name()
            return name is None or cal.model == name

        letters_ok = st["letters"] and model_ok(fq.letters, self.backend)
        probe_ok = st["probe"] and model_ok(fq.probe_cal, self.embedder)
        if check_model:
            for kind, flag, cal, backend in (("letters", st["letters"] and not letters_ok, fq.letters, self.backend),
                                             ("probe", st["probe"] and not probe_ok, fq.probe_cal, self.embedder)):
                if flag and (schema_name, qid, kind) not in self._warned:
                    self._warned.add((schema_name, qid, kind))
                    log.warning("%s/%s: %s calibration was fitted on model %r, the backend serves %r: not used",
                                schema_name, qid, kind, cal.model, backend.model_name())
        return fq, letters_ok, probe_ok

    def _same_question(self, schema: Schema | None, q: Question) -> bool:
        sq = schema.questions.get(q.id) if schema is not None else None
        return sq is not None and sq.signature() == q.signature()

    def question_layout(self, q: Question, schema: Schema | None, layout: str, n_questions: int,
                        check_model: bool = True) -> str:
        """The concrete layout one question of a request is read with. An explicit layout applies as is; `auto` keeps
        the layout of a usable fit, otherwise state_first for two or more questions and question_first for one."""
        if layout in (QUESTION_FIRST, STATE_FIRST):
            return layout
        if self._same_question(schema, q):
            fq, letters_ok, probe_ok = self._usable(schema.name, q.id, None, check_model)
            if fq is not None and (letters_ok or probe_ok):
                return fq.layout
        return resolve_layout(layout, n_questions)

    def probe_index(self) -> dict[str, list[str]]:
        """Questions with a trained, current probe under the schema's configured layout, per schema (model names not
        checked)."""
        out = {}
        for name, s in self.schemas.items():
            layout = self.configured_layout(s)
            out[name] = [qid for qid in s.questions
                         if (st := self._status.get((name, qid), {})).get("probe")
                         and not self._layout_mismatch(st.get("layout", QUESTION_FIRST), layout)]
        return out

    # ---------------------------------------------------------------------------------- requests
    def parse_request(self, body: Any, limits: Limits | None = None) -> DecideRequest:
        if not isinstance(body, dict):
            raise InvalidRequest("the request body must be a JSON object")
        if body.get("state") is None:
            raise InvalidRequest("state is required")
        state = body["state"]
        if not isinstance(state, (str, dict, list)):
            raise InvalidRequest("state must be a string, object or array")
        if limits is not None:
            limits.check_state(state)
        model = body.get("model", DEFAULT_MODEL_ALIAS)
        if model is None:
            model = DEFAULT_MODEL_ALIAS
        if not isinstance(model, str):
            raise InvalidRequest("model must be a string")
        schema = None
        if body.get("schema") is not None:
            name = body["schema"]
            if not isinstance(name, str):
                raise InvalidRequest("schema must be the name of a loaded schema")
            schema = self.schemas.get(name)
            if schema is None:
                loaded = ", ".join(sorted(self.schemas)) or "none"
                raise InvalidRequest(f"unknown schema '{name}' (loaded: {loaded})")
        if body.get("questions") is None:
            if schema is None:
                raise InvalidRequest("questions is required (or name a loaded schema)")
            questions = dict(schema.questions)
        else:
            if limits is not None and isinstance(body["questions"], dict):
                limits.check_questions(len(body["questions"]))
            questions = parse_questions(body["questions"])
        if limits is not None:
            limits.check_questions(len(questions))
        tez = body.get("tez")
        if tez is None:
            tez = {}
        if not isinstance(tez, dict):
            raise InvalidRequest("tez must be an object")
        readout = tez.get("readout")
        readout = "auto" if readout is None else readout
        if readout not in READOUTS:
            raise InvalidRequest(f"tez.readout must be one of auto, letters, probe; got {readout!r}")
        abstain = tez.get("abstain", False)
        abstain = False if abstain is None else abstain
        if not isinstance(abstain, bool):
            raise InvalidRequest("tez.abstain must be true or false")
        alpha = schema.gate_alpha if schema is not None else None
        if "gate" in tez:
            gate = tez["gate"]
            if gate is None or gate is False:
                alpha = None
            elif isinstance(gate, dict):
                if gate.get("alpha") is not None:
                    alpha = parse_alpha(gate["alpha"], "tez.gate.alpha")
                elif alpha is None:
                    raise InvalidRequest("tez.gate.alpha is required (the target error rate among acted decisions)")
            else:
                raise InvalidRequest("tez.gate must be an object like {\"alpha\": 0.05}")
        if tez.get("layout") is not None:
            layout = parse_layout(tez["layout"], "tez.layout")
        else:
            layout = self.configured_layout(schema)
        return DecideRequest(state=state, questions=questions, schema=schema, readout=readout, abstain=abstain,
                             alpha=alpha, model=model, layout=layout)

    def execute(self, body: Any, *, run_id: str | None = None, hooks: Any = None, limits: Limits | None = None,
                parent_run_id: str | None = None, index: int | None = None) -> DecisionContext:
        """Validate a wire-format request, decide it and return the whole DecisionContext (response, per-question
        traces with backend timings, usage, latency). Hooks run around it; on failure on_error runs and the error
        is raised (TezError subclasses carry their HTTP status)."""
        hs = self.hookset(hooks)
        ctx = DecisionContext(run_id=run_id or new_run_id(), request=body, engine=self, parent_run_id=parent_run_id,
                              index=index)
        try:
            req = self.parse_request(body, limits)
            self._load_context(ctx, req)
            hs.emit("on_decide_start", ctx)
            if ctx.skipped:
                ctx.latency_ms = round((time.perf_counter() - ctx.started) * 1000.0, 1)
                if isinstance(ctx.response.get("tez"), dict):
                    ctx.response["tez"]["latency_ms"] = ctx.latency_ms
                ctx.usage = ctx.response.get("usage")
            else:
                req.state = ctx.state
                self._run(req, ctx, hs)
            hs.emit("on_decide_end", ctx)
            return ctx
        except Exception as exc:
            ctx.error = exc
            hs.emit("on_error", ctx)
            raise

    def _load_context(self, ctx: DecisionContext, req: DecideRequest) -> None:
        ctx.state, ctx.questions, ctx.schema = req.state, req.questions, req.schema
        ctx.readout, ctx.abstain, ctx.alpha, ctx.model = req.readout, req.abstain, req.alpha, req.model
        ctx.requested_layout = req.layout
        ctx.layout = resolve_layout(req.layout, len(req.questions))

    def handle(self, body: Any, *, run_id: str | None = None, hooks: Any = None, limits: Limits | None = None) -> dict:
        """Validate a wire-format request and decide it. Raises TezError subclasses (422 / 503)."""
        return self.execute(body, run_id=run_id, hooks=hooks, limits=limits).response

    # ---------------------------------------------------------------------------------- batches
    def handle_batch(self, body: Any, *, run_id: str | None = None, hooks: Any = None,
                     limits: Limits | None = None) -> dict:
        """A batch request {"model", "states": [...], "questions" | "schema" | "json_schema", "tez"}: every state is
        decided like a /v1/systemone request with the shared fields, in input order, each state's questions back to
        back (so the backend's prompt cache keeps the state). Returns {"model", "results", "usage", "tez":
        {"latency_ms", "run_id"}}; a result is the state's response or {"error": {...}}. A bad shared field fails the
        batch (422, 413); an error in one state does not. When the backend is unavailable before any state was decided
        the batch fails (503); after that, the states not yet tried get a backend_unavailable error without a call.
        Item i's run id is "<run_id>.<i>"."""
        t0 = time.perf_counter()
        run_id = run_id or new_run_id()
        if not isinstance(body, dict):
            raise InvalidRequest("the request body must be a JSON object")
        if "state" in body:
            raise InvalidRequest("a batch takes states (an array of states), not state")
        states = body.get("states")
        if not isinstance(states, list) or not states:
            raise InvalidRequest("states must be a non-empty array of states (strings, objects or arrays)")
        if limits is not None:
            limits.check_batch(len(states))
        shared = {k: v for k, v in body.items() if k != "states"}
        self.parse_request({**shared, "state": ""}, limits)      # a bad shared field fails the whole batch, once
        results: list[dict] = []
        used: list[str] = []
        tokens, decided = 0, 0
        for i, state in enumerate(states):
            try:
                ctx = self.execute({**shared, "state": state}, run_id=f"{run_id}.{i}", hooks=hooks, limits=limits,
                                   parent_run_id=run_id, index=i)
            except BackendUnavailable as exc:
                if decided == 0:
                    raise
                results.append(exc.body())
                note = f"not attempted: the backend was unavailable ({exc.message})"
                results += [{"error": {"type": exc.type, "message": note}} for _ in states[i + 1:]]
                break
            except TezError as exc:
                results.append(exc.body())
                continue
            except Exception as exc:              # a hook that raised: this state fails, the batch goes on
                log.exception("batch item %d failed", i)
                results.append(InternalError(f"{exc.__class__.__name__}: {exc}").body())
                continue
            res = ctx.response
            results.append(res)
            decided += 1
            tokens += int((res.get("usage") or {}).get("input_tokens") or 0)
            used += [m.get("readout", "letters") for m in ((res.get("tez") or {}).get("questions") or {}).values()]
        return {"model": self.model_label(used or None), "results": results,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "tez": {"latency_ms": round((time.perf_counter() - t0) * 1000.0, 1), "run_id": run_id}}

    def decide_batch(self, states: list, questions: Mapping | None = None, schema: Any = None, readout: str = "auto",
                     abstain: bool = False, alpha: float | None = None, gate: Any = None,
                     model: str = DEFAULT_MODEL_ALIAS, *, layout: str | None = None, hooks: Any = None,
                     run_id: str | None = None) -> dict:
        """Decide many states against the same questions: the batch response (model, results, usage, tez)."""
        body = self.request_body(None, questions, schema, readout, abstain, alpha, gate, model, layout)
        body.pop("state")
        body["states"] = list(states)
        return self.handle_batch(body, run_id=run_id, hooks=hooks)

    def decide_many(self, states: list, questions: Mapping | None = None, schema: Any = None, readout: str = "auto",
                    abstain: bool = False, alpha: float | None = None, gate: Any = None,
                    model: str = DEFAULT_MODEL_ALIAS, *, layout: str | None = None, hooks: Any = None) -> list[dict]:
        """Decide many states against the same questions; one result per state, in order: the /v1/systemone
        response, or {"error": {"type", "message"}} for a state that failed (the others are still decided)."""
        return self.decide_batch(states, questions, schema, readout, abstain, alpha, gate, model, layout=layout,
                                 hooks=hooks)["results"]

    def decide(self, state: Any, questions: Mapping | None = None, schema: Any = None, readout: str = "auto",
               abstain: bool = False, alpha: float | None = None, gate: Any = None, model: str = DEFAULT_MODEL_ALIAS,
               *, layout: str | None = None, hooks: Any = None, run_id: str | None = None) -> dict:
        """Decide one state. Returns the wire-format response dict (answers, usage, tez block).

        questions  {id: {type, instructions, criteria}} (or Question objects); optional when schema is given
        schema     name of a loaded schema, or a Schema object (loaded on first use)
        readout    auto | letters | probe
        abstain    add the implicit __none__ option to every choice question
        alpha      gate: target error rate among acted decisions (needs a fitted schema); gate=False disables
                   a schema's default gate
        layout     prompt layout: auto | question_first | state_first (default: the schema's, else the engine's)
        hooks      hooks for this call only, after the process-wide and the engine's
        """
        body = self.request_body(state, questions, schema, readout, abstain, alpha, gate, model, layout)
        return self.handle(body, run_id=run_id, hooks=hooks)

    def request_body(self, state: Any, questions: Mapping | None = None, schema: Any = None, readout: str = "auto",
                     abstain: bool = False, alpha: float | None = None, gate: Any = None,
                     model: str = DEFAULT_MODEL_ALIAS, layout: str | None = None) -> dict:
        """The wire-format request body for decide()'s arguments (a Schema object is loaded on first use)."""
        body: dict[str, Any] = {"model": model, "state": state}
        if questions is not None:
            body["questions"] = {qid: (q.to_wire() if isinstance(q, Question) else q) for qid, q in questions.items()}
        if schema is not None:
            if isinstance(schema, Schema):
                if schema.name not in self.schemas:
                    self.add_schemas(schema)
                schema = schema.name
            body["schema"] = schema
        tez: dict[str, Any] = {"readout": readout, "abstain": abstain}
        if alpha is not None:
            tez["gate"] = {"alpha": alpha}
        elif gate is not None:
            tez["gate"] = gate
        if layout is not None:
            tez["layout"] = layout
        body["tez"] = tez
        return body

    def run(self, req: DecideRequest) -> dict:
        """Decide a parsed request without hooks (execute() is the hooked path)."""
        ctx = DecisionContext(run_id=new_run_id(), engine=self)
        self._load_context(ctx, req)
        return self._run(req, ctx, HookSet([]))

    def _run(self, req: DecideRequest, ctx: DecisionContext, hs: HookSet) -> dict:
        t0 = time.perf_counter()
        n = len(req.questions)
        base = resolve_layout(req.layout, n)
        layouts = {qid: self.question_layout(q, req.schema, req.layout, n) for qid, q in req.questions.items()}
        # state-first questions run back to back, so the backend's prompt cache keeps the state between them
        order = [qid for qid in req.questions if layouts[qid] == STATE_FIRST] + \
                [qid for qid in req.questions if layouts[qid] != STATE_FIRST]
        results: dict[str, QuestionResult] = {}
        for qid in order:
            calls: list = []
            tq = time.perf_counter()
            r = self.decide_question(req.questions[qid], req.state, schema=req.schema, readout=req.readout,
                                     abstain=req.abstain, alpha=req.alpha, layout=layouts[qid], n_questions=n, trace=calls)
            if r.layout != base:
                r.meta = {**r.meta, "layout": r.layout}
            results[qid] = r
            ctx.traces[qid] = {"readout": r.readout, "layout": r.layout, "tokens": int(r.tokens),
                               "ms": round((time.perf_counter() - tq) * 1000.0, 3), "calls": calls}
            if hs:
                hs.emit("on_question_end", ctx, qid, r.answer, r.meta)
        answers, metas, used, tokens = {}, {}, [], 0
        for qid in req.questions:
            r = results[qid]
            answers[qid] = r.answer
            metas[qid] = r.meta
            tokens += r.tokens
            used.append(r.readout)
        if len({r.layout for r in results.values()}) > 1:
            ctx.layout = "mixed"
        latency = (time.perf_counter() - t0) * 1000.0
        response = {"model": self.model_label(used), "answers": answers,
                    "usage": {"input_tokens": int(tokens), "output_tokens": 0},
                    "tez": {"latency_ms": round(latency, 1), "questions": metas}}
        ctx.response, ctx.usage, ctx.latency_ms = response, response["usage"], round(latency, 1)
        return response

    def backend_ms(self, ctx: DecisionContext) -> float:
        """Wall time spent in backend calls during a decision (from its traces)."""
        return round(sum(c.get("ms") or 0.0 for t in ctx.traces.values() for c in t.get("calls", [])), 3)

    # ---------------------------------------------------------------------------------- readouts
    def letter_logits(self, q: Question, state: Any, options: list | None = None, shots: list | None = None, *,
                      layout: str = QUESTION_FIRST, trace: list | None = None) -> tuple[np.ndarray, int]:
        """Letter log-probabilities over `options` (default: the question's) and the prompt tokens spent.
        Above 26 options: tournament, and options that left it get -inf. `trace`, when given, receives one record
        per backend call (kind, tokens, wall ms, llama.cpp timings)."""
        options = q.options() if options is None else list(options)
        n = len(options)

        def read(prompt: str, k: int):
            r = self.backend.letters(prompt, k)
            if trace is not None:
                trace.append({"kind": "letters", "tokens": int(r.tokens), "ms": r.ms, "timings": r.timings})
            return r

        if not needs_tournament(n):
            r = read(build_prompt(q, state, self.template, options, shots, layout), n)
            return r.logits, r.tokens
        none_opt = options[-1] if options[-1][0] == NONE_KEY else None
        real = options[:-1] if none_opt else options
        winners, wprobs, tokens = [], [], 0
        for chunk in tournament_plan(len(real)):
            opts = [real[i] for i in chunk] + [TOURNAMENT_NONE]
            r = read(build_prompt(q, state, self.template, opts, None, layout), len(opts))
            tokens += r.tokens
            pc = softmax(r.logits)
            j = int(np.argmax(pc[:-1]))
            winners.append(chunk[j])
            wprobs.append(float(pc[j]))
        fin = pick_finalists(winners, wprobs)
        fopts = [real[i] for i in fin] + ([none_opt] if none_opt else [])
        r = read(build_prompt(q, state, self.template, fopts, None, layout), len(fopts))
        tokens += r.tokens
        z = np.full(n, -np.inf)
        z[fin] = r.logits[: len(fin)]
        if none_opt:
            z[-1] = r.logits[-1]
        return z, tokens

    def probe_prompt(self, q: Question, state: Any, shots: list | None = None, *, layout: str = QUESTION_FIRST) -> str:
        """The prompt whose last-token state a probe reads (the question's own options, no __none__)."""
        return build_prompt(q, state, self.embedder.template, q.options(), shots, layout)

    def decide_question(self, q: Question, state: Any, schema: Schema | None = None, readout: str = "auto",
                        abstain: bool = False, alpha: float | None = None, layout: str = "auto", n_questions: int = 1,
                        trace: list | None = None) -> QuestionResult:
        same = self._same_question(schema, q)
        layout = self.question_layout(q, schema, layout, n_questions) if same else resolve_layout(layout, n_questions)
        fq, letters_ok, probe_ok = self._usable(schema.name, q.id, layout) if same else (None, False, False)
        options, keys = q.options(abstain), q.keys(abstain)
        has_none = len(keys) > len(q.keys())
        shots_letters = schema.shots(q.id, len(options)) if same else []
        shots_probe = schema.shots(q.id) if same else []
        if readout == "probe" and not probe_ok:
            raise InvalidRequest(self._no_probe_reason(schema, q.id, same, layout))
        use_probe = probe_ok and readout != "letters"
        tokens = 0
        p_probe = None
        if use_probe:
            try:
                emb = self.embedder.embed(self.probe_prompt(q, state, shots_probe, layout=layout))
                if trace is not None:
                    trace.append({"kind": "embed", "tokens": int(emb.tokens), "ms": emb.ms, "timings": emb.timings})
                p_probe = fq.probe.predict(emb.vector)
                tokens += emb.tokens
            except (BackendUnavailable, BackendRequestError, ValueError) as exc:
                if readout == "probe":
                    if isinstance(exc, ValueError):
                        raise BackendUnavailable(f"probe for '{q.id}' cannot read this embedding backend: {exc}") from exc
                    raise
                log.warning("probe readout for %s failed (%s); falling back to the letters", q.id, exc)
                use_probe = False
        p_letters = None
        if not use_probe or fq.blend or has_none:
            z, t = self.letter_logits(q, state, options, shots_letters, layout=layout, trace=trace)
            tokens += t
            p_letters = softmax(z, fq.letters.temperature if letters_ok else 1.0)
        if use_probe:
            p = p_probe
            if fq.blend and p_letters is not None:
                prior = p_letters[: len(p_probe)]
                prior = prior / prior.sum() if prior.sum() > 0 else np.full(len(p_probe), 1.0 / len(p_probe))
                p = blend(p_probe, prior, fq.probe.n)
            p = temper(p, fq.probe_cal.temperature)
            if has_none:
                p_none = float(p_letters[-1])
                p = np.append(p * (1.0 - p_none), p_none)
            readout_used, cal = "probe", fq.probe_cal
        else:
            p = p_letters
            readout_used, cal = "letters", (fq.letters if letters_ok else None)
        answer = assemble(q, keys, p)
        meta: dict[str, Any] = {"readout": readout_used}
        p_max = float(np.max(p))
        if alpha is not None:
            if cal is None or (q.type == "choice" and answer["choice"] == NONE_KEY):
                meta["decision"] = "escalate"      # nothing certifies this decision's error rate / no option fits
            else:
                cut, _ = lookup(cal.thresholds, alpha)
                meta["decision"] = gate_decide(p_max, cut)
        if cal is not None:
            meta["p_correct"] = round(p_max, 4)
            meta["calibration_id"] = self.fitted[schema.name].calibration_id
        return QuestionResult(answer=answer, meta=meta, tokens=tokens, readout=readout_used, probabilities=p, keys=keys,
                              layout=layout)

    def _no_probe_reason(self, schema: Schema | None, qid: str, same: bool, layout: str | None = None) -> str:
        base = f"question '{qid}' has no usable probe for tez.readout=probe"
        if schema is None:
            return f"{base} (name a fitted schema, or use readout auto or letters)"
        if not same:
            return f"{base} (it is not the '{schema.name}' schema's question '{qid}' as defined in the schema)"
        st = self._status.get((schema.name, qid))
        if st is None:
            return f"{base} (run tez fit on schema '{schema.name}')"
        if not st.get("probe"):
            return f"{base} ({st.get('reason') or 'no probe was trained for it'})"
        mismatch = self._layout_mismatch(st.get("layout", QUESTION_FIRST), layout) if layout else None
        if mismatch:
            return f"{base} ({mismatch}; request tez.layout {st.get('layout')} or refit with --layout {layout})"
        return f"{base} (fitted on another model)"

    # ---------------------------------------------------------------------------------- endpoints
    def model_label(self, used: list[str] | None = None) -> str:
        readouts = [r for r in ("letters", "probe") if r in (used or ["letters"])]
        name = self.backend.model_name()
        if "probe" in readouts and self.embedder is not self.backend:
            name = f"{name} / {self.embedder.model_name()}"
        return f"tez-{__version__} ({name}, {'+'.join(readouts)})"

    def models(self) -> dict:
        has_probe = any(self.probe_index().values())
        readouts = "letters+probe" if has_probe else "letters"
        return {"models": [{"name": DEFAULT_MODEL_ALIAS,
                            "description": f"Tez local decision engine ({self.backend.model_name()}, {readouts})",
                            "release_date": RELEASE_DATE}]}

    def health(self) -> dict:
        b = self.backend.health()
        out = {"status": "ok" if b.get("ok") else "degraded", "version": __version__, "backend": self.backend.url,
               "template": self.template, "embed_backend": self.embed_backend_url, "schemas": sorted(self.schemas),
               "probes": self.probe_index(), "backend_status": b.get("status")}
        out["model"] = self.backend.model_name() if b.get("ok") else None
        out["layout"] = self.layout
        if self.embedder is not self.backend:
            e = self.embedder.health()
            out["embed_template"] = self.embedder.template
            out["embed_backend_status"] = e.get("status")
        return out

    def schema_summaries(self) -> dict:
        out = []
        for name, s in sorted(self.schemas.items()):
            f = self.fitted.get(name)
            out.append({"name": name, "description": s.description,
                        "questions": {qid: q.type for qid, q in s.questions.items()},
                        "calibration_id": f.calibration_id if f else None,
                        "probes": self.probe_index()[name], "builtin": s.builtin})
        return {"schemas": out}

    def schema_detail(self, name: str) -> dict:
        s = self.schemas.get(name)
        if s is None:
            raise NotFound(f"unknown schema '{name}'")
        f = self.fitted.get(name)
        layout = self.configured_layout(s)
        status = {}
        for qid in s.questions:
            st = self._status.get((name, qid))
            fq = f.questions.get(qid) if f else None
            cal_q = (f.calibration.get("questions") or {}).get(qid, {}) if f else {}
            mismatch = self._layout_mismatch(st.get("layout", QUESTION_FIRST), layout) if st else None
            ready = bool(st and st["probe"] and not mismatch)
            probe = "ready" if ready else ("stale" if fq is not None and fq.probe is not None else "none")
            note = (mismatch if mismatch and fq is not None else None) or (st or {}).get("reason") or cal_q.get("note")
            status[qid] = {"probe": probe, "letters_calibrated": bool(st and st["letters"] and not mismatch),
                           "n_labels": cal_q.get("n_labels"), "note": note,
                           "layout": fq.layout if fq is not None else None}
        manifest = {k: v for k, v in (f.manifest if f else {}).items() if k != "train_hashes"} if f else None
        return {**s.to_wire(), "layout": s.layout, "served_layout": layout,
                "calibration_id": f.calibration_id if f else None, "probes": status,
                "calibration": f.calibration if f else None, "manifest": manifest}

    def feedback_path(self, schema: Schema) -> Path:
        base = self.data_dir if self.data_dir is not None else schema.base_dir / ".tez"
        return base / "feedback" / f"{schema.name}.jsonl"

    def record_feedback(self, body: Any, *, hooks: Any = None) -> dict:
        """Append a corrected label for a past decision to <data-dir>/feedback/<schema>.jsonl (read by tez fit).
        on_feedback hooks see (and may edit) the row before it is written; `run_id` links it to the decision."""
        if not isinstance(body, dict):
            raise InvalidRequest("the request body must be a JSON object")
        name = body.get("schema")
        if not isinstance(name, str) or not name:
            raise InvalidRequest("schema is required")
        schema = self.schemas.get(name)
        if schema is None:
            raise InvalidRequest(f"unknown schema '{name}' (loaded: {', '.join(sorted(self.schemas)) or 'none'})")
        qid = body.get("question")
        if not isinstance(qid, str) or qid not in schema.questions:
            raise InvalidRequest(f"question must be one of the '{name}' schema's questions: {', '.join(schema.questions)}")
        state = body.get("state")
        if state is None or not isinstance(state, (str, dict, list)):
            raise InvalidRequest("state is required (string, object or array)")
        if "label" not in body:
            raise InvalidRequest("label is required")
        label = schema.questions[qid].canonical_label(body["label"], allow_none=True)
        run_id = body.get("run_id")
        if run_id is not None and (not isinstance(run_id, str) or not 0 < len(run_id) <= 128):
            raise InvalidRequest("run_id must be the decision's run id (a string of at most 128 characters)")
        row = {"schema": name, "question": qid, "state": state, "label": label,
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if run_id is not None:
            row["run_id"] = run_id
        self.hookset(hooks).emit("on_feedback", row)
        path = self.feedback_path(schema)
        with self._feedback_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True, "schema": name, "question": qid, "label": label}
