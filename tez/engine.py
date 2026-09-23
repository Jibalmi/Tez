"""The decision engine behind the server, the CLI and the Python API.

One request = one state and many typed questions. Each question is read independently:
  letters  one /completion call (a chunked tournament above 26 options), per-question temperature
  probe    one /embedding call through the question's logistic probe, blended with the letters prior
           (weight n / (n + 10) on the probe) unless the schema was fitted with --no-blend
and gets Jev's answer shape, plus Tez's per-question block (readout, gate decision, p_correct).
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
from .backends import Backend, make_backend
from .errors import BackendRequestError, BackendUnavailable, InvalidRequest, NotFound
from .gate import decide as gate_decide
from .gate import lookup
from .prompt import TOURNAMENT_NONE, build_prompt, fingerprint, needs_tournament, pick_finalists, tournament_plan
from .readout import assemble, blend, softmax, temper
from .schema import NONE_KEY, Question, Schema, load_schemas, parse_alpha, parse_questions

log = logging.getLogger("tez")
READOUTS = ("auto", "letters", "probe")
DEFAULT_MODEL_ALIAS = "tez-latest"


@dataclass
class DecideRequest:
    state: Any
    questions: dict[str, Question]
    schema: Schema | None
    readout: str
    abstain: bool
    alpha: float | None
    model: str


@dataclass
class QuestionResult:
    answer: dict
    meta: dict
    tokens: int
    readout: str
    probabilities: np.ndarray
    keys: list[str]


class Tez:
    """A local decision engine.

        tez = Tez(backend="http://127.0.0.1:8091", template="gemma4", schemas="schemas/")
        tez.decide("Help! My payouts have been failing for 3 days.", schema="support-triage")

    backend        llama-server URL, "fake" (offline demo backend) or a Backend instance
    template       prompt template of the letters model: gemma4 | qwen3
    schemas        a directory of *.yaml schemas, a schema file, Schema objects, or a list of these
    embed_backend  optional separate server for probe features (default: the letters backend)
    """

    def __init__(self, backend: Any = None, template: str = "gemma4", schemas: Any = None, embed_backend: Any = None,
                 embed_template: str | None = None, cache_prompt: bool = True, data_dir: str | Path | None = None,
                 model_name: str | None = None, embed_model_name: str | None = None):
        self.backend: Backend = make_backend(backend, template, cache_prompt, model_name)
        self.template = self.backend.template
        if embed_backend is None:
            self.embedder: Backend = self.backend
            self.embed_backend_url: str | None = None
        else:
            self.embedder = make_backend(embed_backend, embed_template or template, cache_prompt, embed_model_name)
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

    def reload_fitted(self, name: str) -> Fitted | None:
        schema = self.schemas[name]
        self.fitted.pop(name, None)
        for key in [k for k in self._status if k[0] == name]:
            del self._status[key]
        fitted = load_artifacts(schema.artifact_dir)
        if fitted is None:
            return None
        self.fitted[name] = fitted
        for qid, fq in fitted.questions.items():
            self._status[(name, qid)] = self._static_check(schema, qid, fq)
        return fitted

    def _static_check(self, schema: Schema, qid: str, fq: FittedQuestion) -> dict:
        """Is a fitted calibration / probe still valid for the schema as it is now? (model names are checked later)"""
        out = {"letters": False, "probe": False, "reason": None}
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
            elif fq.letters.prompt_sha != fingerprint(sq, self.backend.template, sq.options(), shots):
                reasons.append("the prompt changed since tez fit (instructions, options or examples)")
            else:
                out["letters"] = True
        if fq.probe is not None and fq.probe_cal is not None:
            if fq.probe_cal.template != self.embedder.template:
                reasons.append(f"the probe was fitted with the {fq.probe_cal.template} template")
            elif fq.probe_cal.prompt_sha != fingerprint(sq, self.embedder.template, sq.options(), shots):
                reasons.append("the probe prompt changed since tez fit (instructions, options or examples)")
            else:
                out["probe"] = True
        out["reason"] = "; ".join(dict.fromkeys(reasons)) or fq.note
        return out

    def _usable(self, schema_name: str, qid: str) -> tuple[FittedQuestion | None, bool, bool]:
        """(fitted question, letters calibration usable, probe usable), including the model-name check."""
        fitted = self.fitted.get(schema_name)
        fq = fitted.questions.get(qid) if fitted else None
        st = self._status.get((schema_name, qid))
        if fq is None or st is None:
            return None, False, False
        letters_ok = st["letters"] and fq.letters is not None and fq.letters.model == self.backend.model_name()
        probe_ok = st["probe"] and fq.probe_cal is not None and fq.probe_cal.model == self.embedder.model_name()
        for kind, flag, cal, backend in (("letters", st["letters"] and not letters_ok, fq.letters, self.backend),
                                         ("probe", st["probe"] and not probe_ok, fq.probe_cal, self.embedder)):
            if flag and (schema_name, qid, kind) not in self._warned:
                self._warned.add((schema_name, qid, kind))
                log.warning("%s/%s: %s calibration was fitted on model %r, the backend serves %r: not used",
                            schema_name, qid, kind, cal.model, backend.model_name())
        return fq, letters_ok, probe_ok

    def probe_index(self) -> dict[str, list[str]]:
        """Questions with a trained, current probe, per schema (model names not checked)."""
        return {name: [qid for qid in s.questions if self._status.get((name, qid), {}).get("probe")]
                for name, s in self.schemas.items()}

    # ---------------------------------------------------------------------------------- requests
    def parse_request(self, body: Any) -> DecideRequest:
        if not isinstance(body, dict):
            raise InvalidRequest("the request body must be a JSON object")
        if body.get("state") is None:
            raise InvalidRequest("state is required")
        state = body["state"]
        if not isinstance(state, (str, dict, list)):
            raise InvalidRequest("state must be a string, object or array")
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
            questions = parse_questions(body["questions"])
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
        return DecideRequest(state=state, questions=questions, schema=schema, readout=readout, abstain=abstain,
                             alpha=alpha, model=model)

    def handle(self, body: Any) -> dict:
        """Validate a wire-format request and decide it. Raises TezError subclasses (422 / 503)."""
        return self.run(self.parse_request(body))

    def decide(self, state: Any, questions: Mapping | None = None, schema: Any = None, readout: str = "auto",
               abstain: bool = False, alpha: float | None = None, gate: Any = None, model: str = DEFAULT_MODEL_ALIAS) -> dict:
        """Decide one state. Returns the wire-format response dict (answers, usage, tez block).

        questions  {id: {type, instructions, criteria}} (or Question objects); optional when schema is given
        schema     name of a loaded schema, or a Schema object (loaded on first use)
        readout    auto | letters | probe
        abstain    add the implicit __none__ option to every choice question
        alpha      gate: target error rate among acted decisions (needs a fitted schema); gate=False disables
                   a schema's default gate
        """
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
        body["tez"] = tez
        return self.handle(body)

    def run(self, req: DecideRequest) -> dict:
        t0 = time.perf_counter()
        answers, metas, used = {}, {}, []
        tokens = 0
        for qid, q in req.questions.items():
            r = self.decide_question(q, req.state, schema=req.schema, readout=req.readout, abstain=req.abstain, alpha=req.alpha)
            answers[qid] = r.answer
            metas[qid] = r.meta
            tokens += r.tokens
            used.append(r.readout)
        latency = (time.perf_counter() - t0) * 1000.0
        return {"model": self.model_label(used), "answers": answers,
                "usage": {"input_tokens": int(tokens), "output_tokens": 0},
                "tez": {"latency_ms": round(latency, 1), "questions": metas}}

    # ---------------------------------------------------------------------------------- readouts
    def letter_logits(self, q: Question, state: Any, options: list | None = None, shots: list | None = None) -> tuple[np.ndarray, int]:
        """Letter log-probabilities over `options` (default: the question's) and the prompt tokens spent.
        Above 26 options: tournament, and options that left it get -inf."""
        options = q.options() if options is None else list(options)
        n = len(options)
        if not needs_tournament(n):
            r = self.backend.letters(build_prompt(q, state, self.template, options, shots), n)
            return r.logits, r.tokens
        none_opt = options[-1] if options[-1][0] == NONE_KEY else None
        real = options[:-1] if none_opt else options
        winners, wprobs, tokens = [], [], 0
        for chunk in tournament_plan(len(real)):
            opts = [real[i] for i in chunk] + [TOURNAMENT_NONE]
            r = self.backend.letters(build_prompt(q, state, self.template, opts), len(opts))
            tokens += r.tokens
            pc = softmax(r.logits)
            j = int(np.argmax(pc[:-1]))
            winners.append(chunk[j])
            wprobs.append(float(pc[j]))
        fin = pick_finalists(winners, wprobs)
        fopts = [real[i] for i in fin] + ([none_opt] if none_opt else [])
        r = self.backend.letters(build_prompt(q, state, self.template, fopts), len(fopts))
        tokens += r.tokens
        z = np.full(n, -np.inf)
        z[fin] = r.logits[: len(fin)]
        if none_opt:
            z[-1] = r.logits[-1]
        return z, tokens

    def probe_prompt(self, q: Question, state: Any, shots: list | None = None) -> str:
        """The prompt whose last-token state a probe reads (the question's own options, no __none__)."""
        return build_prompt(q, state, self.embedder.template, q.options(), shots)

    def decide_question(self, q: Question, state: Any, schema: Schema | None = None, readout: str = "auto",
                        abstain: bool = False, alpha: float | None = None) -> QuestionResult:
        sq = schema.questions.get(q.id) if schema is not None else None
        same = sq is not None and sq.signature() == q.signature()
        fq, letters_ok, probe_ok = self._usable(schema.name, q.id) if same else (None, False, False)
        options, keys = q.options(abstain), q.keys(abstain)
        has_none = len(keys) > len(q.keys())
        shots_letters = schema.shots(q.id, len(options)) if same else []
        shots_probe = schema.shots(q.id) if same else []
        if readout == "probe" and not probe_ok:
            raise InvalidRequest(self._no_probe_reason(schema, q.id, same))
        use_probe = probe_ok and readout != "letters"
        tokens = 0
        p_probe = None
        if use_probe:
            try:
                emb = self.embedder.embed(self.probe_prompt(q, state, shots_probe))
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
            z, t = self.letter_logits(q, state, options, shots_letters)
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
        return QuestionResult(answer=answer, meta=meta, tokens=tokens, readout=readout_used, probabilities=p, keys=keys)

    def _no_probe_reason(self, schema: Schema | None, qid: str, same: bool) -> str:
        base = f"question '{qid}' has no usable probe for tez.readout=probe"
        if schema is None:
            return f"{base} (name a fitted schema, or use readout auto or letters)"
        if not same:
            return f"{base} (it is not the '{schema.name}' schema's question '{qid}' as defined in the schema)"
        st = self._status.get((schema.name, qid))
        if st is None:
            return f"{base} (run tez fit on schema '{schema.name}')"
        return f"{base} ({st.get('reason') or 'fitted on another model'})"

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
                        "probes": self.probe_index()[name]})
        return {"schemas": out}

    def schema_detail(self, name: str) -> dict:
        s = self.schemas.get(name)
        if s is None:
            raise NotFound(f"unknown schema '{name}'")
        f = self.fitted.get(name)
        status = {}
        for qid in s.questions:
            st = self._status.get((name, qid))
            fq = f.questions.get(qid) if f else None
            cal_q = (f.calibration.get("questions") or {}).get(qid, {}) if f else {}
            probe = "ready" if st and st["probe"] else ("stale" if fq is not None and fq.probe is not None else "none")
            status[qid] = {"probe": probe, "letters_calibrated": bool(st and st["letters"]),
                           "n_labels": cal_q.get("n_labels"), "note": (st or {}).get("reason") or cal_q.get("note")}
        manifest = {k: v for k, v in (f.manifest if f else {}).items() if k != "train_hashes"} if f else None
        return {**s.to_wire(), "calibration_id": f.calibration_id if f else None, "probes": status,
                "calibration": f.calibration if f else None, "manifest": manifest}

    def feedback_path(self, schema: Schema) -> Path:
        base = self.data_dir if self.data_dir is not None else schema.base_dir / ".tez"
        return base / "feedback" / f"{schema.name}.jsonl"

    def record_feedback(self, body: Any) -> dict:
        """Append a corrected label for a past decision to <data-dir>/feedback/<schema>.jsonl (read by tez fit)."""
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
        row = {"schema": name, "question": qid, "state": state, "label": label,
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        path = self.feedback_path(schema)
        with self._feedback_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True, "schema": name, "question": qid, "label": label}
