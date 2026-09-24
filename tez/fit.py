"""Learning from labels: `tez fit`, `tez suggest` and `tez eval`.

fit, per question (labels from --labels files, recorded feedback and the schema's examples):
  1. split the labelled rows once (seeded): a fit part and a held-out part (default 30 %);
  2. letters: read every row zero-shot; the temperature is fitted on the fit part;
  3. probe (at least 20 labels): embed every row's prompt, train a logistic probe (C = 0.5) on the fit part;
     its temperature is fitted on out-of-fold predictions of the fit part, blended with the letters prior
     (w = n / (n + 10)) exactly as served;
  4. the held-out part is used for nothing but measurement: accuracy, ECE and the conformal gate thresholds,
     so the gate's error guarantee rests on decisions that nothing was fitted on.
The probe keeps the fit-part training only (no refit on the held-out rows) for the same reason.
Rows shown to the model as few-shot examples are never used for fitting or measurement.

A fit reads its prompts in one layout and records it: `--layout`, else the schema's own `layout:` when it names one,
else question_first (the layout every probe and calibration in BENCHMARKS.md was measured with). Under `auto` a
request reads a fitted question in the layout it was fitted under.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ._version import __version__
from .artifacts import load_artifacts, save_artifacts
from .engine import Tez
from .errors import BackendUnavailable, InvalidRequest, TezError
from .gate import DEFAULT_ALPHAS
from .gate import thresholds as gate_thresholds
from .prompt import QUESTION_FIRST, STATE_FIRST, fingerprint
from .readout import BLEND_N0, Probe, blend_weight, ece, fit_temperature_logits, fit_temperature_probs, nll, softmax_rows
from .schema import LAYOUTS, Schema, state_key

MIN_LABELS = 20
HOLDOUT = 0.3
PROBE_C = 0.5
FOLDS = 5


def _say_default(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sklearn(what: str):
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise TezError(f"{what} needs scikit-learn: pip install \"tez-decisions[fit]\"") from exc


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def row_hash(state: Any) -> str:
    return hashlib.sha256(state_key(state).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------------------------ data
def read_jsonl(path: str | Path) -> list[tuple[int, Any]]:
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append((i, json.loads(s)))
            except json.JSONDecodeError as exc:
                raise InvalidRequest(f"{path}:{i}: invalid JSON ({exc.msg})") from exc
    return rows


@dataclass
class Labelled:
    rows: dict[str, dict[str, tuple[Any, int]]]           # qid -> state key -> (state, option index)
    sources: dict = field(default_factory=dict)
    problems: dict = field(default_factory=lambda: defaultdict(int))

    def total(self) -> int:
        return sum(len(v) for v in self.rows.values())


def collect_rows(schema: Schema, label_files: Iterable[str | Path] = (), feedback_file: str | Path | None = None,
                 include_examples: bool = True) -> Labelled:
    """Labelled rows per question. Later sources override earlier ones for the same (state, question):
    schema examples, then label files in order, then feedback (the most recent correction wins)."""
    data = Labelled(rows={qid: {} for qid in schema.questions})

    def add(state: Any, qid: Any, label: Any) -> None:
        q = schema.questions.get(qid) if isinstance(qid, str) else None
        if q is None:
            data.problems["unknown_question"] += 1
            return
        if not isinstance(state, (str, dict, list)):
            data.problems["invalid_state"] += 1
            return
        try:
            idx = q.label_index(label, allow_none=True)
        except InvalidRequest:
            data.problems["invalid_label"] += 1
            return
        if q.type == "choice" and idx == len(q.criteria):
            data.problems["abstain_label"] += 1     # "__none__" corrections are kept in feedback, not trained on
            return
        data.rows[qid][state_key(state)] = (state, idx)

    if include_examples:
        for ex in schema.examples:
            for qid, label in ex["labels"].items():
                add(ex["state"], qid, label)
        data.sources["examples"] = len(schema.examples)
    data.sources["labels"] = {}
    for f in label_files:
        count = 0
        for line, row in read_jsonl(f):
            if not isinstance(row, dict) or "state" not in row:
                raise InvalidRequest(f"{f}:{line}: each row needs a state and labels ({{\"state\": ..., \"labels\": {{...}}}})")
            if isinstance(row.get("labels"), dict):
                for qid, label in row["labels"].items():
                    add(row["state"], qid, label)
            elif "question" in row and "label" in row:     # feedback-style rows are accepted too
                add(row["state"], row["question"], row["label"])
            else:
                raise InvalidRequest(f"{f}:{line}: labels must be an object mapping question ids to labels")
            count += 1
        data.sources["labels"][str(f)] = count
    if feedback_file is not None and Path(feedback_file).exists():
        count = 0
        for _, row in read_jsonl(feedback_file):
            if isinstance(row, dict) and row.get("schema", schema.name) == schema.name and "state" in row:
                add(row["state"], row.get("question"), row.get("label"))
                count += 1
        data.sources["feedback"] = {"file": str(feedback_file), "rows": count}
    for qid in schema.questions:      # a row shown to the model as a worked example would leak its own answer
        for st, _ in schema.shots(qid):
            if data.rows[qid].pop(state_key(st), None) is not None:
                data.problems["few_shot_rows_excluded"] += 1
    return data


# ------------------------------------------------------------------------------------------ probe fitting
def _fit_lr(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # lbfgs convergence notes on raw hidden states
        return LogisticRegression(max_iter=3000, C=PROBE_C).fit(X, y)


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    cls = np.unique(ytr)
    if len(cls) < 2:        # one label value in this fold: predict it with 0.9, as in the probe lab
        P = np.full((len(Xte), k), 0.1 / k)
        P[:, int(cls[0])] += 0.9
        return P
    return Probe.from_sklearn(_fit_lr(Xtr, ytr), k=k, n=len(ytr)).predict_many(Xte)


def _oof(X: np.ndarray, y: np.ndarray, k: int, seed: int, folds: int = FOLDS) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probe probabilities on the fit part, and each row's fold training size."""
    n = len(y)
    P, ntr = np.zeros((n, k)), np.zeros(n)
    rng = np.random.default_rng(seed)
    parts = [p for p in np.array_split(rng.permutation(n), min(folds, n)) if len(p)]
    for f, te in enumerate(parts):
        tr = np.concatenate([parts[g] for g in range(len(parts)) if g != f]) if len(parts) > 1 else te
        ntr[te] = len(tr)
        P[te] = _fit_predict(X[tr], y[tr], X[te], k)
    return P, ntr


def _blend_rows(P: np.ndarray, prior: np.ndarray, n: np.ndarray | float) -> np.ndarray:
    w = np.asarray(n, float) / (np.asarray(n, float) + BLEND_N0)
    w = w[:, None] if np.ndim(w) else w
    out = w * P + (1.0 - w) * prior
    return out / out.sum(axis=1, keepdims=True)


def _temper_rows(P: np.ndarray, t: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return softmax_rows(np.log(P), t)


def _measure(P: np.ndarray, y: np.ndarray, alphas: Iterable[float]) -> dict:
    if len(y) == 0:
        return {"accuracy": None, "ece": None, "nll": None, "thresholds": {f"{a:g}": None for a in sorted(set(alphas))}}
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    wrong = pred != y
    return {"accuracy": float(1 - wrong.mean()), "ece": ece(conf, ~wrong), "nll": nll(P, y),
            "thresholds": gate_thresholds(conf, wrong, alphas)}


def _progress(say: Callable, what: str, i: int, n: int) -> None:
    if n >= 50 and (i + 1) % max(1, n // 5) == 0 and i + 1 < n:
        say(f"    {what} {i + 1}/{n}")


def _cut(model: str | None) -> int | None:
    m = re.search(r"[-_.]l(\d+)$", model or "", flags=re.I)
    return int(m.group(1)) if m else None


def _calibration_id(schema: Schema) -> str:
    base = f"{schema.name}@{datetime.now(timezone.utc).date().isoformat()}"
    prev = load_artifacts(schema.artifact_dir)
    if prev is None or not prev.calibration_id.startswith(base):
        return base
    tail = prev.calibration_id[len(base):]
    n = int(tail[1:]) if tail.startswith(".") and tail[1:].isdigit() else 1
    return f"{base}.{n + 1}"


def fit_layout(schema: Schema, layout: str | None = None) -> str:
    """The layout a fit reads: the one asked for, else the schema's own when it names one, else question_first."""
    if layout is not None and layout not in LAYOUTS:
        raise InvalidRequest(f"layout must be one of {', '.join(LAYOUTS)}, got {layout!r}")
    for choice in (layout, schema.layout):
        if choice in (QUESTION_FIRST, STATE_FIRST):
            return choice
    return QUESTION_FIRST


def fit(tez: Tez, schema: Schema, label_files: Iterable[str | Path] = (), feedback: bool = True,
        holdout: float = HOLDOUT, min_labels: int = MIN_LABELS, seed: int = 0, letters: bool = True,
        use_blend: bool = True, alphas: Iterable[float] = DEFAULT_ALPHAS, say: Callable[[str], None] | None = None,
        layout: str | None = None) -> dict:
    """Train probes and calibrations for a schema and write schemas/.tez/<name>/. Returns calibration.json."""
    _sklearn("tez fit")
    say = say or _say_default
    if not 0.0 < holdout < 1.0:
        raise InvalidRequest("holdout must be between 0 and 1")
    if schema.builtin:
        raise InvalidRequest(f"'{schema.name}' is a built-in preset: copy it into a schema file first "
                             f"(tez presets --show {schema.name} > schemas/{schema.name}.yaml)")
    layout = fit_layout(schema, layout)
    fb = tez.feedback_path(schema) if feedback else None
    data = collect_rows(schema, label_files, fb)
    if data.total() == 0:
        raise InvalidRequest(f"no labelled rows for schema '{schema.name}' (give --labels, add examples to the schema, "
                             f"or record feedback)")
    alphas = sorted(set(float(a) for a in alphas) | ({schema.gate_alpha} if schema.gate_alpha else set()))
    letters_model = tez.backend.model_name() if letters else None
    if letters and letters_model == "unknown":
        raise BackendUnavailable(f"letters backend {tez.backend.url} is unreachable")
    needs_probe = any(len(r) >= min_labels for r in data.rows.values())
    embed_model = tez.embedder.model_name() if needs_probe else None
    if needs_probe and embed_model == "unknown":
        raise BackendUnavailable(f"embedding backend {tez.embedder.url} is unreachable")
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cal_id = _calibration_id(schema)
    say(f"fitting '{schema.name}' ({data.total()} labelled decisions, calibration {cal_id}, {layout} layout)")
    questions, probes, skipped, counts, hashes = {}, {}, {}, {}, {}
    embed_dim = None
    for qid, q in schema.questions.items():
        rows = list(data.rows[qid].values())
        n, k, keys = len(rows), len(q.keys()), q.keys()
        entry: dict[str, Any] = {"type": q.type, "options": keys, "n_labels": n}
        if n == 0:
            entry["note"] = "no labels"
            skipped[qid] = "no labels"
            questions[qid] = entry
            say(f"  {qid}: no labels, left zero-shot")
            continue
        states = [s for s, _ in rows]
        y = np.array([i for _, i in rows], dtype=int)
        perm = np.random.default_rng(seed + _stable_int(qid)).permutation(n)
        n_cal = min(max(int(round(holdout * n)), 1), n - 1) if n >= 2 else 0
        cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
        shots = schema.shots(qid)
        counts[qid] = {"total": n, "fit": int(len(fit_idx)), "held_out": int(len(cal_idx)),
                       "by_label": {keys[i]: int((y == i).sum()) for i in range(k) if (y == i).any()}}
        say(f"  {qid} ({q.type}, {k} options): {n} labels, {len(fit_idx)} to fit, {len(cal_idx)} held out")
        Zl, t_letters = None, 1.0
        if letters:
            Z = []
            for i, s in enumerate(states):
                Z.append(tez.letter_logits(q, s, None, shots, layout=layout)[0])
                _progress(say, "letters", i, n)
            Zl = np.stack(Z)
            t_letters = fit_temperature_logits(Zl[fit_idx], y[fit_idx])
            m = _measure(softmax_rows(Zl[cal_idx], t_letters), y[cal_idx], alphas)
            raw = softmax_rows(Zl[cal_idx], 1.0)
            entry["letters"] = {"model": letters_model, "template": tez.backend.template, "layout": layout,
                                "prompt_sha": fingerprint(q, tez.backend.template, q.options(), shots, layout),
                                "temperature": round(t_letters, 4), "n_fit": int(len(fit_idx)), "n_held_out": int(len(cal_idx)),
                                "accuracy": m["accuracy"], "ece": m["ece"],
                                "ece_uncalibrated": ece(raw.max(1), raw.argmax(1) == y[cal_idx]) if len(cal_idx) else None,
                                "nll": m["nll"], "thresholds": m["thresholds"]}
            acc = "n/a" if m["accuracy"] is None else f"{m['accuracy']:.3f}"
            say(f"    letters: held-out accuracy {acc}, temperature {t_letters:.2f}")
        fit_classes = np.unique(y[fit_idx])
        if n < min_labels:
            reason = f"only {n} labels (< {min_labels}): probe skipped" + (", letters calibrated only" if letters else "")
        elif len(fit_classes) < 2:
            reason = f"all {len(fit_idx)} training labels are '{keys[int(fit_classes[0])]}': probe skipped"
        else:
            reason = None
        if reason:
            entry["note"] = reason
            skipped[qid] = reason
            say(f"    {reason}")
            questions[qid] = entry
            continue
        X = []
        for i, s in enumerate(states):
            X.append(tez.embedder.embed(tez.probe_prompt(q, s, shots, layout=layout)).vector)
            _progress(say, "embeddings", i, n)
        X = np.stack(X).astype(np.float64)
        embed_dim = int(X.shape[1])
        Xf, yf = X[fit_idx], y[fit_idx]
        blend_on = bool(use_blend and Zl is not None)
        P_oof, n_tr = _oof(Xf, yf, k, seed + 1 + _stable_int(qid))
        if blend_on:
            P_oof = _blend_rows(P_oof, softmax_rows(Zl[fit_idx], t_letters), n_tr)
        t_probe = fit_temperature_probs(P_oof, yf)
        probe = Probe.from_sklearn(_fit_lr(Xf, yf), k=k, n=int(len(fit_idx)))
        Pc = probe.predict_many(X[cal_idx])
        if blend_on and len(cal_idx):
            Pc = _blend_rows(Pc, softmax_rows(Zl[cal_idx], t_letters), float(len(fit_idx)))
        Pc = _temper_rows(Pc, t_probe) if len(cal_idx) else Pc
        m = _measure(Pc, y[cal_idx], alphas)
        entry["probe"] = {"model": embed_model, "template": tez.embedder.template, "layout": layout,
                          "prompt_sha": fingerprint(q, tez.embedder.template, q.options(), shots, layout),
                          "dim": embed_dim, "n_train": int(len(fit_idx)), "n_held_out": int(len(cal_idx)), "C": PROBE_C,
                          "blend": blend_on, "weight": round(blend_weight(len(fit_idx)), 4) if blend_on else 1.0,
                          "temperature": round(t_probe, 4), "accuracy": m["accuracy"], "ece": m["ece"], "nll": m["nll"],
                          "thresholds": m["thresholds"]}
        probes[qid] = probe
        hashes[qid] = [row_hash(states[i]) for i in fit_idx]
        questions[qid] = entry
        acc = "n/a" if m["accuracy"] is None else f"{m['accuracy']:.3f}"
        kind = f"probe blended with letters (w={blend_weight(len(fit_idx)):.2f})" if blend_on else "probe"
        say(f"    {kind}: held-out accuracy {acc}, temperature {t_probe:.2f}")
        la = (entry.get("letters") or {}).get("accuracy")
        if la is not None and m["accuracy"] is not None and m["accuracy"] + 0.02 < la and len(cal_idx) >= 20:
            say("    note: the probe is below the zero-shot letters on the held-out rows; consider readout=letters or more labels")
    calibration = {"schema": schema.name, "calibration_id": cal_id, "created": created, "tez_version": __version__,
                   "holdout": holdout, "min_labels": min_labels, "blend_n0": BLEND_N0, "alphas": alphas, "layout": layout,
                   "questions": questions}
    manifest = {"schema": schema.name, "calibration_id": cal_id, "created": created, "tez_version": __version__,
                "backend": tez.backend.url, "template": tez.backend.template, "model": letters_model, "layout": layout,
                "embed_backend": tez.embedder.url, "embed_template": tez.embedder.template, "embed_model": embed_model,
                "embed_dim": embed_dim, "layer": "last-token state (llama-server --embeddings --pooling last)",
                "cut": _cut(embed_model), "label_counts": counts, "sources": data.sources, "problems": dict(data.problems),
                "probes": list(probes), "skipped": skipped,
                "settings": {"holdout": holdout, "min_labels": min_labels, "C": PROBE_C, "folds": FOLDS, "seed": seed,
                             "letters": letters, "blend": use_blend, "blend_n0": BLEND_N0, "layout": layout},
                "train_hashes": hashes}
    save_artifacts(schema.artifact_dir, calibration, manifest, probes)
    if schema.name in tez.schemas:
        tez.reload_fitted(schema.name)
    say(f"wrote {schema.artifact_dir}")
    return calibration


# ------------------------------------------------------------------------------------------ suggest
def read_states(path: str | Path) -> list[Any]:
    """Unlabelled states: JSONL rows with a "state" field, bare JSON values, or plain text lines."""
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.rstrip("\r\n")
            if not s.strip():
                continue
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                out.append(s)
                continue
            if isinstance(v, dict) and "state" in v:
                out.append(v["state"])
            elif isinstance(v, (str, dict, list)):
                out.append(v)
            else:
                out.append(s)
    return out


def suggest(tez: Tez, schema: Schema, states: list[Any], n: int = 25, question: str | None = None, seed: int = 0,
            say: Callable[[str], None] | None = None) -> list[dict]:
    """The n most typical states to label first: k-means (n clusters) on PCA-32 of the embeddings, the member
    nearest each centre, largest cluster first. Each state is represented by the mean of its per-question
    last-token states (L2-normalised), or by one question's with `question`."""
    _sklearn("tez suggest")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    say = say or _say_default
    if question is not None and question not in schema.questions:
        raise InvalidRequest(f"unknown question '{question}' in schema '{schema.name}'")
    seen, rows = set(), []
    for s in states:
        key = state_key(s)
        if key not in seen:
            seen.add(key)
            rows.append(s)
    if n <= 0 or not rows:
        return []
    qids = [question] if question else list(schema.questions)
    say(f"embedding {len(rows)} states x {len(qids)} question(s)")
    X = []
    for i, s in enumerate(rows):
        vs = []
        for qid in qids:
            v = tez.embedder.embed(tez.probe_prompt(schema.questions[qid], s, schema.shots(qid))).vector.astype(np.float64)
            vs.append(v / (np.linalg.norm(v) + 1e-9))
        X.append(np.mean(vs, axis=0))
        _progress(say, "embedded", i, len(rows))
    X = np.stack(X)
    n = min(n, len(rows))
    if n == len(rows):
        order, sizes, labels = list(range(len(rows))), np.ones(len(rows), int), np.arange(len(rows))
        picks = [(i, i) for i in order]
    else:
        Z = PCA(n_components=min(32, len(rows), X.shape[1]), random_state=seed).fit_transform(X)
        km = KMeans(n_clusters=n, n_init=3, random_state=seed).fit(Z)
        labels = km.labels_
        sizes = np.bincount(labels, minlength=n)
        used: set[int] = set()
        picks = []
        for c in np.argsort(-sizes, kind="stable"):
            d = np.linalg.norm(Z - km.cluster_centers_[c], axis=1)
            members = [int(i) for i in np.argsort(d) if labels[i] == c and int(i) not in used]
            rest = [int(i) for i in np.argsort(d) if int(i) not in used]
            pick = members[0] if members else (rest[0] if rest else None)
            if pick is not None:
                used.add(pick)
                picks.append((pick, int(c)))
    return [{"state": rows[i], "labels": {}, "_suggest": {"rank": r + 1, "cluster_size": int(sizes[c]), "row": i}}
            for r, (i, c) in enumerate(picks)]


# ------------------------------------------------------------------------------------------ eval
def evaluate(tez: Tez, schema: Schema, label_files: Iterable[str | Path], readout: str = "auto", alpha: float | None = None,
             include_seen: bool = False, say: Callable[[str], None] | None = None, layout: str | None = None) -> dict:
    """Accuracy and ECE per question on labelled rows, with the schema's fitted probes and calibration.
    Rows a probe was trained on are skipped for that question unless include_seen. Each question is read in the
    layout a request with all of the schema's questions would read it (`layout` overrides the schema's)."""
    say = say or _say_default
    if layout is not None and layout not in LAYOUTS:
        raise InvalidRequest(f"layout must be one of {', '.join(LAYOUTS)}, got {layout!r}")
    requested = layout or tez.configured_layout(schema)
    n_all = len(schema.questions)
    data = collect_rows(schema, label_files, None, include_examples=False)
    if data.total() == 0:
        raise InvalidRequest("no labelled rows to evaluate")
    fitted = tez.fitted.get(schema.name)
    trained = {qid: set(h) for qid, h in ((fitted.manifest.get("train_hashes") or {}) if fitted else {}).items()}
    by_state: dict[str, tuple[Any, dict[str, int]]] = {}
    for qid, rows in data.rows.items():
        for key, (state, idx) in rows.items():
            by_state.setdefault(key, (state, {}))[1][qid] = idx
    rec: dict[str, list] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for j, (state, labels) in enumerate(by_state.values()):
        h = row_hash(state)
        for qid, gold in labels.items():
            q = schema.questions[qid]
            r = tez.decide_question(q, state, schema=schema, readout=readout, abstain=False, alpha=alpha,
                                    layout=tez.question_layout(q, schema, requested, n_all))
            if r.readout == "probe" and not include_seen and h in trained.get(qid, ()):
                skipped[qid] += 1
                continue
            rec[qid].append((r.probabilities, gold, r.readout, r.meta.get("decision")))
        _progress(say, "evaluated", j, len(by_state))
    out: dict[str, Any] = {"schema": schema.name, "readout": readout, "alpha": alpha, "layout": requested, "questions": {}}
    all_conf, all_ok = [], []
    for qid in schema.questions:
        items = rec.get(qid, [])
        if not items:
            if skipped.get(qid):
                out["questions"][qid] = {"n": 0, "skipped_seen_in_training": skipped[qid]}
            continue
        P = [np.asarray(p, float) for p, _, _, _ in items]
        gold = np.array([g for _, g, _, _ in items])
        conf = np.array([p.max() for p in P])
        ok = np.array([int(np.argmax(p)) == g for p, g in zip(P, gold)])
        res = {"n": len(items), "accuracy": float(ok.mean()), "ece": ece(conf, ok), "mean_confidence": float(conf.mean()),
               "readouts": dict(sorted({r: sum(1 for _, _, rr, _ in items if rr == r) for r in {x[2] for x in items}}.items()))}
        if skipped.get(qid):
            res["skipped_seen_in_training"] = skipped[qid]
        if alpha is not None:
            acted = np.array([d == "act" for _, _, _, d in items])
            res["acted"] = float(acted.mean())
            res["error_among_acted"] = float((~ok[acted]).mean()) if acted.any() else None
        out["questions"][qid] = res
        all_conf.extend(conf.tolist())
        all_ok.extend(ok.tolist())
    if all_ok:
        out["overall"] = {"n": len(all_ok), "accuracy": float(np.mean(all_ok)), "ece": ece(all_conf, all_ok)}
    return out
