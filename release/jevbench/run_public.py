#!/usr/bin/env python3
"""Run the public JevBench tiers through a /v1/systemone endpoint and score them like experiments/bench_jevbench.py.

Standard library only, so it runs next to any server that speaks TypeSafe's wire format (docs/API.md). One request
at a time, no concurrency, caller wall time including the network: the timing rule the JevBench README states for
self-hosted endpoints.

  python release/jevbench/run_public.py                          # Tez at http://127.0.0.1:8787/v1/systemone
  python release/jevbench/run_public.py --endpoint http://127.0.0.1:8799/v1/systemone --out stub_check.json
  python release/jevbench/run_public.py --tiers original --limit 5  # smoke test

Items: data/jevbench/{original,easy,hard}.jsonl, the public tiers (original is the leaderboard's "standard" tier).
Each item is one request with one question, sent as stored (type, instructions, criteria); choice options go in
the item's `labels` order, as bench_jevbench.py orders them (--criteria-order stored keeps the file's key order).

Scoring, identical to experiments/bench_jevbench.py:
  intelligence  max(0, (accuracy - chance) / (1 - chance)) * 100 per tier, chance = mean of 1 / |labels|
  ece15         ECE with 15 equal-width bins on the max probability (experiments/bench_h2h.py ece15)
  nll           mean of -log p(gold)
  speed_score   100 - 20 log10(median seconds / 0.1)
  by_family, and pair_consistency on the original tier (paraphrase pairs share a group)
Added here, not in bench_jevbench.py:
  validity      a distribution must cover exactly the label set, lie in [0, 1] and sum to 1; sums within 2 % are
                renormalised, anything else is invalid and counts as wrong (JevBench README, v1.4.0). Transport
                errors also count as wrong (bench_jevbench.py skipped them; with zero errors the two agree).
  p95_s, speed_score_p95 and speed_axis = mean of score(p50) and score(p95) (JevBench README, v1.4.0). The README
                also says self-hosted latency is "adjusted x2 (+0.15 s on our own servers)"; speed_axis_selfhosted
                applies 2 * s + 0.15 s, our reading of that sentence, not a published formula.
  brier, mean_confidence, and gold_fidelity: mean total variation distance to provenance.gold_probs on the hard
                items that carry one (informative only; not the leaderboard's calibration formula).
  public_weighted.intelligence: tier weights hard 0.30, easy 0.14, standard 0.28 renormalised over the public tiers
                run (the judge tier, 0.28, is not public). Indicative only: the leaderboard scores held-out items.

Writes results/jevbench_tez_server.json by default. Exit status 0 when every item got a valid answer, 1 otherwise
(the results file is written either way).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import http.client
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "http://127.0.0.1:8787/v1/systemone"
DEFAULT_OUT = ROOT / "results" / "jevbench_tez_server.json"
TIERS = ("original", "easy", "hard")
TIER_NAMES = {"original": "standard (original)", "easy": "easy", "hard": "hard"}
TIER_WEIGHTS = {"hard": 0.30, "easy": 0.14, "original": 0.28}   # leaderboard: hard 30, easy 14, standard 28, judge 28
YES = {"yes", "true"}
RENORM_BAND = 0.02
EPS = 1e-9
RUNNER_VERSION = "1.0"


# ----------------------------------------------------------------------------- data
def _parse(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:  # noqa: BLE001
        return json.loads(v)


def load_tier(data_dir: Path, tier: str, criteria_order: str = "labels"):
    """Items of one public tier as request-ready dicts, plus the file's SHA-256."""
    raw = (data_dir / f"{tier}.jsonl").read_bytes()
    items = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels = [str(x) for x in _parse(r["labels"])]
        q = _parse(r["question"])
        qtype = q.get("type") or ("noul" if set(labels) <= {"no", "yes", "false", "true"} else "choice")
        criteria = q.get("criteria")
        if qtype == "choice":
            criteria = criteria if isinstance(criteria, dict) else {}
            if criteria_order == "labels" or set(map(str, criteria)) != set(labels):
                criteria = {lab: criteria.get(lab) for lab in labels}
        elif qtype == "score" and not isinstance(criteria, list):
            criteria = list(labels)
        items.append(dict(id=r["id"], tier=tier, family=r.get("family"), group=r.get("group"), qtype=qtype,
                          labels=labels, gold=labels.index(str(r["expected"])), state=r["state"],
                          question={"type": qtype, "instructions": q.get("instructions", ""), "criteria": criteria},
                          gold_probs=(r.get("provenance") or {}).get("gold_probs")))
    return items, hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------------------- HTTP
class Client:
    """A persistent HTTP/1.1 connection (like requests.Session), standard library only."""

    def __init__(self, endpoint: str, api_key: str | None, timeout: float):
        u = urlsplit(endpoint)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise SystemExit(f"not an http(s) URL: {endpoint}")
        self.https = u.scheme == "https"
        self.host, self.port = u.hostname, u.port or (443 if self.https else 80)
        self.path = (u.path or "/v1/systemone") + (f"?{u.query}" if u.query else "")
        self.base = u.path.rsplit("/v1/systemone", 1)[0] if u.path.endswith("/v1/systemone") else None
        self.timeout = timeout
        self.conn = None
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _connection(self):
        if self.conn is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            self.conn = cls(self.host, self.port, timeout=self.timeout)
        return self.conn

    def _drop(self):
        if self.conn is not None:
            self.conn.close()
        self.conn = None

    def call(self, method: str, path: str, payload=None):
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in (0, 1):
            try:
                conn = self._connection()
                conn.request(method, path, body=body, headers=self.headers)
                resp = conn.getresponse()
                return resp.status, resp.read()
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, ConnectionResetError,
                    BrokenPipeError, ConnectionAbortedError):
                self._drop()                       # a kept-alive connection the server closed: reconnect once
                if attempt:
                    raise
            except (OSError, http.client.HTTPException):
                self._drop()
                raise
        raise RuntimeError("unreachable")

    def get_json(self, suffix: str):
        """Best-effort GET of /v1/models or /healthz for the manifest."""
        if self.base is None:
            return {"skipped": "endpoint path does not end in /v1/systemone"}
        try:
            status, data = self.call("GET", self.base + suffix)
            return json.loads(data.decode("utf-8")) if status == 200 else {"status": status, "body": data.decode("utf-8", "replace")[:500]}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------------------- answers
class InvalidAnswer(Exception):
    pass


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def distribution(item: dict, answer) -> list[float]:
    """The answer as probabilities in the item's label order, or InvalidAnswer (JevBench's validity rule)."""
    labels, qtype = item["labels"], item["qtype"]
    if not isinstance(answer, dict):
        raise InvalidAnswer("no answer for the question")
    if answer.get("type") not in (None, qtype):
        raise InvalidAnswer(f"answer type {answer.get('type')!r}, expected {qtype!r}")
    if qtype == "noul":
        p = answer.get("noul")
        if not _number(p) or not -EPS <= p <= 1 + EPS:
            raise InvalidAnswer(f"noul must be a number in [0, 1], got {p!r}")
        p = min(1.0, max(0.0, float(p)))
        return [p if lab in YES else 1.0 - p for lab in labels]
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        raise InvalidAnswer("`probabilities` missing")
    got = {str(k): v for k, v in probs.items()}
    if set(got) != set(labels) or len(got) != len(labels):
        raise InvalidAnswer(f"probabilities cover {sorted(got)}, not exactly the label set {sorted(labels)}")
    out = []
    for lab in labels:
        v = got[lab]
        if not _number(v) or not -EPS <= v <= 1 + EPS:
            raise InvalidAnswer(f"probability of {lab!r} is {v!r}, not in [0, 1]")
        out.append(min(1.0, max(0.0, float(v))))
    total = sum(out)
    if abs(total - 1.0) > RENORM_BAND:
        raise InvalidAnswer(f"probabilities sum to {total:.4f}, outside the 2 % band")
    return [v / total for v in out]


def argmax(p: list[float]) -> int:
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best                                    # first maximum, as numpy.argmax


# ----------------------------------------------------------------------------- scoring
def ece15(conf, corr, bins: int = 15) -> float:
    """Identical to experiments/bench_h2h.py ece15: numpy.linspace edges, bins (lo, hi]."""
    n = len(conf)
    if n == 0:
        return float("nan")
    step = 1.0 / bins
    edges = [i * step for i in range(bins)] + [1.0]
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(conf) if lo < c <= hi]
        if idx:
            e += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(corr[i] for i in idx) / len(idx))
    return e


def percentile(xs, q: float) -> float:
    """numpy.percentile with the default linear interpolation."""
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def speed_score(seconds: float) -> float:
    """bench_jevbench.py: 100 - 20 log10(s / 0.1 s)."""
    return 100 - 20 * math.log10(max(seconds, 1e-3) / 0.1)


def _mean(xs):
    return statistics.fmean(xs) if xs else None


def score_tier(rows: list[dict], tier: str) -> dict:
    """Summary of one tier. Rows need status ('ok' | 'invalid' | 'error'), gold, k, family, group, s (seconds or
    None) and, when ok, probabilities (label order) and pred."""
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    correct = [r["status"] == "ok" and r["pred"] == r["gold"] for r in rows]
    acc = sum(correct) / n
    chance = statistics.fmean(1 / r["k"] for r in rows)
    conf = [max(r["probabilities"]) for r in ok]
    corr = [float(r["pred"] == r["gold"]) for r in ok]
    secs = [r["s"] for r in rows if r.get("s") is not None]
    p50 = statistics.median(secs) if secs else float("nan")
    p95 = percentile(secs, 95)
    out = dict(
        n=n, n_valid=len(ok), n_invalid=sum(r["status"] == "invalid" for r in rows),
        n_errors=sum(r["status"] == "error" for r in rows),
        accuracy=acc, chance=chance, intelligence=max(0.0, (acc - chance) / (1 - chance)) * 100,
        ece15=ece15(conf, corr) if ok else None,
        nll=_mean([-math.log(max(r["probabilities"][r["gold"]], 1e-12)) for r in ok]),
        brier=_mean([sum((p - (1.0 if i == r["gold"] else 0.0)) ** 2 for i, p in enumerate(r["probabilities"])) for r in ok]),
        mean_confidence=_mean(conf),
        median_s=p50, p95_s=p95, speed_score=speed_score(p50) if secs else None,
        speed_score_p95=speed_score(p95) if secs else None,
        speed_axis=(speed_score(p50) + speed_score(p95)) / 2 if secs else None,
        speed_axis_selfhosted=(speed_score(2 * p50 + 0.15) + speed_score(2 * p95 + 0.15)) / 2 if secs else None,
        by_family={f: statistics.fmean(c for r, c in zip(rows, correct) if r["family"] == f)
                   for f in sorted({r["family"] for r in rows}, key=str)},
    )
    if tier == "original":                         # 36 paraphrase pairs share a group
        groups: dict = {}
        for r, c in zip(rows, correct):
            groups.setdefault(r["group"], []).append(c)
        pairs = [v for v in groups.values() if len(v) == 2]
        out["pair_consistency"] = statistics.fmean(a == b for a, b in pairs) if pairs else None
    fid = [r for r in ok if r.get("gold_probs")]
    if fid:
        tvd = [0.5 * sum(abs(p - float(r["gold_probs"].get(lab, 0.0))) for p, lab in zip(r["probabilities"], r["labels"]))
               for r in fid]
        out["gold_fidelity"] = dict(n=len(fid), mean_total_variation=statistics.fmean(tvd),
                                    note="informative only; not the leaderboard's calibration formula")
    return out


def public_weighted(summary: dict) -> dict | None:
    tiers = [t for t in summary if t in TIER_WEIGHTS]
    if not tiers:
        return None
    w = sum(TIER_WEIGHTS[t] for t in tiers)
    return dict(intelligence=sum(TIER_WEIGHTS[t] * summary[t]["intelligence"] for t in tiers) / w,
                weights={t: TIER_WEIGHTS[t] for t in tiers}, tiers=tiers,
                note="leaderboard tier weights (hard 0.30, easy 0.14, standard 0.28) renormalised over the public "
                     "tiers run; the judge tier (0.28) is not public. Indicative only: the leaderboard uses held-out items.")


def _f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def markdown(summary: dict, weighted: dict | None) -> str:
    lines = ["| tier (n) | accuracy | chance | intelligence | ECE-15 | NLL | p50 s | p95 s | speed score p50 / p95 | invalid + errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, s in summary.items():
        lines.append(f"| {TIER_NAMES.get(t, t)} ({s['n']}) | {_f(s['accuracy'])} | {_f(s['chance'])} | {_f(s['intelligence'], 1)} | "
                     f"{_f(s['ece15'])} | {_f(s['nll'])} | {_f(s['median_s'])} | {_f(s['p95_s'])} | "
                     f"{_f(s['speed_score'], 1)} / {_f(s['speed_score_p95'], 1)} | {s['n_invalid'] + s['n_errors']} |")
    if weighted:
        lines.append("")
        lines.append(f"Public-tier intelligence, leaderboard weights over {', '.join(weighted['tiers'])} "
                     f"(judge tier not public): {_f(weighted['intelligence'], 1)} (indicative only).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- run
def run_item(client: Client, item: dict, model: str, qid: str) -> dict:
    payload = {"model": model, "state": item["state"], "questions": {qid: item["question"]}}
    row = dict(id=item["id"], tier=item["tier"], family=item["family"], group=item["group"], qtype=item["qtype"],
               labels=item["labels"], gold=item["gold"], k=len(item["labels"]), gold_probs=item["gold_probs"])
    t0 = time.perf_counter()
    try:
        status, data = client.call("POST", client.path, payload)
    except Exception as exc:  # noqa: BLE001
        return dict(row, status="error", reason=f"{type(exc).__name__}: {exc}", s=None)
    row["s"] = time.perf_counter() - t0
    if status != 200:
        return dict(row, status="error", reason=f"HTTP {status}: {data.decode('utf-8', 'replace')[:300]}")
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return dict(row, status="error", reason=f"response is not JSON: {exc}")
    answer = (body.get("answers") or {}).get(qid) if isinstance(body, dict) else None
    tez = body.get("tez") if isinstance(body, dict) and isinstance(body.get("tez"), dict) else {}
    row.update(model=body.get("model") if isinstance(body, dict) else None, server_latency_ms=tez.get("latency_ms"),
               readout=((tez.get("questions") or {}).get(qid) or {}).get("readout"))
    try:
        p = distribution(item, answer)
    except InvalidAnswer as exc:
        return dict(row, status="invalid", reason=str(exc), answer=answer)
    row.update(status="ok", probabilities=p, pred=argmax(p))
    if item["qtype"] == "choice" and isinstance(answer.get("choice"), str):
        row["choice_matches_argmax"] = answer["choice"] == item["labels"][row["pred"]]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Public JevBench tiers through a /v1/systemone endpoint.")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "jevbench"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default="tez-latest", help="the request's `model` field")
    ap.add_argument("--api-key", default=os.environ.get("TEZ_API_KEY"), help="sent as a Bearer token (default $TEZ_API_KEY)")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    ap.add_argument("--warmup", type=int, default=1, help="unscored requests sent before timing starts")
    ap.add_argument("--limit", type=int, default=0, help="first N items per tier only (smoke tests)")
    ap.add_argument("--criteria-order", choices=("labels", "stored"), default="labels")
    ap.add_argument("--question-id", default="decision")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    data_dir = Path(args.data_dir)
    loaded, data_manifest = {}, {}
    for t in tiers:
        items, sha = load_tier(data_dir, t, args.criteria_order)
        data_manifest[t] = dict(file=f"{t}.jsonl", sha256=sha, items=len(items))
        loaded[t] = items[: args.limit] if args.limit else items

    client = Client(args.endpoint, args.api_key, args.timeout)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    models, health = client.get_json("/v1/models"), client.get_json("/healthz")
    first = loaded[tiers[0]] if tiers else []
    for item in first[: max(0, args.warmup)]:
        w = run_item(client, item, args.model, args.question_id)
        if w["status"] == "error":
            print(f"warm-up request failed: {w['reason']}")

    summary, rows = {}, []
    for t in tiers:
        recs = []
        for i, item in enumerate(loaded[t], 1):
            r = run_item(client, item, args.model, args.question_id)
            if r["status"] != "ok" and sum(x["status"] != "ok" for x in recs) < 5:
                print(f"  {r['status'].upper()} {r['id']}: {r['reason'][:160]}")
            recs.append(r)
            if i % 25 == 0 or i == len(loaded[t]):
                print(f"  {t}: {i}/{len(loaded[t])}", flush=True)
        if not recs:
            continue
        summary[t] = score_tier(recs, t)
        rows += recs
        s = summary[t]
        print(f"{t:9s} n={s['n']} acc={_f(s['accuracy'])} chance={_f(s['chance'])} intelligence={_f(s['intelligence'], 1)} "
              f"ece15={_f(s['ece15'])} nll={_f(s['nll'])} p50={_f(s['median_s'])}s p95={_f(s['p95_s'])}s "
              f"speed={_f(s['speed_score'], 1)} invalid={s['n_invalid']} errors={s['n_errors']}"
              + (f" pairs={_f(s['pair_consistency'])}" if s.get("pair_consistency") is not None else ""), flush=True)

    weighted = public_weighted(summary)
    table = markdown(summary, weighted)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = dict(
        runner="release/jevbench/run_public.py", runner_version=RUNNER_VERSION, started_utc=started, finished_utc=finished,
        endpoint=args.endpoint, request_model=args.model, server_models=sorted({r["model"] for r in rows if r.get("model")}),
        models_endpoint=models, healthz=health, data_dir=str(data_dir), data=data_manifest, tiers=tiers,
        criteria_order=args.criteria_order, question_id=args.question_id, warmup=args.warmup, limit=args.limit or None,
        timeout_s=args.timeout, timing="caller wall time per request including the network; one request at a time",
        python=platform.python_version(), platform=platform.platform(),
        choice_label_disagreements=sum(1 for r in rows if r.get("choice_matches_argmax") is False),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, public_weighted=weighted, manifest=manifest, markdown_table=table,
                                   rows=rows), indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print(table)
    print(f"\nwrote {out}")
    bad = sum(s["n_invalid"] + s["n_errors"] for s in summary.values())
    if bad:
        print(f"WARNING: {bad} item(s) had no valid answer and were scored as wrong.")
    return 1 if bad or not summary else 0


if __name__ == "__main__":
    sys.exit(main())
