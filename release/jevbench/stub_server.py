#!/usr/bin/env python3
"""A stand-in /v1/systemone server (TypeSafe's wire format, docs/API.md). Standard library only.

It answers every question with a UNIFORM distribution in the exact response shape of docs/API.md, so a client
(release/jevbench/run_public.py, a Jev SDK, the site playground) can be tested before a real model is running.
It knows nothing: every number it returns is chance by construction. Requests are validated the way docs/API.md
describes them (422 with {"error": {...}} for malformed input, 401 when --api-key is set and the key is wrong).

  python release/jevbench/stub_server.py                        # http://127.0.0.1:8799/v1/systemone
  python release/jevbench/stub_server.py --port 8799 --delay-ms 20
  python release/jevbench/stub_server.py --corrupt-every 5      # every 5th answer is an invalid distribution

Endpoints: POST /v1/systemone, GET /v1/models, GET /healthz. The default port is not Tez's 8787, so a client left
on its default address never mistakes the stub for the real server.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TYPES = ("noul", "choice", "score")
MODEL = "tez-stub (uniform probabilities, no model)"


class Invalid(Exception):
    """A request that the real server would reject with 422."""


def validate(body: object) -> dict:
    """Check a /v1/systemone request against docs/API.md and return its questions."""
    if not isinstance(body, dict):
        raise Invalid("the request body must be a JSON object")
    if "state" not in body:
        raise Invalid("`state` is required")
    if not isinstance(body["state"], (str, dict, list)):
        raise Invalid("`state` must be a string, an object or an array")
    if "model" in body and not isinstance(body["model"], str):
        raise Invalid("`model` must be a string")
    if body.get("tez") is not None and not isinstance(body["tez"], dict):
        raise Invalid("`tez` must be an object")
    questions = body.get("questions")
    if questions is None:
        if body.get("schema") is not None:
            raise Invalid(f"unknown schema {body['schema']!r}: the stub loads no schemas")
        raise Invalid("`questions` is required")
    if not isinstance(questions, dict) or not questions:
        raise Invalid("`questions` must be a non-empty object mapping ids to questions")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise Invalid(f"question {qid!r} must be an object")
        qtype = q.get("type")
        if qtype not in TYPES:
            raise Invalid(f"question {qid!r}: `type` must be one of noul, choice, score")
        if q.get("instructions") is not None and not isinstance(q["instructions"], (str, dict, list)):
            raise Invalid(f"question {qid!r}: `instructions` must be a string, an object or an array")
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise Invalid(f"question {qid!r}: a choice needs `criteria` mapping 2 to 255 labels to descriptions")
            for label, desc in criteria.items():
                if desc is not None and not isinstance(desc, (str, dict, list)):
                    raise Invalid(f"question {qid!r}: the description of {label!r} must be a string or null")
        elif qtype == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise Invalid(f"question {qid!r}: a score needs `criteria` as a list of 2 to 10 levels")
        elif criteria is not None and not isinstance(criteria, dict):
            raise Invalid(f"question {qid!r}: noul `criteria` must be an object or null")
    return questions


def confidence(k: int, p_max: float) -> float:
    """docs/API.md: confidence = (k * p_max - 1) / (k - 1)."""
    return max(0.0, (k * p_max - 1.0) / (k - 1))


def uniform_answer(question: dict, abstain: bool) -> dict:
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "noul", "noul": 0.5}
    if qtype == "choice":
        labels = [str(label) for label in question["criteria"]] + (["__none__"] if abstain else [])
        k = len(labels)
        p = 1.0 / k
        return {"type": "choice", "choice": labels[0], "probabilities": {label: p for label in labels},
                "confidence": confidence(k, p)}
    levels = question["criteria"]
    k = len(levels)
    p = 1.0 / k
    return {"type": "score", "score": sum(i * p for i in range(k)),
            "legend": {str(i): (lv if isinstance(lv, str) else json.dumps(lv)) for i, lv in enumerate(levels)},
            "probabilities": {str(i): p for i in range(k)}, "confidence": confidence(k, p)}


def corrupt(answer: dict, n: int) -> dict:
    """Make an answer that breaks JevBench's validity rule, cycling through the ways a server can get it wrong."""
    if answer["type"] == "noul":
        return dict(answer, noul=1.2) if n % 2 == 0 else dict(answer, noul=None)
    probs = dict(answer["probabilities"])
    kind = n % 3
    if kind == 0:                                  # sums to 0.9: outside the 2 % renormalisation band
        probs = {k: v * 0.9 for k, v in probs.items()}
    elif kind == 1:                                # a label is missing
        probs.pop(next(reversed(probs)))
    else:                                          # a label that is not in the question
        probs["__extra__"] = 0.0
    return dict(answer, probabilities=probs)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                 # keep-alive, like a real server
    server_version = "tez-stub/1"

    def log_message(self, fmt, *args):             # quiet unless --verbose
        if self.server.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, obj: object | None) -> None:
        data = b"" if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        if obj is not None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _error(self, status: int, etype: str, message: str) -> None:
        with self.server.lock:
            self.server.stats[str(status)] = self.server.stats.get(str(status), 0) + 1
        self._send(status, {"error": {"type": etype, "message": message}})

    def _authorised(self) -> bool:
        key = self.server.api_key
        return not key or self.headers.get("Authorization", "") == f"Bearer {key}"

    def do_OPTIONS(self):                          # CORS preflight (docs/API.md: CORS open by default)
        self._send(204, None)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/healthz":
            self._send(200, {"status": "ok", "version": "stub", "backend": None, "template": None,
                             "embed_backend": None, "schemas": [], "probes": {}})
        elif path == "/v1/models":
            self._send(200, {"models": [{"name": "tez-latest", "description": MODEL,
                                         "release_date": self.server.started}]})
        else:
            self._error(404, "not_found", f"no route for GET {path}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/v1/systemone":
            return self._error(404, "not_found", f"no route for POST {path}")
        if not self._authorised():
            return self._error(401, "authentication_error", "missing or invalid API key")
        t0 = time.perf_counter()
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._error(422, "invalid_request", f"the body is not valid JSON: {exc}")
        try:
            questions = validate(body)
        except Invalid as exc:
            return self._error(422, "invalid_request", str(exc))
        if self.server.delay_ms:
            time.sleep(self.server.delay_ms / 1000.0)
        abstain = bool((body.get("tez") or {}).get("abstain"))
        answers = {}
        for qid, q in questions.items():
            answer = uniform_answer(q, abstain)
            with self.server.lock:
                self.server.decisions += 1
                n = self.server.decisions
            if self.server.corrupt_every and n % self.server.corrupt_every == 0:
                answer = corrupt(answer, n // self.server.corrupt_every)
            answers[qid] = answer
        resp = {"model": MODEL, "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}
        if self.server.tez_block:
            resp["tez"] = {"latency_ms": round((time.perf_counter() - t0) * 1000.0, 3), "questions": {}}
        with self.server.lock:
            self.server.stats["200"] = self.server.stats.get("200", 0) + 1
        self._send(200, resp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Uniform-probability stand-in for a /v1/systemone server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--api-key", default=None, help="require `Authorization: Bearer <key>` (as `tez serve --api-key`)")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="sleep this long per request, to exercise speed scoring")
    ap.add_argument("--corrupt-every", type=int, default=0,
                    help="make every Nth answer an invalid distribution, to test a client's validity checks (0 = never)")
    ap.add_argument("--no-tez-block", action="store_true", help="omit the optional `tez` block (strict Jev shape)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.verbose = args.verbose
    server.api_key = args.api_key
    server.delay_ms = args.delay_ms
    server.corrupt_every = max(0, args.corrupt_every)
    server.tez_block = not args.no_tez_block
    server.started = date.today().isoformat()
    server.lock = threading.Lock()
    server.decisions = 0
    server.stats = {}
    print(f"stub /v1/systemone listening on http://{args.host}:{args.port}/v1/systemone "
          f"(uniform probabilities; corrupt every {server.corrupt_every or 'never'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"stub stopped: {server.decisions} decisions served, responses by status {server.stats}", flush=True)


if __name__ == "__main__":
    main()
