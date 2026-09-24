"""Batches (Tez.decide_many, POST /v1/systemone/batch, RemoteTez.decide_batch, tez decide --states-file) and the
request limits (413 payload_too_large)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import DOCS_REQUEST
from stub_http import StubServer
from tez import BaseHook, DecisionLog, FakeBackend, InvalidRequest, Tez
from tez.cli import build_parser, main
from tez.engine import Limits
from tez.errors import BackendUnavailable, PayloadTooLarge
from tez.integrations import RemoteTez
from tez.prompt import state_prefix
from tez.server import create_app

QUESTIONS = DOCS_REQUEST["questions"]
STATES = ["Help! My payouts have been failing for 3 days.", "The app crashes on start", {"text": "pricing for 50 seats"}]


class FailAfter(FakeBackend):
    """A fake backend that goes down after `n` letter readouts."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def letters(self, prompt, k):
        if self.calls["letters"] >= self.n:
            self.calls["letters"] += 1
            raise BackendUnavailable("backend went away")
        return super().letters(prompt, k)


# ---------------------------------------------------------------------------------------------- the engine
def test_decide_many_matches_decide_in_order():
    tez = Tez(backend="fake")
    results = tez.decide_many(STATES, questions=QUESTIONS)
    assert len(results) == 3
    for state, res in zip(STATES, results):
        assert res["answers"] == tez.decide(state, questions=QUESTIONS)["answers"]
    assert results[0]["answers"]["topic"]["choice"] == "billing" and results[1]["answers"]["topic"]["choice"] == "technical"


def test_batch_response_shape_and_state_major_order():
    fb = FakeBackend()
    res = Tez(backend=fb).decide_batch(STATES[:2], questions=QUESTIONS)
    assert list(res) == ["model", "results", "usage", "tez"] and set(res["tez"]) == {"latency_ms", "run_id"}
    assert res["model"].endswith("(fake, letters)")
    assert res["usage"] == {"input_tokens": sum(r["usage"]["input_tokens"] for r in res["results"]), "output_tokens": 0}
    # each state's questions run back to back, state first, so the state stays in the backend's cache
    assert all(p.startswith(state_prefix(STATES[0], "gemma4")) for p in fb.prompts[:3])
    assert all(p.startswith(state_prefix(STATES[1], "gemma4")) for p in fb.prompts[3:])


def test_per_state_errors_do_not_fail_the_batch():
    results = Tez(backend="fake").decide_many(["fine", 42, None, "also fine"], questions=QUESTIONS)
    assert "answers" in results[0] and "answers" in results[3]
    assert results[1] == {"error": {"type": "invalid_request", "message": "state must be a string, object or array"}}
    assert results[2]["error"]["message"] == "state is required"


@pytest.mark.parametrize("body, fragment", [
    ({"questions": QUESTIONS}, "states must be a non-empty array"),
    ({"states": [], "questions": QUESTIONS}, "states must be a non-empty array"),
    ({"states": "one", "questions": QUESTIONS}, "states must be a non-empty array"),
    ({"state": "x", "states": ["y"], "questions": QUESTIONS}, "not state"),
    ({"states": ["x"], "questions": {"q": {"type": "maybe", "instructions": "x"}}}, "type must be one of"),
    ({"states": ["x"]}, "questions is required"),
    ({"states": ["x"], "questions": QUESTIONS, "tez": {"layout": "sideways"}}, "tez.layout"),
    ([1, 2], "JSON object"),
])
def test_bad_shared_fields_fail_the_whole_batch(body, fragment):
    with pytest.raises(InvalidRequest, match=fragment):
        Tez(backend="fake").handle_batch(body)


def test_backend_down_before_any_decision_fails_the_batch():
    with pytest.raises(BackendUnavailable):
        Tez(backend=FakeBackend(fail=True)).decide_many(STATES, questions=QUESTIONS)


def test_backend_down_mid_batch_stops_calling_it():
    fb = FailAfter(4)                              # the first state's 3 questions, then one more
    results = Tez(backend=fb).decide_many(STATES, questions=QUESTIONS)
    assert "answers" in results[0]
    assert results[1] == {"error": {"type": "backend_unavailable", "message": "backend went away"}}
    assert results[2]["error"]["type"] == "backend_unavailable" and results[2]["error"]["message"].startswith("not attempted")
    assert fb.calls["letters"] == 5                 # 3 for the first state, 1 + 1 refused for the second, none after


def test_hooks_see_every_item_with_its_own_run_id(tmp_path: Path):
    seen = []

    class Rec(BaseHook):
        def on_decide_end(self, ctx):
            seen.append((ctx.run_id, ctx.parent_run_id, ctx.index))

    log_path = tmp_path / "d.jsonl"
    res = Tez(backend="fake", hooks=[Rec(), DecisionLog(log_path)]).decide_batch(STATES, questions=QUESTIONS)
    rid = res["tez"]["run_id"]
    assert seen == [(f"{rid}.{i}", rid, i) for i in range(3)]
    assert [json.loads(line)["run_id"] for line in log_path.read_text(encoding="utf-8").splitlines()] == \
           [f"{rid}.{i}" for i in range(3)]


def test_a_refused_batch_reaches_on_error_once_with_the_batch_run_id():
    seen = []

    class Rec(BaseHook):
        def on_decide_start(self, ctx):
            seen.append(("start", ctx.run_id))

        def on_error(self, ctx):
            seen.append((ctx.run_id, type(ctx.error).__name__, ctx.index, ctx.request is not None))

    tez = Tez(backend="fake", hooks=[Rec()])
    with pytest.raises(InvalidRequest):
        tez.handle_batch({"states": ["x"], "questions": {"q": {"type": "maybe", "instructions": "x"}}}, run_id="b1")
    with pytest.raises(PayloadTooLarge):
        tez.handle_batch({"states": STATES, "questions": QUESTIONS}, run_id="b2", limits=Limits(max_batch=2))
    assert seen == [("b1", "InvalidRequest", None, True), ("b2", "PayloadTooLarge", None, True)]
    r = TestClient(create_app(tez)).post("/v1/systemone/batch", json={"states": [], "questions": QUESTIONS})
    assert r.status_code == 422 and seen[-1] == (r.headers["x-tez-run-id"], "InvalidRequest", None, True)


def test_a_hook_failing_on_one_item_fails_only_that_item():
    class Picky(BaseHook):
        def on_decide_start(self, ctx):
            if ctx.index == 1:
                raise RuntimeError("not this one")

    results = Tez(backend="fake", hooks=[Picky()]).decide_many(STATES, questions=QUESTIONS)
    assert "answers" in results[0] and "answers" in results[2]
    assert results[1] == {"error": {"type": "internal_error", "message": "RuntimeError: not this one"}}


def test_limits():
    tez = Tez(backend="fake")
    with pytest.raises(PayloadTooLarge, match="too many states in one batch: 3 .* --max-batch"):
        tez.handle_batch({"states": STATES, "questions": QUESTIONS}, limits=Limits(max_batch=2))
    with pytest.raises(PayloadTooLarge, match="too many questions: 3 .* --max-questions"):
        tez.handle_batch({"states": STATES, "questions": QUESTIONS}, limits=Limits(max_questions=2))
    res = tez.handle_batch({"states": ["short", "x" * 101], "questions": QUESTIONS}, limits=Limits(max_state_chars=100))
    assert "answers" in res["results"][0] and res["results"][1]["error"]["type"] == "payload_too_large"
    with pytest.raises(PayloadTooLarge, match="state is too large: 21 characters"):     # objects count as JSON
        tez.handle({"state": {"k": "a" * 12}, "questions": QUESTIONS}, limits=Limits(max_state_chars=20))
    assert Limits.unlimited() == Limits(None, None, None, None)
    tez.handle({"state": "x" * 100_000, "questions": QUESTIONS})          # the Python API has no limits by default


# ---------------------------------------------------------------------------------------------- HTTP
def test_batch_endpoint():
    c = TestClient(create_app(Tez(backend="fake")))
    r = c.post("/v1/systemone/batch", json={"states": STATES, "questions": QUESTIONS, "tez": {"readout": "letters"}})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3 and body["tez"]["run_id"] == r.headers["x-tez-run-id"]
    assert r.headers["server-timing"].startswith("tez;dur=")
    assert "POST /v1/systemone/batch" in c.get("/").json()["endpoints"]
    r = c.post("/v1/systemone/batch", json={"states": [], "questions": QUESTIONS})
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request"
    down = TestClient(create_app(Tez(backend=FakeBackend(fail=True))))
    assert down.post("/v1/systemone/batch", json={"states": ["x"], "questions": QUESTIONS}).status_code == 503


def test_http_limits_are_413():
    c = TestClient(create_app(Tez(backend="fake"), limits=Limits(max_body_bytes=2000, max_questions=2,
                                                                 max_state_chars=50, max_batch=2)))
    for path, body, fragment in [
        ("/v1/systemone", {"state": "x", "questions": QUESTIONS}, "too many questions: 3"),
        ("/v1/systemone", {"state": "y" * 51, "questions": {"q": QUESTIONS["is_urgent"]}}, "state is too large"),
        ("/v1/systemone/batch", {"states": ["a", "b", "c"], "questions": {"q": QUESTIONS["is_urgent"]}}, "too many states"),
        ("/v1/systemone", {"state": "z" * 3000, "questions": {"q": QUESTIONS["is_urgent"]}}, "over 2,000 bytes"),
    ]:
        r = c.post(path, json=body)
        assert r.status_code == 413, (path, r.text)
        assert r.json()["error"]["type"] == "payload_too_large" and fragment in r.json()["error"]["message"]

    def chunks():                                   # no Content-Length: the stream is cut at the limit
        yield b'{"state": "'
        for _ in range(10):
            yield b"w" * 500
        yield b'"}'

    r = c.post("/v1/systemone", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and "over 2,000 bytes" in r.json()["error"]["message"]
    ok = c.post("/v1/systemone", json={"state": "fine", "questions": {"q": QUESTIONS["is_urgent"]}})
    assert ok.status_code == 200
    r = c.post("/v1/feedback", json={"schema": "s", "question": "q", "state": "f" * 60, "label": "x"})
    assert r.status_code == 413


def test_remote_batch_matches_in_process():
    engine = Tez(backend="fake")
    with StubServer(engine) as srv:
        remote = RemoteTez(srv.url)
        batch = remote.decide_batch(STATES, questions=QUESTIONS, readout="letters")
        many = remote.decide_many(STATES[:1], questions=QUESTIONS)
    local = engine.decide_many(STATES, questions=QUESTIONS)
    assert [r["answers"] for r in batch["results"]] == [r["answers"] for r in local]
    assert many[0]["answers"] == local[0]["answers"]
    assert srv.requests[0]["path"] == "/v1/systemone/batch"
    assert srv.requests[0]["body"] == {"states": STATES, "questions": QUESTIONS, "tez": {"readout": "letters"}}
    assert srv.requests[1]["body"] == {"states": STATES[:1], "questions": QUESTIONS}


# ---------------------------------------------------------------------------------------------- CLI
def test_cli_states_file(tmp_path: Path, capsys):
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"state": "The app crashes on start"}\n{"state": {"text": "refund please"}}\nplain text line\n\n',
                    encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert main(["decide", "--backend", "fake", "--questions", str(qfile), "--states-file", str(rows), "--out", str(out)]) == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and all("answers" in x for x in lines)
    assert lines[0]["answers"]["topic"]["choice"] == "technical"
    assert "decided 3 of 3 states -> " in capsys.readouterr().err
    assert main(["decide", "--backend", "fake", "--questions", str(qfile), "--states-file", str(rows)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_cli_positional_text_and_state_sources(tmp_path: Path, capsys):
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    assert main(["decide", "My invoice is wrong", "--backend", "fake", "--questions", str(qfile), "--compact"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"]["topic"]["choice"] == "billing"
    for argv in (["decide", "--backend", "fake", "--questions", str(qfile)],
                 ["decide", "text", "--state", "other", "--backend", "fake", "--questions", str(qfile)],
                 ["decide", "text", "--out", "o.jsonl", "--backend", "fake", "--questions", str(qfile)]):
        assert main(argv) == 2
        assert "error:" in capsys.readouterr().err
    args = build_parser().parse_args(["serve", "--max-batch", "0", "--max-questions", "8"])
    assert (args.max_batch, args.max_questions, args.max_body_bytes, args.max_state_chars) == (0, 8, 2 * 1024 * 1024, 50_000)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--max-batch", "-1"])
