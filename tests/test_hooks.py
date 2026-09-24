"""Hooks (tez/hooks.py, docs/HOOKS.md): the lifecycle, ordering, skip, the failure rules, and the built-in hooks."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import DOCS_REQUEST, SYNTH_SCHEMA_YAML, synth_rows, write_jsonl
from stub_llama import StubLlama
from tez import (BaseHook, Cache, DecisionContext, DecisionLog, FakeBackend, InvalidRequest, Metrics, OTelHook, Redact,
                 Tez, default_hooks, set_default_hooks)
from tez.cli import build_parser, main
from tez.errors import BackendUnavailable
from tez.fit import fit
from tez.hooks import HookSet, load_hook, normalise_hooks, predicted_label
from tez.server import create_app

QUESTIONS = DOCS_REQUEST["questions"]
STATE = DOCS_REQUEST["state"]


class Recorder(BaseHook):
    def __init__(self, name="rec", log=None):
        self.name = name
        self.log = log if log is not None else []
        self.contexts: list[DecisionContext] = []

    def on_decide_start(self, ctx):
        self.log.append((self.name, "start"))
        self.contexts.append(ctx)

    def on_question_end(self, ctx, qid, answer, meta):
        self.log.append((self.name, "question", qid, answer["type"], meta["readout"]))

    def on_decide_end(self, ctx):
        self.log.append((self.name, "end", ctx.skipped))

    def on_error(self, ctx):
        self.log.append((self.name, "error", type(ctx.error).__name__))

    def on_feedback(self, row):
        self.log.append((self.name, "feedback", row["question"]))


class Boom(BaseHook):
    def __init__(self, event):
        self.event = event

    def on_decide_start(self, ctx):
        if self.event == "start":
            raise RuntimeError("start hook broke")

    def on_decide_end(self, ctx):
        if self.event == "end":
            raise RuntimeError("end hook broke")

    def on_error(self, ctx):
        raise RuntimeError("error hook broke")

    def on_feedback(self, row):
        if self.event == "feedback":
            raise RuntimeError("feedback hook broke")


@pytest.fixture(autouse=True)
def no_default_hooks():
    set_default_hooks(None)
    yield
    set_default_hooks(None)


# ---------------------------------------------------------------------------------------------- lifecycle
def test_lifecycle_and_context():
    rec = Recorder()
    res = Tez(backend="fake", hooks=[rec]).decide(STATE, questions=QUESTIONS, alpha=0.1)
    assert rec.log == [("rec", "start"),
                       ("rec", "question", "is_urgent", "noul", "letters"),
                       ("rec", "question", "topic", "choice", "letters"),
                       ("rec", "question", "anger", "score", "letters"),
                       ("rec", "end", False)]
    ctx = rec.contexts[0]
    assert len(ctx.run_id) == 32 and ctx.response is res and ctx.request["state"] == STATE
    assert ctx.state == STATE and list(ctx.questions) == ["is_urgent", "topic", "anger"] and ctx.schema is None
    assert (ctx.readout, ctx.alpha, ctx.layout, ctx.requested_layout) == ("auto", 0.1, "state_first", "auto")
    assert ctx.usage == res["usage"] and ctx.latency_ms == res["tez"]["latency_ms"] and ctx.error is None
    trace = ctx.traces["topic"]
    assert trace["readout"] == "letters" and trace["layout"] == "state_first" and trace["tokens"] > 0
    assert [c["kind"] for c in trace["calls"]] == ["letters"] and "cache_n" in trace["calls"][0]["timings"]
    assert ctx.traces["anger"]["calls"][0]["timings"]["cache_n"] > 0          # the state was cached by the earlier question


def test_order_defaults_then_engine_then_call_and_add_remove():
    log: list = []
    d, e, c = Recorder("default", log), Recorder("engine", log), Recorder("call", log)
    set_default_hooks([d])
    assert default_hooks() == [d]
    tez = Tez(backend="fake", hooks=e)
    tez.decide("x", questions={"q": QUESTIONS["is_urgent"]}, hooks=[c])
    assert [entry[0] for entry in log if entry[1] == "start"] == ["default", "engine", "call"]
    extra = Recorder("extra", log)
    assert tez.add_hook(extra) is tez and tez.hooks == [e, extra]
    assert tez.remove_hook(e) is True and tez.remove_hook(e) is False and tez.hooks == [extra]
    log.clear()
    tez.decide("x", questions={"q": QUESTIONS["is_urgent"]})
    assert [entry[0] for entry in log if entry[1] == "start"] == ["default", "extra"]


def test_hook_validation():
    with pytest.raises(TypeError, match="instantiate Metrics"):
        Tez(backend="fake", hooks=[Metrics])
    with pytest.raises(TypeError, match="not a hook"):
        normalise_hooks(object())
    assert normalise_hooks(None) == [] and len(normalise_hooks(Metrics())) == 1


def test_skip_answers_without_the_backend():
    class Canned(BaseHook):
        def on_decide_start(self, ctx):
            ctx.skip({"model": "canned", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
                      "tez": {"latency_ms": 999.0, "questions": {}}})

    fb = FakeBackend()
    rec = Recorder()
    res = Tez(backend=fb, hooks=[Canned(), rec]).decide("x", questions=QUESTIONS)
    assert res["model"] == "canned" and fb.calls["letters"] == 0
    assert res["tez"]["latency_ms"] < 999.0                                     # the time actually spent
    assert rec.log == [("rec", "start"), ("rec", "end", True)]                  # no question events after a skip
    with pytest.raises(TypeError):
        DecisionContext(run_id="r").skip("not a dict")


def test_on_error_for_invalid_requests_and_backend_failures():
    rec = Recorder()
    tez = Tez(backend="fake", hooks=[rec])
    with pytest.raises(InvalidRequest):
        tez.handle({"questions": QUESTIONS})
    assert rec.log[-1] == ("rec", "error", "InvalidRequest") and rec.contexts == []
    rec2 = Recorder()
    with pytest.raises(BackendUnavailable):
        Tez(backend=FakeBackend(fail=True), hooks=[rec2]).decide("x", questions=QUESTIONS)
    assert rec2.log == [("rec", "start"), ("rec", "error", "BackendUnavailable")]


def test_failure_rules():
    # hooks_raise=True (default): the hook's exception fails the decision; on_error hooks see it
    rec = Recorder()
    with pytest.raises(RuntimeError, match="start hook broke"):
        Tez(backend="fake", hooks=[Boom("start"), rec]).decide("x", questions={"q": QUESTIONS["is_urgent"]})
    assert rec.log == [("rec", "error", "RuntimeError")]
    with pytest.raises(RuntimeError, match="end hook broke"):
        Tez(backend="fake", hooks=[Boom("end")]).decide("x", questions={"q": QUESTIONS["is_urgent"]})
    # an on_error hook that raises never masks the original error
    with pytest.raises(BackendUnavailable):
        Tez(backend=FakeBackend(fail=True), hooks=[Boom("none")]).decide("x", questions={"q": QUESTIONS["is_urgent"]})
    # hooks_raise=False: logged, recorded on the context, and the decision goes on
    rec = Recorder()
    tez = Tez(backend="fake", hooks=[Boom("start"), Boom("end"), rec], hooks_raise=False)
    res = tez.decide("x", questions={"q": QUESTIONS["is_urgent"]})
    assert res["answers"]["q"]["type"] == "noul"
    assert [(name, event) for name, event, _ in rec.contexts[0].hook_errors] == [("Boom", "on_decide_start"),
                                                                                  ("Boom", "on_decide_end")]


def test_a_hook_may_reject_a_request_with_a_tez_error():
    class Policy(BaseHook):
        def on_decide_start(self, ctx):
            if "forbidden" in str(ctx.state):
                raise InvalidRequest("this state is not allowed here")

    c = TestClient(create_app(Tez(backend="fake", hooks=[Policy()])))
    r = c.post("/v1/systemone", json={"state": "forbidden words", "questions": {"q": QUESTIONS["is_urgent"]}})
    assert r.status_code == 422 and r.json()["error"]["message"] == "this state is not allowed here"


def test_on_feedback(schema_dir: Path):
    rec = Recorder()
    tez = Tez(backend="fake", schemas=schema_dir, hooks=[Redact(), rec])
    out = tez.record_feedback({"schema": "support-triage", "question": "topic", "label": "billing", "run_id": "abc",
                               "state": "Refund to ana@example.com please"})
    assert out == {"ok": True, "schema": "support-triage", "question": "topic", "label": "billing"}
    row = json.loads(tez.feedback_path(tez.schemas["support-triage"]).read_text(encoding="utf-8").splitlines()[-1])
    assert row["state"] == "Refund to [EMAIL] please" and row["run_id"] == "abc" and rec.log == [("rec", "feedback", "topic")]
    failing = Tez(backend="fake", schemas=schema_dir, hooks=[Boom("feedback")])
    before = failing.feedback_path(failing.schemas["support-triage"]).read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        failing.record_feedback({"schema": "support-triage", "question": "topic", "state": "s", "label": "billing"})
    assert failing.feedback_path(failing.schemas["support-triage"]).read_text(encoding="utf-8") == before
    with pytest.raises(InvalidRequest, match="run_id"):
        tez.record_feedback({"schema": "support-triage", "question": "topic", "state": "s", "label": "billing", "run_id": 5})


# ---------------------------------------------------------------------------------------------- DecisionLog
def test_decision_log_rows_feed_tez_fit(tmp_path: Path):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    log_path = tmp_path / "logs" / "decisions.jsonl"
    dlog = DecisionLog(log_path)
    tez = Tez(backend=FakeBackend(), schemas=tmp_path, hooks=[dlog])
    rows = synth_rows(30, seed=6, anger_every=0)
    for r in rows:
        tez.decide(r["state"], schema="synth")
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 30 == dlog.written
    first = lines[0]
    assert first["state"] == rows[0]["state"] and first["labels"] == {} and first["schema"] == "synth"
    assert set(first["predicted"]) == {"topic", "is_urgent", "anger"} and first["predicted"]["is_urgent"] in ("true", "false")
    assert isinstance(first["predicted"]["anger"], int) and 0 <= first["decisions"]["topic"]["confidence"] <= 1
    assert len(first["run_id"]) == 32 and first["cached"] is False and first["model"].startswith("tez-")
    with pytest.raises(InvalidRequest, match="no labelled rows"):          # unreviewed rows are never trained on
        fit(tez, tez.schemas["synth"], [log_path], say=lambda m: None)
    for line, r in zip(lines, rows):                                        # a reviewer fills in the labels
        line["labels"] = {"topic": r["labels"]["topic"]}
    reviewed = write_jsonl(tmp_path / "reviewed.jsonl", lines)
    cal = fit(tez, tez.schemas["synth"], [reviewed], say=lambda m: None)
    assert cal["questions"]["topic"]["n_labels"] == 30


def test_decision_log_sampling(tmp_path: Path):
    none = DecisionLog(tmp_path / "none.jsonl", sample=0.0)
    some = DecisionLog(tmp_path / "some.jsonl", sample=0.5, seed=1)
    tez = Tez(backend="fake", hooks=[none, some])
    for i in range(40):
        tez.decide(f"message {i}", questions={"q": QUESTIONS["is_urgent"]})
    assert none.written == 0 and not (tmp_path / "none.jsonl").exists()
    assert 5 < some.written < 35
    for bad in (-0.1, 1.5, True, "all"):
        with pytest.raises(ValueError):
            DecisionLog(tmp_path / "x.jsonl", sample=bad)


def test_predicted_label():
    assert predicted_label({"type": "noul", "noul": 0.7}) == "true"
    assert predicted_label({"type": "choice", "choice": "b"}) == "b"
    assert predicted_label({"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}) == 2


# ---------------------------------------------------------------------------------------------- Redact
def test_redact_patterns():
    r = Redact()
    text = ("Mail ana.silva@example.co.uk or call +1 (555) 123-4567; card 4111 1111 1111 1111; "
            "IBAN DE89 3704 0044 0532 0130 00. Invoice 4411 dated 2026-09-24, order 12345678.")
    out = r.redact(text)
    assert out == ("Mail [EMAIL] or call [PHONE]; card [CARD]; IBAN [IBAN]. Invoice 4411 dated 2026-09-24, "
                   "order 12345678.")
    assert r.redact("4111 1111 1111 1112") == "4111 1111 1111 1112"            # fails the Luhn check: not a card
    assert r.redact({"from": "bo@x.io", "lines": ["ok", "call 0044 20 7946 0958"], "n": 3}) == \
        {"from": "[EMAIL]", "lines": ["ok", "call [PHONE]"], "n": 3}
    custom = Redact([r"ACME-\d+", "email"], replacement="***")
    assert custom.redact("ticket ACME-991 from a@b.io") == "ticket *** from ***"
    with pytest.raises(ValueError, match="invalid redaction pattern"):
        Redact(["(unclosed"])


@pytest.mark.parametrize("prefix", ["-" * 70, "_" * 60, "%20" * 22, "a" * 64, "x.y" * 40, "+" * 500])
def test_redact_an_address_glued_to_a_long_run_of_address_characters(prefix):
    """A local part is at most 64 characters: the last 64 before the @ go with the address, whatever comes before."""
    out = Redact(["email"]).redact(f"Reply to {prefix}john.smith@example.com today")
    assert "john.smith" not in out and "example.com" not in out and "[EMAIL]" in out
    kept = out[len("Reply to "):out.index("[EMAIL]")]
    assert prefix.startswith(kept) and len(prefix) + len("john.smith") - len(kept) <= 64


def test_redact_keeps_address_boundaries():
    r = Redact(["email"])
    assert r.redact("a@b.co@c.com, x@@y.com and caféjohn@x.io") == "[EMAIL]@c.com, x@@y.com and café[EMAIL]"
    assert r.redact("no address @example.com or a@b") == "no address @example.com or a@b"


@pytest.mark.parametrize("kind", ["letters", "dotted", "digits", "iban-like", "spaced-digits", "parentheses", "dashes",
                                  "percent-20", "glued-addresses", "at-domains", "a-at", "at-a.bc", "long-labels"])
def test_redact_is_linear_on_long_runs(kind):
    """A server state can be 50,000 characters of anything: no pattern may backtrack quadratically (the email pattern
    once took seconds on a run of letters)."""
    import time
    text = {"letters": "a" * 50_000, "dotted": "a." * 25_000, "digits": "1" * 50_000 + "x",
            "iban-like": "DE" + "1" * 50_000, "spaced-digits": "1 " * 25_000 + "x", "parentheses": "(" * 50_000,
            "dashes": "-" * 50_000, "percent-20": "%20" * 16_666, "glued-addresses": ("x" * 70 + "jo@ex.com") * 630,
            "at-domains": ("@" + "a" * 63 + ".") * 770, "a-at": "a@" * 25_000, "at-a.bc": "@a.bc" * 10_000,
            "long-labels": ("a@" + ("b" * 63 + ".") * 9 + "1 ") * 84}[kind]
    t0 = time.perf_counter()
    Redact().redact(text)
    assert time.perf_counter() - t0 < 1.0


def test_redact_runs_before_the_model_reads_the_state():
    fb = FakeBackend()
    Tez(backend=fb, hooks=[Redact(["email"])]).decide("Write to ana@example.com", questions={"q": QUESTIONS["is_urgent"]})
    assert "ana@example.com" not in fb.prompts[0] and "Write to [EMAIL]" in fb.prompts[0]


# ---------------------------------------------------------------------------------------------- Cache
def test_cache_hits_skip_the_backend_and_respect_the_key():
    fb = FakeBackend()
    cache = Cache(maxsize=2)
    rec = Recorder()
    tez = Tez(backend=fb, hooks=[cache, rec])
    first = tez.decide(STATE, questions=QUESTIONS)
    calls = fb.calls["letters"]
    again = tez.decide(STATE, questions=QUESTIONS)
    assert fb.calls["letters"] == calls and again["answers"] == first["answers"]
    assert again["tez"]["cached"] is True and again["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert "cached" not in first["tez"] and rec.log[-1] == ("rec", "end", True)
    tez.decide(STATE, questions=QUESTIONS, readout="letters", alpha=0.1)        # other options: another key
    assert fb.calls["letters"] == calls + 3 and cache.stats()["misses"] == 2 and cache.stats()["hits"] == 1
    tez.decide("another state", questions=QUESTIONS)
    assert len(cache) == 2                                                      # least recently used evicted
    tez.decide(STATE, questions=QUESTIONS)
    assert cache.stats()["hits"] == 1                                           # it was the evicted one
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(ValueError):
        Cache(maxsize=0)


def test_cache_is_invalidated_by_a_new_fit(tmp_path: Path):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    fb = FakeBackend()
    cache = Cache()
    tez = Tez(backend=fb, schemas=tmp_path, hooks=[cache])
    tez.decide("invoice refund", schema="synth")
    fit(tez, tez.schemas["synth"], [write_jsonl(tmp_path / "l.jsonl", synth_rows(60, seed=2, anger_every=0))],
        say=lambda m: None)
    res = tez.decide("invoice refund", schema="synth")
    assert "cached" not in res["tez"] and res["tez"]["questions"]["topic"]["readout"] == "probe"


def test_a_shared_cache_keys_on_the_engine_settings_and_the_model_name():
    cache = Cache()
    set_default_hooks([cache])                                   # process-wide: every engine below shares it
    gemma = "gemma-4-12b-q8_0"
    q = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}

    def run(tez):
        return tez.decide("Server down, customers blocked!", questions=q)

    run(Tez(backend="fake", model_name=gemma, default_temperature="off"))
    assert run(Tez(backend="fake", model_name=gemma, default_temperature="off"))["tez"]["cached"] is True
    tempered = run(Tez(backend="fake", model_name=gemma))       # default temperature auto: 6.01 for this model
    assert "cached" not in tempered["tez"] and tempered["tez"]["questions"]["urgent"]["temperature"] == 6.01
    for other in (Tez(backend="fake", model_name="another-model", default_temperature="off"),
                  Tez(backend="fake", model_name=gemma, default_temperature="off", embed_backend="fake",
                      embed_template="qwen3")):
        assert "cached" not in run(other)["tez"]
    with StubLlama() as stub:
        wide = Tez(backend=stub.url, n_probs=200)
        first = run(wide)
        narrow = run(Tez(backend=stub.url, n_probs=1))            # letters past the first get the floor
        assert "cached" not in narrow["tez"] and narrow["answers"] != first["answers"]
        stub.model_path = "/models/another-model-Q8_0.gguf"       # the model is swapped behind the same URL
        assert "cached" not in run(Tez(backend=stub.url, n_probs=200))["tez"]    # an engine reading the new name misses
        assert run(wide)["tez"]["cached"] is True                 # one that read the old name keeps it (documented) ...
        cache.clear()
        assert "cached" not in run(wide)["tez"]                   # ... so a swap needs Cache.clear()


# ---------------------------------------------------------------------------------------------- Metrics
def test_metrics():
    m = Metrics()
    tez = Tez(backend="fake", hooks=[Cache(), m])
    tez.decide(STATE, questions=QUESTIONS, alpha=0.1)
    tez.decide(STATE, questions=QUESTIONS, alpha=0.1)
    with pytest.raises(InvalidRequest):
        tez.handle({"state": "x"})
    s = m.snapshot()
    assert s["decisions"] == 2 and s["skipped"] == 1 and s["questions"] == 3 and s["errors"] == 1
    assert s["errors_by_type"] == {"invalid_request": 1} and s["readouts"] == {"letters": 3}
    assert s["gate"] == {"escalate": 3} and s["layouts"] == {"state_first": 3} and s["input_tokens"] > 0
    assert s["latency_ms"]["count"] == 2 and s["latency_ms"]["p50"] is not None
    text = m.prometheus()
    assert "tez_decisions_total 2" in text and 'tez_errors_total{type="invalid_request"} 1' in text
    assert 'tez_questions_total{readout="letters"} 3' in text and "tez_latency_ms_count 2" in text
    m.reset()
    assert m.snapshot()["decisions"] == 0


# ---------------------------------------------------------------------------------------------- OTelHook
class FakeSpan:
    def __init__(self, name, attributes):
        self.name, self.attributes, self.events, self.ended, self.exceptions = name, dict(attributes), [], False, []

    def add_event(self, name, attributes):
        self.events.append((name, attributes))

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class FakeTracer:
    def __init__(self):
        self.spans: list[FakeSpan] = []

    def start_span(self, name, attributes=None):
        self.spans.append(FakeSpan(name, attributes or {}))
        return self.spans[-1]


def test_otel_hook_with_a_tracer():
    tracer = FakeTracer()
    tez = Tez(backend="fake", hooks=[OTelHook(tracer=tracer)])
    tez.decide(STATE, questions=QUESTIONS)
    span = tracer.spans[0]
    assert span.name == "tez.decide" and span.ended and span.attributes["tez.questions"] == 3
    assert span.attributes["tez.layout"] == "state_first" and span.attributes["tez.input_tokens"] > 0
    assert [e[1]["tez.question"] for e in span.events] == ["is_urgent", "topic", "anger"]
    assert all(STATE not in str(v) for v in span.attributes.values())          # the state is never recorded
    with pytest.raises(BackendUnavailable):
        Tez(backend=FakeBackend(fail=True), hooks=[OTelHook(tracer=tracer)]).decide("x", questions=QUESTIONS)
    assert tracer.spans[-1].ended and isinstance(tracer.spans[-1].exceptions[0], BackendUnavailable)


def test_otel_hook_with_the_sdk():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    Tez(backend="fake", hooks=[OTelHook(tracer=provider.get_tracer("tez"))]).decide("x", questions=QUESTIONS)
    (span,) = exporter.get_finished_spans()
    assert span.name == "tez.decide" and span.attributes["tez.questions"] == 3 and len(span.events) == 3


def test_otel_hook_names_the_extra_when_opentelemetry_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with pytest.raises(ImportError, match=r'tez-decisions\[otel\]'):
        OTelHook()


# ---------------------------------------------------------------------------------------------- loading and the server
def test_load_hook(tmp_path: Path, monkeypatch):
    (tmp_path / "myhooks.py").write_text(textwrap.dedent("""
        from tez import BaseHook, Metrics
        class Counter(BaseHook):
            def __init__(self):
                self.n = 0
            def on_decide_end(self, ctx):
                self.n += 1
        shared = Metrics()
        def make():
            return Counter()
        not_a_hook = 3
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert type(load_hook("myhooks:Counter")).__name__ == "Counter"
    import myhooks
    assert load_hook("myhooks:shared") is myhooks.shared
    assert type(load_hook("myhooks:make")).__name__ == "Counter"
    for bad, fragment in [("myhooks", "package.module:object"), ("myhooks:nothing", "has no"),
                          ("no_such_module_xyz:Hook", "cannot import")]:
        with pytest.raises(ValueError, match=fragment):
            load_hook(bad)
    with pytest.raises(TypeError, match="not a hook"):
        load_hook("myhooks:not_a_hook")
    (tmp_path / "here").mkdir()
    (tmp_path / "here" / "localhooks_xyz.py").write_text("from tez import Metrics\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "here")                     # a file here is not importable: the path is not widened
    with pytest.raises(ValueError, match=r"localhooks_xyz is in the current directory.*PYTHONPATH=\."):
        load_hook("localhooks_xyz:Metrics")
    with pytest.raises(ValueError, match="must be installed or on PYTHONPATH"):
        load_hook("elsewhere_xyz:Hook")
    args = build_parser().parse_args(["serve", "--hook", "myhooks:Counter", "--hook", "myhooks:shared",
                                      "--decision-log", str(tmp_path / "d.jsonl"), "--hook-errors", "log"])
    from tez.cli import _make_tez
    tez = _make_tez(args)
    assert [type(h).__name__ for h in tez.hooks] == ["Counter", "Metrics", "DecisionLog"] and tez.hooks_raise is False


def test_cli_decide_writes_the_decision_log(tmp_path: Path, capsys):
    path = tmp_path / "decisions.jsonl"
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps({"q": QUESTIONS["is_urgent"]}), encoding="utf-8")
    assert main(["decide", "--backend", "fake", "--questions", str(qfile), "--state", "now!", "--decision-log", str(path)]) == 0
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["state"] == "now!" and row["labels"] == {} and list(row["predicted"]) == ["q"]
    capsys.readouterr()
    assert main(["decide", "--backend", "fake", "--questions", str(qfile), "--state", "x", "--hook", "nope"]) == 2
    assert "package.module:object" in capsys.readouterr().err


def test_server_run_id_headers_match_the_hooks():
    rec = Recorder()
    c = TestClient(create_app(Tez(backend="fake", hooks=[rec])))
    r = c.post("/v1/systemone", json=DOCS_REQUEST)
    assert r.status_code == 200
    rid = r.headers["x-tez-run-id"]
    assert rid == rec.contexts[0].run_id == r.headers["x-typesafe-request-id"] and r.headers["x-tez-layout"] == "state_first"
    timing = r.headers["server-timing"]
    assert timing.startswith("tez;dur=") and "backend;dur=" in timing and "total;dur=" in timing
    bad = c.post("/v1/systemone", json={"state": "x"})
    assert bad.status_code == 422 and len(bad.headers["x-tez-run-id"]) == 32 and rec.log[-1][1] == "error"
    for path in ("/v1/models", "/healthz", "/nope"):
        resp = c.get(path)
        assert len(resp.headers["x-typesafe-request-id"]) == 32 and "total;dur=" in resp.headers["server-timing"]
        assert "x-tez-run-id" not in resp.headers
    assert c.get("/healthz").headers["x-typesafe-request-id"] != c.get("/healthz").headers["x-typesafe-request-id"]


def test_server_hook_failure_is_a_500_with_headers():
    c = TestClient(create_app(Tez(backend="fake", hooks=[Boom("end")])), raise_server_exceptions=False)
    r = c.post("/v1/systemone", json={"state": "x", "questions": {"q": QUESTIONS["is_urgent"]}},
               headers={"Origin": "http://localhost:8000"})
    assert r.status_code == 500
    assert r.json() == {"error": {"type": "internal_error", "message": "RuntimeError: end hook broke"}}
    assert "access-control-allow-origin" in r.headers and "x-typesafe-request-id" in r.headers


def test_hookset_is_falsy_without_hooks():
    assert not HookSet([]) and HookSet([Metrics()])
