"""The in-process backend (tez/inproc.py) and the engine's batched reads, against a stub of llama.cpp's library
(tests/stub_libllama.py): no GPU, no model file contents, no llama.cpp needed."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from conftest import DOCS_REQUEST, SYNTH_SCHEMA_YAML, synth_rows, write_jsonl
from stub_libllama import StubApi, embed_ref, letters_ref, tokenize
from tez import BackendRequestError, BackendUnavailable, InprocBackend, LlamaCppBackend, Tez
from tez.backends import make_backend
from tez.cli import build_parser, main
from tez.doctor import run_inproc_doctor
from tez.hooks import BaseHook
from tez.inproc import find_library, lib_file, options_from_env
from tez.prompt import build_prompt
from tez.schema import parse_question

QUESTIONS = DOCS_REQUEST["questions"]
STATE = DOCS_REQUEST["state"]
BLOB = "sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"


@pytest.fixture
def gguf(tmp_path: Path) -> Path:
    """A model file named like an Ollama blob (hash name): the model name then comes from its metadata."""
    p = tmp_path / BLOB
    p.write_bytes(b"GGUF")
    return p


def backend(path: Path, api: StubApi | None = None, **kw) -> InprocBackend:
    return InprocBackend(path, "gemma4", api=api or StubApi(), **kw)


def prompts_about(state: str, n: int = 5) -> tuple[list[str], list[int]]:
    """n state-first prompts about one state (the questions of one request), and their option counts."""
    qs = [parse_question(f"q{i}", {"type": "choice", "instructions": f"Which team handles it? (question {i})",
                                   "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}}
                         if i % 2 == 0 else {"type": "noul", "instructions": f"Is it urgent? (question {i})"})
          for i in range(n)]
    return [build_prompt(q, state, "gemma4", layout="state_first") for q in qs], [len(q.options()) for q in qs]


# ---------------------------------------------------------------------------------------------- library discovery
def test_find_library_explicit_env_then_path(tmp_path: Path):
    a, b, c = (tmp_path / x for x in "abc")
    for d in (a, b, c):
        d.mkdir()
        (d / lib_file("llama")).write_bytes(b"")
    server = c / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    server.write_bytes(b"")
    server.chmod(0o755)
    env = {"TEZ_LLAMA_LIB": str(b), "PATH": str(c)}
    assert find_library(a, env) == a.resolve()
    assert find_library(a / lib_file("llama"), env) == a.resolve()          # the library file itself
    assert find_library(None, env) == b.resolve()
    assert find_library(None, {"PATH": str(c)}) == c.resolve()               # next to llama-server
    with pytest.raises(BackendUnavailable, match=r"--llama-lib"):
        find_library(tmp_path / "missing", env)                              # a wrong explicit setting is an error
    with pytest.raises(BackendUnavailable, match="TEZ_LLAMA_LIB"):
        find_library(None, {"TEZ_LLAMA_LIB": str(tmp_path), "PATH": str(c)})
    with pytest.raises(BackendUnavailable, match="was not found.*b11100"):
        find_library(None, {"PATH": str(tmp_path / "nowhere")})


def test_options_from_env():
    env = {"TEZ_LLAMA_LIB": "/opt/llama", "TEZ_N_CTX": "8192", "TEZ_N_GPU_LAYERS": "0"}
    assert options_from_env(env.get) == {"lib": "/opt/llama", "n_ctx": 8192, "n_gpu_layers": 0}
    with pytest.raises(ValueError, match="TEZ_N_BATCH"):
        options_from_env({"TEZ_N_BATCH": "lots"}.get)


# ---------------------------------------------------------------------------------------------- loading
def test_lazy_load_names_and_info(gguf: Path):
    api = StubApi()
    b = make_backend(f"inproc:{gguf}", inproc={"n_ctx": 1024}, model_name=None)
    b._api = api
    assert isinstance(b, InprocBackend) and b.url == f"inproc:{gguf}" and not b.loaded
    assert b.known_model_name() is None                       # a hash-named blob: the name needs the metadata
    assert api.loads == 0 and b.health() == {"ok": True, "status": "not loaded"}
    assert b.model_name() == "gemma-4-12b-q8_0"               # what LlamaCppBackend derives from llama-server's /props
    assert api.loads == 1 and b.loaded and b.health()["status"] == "ok"
    info = b.info()
    assert set(info) == {"model_path", "model_alias", "ftype", "n_params", "n_embd", "n_ctx", "build"}
    assert info["ftype"] == "Q8_0" and info["n_ctx"] == 1024 and info["build"].startswith("b11100-7ab4ee7ba")
    named = InprocBackend(gguf.with_name("Qwen3.5-4B-Q8_0.gguf"), "qwen3", api=StubApi())
    assert named.known_model_name() == "qwen3.5-4b-q8_0"


def test_context_params_are_set_explicitly(gguf: Path):
    api = StubApi()
    backend(gguf, api, n_ctx=2048, n_seq_max=8).load()
    cp = api.ctx_params
    assert cp["n_outputs_max_per_seq"] == 8 and cp["n_outputs_max"] == 8     # the library's default per sequence is 1
    assert (cp["n_ctx"], cp["n_batch"], cp["n_ubatch"], cp["n_seq_max"]) == (2048, 2048, 512, 8)
    assert cp["embeddings"] is False and cp["pooling_type"] == 0 and cp["kv_unified"] and cp["swa_full"]
    assert api.model_params["n_gpu_layers"] == -1


def test_no_gpu_fails_loudly_unless_cpu_is_asked_for(gguf: Path):
    api = StubApi(registries=(("CPU", 1),), preload_errors={"ggml-cuda.dll": "[WinError 126] module not found"})
    with pytest.raises(BackendUnavailable, match=r"no GPU.*WinError 126.*--n-gpu-layers 0"):
        backend(gguf, api).load()
    assert api.loads == 0                                     # refused before loading the weights
    b = backend(gguf, api, n_gpu_layers=0).load()
    assert b.loaded and api.model_params["n_gpu_layers"] == 0


def test_missing_model_and_bad_settings(tmp_path: Path):
    with pytest.raises(BackendUnavailable, match="no GGUF file"):
        backend(tmp_path / "none.gguf").letters("x", 2)
    with pytest.raises(ValueError, match="n_ctx"):
        InprocBackend(tmp_path / "m.gguf", n_ctx=0)
    with pytest.raises(ValueError, match="inproc:PATH"):
        make_backend("inproc:")
    with pytest.raises(ValueError, match="unknown in-process setting"):
        make_backend("inproc:m.gguf", inproc={"n_threads_x": 2})


# ---------------------------------------------------------------------------------------------- single reads
def test_letters_are_exact_log_probabilities(gguf: Path):
    b = backend(gguf)
    p = build_prompt(parse_question("t", QUESTIONS["topic"]), STATE, "gemma4")
    r = b.letters(p, 3)
    assert np.allclose(r.logits, letters_ref(p, 3), atol=1e-5)
    assert r.tokens == len(tokenize(p)) and r.timings["prompt_n"] == r.tokens and r.timings["cache_n"] == 0
    e = b.embed(p)
    assert np.allclose(e.vector, embed_ref(p)) and e.vector.dtype == np.float32


def test_single_reads_reuse_the_prefix_like_a_prompt_cache(gguf: Path):
    api = StubApi()
    b = backend(gguf, api)
    (p1, p2, *_), (k1, k2, *_) = prompts_about(STATE)
    b.letters(p1, k1)
    r = b.letters(p2, k2)
    shared = next(i for i, (x, y) in enumerate(zip(tokenize(p1), tokenize(p2))) if x != y)
    assert r.timings["cache_n"] == shared and r.timings["prompt_n"] == len(tokenize(p2)) - shared
    assert np.allclose(r.logits, letters_ref(p2, k2), atol=1e-5)
    assert (0, shared, -1) in api.seq_rms                    # the tail of the previous prompt was removed
    again = b.letters(p2, k2)                                 # the same prompt: only its last token is evaluated again
    assert again.timings["prompt_n"] == 1 and np.allclose(again.logits, r.logits)
    off = backend(gguf, StubApi(), cache_prompt=False)
    off.letters(p1, k1)
    assert off.letters(p2, k2).timings["cache_n"] == 0


def test_hybrid_models_never_roll_back(gguf: Path):
    api = StubApi(hybrid=True)
    b = backend(gguf, api)
    ps, ks = prompts_about(STATE, 4)
    for p, k in zip(ps, ks):
        assert np.allclose(b.letters(p, k).logits, letters_ref(p, k), atol=1e-5)
    got = b.read_many(ps, ks)
    for (r, _), p, k in zip(got, ps, ks):
        assert np.allclose(r.logits, letters_ref(p, k), atol=1e-5)
    ext = b.letters(ps[0] + " more", 2)                       # an exact extension of nothing held: a fresh read
    assert np.allclose(ext.logits, letters_ref(ps[0] + " more", 2), atol=1e-5)
    assert api.partial_rm_refused == [] and all(p0 <= 0 and p1 < 0 for _, p0, p1 in api.seq_rms)


def test_context_overflow_is_a_request_error_and_the_backend_recovers(gguf: Path):
    b = backend(gguf, n_ctx=64)
    with pytest.raises(BackendRequestError, match="context size"):
        b.letters("x" * 200, 2)
    assert np.allclose(b.letters("short prompt", 2).logits, letters_ref("short prompt", 2), atol=1e-5)


# ---------------------------------------------------------------------------------------------- batched reads
def test_batched_read_matches_single_reads_in_one_decode(gguf: Path):
    api = StubApi()
    b = backend(gguf, api)
    ps, ks = prompts_about(STATE, 6)
    got = b.read_many(ps, ks)
    for (r, e), p, k in zip(got, ps, ks):
        assert e is None and np.allclose(r.logits, letters_ref(p, k), atol=1e-5)
    prefix = got[0][0].timings["batch_prefix_n"]
    assert prefix > len(tokenize(STATE)) and all(r.timings["batch_n"] == 6 for r, _ in got)
    assert [d["outputs"] for d in api.decodes] == [0, 6]      # the prefix once, then every suffix in ONE decode
    assert api.decodes[1]["seqs"] == [1, 2, 3, 4, 5, 6] and len(api.seq_cps) == 6
    assert api.held() == {0: prefix}                          # only the prefix is left, on sequence 0
    evaluated = sum(r.timings["prompt_n"] for r, _ in got)
    assert evaluated == sum(len(tokenize(p)) for p in ps) - 5 * prefix == got[0][0].timings["batch_prompt_n"]
    api.decodes.clear()
    b.read_many(ps[:3], ks[:3])                               # the same state again: the prefix is reused from sequence 0
    assert [d["outputs"] for d in api.decodes] == [3]


def test_batched_embeddings_read_only_the_last_tokens_with_embeddings_on(gguf: Path):
    api = StubApi()
    b = backend(gguf, api)
    ps, ks = prompts_about(STATE, 4)
    got = b.read_many(ps, [ks[0], 0, ks[2], ks[3]], [True, True, False, False])
    assert np.allclose(got[0][0].logits, letters_ref(ps[0], ks[0]), atol=1e-5)          # letters and state of one prompt
    assert np.allclose(got[0][1].vector, embed_ref(ps[0])) and got[1][0] is None
    assert np.allclose(got[1][1].vector, embed_ref(ps[1]))
    assert got[2][1] is None and np.allclose(got[3][0].logits, letters_ref(ps[3], ks[3]), atol=1e-5)
    with_embd = [d for d in api.decodes if d["embeddings"]]
    assert len(with_embd) == 1 and with_embd[0]["n"] == 2 == with_embd[0]["outputs"]   # the two last tokens only


def test_batches_split_into_waves_when_sequences_or_the_cache_run_out(gguf: Path):
    ps, ks = prompts_about(STATE, 7)
    for kw in ({"n_seq_max": 3}, {"n_ctx": 1100}, {"n_batch": 40}):
        api = StubApi()
        got = backend(gguf, api, **kw).read_many(ps, ks, [i % 3 == 0 for i in range(7)])
        for (r, e), p, k, i in zip(got, ps, ks, range(7)):
            assert np.allclose(r.logits, letters_ref(p, k), atol=1e-5), kw
            assert (e is not None) == (i % 3 == 0) and (e is None or np.allclose(e.vector, embed_ref(p)))
        assert len(api.decodes) > 2 and max(max(d["seqs"]) for d in api.decodes) < api.ctx_params["n_seq_max"]


def test_duplicate_prompts_are_read_once(gguf: Path):
    api = StubApi()
    ps, ks = prompts_about(STATE, 2)
    got = backend(gguf, api).read_many([ps[0], ps[1], ps[0]], [ks[0], ks[1], ks[0]])
    assert got[0][0].timings["batch_n"] == 2 and np.allclose(got[0][0].logits, got[2][0].logits)
    assert api.decodes[-1]["outputs"] == 2


def test_prompts_without_a_common_prefix_and_bad_arguments(gguf: Path):
    b = backend(gguf)
    got = b.read_many(["alpha", "beta"], [2, 2])              # only BOS in common
    assert [np.allclose(r.logits, letters_ref(p, 2), atol=1e-5) for (r, _), p in zip(got, ["alpha", "beta"])] == [True] * 2
    with pytest.raises(ValueError, match="letters readout"):
        b.read_many(["a", "b"], [0, 2])
    with pytest.raises(ValueError, match="same length"):
        b.read_many(["a"], [1, 2])
    assert b.read_many([], []) == []


def test_concurrent_calls_are_serialised(gguf: Path):
    b = backend(gguf, StubApi(decode_delay=0.002))
    ps, ks = prompts_about(STATE, 6)
    errors, out = [], {}

    def work(i: int) -> None:
        try:
            out[i] = b.letters(ps[i], ks[i]).logits if i % 2 else b.read_many(ps, ks)[i][0].logits
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(np.allclose(out[i], letters_ref(ps[i], ks[i]), atol=1e-5) for i in range(6))


# ---------------------------------------------------------------------------------------------- the engine
class Capture(BaseHook):
    def __init__(self):
        self.contexts = []

    def on_decide_end(self, ctx):
        self.contexts.append(ctx)


def twin_engines(gguf: Path, **kw) -> tuple[Tez, Tez]:
    """The same stub model behind two engines: one reads in batches, the other question by question."""
    batched = Tez(backend=backend(gguf, StubApi()), **kw)
    single = Tez(backend=backend(gguf, StubApi()), **kw)
    single.backend.batched = False
    return batched, single


def same_answers(a: dict, b: dict) -> None:
    assert a["answers"].keys() == b["answers"].keys()
    for qid, x in a["answers"].items():
        y = b["answers"][qid]
        assert x.keys() == y.keys()
        for k, v in x.items():
            if isinstance(v, dict) and all(isinstance(x, float) for x in v.values()):
                assert v.keys() == y[k].keys() and np.allclose(list(v.values()), list(y[k].values()), atol=1e-9)
            elif isinstance(v, float):
                assert v == pytest.approx(y[k], abs=1e-9)
            else:
                assert v == y[k]
    assert a["tez"]["questions"] == b["tez"]["questions"] and a["usage"] == b["usage"]


def test_engine_reads_a_multi_question_request_in_one_batch(gguf: Path):
    batched, single = twin_engines(gguf)
    cap = Capture()
    res = batched.decide(STATE, questions=QUESTIONS, hooks=[cap])
    same_answers(res, single.decide(STATE, questions=QUESTIONS))
    assert res["model"].startswith("tez-") and "gemma-4-12b-q8_0" in res["model"]
    assert all("temperature" in m for m in res["tez"]["questions"].values())    # the measured unfitted defaults
    ctx = cap.contexts[0]
    (batch,) = ctx.batches
    assert batch["kind"] == "read_many" and batch["group"] == "state" and batch["prompts"] == 3
    assert batch["questions"] == list(QUESTIONS) and batch["tokens"] == res["usage"]["input_tokens"] and batch["ms"] > 0
    for qid in QUESTIONS:
        (call,) = ctx.traces[qid]["calls"]
        assert call["kind"] == "letters" and call["batch"] == 0 and call["tokens"] > 0 and call["ms"] > 0
        assert call["timings"]["batch_n"] == 3
    assert batched.backend_ms(ctx) == pytest.approx(batch["ms"])
    assert [d["outputs"] for d in batched.backend._api.decodes] == [0, 3]


def test_engine_one_question_and_tournaments_read_singly(gguf: Path):
    batched, single = twin_engines(gguf)
    one = {"topic": QUESTIONS["topic"]}
    cap = Capture()
    same_answers(batched.decide(STATE, questions=one, hooks=[cap]), single.decide(STATE, questions=one))
    assert cap.contexts[0].batches == []
    big = {"topic": QUESTIONS["topic"], "urgent": QUESTIONS["is_urgent"],
           "which": {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": None for i in range(30)}}}
    res = batched.decide(STATE, questions=big, hooks=[cap])
    same_answers(res, single.decide(STATE, questions=big))
    ctx = cap.contexts[1]
    assert ctx.batches[0]["questions"] == ["topic", "urgent"]                   # the tournament read on its own
    assert all("batch" not in c for c in ctx.traces["which"]["calls"]) and len(ctx.traces["which"]["calls"]) == 3


def test_engine_question_first_and_abstain_are_batched_too(gguf: Path):
    batched, single = twin_engines(gguf)
    for tez_opts in ({"layout": "question_first"}, {"abstain": True}):
        cap = Capture()
        body = {"state": STATE, "questions": QUESTIONS, "tez": tez_opts}
        res = batched.handle(body, hooks=[cap])
        same_answers(res, single.handle(body))
        assert cap.contexts[0].batches[0]["group"] == ("rest" if "layout" in tez_opts else "state")


def test_engine_batches_probe_reads_and_matches_single_reads(gguf: Path, tmp_path: Path):
    from tez.fit import fit
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    labels = write_jsonl(tmp_path / "labels.jsonl", synth_rows(40, seed=3, anger_every=0))
    batched, single = twin_engines(gguf, schemas=tmp_path)
    fit(batched, batched.schemas["synth"], [labels], say=lambda m: None, layout="state_first")
    single.reload_fitted("synth")
    cap = Capture()
    state = "invoice refund charged immediately"
    res = batched.decide(state, schema="synth", hooks=[cap])
    same_answers(res, single.decide(state, schema="synth"))
    assert res["tez"]["questions"]["topic"]["readout"] == "probe"
    ctx = cap.contexts[0]
    kinds = sorted(c["kind"] for c in ctx.traces["topic"]["calls"])
    assert kinds == ["embed", "letters"] and all(c["batch"] == 0 for c in ctx.traces["topic"]["calls"])
    assert ctx.batches[0]["prompts"] == 3                     # the probe prompt is the letters prompt: read once


def test_batch_endpoint_reads_each_state_in_one_batch(gguf: Path):
    batched, single = twin_engines(gguf)
    cap = Capture()
    states = [STATE, "The app crashes when I log in.", "What does the enterprise plan cost?"]
    got = batched.decide_many(states, questions=QUESTIONS, hooks=[cap])
    for a, b in zip(got, single.decide_many(states, questions=QUESTIONS)):
        same_answers(a, b)
    assert [len(c.batches) for c in cap.contexts] == [1, 1, 1]


def test_plan_marks_batched_reads_without_loading(gguf: Path):
    b = backend(gguf)
    plan = Tez(backend=b).plan({"state": STATE, "questions": QUESTIONS})
    assert not b.loaded and b._api.loads == 0
    assert all(q["batch"] == "state" for q in plan["questions"].values()) and plan["totals"]["batches"] == 1
    assert any(n.startswith("in-process: 3 reads of 3 question(s)") for n in plan["notes"])
    assert not any("--swa-full" in n for n in plan["notes"])
    http = Tez(backend=LlamaCppBackend("http://127.0.0.1:9")).plan({"state": STATE, "questions": QUESTIONS})
    assert "batch" not in http["questions"]["topic"] and "batches" not in http["totals"]


def test_embed_backend_naming_the_same_model_is_not_loaded_twice(gguf: Path):
    tez = Tez(backend=f"inproc:{gguf}", embed_backend=f"inproc:{gguf}")
    assert tez.embedder is tez.backend and tez.embed_backend_url is None


# ---------------------------------------------------------------------------------------------- CLI and doctor
@pytest.fixture
def stub_library(monkeypatch, tmp_path: Path):
    """tez.inproc.load_api returns a stub for any directory holding a file named like llama.cpp's library."""
    lib = tmp_path / "llama"
    lib.mkdir()
    (lib / lib_file("llama")).write_bytes(b"")
    api = StubApi()
    monkeypatch.setattr("tez.inproc.load_api", lambda directory, verbose=False: api)
    return lib, api


def test_cli_decide_and_env_settings(gguf: Path, stub_library, capsys):
    import json
    lib, api = stub_library
    assert main(["decide", STATE, "--backend", f"inproc:{gguf}", "--llama-lib", str(lib), "--preset", "support-triage",
                 "--n-ctx", "2048", "--compact"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert set(res["answers"]) == {"topic", "urgent", "frustration"} and api.ctx_params["n_ctx"] == 2048
    args = build_parser({"TEZ_N_CTX": "8192", "TEZ_N_BATCH": "1024", "TEZ_N_GPU_LAYERS": "0", "TEZ_LLAMA_LIB": "/x"}) \
        .parse_args(["decide", "--backend", "inproc:m.gguf", "x"])
    assert (args.n_ctx, args.n_batch, args.n_gpu_layers, args.llama_lib) == (8192, 1024, 0, "/x")
    with pytest.raises(SystemExit):
        build_parser({"TEZ_N_CTX": "big"}).parse_args(["decide", "x"])


def test_serve_loads_the_model_at_start_up(gguf: Path, stub_library, capsys):
    from tez.cli import _load_inproc
    lib, api = stub_library
    tez = Tez(backend=f"inproc:{gguf}", inproc={"lib": lib})
    _load_inproc(tez)
    assert tez.backend.loaded and "loaded gemma-4-12b-q8_0 in this process" in capsys.readouterr().err


def test_inproc_doctor(gguf: Path, stub_library, capsys):
    lib, api = stub_library
    checks = {c.name: c for c in run_inproc_doctor(f"inproc:{gguf}", "gemma4", {"lib": lib})}
    for name in ("library", "gpu", "model", "template", "letters", "batch", "embeddings"):
        assert checks[name].status == "ok", (name, checks[name])
    assert "b11100" in checks["library"].detail and "Stub GPU" in checks["gpu"].detail
    assert "same answers" in checks["batch"].detail and checks["process"].status == "info"
    wrong = {c.name: c for c in run_inproc_doctor(f"inproc:{gguf}", "qwen3", backend=backend(gguf))}
    assert wrong["template"].status == "fail" and "--template gemma4" in wrong["template"].fix
    old = {c.name: c for c in run_inproc_doctor("", backend=InprocBackend(gguf, api=StubApi(commit="abc1234", hybrid=True)))}
    assert old["library"].status == "warn" and old["memory"].status == "info"
    nogpu = run_inproc_doctor("", backend=InprocBackend(gguf, api=StubApi(registries=(("CPU", 1),))))
    assert [(c.name, c.status) for c in nogpu] == [("gpu", "fail")]
    assert run_inproc_doctor(f"inproc:{gguf.parent / 'none.gguf'}")[0].detail.startswith("no GGUF file")
    missing = run_inproc_doctor(f"inproc:{gguf}", options={"lib": gguf.parent / "nolib"})
    assert missing[0].name == "library" and missing[0].status == "fail"
    assert main(["doctor", "--backend", f"inproc:{gguf}", "--llama-lib", str(lib)]) == 0
    assert "0 failed" in capsys.readouterr().out


def test_mcp_engine_reads_the_in_process_settings(gguf: Path):
    from tez.mcp_server import engine_from_env
    tez = engine_from_env({"TEZ_BACKEND": f"inproc:{gguf}", "TEZ_N_CTX": "8192", "TEZ_N_GPU_LAYERS": "0",
                           "TEZ_LLAMA_LIB": str(gguf.parent)})
    b = tez.backend
    assert isinstance(b, InprocBackend) and (b.n_ctx, b.n_batch, b.n_gpu_layers, b.lib) == (8192, 8192, 0, str(gguf.parent))
    assert not b.loaded
