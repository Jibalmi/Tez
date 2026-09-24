"""The in-process backend on a real model. Skipped unless TEZ_TEST_GGUF (a GGUF file) and TEZ_LLAMA_LIB (the directory of
llama.cpp's library, from a release archive) are set; marked `live`, so the offline suite leaves it out.

    TEZ_TEST_GGUF=models/Qwen3.5-4B-Q8_0.gguf TEZ_LLAMA_LIB=C:/llama.cpp TEZ_TEST_TEMPLATE=qwen3 python -m pytest -m live tests/test_inproc_live.py

With TEZ_TEST_LLAMA_URL, a llama-server serving the same GGUF, it also checks that the in-process letters match the
server's. The GPU then holds the model twice: use a small one. TEZ_TEST_TEMPLATE (default gemma4) is the model's prompt
template, TEZ_TEST_N_GPU_LAYERS (default -1, all) the layers on the GPU."""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import DOCS_REQUEST
from tez import InprocBackend, LlamaCppBackend, Tez
from tez.prompt import build_prompt
from tez.readout import softmax
from tez.schema import parse_question

GGUF = os.environ.get("TEZ_TEST_GGUF")
LIB = os.environ.get("TEZ_LLAMA_LIB")
URL = os.environ.get("TEZ_TEST_LLAMA_URL")
TEMPLATE = os.environ.get("TEZ_TEST_TEMPLATE", "gemma4")

pytestmark = [pytest.mark.live, pytest.mark.skipif(not (GGUF and LIB), reason="set TEZ_TEST_GGUF and TEZ_LLAMA_LIB")]

STATES = [DOCS_REQUEST["state"],
          "The app crashes every time I open the invoices page, since this morning's update. We cannot bill anyone.",
          {"ticket": {"subject": "Upgrade", "messages": [{"from": "customer", "text": "What would 40 more seats cost?"}]}}]
QUESTIONS = [parse_question(qid, q) for qid, q in DOCS_REQUEST["questions"].items()]


def prompts(layout: str) -> list[tuple[str, int]]:
    return [(build_prompt(q, s, TEMPLATE, layout=layout), len(q.options())) for s in STATES for q in QUESTIONS]


@pytest.fixture(scope="module")
def backend():
    b = InprocBackend(GGUF, TEMPLATE, lib=LIB, n_gpu_layers=int(os.environ.get("TEZ_TEST_N_GPU_LAYERS", "-1")))
    b.load()
    yield b
    b.close()


def test_batched_reads_match_single_reads(backend):
    for layout in ("state_first", "question_first"):
        for s in STATES:
            ps = [(build_prompt(q, s, TEMPLATE, layout=layout), len(q.options())) for q in QUESTIONS]
            single = [backend.letters(p, k).logits for p, k in ps]
            batch = backend.read_many([p for p, _ in ps], [k for _, k in ps])
            for a, (b, _) in zip(single, batch):
                assert int(np.argmax(a)) == int(np.argmax(b.logits))
                assert np.max(np.abs(a - b.logits)) < 0.25          # the same numbers up to batch-shape numerics
            if layout == "state_first":
                assert batch[0][0].timings["batch_prefix_n"] > 20   # instructions + state read once


def test_letters_are_log_probabilities_of_letter_tokens(backend):
    for p, k in prompts("state_first"):
        z = backend.letters(p, k).logits
        assert np.all(z <= 0) and 0.5 < float(np.exp(z).sum()) <= 1.0 + 1e-6   # the model answers with a letter


def test_states_match_between_single_and_batched_reads(backend):
    ps = [p for p, _ in prompts("state_first")[:3]]
    single = [backend.embed(p).vector for p in ps]
    batch = backend.read_many(ps, [0, 0, 0], [True, True, True])
    for a, (_, b) in zip(single, batch):
        assert a.shape == b.vector.shape == (backend.details()["n_embd"],)
        cos = float(a @ b.vector / (np.linalg.norm(a) * np.linalg.norm(b.vector)))
        assert cos > 0.999


def test_engine_decides_the_docs_request(backend):
    tez = Tez(backend=backend, template=TEMPLATE)
    res = tez.handle(DOCS_REQUEST)
    assert res["answers"]["topic"]["choice"] == "billing"
    assert abs(sum(res["answers"]["topic"]["probabilities"].values()) - 1) < 1e-9
    assert res["usage"]["input_tokens"] > 100


@pytest.mark.skipif(not URL, reason="set TEZ_TEST_LLAMA_URL to a llama-server serving the same GGUF")
def test_letters_match_llama_server(backend):
    http = LlamaCppBackend(URL, template=TEMPLATE)
    server, ours = http.info(), backend.info()
    assert (server["n_params"], server["ftype"]) == (ours["n_params"], ours["ftype"]), "llama-server serves another model"
    worst = 0.0
    for layout in ("question_first", "state_first"):
        for p, k in prompts(layout):
            a, b = backend.letters(p, k).logits, http.letters(p, k).logits
            assert int(np.argmax(a)) == int(np.argmax(b)), p
            worst = max(worst, float(np.max(np.abs(softmax(a) - softmax(b)))))
    assert worst < 0.05, f"letter probabilities differ by up to {worst:.3f}"
