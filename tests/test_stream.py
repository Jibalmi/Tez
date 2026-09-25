"""Streaming decisions (tez/stream.py) and the in-process stream reader (tez/inproc.py InprocStreamReader), against the
stub of llama.cpp's library (tests/stub_libllama.py) and FakeBackend: no GPU, no model."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from conftest import write_jsonl
from stub_libllama import StubApi, tokenize
from tez import FakeBackend, InprocBackend, InvalidRequest, Tez
from tez.prompt import QUESTION_FIRST, build_prompt
from tez.readout import softmax
from tez.schema import load_schema, parse_schema
from tez.stream import Action, Policy, StreamSession, parse_action

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "data" / "voice" / "schema.yaml"
BLOB = "sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"   # Gemma 4 12B Q8_0 by metadata


def _stream_policy():
    spec = importlib.util.spec_from_file_location("stream_policy", ROOT / "experiments" / "stream_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def gguf(tmp_path: Path) -> Path:
    p = tmp_path / BLOB
    p.write_bytes(b"GGUF")
    return p


def inproc(gguf: Path, api: StubApi | None = None, **kw) -> InprocBackend:
    return InprocBackend(gguf, "gemma4", api=api or StubApi(), n_seq_max=kw.pop("n_seq_max", 8), **kw)


def engines(gguf: Path, schemas=VOICE, api: StubApi | None = None, **kw) -> tuple[Tez, Tez]:
    """A streaming engine and a reference engine that reads every prompt whole, on the same stub model."""
    return (Tez(backend=inproc(gguf, api), schemas=schemas, **kw), Tez(backend=inproc(gguf), schemas=schemas, **kw))


def reference(ref: Tez, session: StreamSession, state: str):
    return ref.decide_question(session.question, state, ref.schemas.get(session.schema.name) if session.schema else None,
                               session.readout, session.abstain, session.alpha, QUESTION_FIRST, 1)


WORDS = "could you open the notes app and type buy milk".split()


# ---------------------------------------------------------------------------------------------- parity
def test_every_word_matches_decide_question_with_its_temperature(gguf: Path):
    tez, ref = engines(gguf)
    s = tez.stream(schema="voice-desktop")
    assert s.incremental and s.question.id == "action" and not s.abstain
    assert tez.backend.model_name() == "gemma-4-12b-q8_0"
    for w in WORDS:
        u = s.feed(w)
        r = reference(ref, s, u.state)
        assert u.decision == r.keys[int(np.argmax(r.probabilities))]
        assert np.allclose(list(u.probabilities.values()), r.probabilities, atol=1e-6)
        assert u.meta == r.meta and u.meta["temperature"] == 2.66      # 16 options shown: the measured default
        assert u.confidence == pytest.approx(float(np.max(r.probabilities)))


def test_default_temperature_off_and_a_number_follow_the_engine(gguf: Path):
    for setting, t in (("off", 1.0), (3.5, 3.5)):
        tez, ref = engines(gguf, default_temperature=setting)
        s = tez.stream(schema="voice-desktop")
        u = s.feed("pause")
        r = reference(ref, s, "pause")
        assert np.allclose(list(u.probabilities.values()), r.probabilities, atol=1e-6)
        assert u.meta.get("temperature", 1.0) == t
        prompt = build_prompt(s.question, "pause", "gemma4", s.setup.options, None, QUESTION_FIRST)
        z = ref.backend.letters(prompt, len(s.setup.options)).logits
        assert np.allclose(list(u.probabilities.values()), softmax(z, t), atol=1e-6)


def test_only_the_new_word_and_the_tail_are_evaluated(gguf: Path):
    api = StubApi()
    tez = Tez(backend=inproc(gguf, api), schemas=VOICE)
    s = tez.stream(schema="voice-desktop").load()
    head = api.decodes[:]
    assert sum(d["n"] for d in head) > 500 and all(d["outputs"] == 0 for d in head)
    tail = len(s._parts[0][2].encode())
    for i, w in enumerate(WORDS):
        n0 = len(api.decodes)
        u = s.feed(w)
        decide, commit = api.decodes[n0], api.decodes[n0 + 1:]
        piece = len((" " if i else "") + w) + (0 if i else 1)    # the first word also reads the head's seam token
        assert decide["n"] == piece + tail and decide["outputs"] == 1          # the new word + the answer tail
        assert sum(d["n"] for d in commit) == piece and all(d["outputs"] == 0 for d in commit)
        assert u.timings["tokens"] == piece + tail and u.timings["commit_tokens"] == piece
    s.end()
    s.reset()
    n0 = len(api.decodes)
    s.feed("pause")
    assert api.decodes[n0]["n"] == len("pause") + 1 + tail                   # the head is not evaluated again
    assert s._reader.restored == 0


def test_the_same_session_twice_and_a_new_session_reuse_nothing_stale(gguf: Path):
    tez, ref = engines(gguf)
    s = tez.stream(schema="voice-desktop")
    for text in ("open notes", "turn it up a bit", "open notes"):
        s.reset()
        for w in text.split():
            u = s.feed(w)
        assert np.allclose(list(u.probabilities.values()), reference(ref, s, text).probabilities, atol=1e-6)
    s.close()
    assert tez.backend._reserved == []
    t = tez.stream(schema="voice-desktop")
    u = t.feed("pause")
    assert np.allclose(list(u.probabilities.values()), reference(ref, t, "pause").probabilities, atol=1e-6)


def test_hybrid_models_never_roll_back(gguf: Path):
    api = StubApi(hybrid=True)
    tez, ref = engines(gguf, api=api)
    s = tez.stream(schema="voice-desktop")
    for w in WORDS:
        u = s.feed(w)
    assert np.allclose(list(u.probabilities.values()), reference(ref, s, " ".join(WORDS)).probabilities, atol=1e-6)
    assert api.partial_rm_refused == []
    assert all(p0 <= 0 and p1 < 0 for _, p0, p1 in api.seq_rms)


def test_other_reads_between_words_leave_the_stream_intact(gguf: Path):
    tez, ref = engines(gguf)
    s = tez.stream(schema="voice-desktop")
    s.feed("open")
    tez.decide("a message about billing", questions={f"q{i}": {"type": "noul", "instructions": f"Q{i}?"}
                                                        for i in range(9)})     # batched waves avoid the stream's ids
    tez.backend.letters(build_prompt(s.question, "something else", "gemma4"), 3)   # a single read on sequence 0
    u = s.feed("notes")
    assert np.allclose(list(u.probabilities.values()), reference(ref, s, "open notes").probabilities, atol=1e-6)
    tez.backend._reset()                                  # a failed read elsewhere clears the cache
    u = s.feed("please")
    assert np.allclose(list(u.probabilities.values()), reference(ref, s, "open notes please").probabilities, atol=1e-6)


def test_a_word_that_changes_earlier_tokens_rebuilds_from_the_head(gguf: Path):
    tez, ref = engines(gguf)
    s = tez.stream(schema="voice-desktop")
    s.feed("open")
    s.words[-1] = "shut"                                   # a revised transcript
    u = s.feed("notes")
    assert s._reader.restored == 1
    assert np.allclose(list(u.probabilities.values()), reference(ref, s, "shut notes").probabilities, atol=1e-6)


def test_too_few_sequences_is_a_clear_error(gguf: Path):
    tez = Tez(backend=inproc(gguf, n_seq_max=3), schemas=VOICE)
    with pytest.raises(Exception, match="n_seq_max"):
        tez.stream(schema="voice-desktop").feed("open")


def test_probe_readouts_stream_too(gguf: Path, tmp_path: Path):
    """A fitted probe on an abstaining question reads two prompts per word (the probe's and the letters' with
    __none__), both in the one decode, and still answers as decide_question does."""
    from tez.fit import fit
    yaml = VOICE.read_text(encoding="utf-8").replace("      none: \"Out of scope", "      # none: \"Out of scope")
    (tmp_path / "voice.yaml").write_text(yaml, encoding="utf-8")
    rng = random.Random(4)
    schema = load_schema(tmp_path / "voice.yaml")
    labels = list(schema.questions["action"].criteria)
    rows = [{"state": " ".join(rng.sample(WORDS, 4)), "labels": {"action": labels[i % len(labels)]}} for i in range(60)]
    lab = write_jsonl(tmp_path / "labels.jsonl", rows)
    api = StubApi()
    tez = Tez(backend=inproc(gguf, api, n_ctx=16384, n_outputs=256), schemas=tmp_path)   # the stub's tokens are bytes
    fit(tez, tez.schemas["voice-desktop"], [lab], say=lambda m: None)
    ref = Tez(backend=inproc(gguf, n_ctx=16384), schemas=tmp_path)
    s = tez.stream(schema="voice-desktop").load()
    assert s.abstain and [p[0] for p in s._parts] == ["embed", "letters"]
    for w in WORDS[:5]:
        n0 = len(api.decodes)
        u = s.feed(w)
        assert api.decodes[n0]["embeddings"] and len(api.decodes[n0]["seqs"]) == 2       # both prompts, one decode
        r = reference(ref, s, u.state)
        assert u.readout == "probe" and np.allclose(list(u.probabilities.values()), r.probabilities, atol=1e-6)
    few = Tez(backend=inproc(gguf, n_ctx=16384, n_outputs=8), schemas=tmp_path)     # outputs run short: two decodes
    t = few.stream(schema="voice-desktop")
    for w in WORDS[:3]:
        u = t.feed(w)
    assert np.allclose(list(u.probabilities.values()), reference(ref, t, u.state).probabilities, atol=1e-6)


def test_a_probe_on_the_letters_prompt_reads_one_sequence(gguf: Path, tmp_path: Path):
    """An explicit none option: the probe's prompt is the letters prompt, so one sequence gives both readouts
    (blend), or the probe alone (fitted without the blend)."""
    from tez.fit import fit
    (tmp_path / "voice.yaml").write_text(VOICE.read_text(encoding="utf-8"), encoding="utf-8")
    rng = random.Random(5)
    labels = list(load_schema(tmp_path / "voice.yaml").questions["action"].criteria)
    rows = [{"state": " ".join(rng.sample(WORDS, 3)), "labels": {"action": labels[i % len(labels)]}} for i in range(48)]
    lab = write_jsonl(tmp_path / "labels.jsonl", rows)
    for blend, parts in ((True, ["both"]), (False, ["embed"])):
        tez = Tez(backend=inproc(gguf, n_ctx=16384, n_outputs=128), schemas=tmp_path)
        fit(tez, tez.schemas["voice-desktop"], [lab], say=lambda m: None, use_blend=blend)
        ref = Tez(backend=inproc(gguf, n_ctx=16384), schemas=tmp_path)
        s = tez.stream(schema="voice-desktop")
        assert not s.abstain and [p[0] for p in s._parts] == parts
        for w in WORDS[:4]:
            u = s.feed(w)
            assert u.readout == "probe"
            assert np.allclose(list(u.probabilities.values()), reference(ref, s, u.state).probabilities, atol=1e-6)


def test_http_style_backends_read_whole_prompts():
    fb = FakeBackend()
    tez = Tez(backend=fb, schemas=VOICE)
    s = tez.stream(schema="voice-desktop")
    assert not s.incremental
    for w in "please pause the music".split():
        u = s.feed(w)
    assert fb.calls["letters"] == 4
    r = tez.decide_question(s.question, u.state, tez.schemas["voice-desktop"], layout=QUESTION_FIRST)
    assert u.decision == "pause_media" == r.answer["choice"]


# ---------------------------------------------------------------------------------------------- the policy
def scripted(traj_by_text: dict[str, tuple[str, float]], keys: list[str]):
    """letters_fn for FakeBackend: the transcript in the prompt picks (decision, probability)."""
    def fn(prompt: str, k: int):
        text = prompt.split("Input:\n", 1)[1].split("<turn|>", 1)[0]
        pred, p = traj_by_text[text]
        z = np.full(k, np.log((1.0 - p) / (k - 1)))
        z[keys.index(pred)] = np.log(p)
        return z
    return fn


def run_policy(traj: list[tuple[str, float]], policy: Policy | None = None, **kw) -> tuple[StreamSession, list]:
    schema = load_schema(VOICE)
    keys = list(schema.questions["action"].criteria)
    words = [f"w{i}" for i in range(1, len(traj) + 1)]
    by_text = {" ".join(words[:k]): traj[k - 1] for k in range(1, len(traj) + 1)}
    tez = Tez(backend=FakeBackend(letters_fn=scripted(by_text, keys)), schemas=[schema])
    fired: list[Action] = []
    s = tez.stream(schema="voice-desktop", policy=policy, on_action=fired.append, **kw)
    for w in words:
        s.feed(w)
    s.end()
    return s, fired


def test_the_policy_reproduces_stream_policy_py():
    sp = _stream_policy()
    keys = list(load_schema(VOICE).questions["action"].criteria)
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 7)
        focus = rng.sample(keys, 3)
        traj = [(rng.choice(focus if rng.random() < 0.8 else keys), rng.choice([0.5, 0.85, 0.899, 0.901, 0.95, 0.99]))
                for _ in range(n)]
        expected = sp.executed_sequence([{"k": k, "pred": a, "pmax": p} for k, (a, p) in enumerate(traj, 1)], 0.9, 2, 2, 2)
        s, fired = run_policy(traj)
        assert [(a.word, a.name) for a in fired] == expected, traj
        assert [(a.word, a.name) for a in s.actions] == expected


def test_policy_replay_on_logged_steps_matches_the_session_and_stream_policy_py():
    sp = _stream_policy()
    keys = list(load_schema(VOICE).questions["action"].criteria)
    pol = Policy.from_schema(load_schema(VOICE), None)
    rng = random.Random(11)
    for _ in range(200):
        traj = [(rng.choice(keys), rng.choice([0.5, 0.95, 0.99])) for _ in range(rng.randint(1, 6))]
        expected = sp.executed_sequence([{"k": k, "pred": a, "pmax": p} for k, (a, p) in enumerate(traj, 1)], 0.9, 2, 2, 2)
        got = pol.replay([(a, p, k) for k, (a, p) in enumerate(traj, 1)])
        assert [(k, name) for _, k, _, name in got] == expected


def test_none_on_a_prefix_waits_and_none_at_the_end_hands_off():
    s, fired = run_policy([("none", 0.99), ("none", 0.99)])
    assert fired == [] and s.updates[-1].handoff and s.updates[-1].none and s.updates[-1].final
    s, fired = run_policy([("none", 0.99), ("open_notes", 0.95), ("none", 0.97)])
    assert [a.name for a in fired] == ["open_notes"] and s.updates[-1].handoff


def test_classes_early_ancestor_deferred_and_gated():
    s, fired = run_policy([("open_notes", 0.95)])
    assert [(a.kind, a.name) for a in fired] == [("early", "open_notes")] and s.updates[0].committed
    s, fired = run_policy([("play_spotify", 0.95), ("play_spotify", 0.95), ("play_spotify", 0.95)])
    assert [(a.kind, a.name, a.word) for a in fired] == [("ancestor", "open_spotify", 2), ("end", "play_spotify", 3)]
    s, fired = run_policy([("type_text", 0.99), ("type_text", 0.99)])
    assert [(a.kind, a.name, a.word) for a in fired] == [("end", "type_text", 2)]
    pol = Policy.from_schema(load_schema(VOICE), None)
    pol.actions["close_app"] = parse_action("close_app", "gated")
    s, fired = run_policy([("close_app", 0.99), ("close_app", 0.99)], policy=pol)
    assert [(a.kind, a.name, a.gated) for a in fired] == [("approval", "close_app", True)]


def test_undo_and_retraction():
    undone: list[Action] = []
    s, fired = run_policy([("open_youtube", 0.95), ("play_spotify", 0.95), ("play_spotify", 0.95)], on_undo=undone.append)
    assert [a.name for a in fired] == ["open_youtube", "open_spotify", "play_spotify"]
    u = s.undo()
    assert u.kind == "undo" and u.undoes.name == "play_spotify" and undone == [u] and s.undo(u) is None
    pol = Policy.from_schema(load_schema(VOICE), None, retract=True)
    s, fired = run_policy([("open_youtube", 0.95), ("play_spotify", 0.95), ("play_spotify", 0.95)], policy=pol,
                          on_undo=undone.append)
    kinds = [(a.kind, a.name) for a in s.actions]
    assert kinds == [("early", "open_youtube"), ("ancestor", "open_spotify"), ("undo", "open_youtube"),
                     ("end", "play_spotify")]
    assert pol.consistent("open_spotify", "play_spotify") and not pol.consistent("open_youtube", "play_spotify")


def test_residual_mode_reads_what_follows_with_a_prior_line(gguf: Path):
    tez, ref = engines(gguf)
    pol = Policy(tau=0.0, default_class="R0", residual=True)       # everything fires at once
    s = tez.stream(schema="voice-desktop", policy=pol)
    for w in WORDS:
        u = s.feed(w)
        if u.committed:
            break
    assert u.committed and u.state == " ".join(s.words)
    name = u.action.name
    for w in ("and", "then"):
        u = s.feed(w)
        assert u.state.startswith(f"Already done earlier in this utterance: {name}.")
        assert np.allclose(list(u.probabilities.values()), reference(ref, s, u.state).probabilities, atol=1e-6)
        if u.committed:
            name = u.action.name


# ---------------------------------------------------------------------------------------------- recogniser text
def test_partial_words_are_read_but_only_complete_words_are_committed(gguf: Path):
    api = StubApi()
    tez, ref = engines(gguf, api=api)
    s = tez.stream(schema="voice-desktop").load()
    steps = [("op", False), ("open", False), ("open no", False), ("open notes", False), ("open notes", True),
             ("open notes and", False)]
    for text, final in steps:
        u = s.update(text, final=final)
        assert u.partial == (not final) and u.transcript == text
        assert np.allclose(list(u.probabilities.values()), reference(ref, s, text).probabilities, atol=1e-6)
        done = text.split() if final else text.split()[:-1]
        assert s._reader.committed_tokens[0] == (len(" ".join(done)) + 1 if done else 0)   # +1: the head's seam token
    assert s.update("open notes and", final=False) is None                              # nothing changed
    assert s._reader.restored == 0                                                       # growing words never rebuild


def test_stability_counts_words_not_readings():
    """Two readings of one word ("pau", "pause") are one word: an R0 action that needs two stable words waits."""
    schema = load_schema(VOICE)
    keys = list(schema.questions["action"].criteria)
    table = {"pau": ("pause_media", 0.95), "pause": ("pause_media", 0.95), "pause it": ("pause_media", 0.95)}
    tez = Tez(backend=FakeBackend(letters_fn=scripted(table, keys)), schemas=[schema])
    s = tez.stream(schema="voice-desktop")
    assert s.update("pau").stable == 1 and not s.updates[-1].committed
    u = s.update("pause")
    assert u.stable == 1 and not u.committed                        # the same word, read again
    u = s.update("pause it")
    assert u.stable == 2 and u.committed and u.action.name == "pause_media"


# ---------------------------------------------------------------------------------------------- set-up errors
def test_actions_block_is_validated():
    base = {"name": "x", "questions": {"a": {"type": "choice", "instructions": "Which?", "criteria": {"p": None, "q": None}},
                                       "b": {"type": "noul", "instructions": "Yes?"}}}
    assert parse_schema({**base, "actions": {"p": "R0"}}).actions == {"p": "R0"}
    with pytest.raises(InvalidRequest, match="actions must map"):
        parse_schema({**base, "actions": ["p"]})
    with pytest.raises(InvalidRequest, match="class must be"):
        parse_action("p", {"class": "R9"})
    with pytest.raises(InvalidRequest, match="unknown settings"):
        parse_action("p", {"clss": "R0"})
    with pytest.raises(InvalidRequest, match="stable"):
        parse_action("p", {"class": "R0", "stable": 0})
    assert parse_action("p", "refinable").cls == "R1" and parse_action("p", "refinable").stable == 2
    tez = Tez(backend=FakeBackend(), schemas=[parse_schema({**base, "actions": {"z": "R0"}})])
    with pytest.raises(InvalidRequest, match="not an option"):
        tez.stream(schema="x", question="a")
    tez = Tez(backend=FakeBackend(), schemas=[parse_schema({**base, "actions": {"p": {"class": "R1", "refines": "r"}}})])
    with pytest.raises(InvalidRequest, match="refines"):
        tez.stream(schema="x", question="a")


def test_session_errors():
    base = {"name": "x", "questions": {"a": {"type": "choice", "instructions": "Which?", "criteria": {"p": None, "q": None}},
                                       "c": {"type": "choice", "instructions": "Or?", "criteria": {"r": None, "s": None}},
                                       "b": {"type": "noul", "instructions": "Yes?"}}}
    tez = Tez(backend=FakeBackend(), schemas=[parse_schema(base)])
    with pytest.raises(InvalidRequest, match="name the question"):
        tez.stream(schema="x")
    with pytest.raises(InvalidRequest, match="choice question"):
        tez.stream(schema="x", question="b")
    with pytest.raises(InvalidRequest, match="unknown schema"):
        tez.stream(schema="nope")
    s = tez.stream(schema="x", question="a")
    assert s.abstain                                      # no explicit none option: __none__ is added
    with pytest.raises(InvalidRequest, match="non-empty"):
        s.feed("  ")
    u = s.end()
    assert u.handoff and u.decision == "__none__"
    with pytest.raises(InvalidRequest, match="ended"):
        s.feed("p")
    with pytest.raises(InvalidRequest, match="already"):
        s.end()
    s.reset()
    assert s.feed("p").k == 1


def test_stub_tokeniser_matches_the_seam_assumption():
    """The stream's head is cut one token before the first token two different states disagree on."""
    a = tokenize("Input:\nopen notes")
    b = tokenize("Input:\nwhat time is it")
    assert a[: len(tokenize("Input:\n"))] == b[: len(tokenize("Input:\n"))]
