"""ASR stage benchmark on the synthesised command WAVs (data/voice/wav, 16 kHz SAPI).

For each wav, audio is fed in --chunk-ms pieces exactly as the live loop would receive it and
faster-whisper is re-run on the growing buffer after every chunk (no VAD, greedy, no context).
Reported per model:
  * partial-decode latency p50/p95 (ms) as a function of buffer length
  * final transcript WER against the command text (normalised) and exact-match rate
  * "stable-prefix" fraction: how much of the final transcript's words were already present in
    the partial at 50 % / 75 % of the utterance -- what the decision layer actually sees early
  * the intent decision on the ASR final transcript vs on the ground-truth text, if --server is
    given (does ASR noise change the routed intent?)
Synthetic speech is clean; treat these as lower bounds on error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def add_nvidia_dlls():
    """CTranslate2 (faster-whisper) wants CUDA 12 cuBLAS/cuDNN; the pip wheels ship them under
    site-packages/nvidia/*/bin but Windows will not find them unless the dirs are registered."""
    import os, site, sys  # noqa: E401
    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            for d in nv.glob("*/bin"):
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


add_nvidia_dlls()


def norm(s: str) -> list[str]:
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[len(ref), len(hyp)] / max(1, len(ref))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--server", default="", help="optional llama-server for ASR-vs-text intent agreement")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    m = WhisperModel(args.model, device="cuda", compute_type=args.compute)
    m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1, language="en")

    def decode(a):
        t0 = time.perf_counter()
        segs, _ = m.transcribe(a, beam_size=1, language="en", vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segs).strip(), (time.perf_counter() - t0) * 1000

    dec = None
    if args.server:
        spec = importlib.util.spec_from_file_location("vf", ROOT / "experiments" / "run_voice_fast.py")
        vf = importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)  # type: ignore[union-attr]
        intents = json.loads((ROOT / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
        ids = [i["id"] for i in intents]

        def dec(text):
            r = vf.score(args.server, vf.build_prompt(text, intents, "final", "last"), len(ids), 200, True)
            p = r["probabilities"]; i = max(range(len(p)), key=lambda j: p[j])
            return ids[i], p[i]

    cmds = [json.loads(l) for l in (ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cmds = cmds[: args.limit]
    n = int(16000 * args.chunk_ms / 1000)
    rows, lat_by_len = [], {}
    for c in cmds:
        audio, sr = sf.read(ROOT / "data" / "voice" / "wav" / f"{c['id']}.wav", dtype="float32")
        assert sr == 16000
        ref = norm(c["text"])
        partials = []
        for end in range(n, len(audio) + n, n):
            buf = audio[:end]
            text, ms = decode(buf)
            partials.append((end / 16000, norm(text), ms))
            lat_by_len.setdefault(round(end / 16000, 1), []).append(ms)
        final = partials[-1][1]
        dur = len(audio) / 16000
        def prefix_frac(at):
            t = at * dur
            hyp = next((p[1] for p in partials if p[0] >= t), partials[-1][1])
            k = 0
            while k < min(len(hyp), len(final)) and hyp[k] == final[k]:
                k += 1
            return k / max(1, len(final))
        row = dict(id=c["id"], style=c["style"], ref=" ".join(ref), hyp=" ".join(final), wer=wer(ref, final), exact=ref == final,
                   dur_s=dur, n_partials=len(partials), stable_prefix_50=prefix_frac(0.5), stable_prefix_75=prefix_frac(0.75),
                   final_decode_ms=partials[-1][2])
        if dec:
            pi, pp = dec(" ".join(final)); ti, tp = dec(c["text"])
            row.update(intent_from_asr=pi, intent_from_text=ti, intent_agree=pi == ti, asr_intent_correct=pi == c["intent"], text_intent_correct=ti == c["intent"])
        rows.append(row)
    all_ms = [ms for v in lat_by_len.values() for ms in v]
    summary = dict(
        model=args.model, compute=args.compute, chunk_ms=args.chunk_ms, n=len(rows),
        wer_mean=statistics.mean(r["wer"] for r in rows), exact_rate=statistics.mean(r["exact"] for r in rows),
        wer_by_style={s: round(statistics.mean(r["wer"] for r in rows if r["style"] == s), 3) for s in sorted({r["style"] for r in rows})},
        partial_ms_p50=pct(all_ms, 0.5), partial_ms_p95=pct(all_ms, 0.95),
        partial_ms_by_buffer_s={k: round(statistics.median(v), 1) for k, v in sorted(lat_by_len.items())},
        stable_prefix_at_50pct=statistics.mean(r["stable_prefix_50"] for r in rows),
        stable_prefix_at_75pct=statistics.mean(r["stable_prefix_75"] for r in rows),
    )
    if dec:
        summary.update(intent_agree_asr_vs_text=statistics.mean(r["intent_agree"] for r in rows),
                       intent_acc_from_asr=statistics.mean(r["asr_intent_correct"] for r in rows),
                       intent_acc_from_text=statistics.mean(r["text_intent_correct"] for r in rows))
    out = Path(args.out)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=lambda o: round(o, 4) if isinstance(o, float) else o))


if __name__ == "__main__":
    main()
