"""Fast voice intent routing: transcript LAST so the constant part of the prompt (instruction
+ the 16 actions) is a KV-cache prefix and each new word costs only its own tokens.

Compared with run_voice.py (SemIf payload, transcript first => nothing shareable, full prompt
re-evaluated every time), this variant answers the two questions that matter for a
sub-50 ms voice loop:
  1. accuracy: does moving the transcript after the options (and telling the model the
     transcript may be cut off) change the decision quality?        (--order, --variant)
  2. latency:  with llama-server cache_prompt=true, how many tokens are actually evaluated
     per streamed word, and how many ms is that on this GPU?         (prompt_n, prompt_ms)

Readout is unchanged: one forward pass, top_logprobs at the answer position, restricted to
the 16 option letters, softmax. Nothing decoded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOP"

spec = importlib.util.spec_from_file_location("stream_analysis", ROOT / "experiments" / "stream_analysis.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)  # type: ignore[union-attr]

HEADER = {
    "final": ("You are the command router of a voice-controlled computer assistant. Read the user's "
              "transcript and decide which ONE action they are asking for. Answer with the letter only. "
              "Choose 'none' if the request is chit-chat, a question, out of scope, or too unclear to act on."),
    "partial": ("You are the command router of a voice-controlled computer assistant. The user is still "
                "speaking: the transcript may be cut off mid-sentence. Decide which ONE action they are asking "
                "for as soon as it is clear. Answer with the letter only. Choose 'none' if the request is "
                "chit-chat, a question, out of scope, or if what has been said so far is not yet enough to "
                "know which action to take."),
}
TAIL = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def options_block(intents):
    return "\n".join(f"{LETTERS[i]}. {it['id']}: {it['description']}" for i, it in enumerate(intents))


def build_prompt(text: str, intents, variant: str, order: str, prior: str | None = None) -> str:
    """prior: the action already executed earlier in this utterance (multi-step commands); it is
    placed before the transcript so the cached prefix only changes when an action fires."""
    head = HEADER[variant]
    opts = "Actions:\n" + options_block(intents)
    tr = f'Transcript{" so far" if variant == "partial" else ""}: "{text}"'
    if prior:
        tr = f"Already done earlier in this utterance: {prior}. The transcript below is what followed.\n{tr}"
    if order == "last":
        user = f"{head}\n\n{opts}\n\n{tr}"
    else:
        user = f"{tr}\n\n{head}\n\n{opts}"
    return f"<|turn>user\n{user}{TAIL}"


def softmax(xs):
    m = max(xs)
    e = [pow(2.718281828459045, x - m) for x in xs]
    z = sum(e)
    return [v / z for v in e]


SESSION = requests.Session()  # keep-alive: a new TCP connection per decision costs ~10 ms on Windows


def score(server, prompt, n_letters, n_probs, cache):
    body = {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0, "cache_prompt": cache, "samplers": []}
    t0 = time.perf_counter()
    data = SESSION.post(f"{server}/completion", json=body, timeout=300).json()
    http_ms = (time.perf_counter() - t0) * 1000
    tops = data["completion_probabilities"][0]["top_logprobs"]
    lp = {}
    for t in tops:
        lp.setdefault(t["token"], t["logprob"])
    floor = min(t["logprob"] for t in tops) - 2.0
    letters = LETTERS[:n_letters]
    logprobs = [lp.get(L, floor) for L in letters]
    missing = [L for L in letters if L not in lp]
    tm = data.get("timings", {})
    return dict(probabilities=softmax(logprobs), option_logprobs=logprobs, missing_letters=missing,
                top1_token=tops[0]["token"], top1_is_slot=tops[0]["token"] in letters,
                prompt_n=tm.get("prompt_n"), prompt_ms=tm.get("prompt_ms"), http_ms=http_ms)


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default="partial", choices=list(HEADER))
    ap.add_argument("--order", default="last", choices=["last", "first"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--n-probs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cache = not args.no_cache

    intents = json.loads((ROOT / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
    ids = [i["id"] for i in intents]
    cmds = [json.loads(l) for l in (ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cmds = cmds[: args.limit]
    props = requests.get(f"{args.server}/props", timeout=30).json()

    # warm the shared prefix once so the first measured request is representative
    score(args.server, build_prompt("warm up", intents, args.variant, args.order), len(ids), args.n_probs, cache)

    full, stream = [], []
    t0 = time.perf_counter()
    for n, cmd in enumerate(cmds, 1):
        words = cmd["text"].split()
        if not args.no_stream:
            traj = []
            for k in range(1, len(words) + 1):
                prefix = " ".join(words[:k])
                r = score(args.server, build_prompt(prefix, intents, args.variant, args.order), len(ids), args.n_probs, cache)
                p = r["probabilities"]
                traj.append(dict(k=k, prefix=prefix, pred=ids[argmax(p)], pmax=max(p), probabilities=p,
                                 prompt_n=r["prompt_n"], prompt_ms=r["prompt_ms"], http_ms=r["http_ms"],
                                 missing_letters=r["missing_letters"], top1_is_slot=r["top1_is_slot"]))
            stream.append(dict(id=cmd["id"], gold=cmd["intent"], style=cmd["style"], n_words=len(words),
                               human_commit_word=cmd.get("commit_word", -1), trajectory=traj))
            rec = dict(traj[-1])  # the full utterance IS the last prefix; identical prompt
            rec.pop("k"); rec.pop("prefix")
        else:
            rec = score(args.server, build_prompt(cmd["text"], intents, args.variant, args.order), len(ids), args.n_probs, cache)
            rec["pred"] = ids[argmax(rec["probabilities"])]; rec["pmax"] = max(rec["probabilities"])
        rec.update(id=cmd["id"], text=cmd["text"], gold_intent=cmd["intent"], pred_intent=rec["pred"], style=cmd["style"],
                   n_words=len(words), then=cmd.get("then"), alt=cmd.get("alt"), human_commit_word=cmd.get("commit_word", -1),
                   option_ids=ids)
        full.append(rec)
        if n % 40 == 0 or n == len(cmds):
            print(f"  {n}/{len(cmds)}  {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full), encoding="utf-8")
    if stream:
        out.with_name(out.stem + "_stream.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in stream), encoding="utf-8")

    # ---- accuracy
    def ok(r):
        return r["pred_intent"] == r["gold_intent"]
    acc = statistics.mean(ok(r) for r in full)
    acc_alt = statistics.mean(ok(r) or (bool(r.get("alt")) and r["pred_intent"] == r["alt"]) for r in full)
    acc_any = statistics.mean(ok(r) or (bool(r.get("alt")) and r["pred_intent"] == r["alt"]) or (bool(r.get("then")) and r["pred_intent"] == r["then"]) for r in full)
    by_style = defaultdict(list)
    for r in full:
        by_style[r["style"]].append(ok(r))
    none_rows = [r for r in full if r["gold_intent"] == "none"]
    none_pred = [r for r in full if r["pred_intent"] == "none"]
    none_tp = sum(1 for r in none_rows if r["pred_intent"] == "none")
    summary = dict(
        tag=args.tag, variant=args.variant, order=args.order, cache=cache, n=len(full), wall_s=wall,
        intent_accuracy=acc, intent_accuracy_alt_ok=acc_alt, intent_accuracy_alt_or_then_ok=acc_any,
        accuracy_by_style={s: statistics.mean(v) for s, v in by_style.items()},
        none_recall=none_tp / len(none_rows) if none_rows else None,
        none_precision=none_tp / len(none_pred) if none_pred else None,
        full_prompt_n_p50=pct([r["prompt_n"] for r in full], 0.5),
        full_prompt_ms_p50=pct([r["prompt_ms"] for r in full], 0.5),
        full_http_ms_p50=pct([r["http_ms"] for r in full], 0.5),
        missing_letter_rows=sum(1 for r in full if r["missing_letters"]),
        top1_not_slot_rows=sum(1 for r in full if not r["top1_is_slot"]),
        errors=[dict(id=r["id"], text=r["text"], gold=r["gold_intent"], pred=r["pred_intent"], p=round(r["pmax"], 3), style=r["style"]) for r in full if not ok(r)],
    )
    # ---- latency of streaming: incremental cost per word (k >= 2), i.e. what a live loop pays
    if stream:
        inc_n = [t["prompt_n"] for s in stream for t in s["trajectory"] if t["k"] >= 2]
        inc_ms = [t["prompt_ms"] for s in stream for t in s["trajectory"] if t["k"] >= 2]
        inc_http = [t["http_ms"] for s in stream for t in s["trajectory"] if t["k"] >= 2]
        first_n = [s["trajectory"][0]["prompt_n"] for s in stream]
        first_ms = [s["trajectory"][0]["prompt_ms"] for s in stream]
        summary["stream_latency"] = dict(
            incremental_prompt_n_p50=pct(inc_n, 0.5), incremental_prompt_n_p95=pct(inc_n, 0.95),
            incremental_prompt_ms_p50=pct(inc_ms, 0.5), incremental_prompt_ms_p95=pct(inc_ms, 0.95),
            incremental_http_ms_p50=pct(inc_http, 0.5), incremental_http_ms_p95=pct(inc_http, 0.95),
            first_word_prompt_n_p50=pct(first_n, 0.5), first_word_prompt_ms_p50=pct(first_ms, 0.5),
            requests=len(inc_n) + len(first_n),
        )
        # ---- commit policies (non-none rule), accept-set = gold + then + alt
        F = {r["id"]: r for r in full}
        actionable = [s for s in stream if s["gold"] != "none"]
        pol = {}
        for tau in (0.9, 0.99):
            for m in (1, 2):
                commits = correct = correct_acc = early = 0
                pos, saved = [], []
                for s in actionable:
                    t = sa.commit_point(s["trajectory"], tau, m, 1, skip_none=True)
                    if t is None:
                        continue
                    f = F[s["id"]]
                    acc_set = {s["gold"]} | ({f["then"]} if f.get("then") else set()) | ({f["alt"]} if f.get("alt") else set())
                    commits += 1; correct += t["pred"] == s["gold"]; correct_acc += t["pred"] in acc_set
                    pos.append(t["k"]); saved.append(s["n_words"] - t["k"])
                    early += s["human_commit_word"] > 0 and t["k"] <= s["human_commit_word"]
                pol[f"nonnone tau={tau} stable={m}"] = dict(
                    commit_rate=commits / len(actionable), acc_at_commit=correct / commits if commits else None,
                    acc_at_commit_accepting_then_alt=correct_acc / commits if commits else None,
                    mean_commit_word=statistics.mean(pos) if pos else None, mean_words_saved=statistics.mean(saved) if saved else None,
                    at_or_before_human=early / commits if commits else None)
        summary["commit_policies"] = pol
        # false commits on out-of-scope rows: a confident non-none on any prefix of a 'none' utterance
        oos = [s for s in stream if s["gold"] == "none"]
        fc = sum(1 for s in oos if sa.commit_point(s["trajectory"], 0.9, 1, 1, skip_none=True) is not None)
        summary["false_commit_on_oos_rows_tau0.9"] = dict(rows=len(oos), false_commits=fc)

    manifest = dict(tag=args.tag, model_path=props.get("model_path"), n_ctx=props.get("default_generation_settings", {}).get("n_ctx"),
                    variant=args.variant, order=args.order, cache=cache, n_probs=args.n_probs, rows=len(full),
                    data_sha256=hashlib.sha256((ROOT / "data" / "voice" / "commands.jsonl").read_bytes()).hexdigest(),
                    prompt_example_sha256=hashlib.sha256(build_prompt("open notes", intents, args.variant, args.order).encode()).hexdigest(),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k != "errors"}
    printable["n_errors"] = len(summary["errors"])
    print(json.dumps(printable, indent=2, default=lambda o: round(o, 4) if isinstance(o, float) else o))


if __name__ == "__main__":
    main()
