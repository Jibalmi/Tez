"""Voice-command intent routing as a typed decision, scored on full utterances and on
word-by-word streaming prefixes to measure how early a calibrated decision can commit.

Each utterance becomes a SemIf-style row:
  state    = {"transcript": "<partial or full transcript>"}
  question = which command is the user giving (or none)
  options  = the 16 intents from data/voice/intents.json (letters A-P)
and is scored through llama-server exactly like the benchmark rows (run_direct.score_row).

Streaming: for every prefix of k words we record the distribution; a decision COMMITS at the
first prefix whose max-probability >= tau. We report, per tau: commit rate, accuracy at commit,
mean commit position, fraction that commit before the human commit word, and words saved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_direct", ROOT / "experiments" / "run_direct.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)  # type: ignore[union-attr]

QUESTION = ("Which command is the user giving to the computer assistant? Choose the single best "
            "matching action. Choose 'none' if the request is chit-chat, a question, out of scope, "
            "or too unclear to act on.")


def load_voice():
    intents = json.loads((ROOT / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
    rows = [json.loads(l) for l in (ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return intents, rows


def make_row(cmd: dict, transcript: str, intents: list[dict], suffix: str = "") -> dict:
    ids = [i["id"] for i in intents]
    return {
        "id": cmd["id"] + suffix,
        "state": {"transcript": transcript},
        "question": QUESTION,
        "options": [{"id": i["id"], "description": i["description"]} for i in intents],
        "label": ids.index(cmd["intent"]),
        "family": cmd["style"],
        "group_id": cmd["id"],
    }


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--template", default="gemma4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stream", action="store_true", help="also score every word prefix")
    ap.add_argument("--n-probs", type=int, default=200)
    ap.add_argument("--from-files", action="store_true", help="skip scoring; summarise existing --out files")
    args = ap.parse_args()

    intents, cmds = load_voice()
    if args.limit:
        cmds = cmds[: args.limit]
    ids = [i["id"] for i in intents]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    full_recs, stream_recs = [], []
    t0 = time.perf_counter()
    if args.from_files:
        full_recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        sp = out.with_name(out.stem + "_stream.jsonl")
        if sp.exists():
            stream_recs = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        cmds = []
    for n, cmd in enumerate(cmds, 1):
        words = cmd["text"].split()
        rec = rd.score_row(args.server, make_row(cmd, cmd["text"], intents), args.n_probs, False, args.template)
        rec.update(gold_intent=cmd["intent"], style=cmd["style"], n_words=len(words),
                   then=cmd.get("then"), alt=cmd.get("alt"), human_commit_word=cmd.get("commit_word", -1),
                   pred_intent=ids[argmax(rec["probabilities"])], text=cmd["text"])
        full_recs.append(rec)
        if args.stream:
            traj = []
            for k in range(1, len(words) + 1):
                prefix = " ".join(words[:k])
                r = rd.score_row(args.server, make_row(cmd, prefix, intents, f"#{k}"), args.n_probs, False, args.template)
                p = r["probabilities"]
                traj.append(dict(k=k, prefix=prefix, pred=ids[argmax(p)], pmax=max(p), probabilities=p, prompt_ms=r["prompt_ms"]))
            stream_recs.append(dict(id=cmd["id"], gold=cmd["intent"], style=cmd["style"], n_words=len(words),
                                    human_commit_word=cmd.get("commit_word", -1), trajectory=traj))
        if n % 20 == 0 or n == len(cmds):
            print(f"  {n}/{len(cmds)}  {time.perf_counter()-t0:.0f}s", flush=True)

    with out.open("w", encoding="utf-8") as f:
        for r in full_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if stream_recs:
        with out.with_name(out.stem + "_stream.jsonl").open("w", encoding="utf-8") as f:
            for r in stream_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- summary: full-utterance
    acc = statistics.mean(r["pred_intent"] == r["gold_intent"] for r in full_recs)
    by_style = defaultdict(list)
    for r in full_recs:
        by_style[r["style"]].append(r["pred_intent"] == r["gold_intent"])
    # count an 'ambiguous' row as correct if it predicts the alt too
    acc_alt = statistics.mean(bool((r["pred_intent"] == r["gold_intent"]) or (r.get("alt") and r["pred_intent"] == r["alt"])) for r in full_recs)
    none_rows = [r for r in full_recs if r["gold_intent"] == "none"]
    none_pred = [r for r in full_recs if r["pred_intent"] == "none"]
    none_tp = sum(1 for r in none_rows if r["pred_intent"] == "none")
    lat = sorted(r["prompt_ms"] for r in full_recs)
    conf_correct = [max(r["probabilities"]) for r in full_recs if r["pred_intent"] == r["gold_intent"]]
    conf_wrong = [max(r["probabilities"]) for r in full_recs if r["pred_intent"] != r["gold_intent"]]
    summary = dict(
        tag=args.tag, n=len(full_recs), intent_accuracy=acc, intent_accuracy_alt_ok=acc_alt,
        accuracy_by_style={s: statistics.mean(v) for s, v in by_style.items()},
        none_recall=none_tp / len(none_rows) if none_rows else None,
        none_precision=none_tp / len(none_pred) if none_pred else None,
        mean_conf_correct=statistics.mean(conf_correct) if conf_correct else None,
        mean_conf_wrong=statistics.mean(conf_wrong) if conf_wrong else None,
        prompt_ms_p50=lat[len(lat) // 2], prompt_ms_p95=lat[int(0.95 * (len(lat) - 1))],
        mean_prompt_tokens=statistics.mean(r["prompt_n"] for r in full_recs),
        errors=[dict(id=r["id"], text=r["text"], gold=r["gold_intent"], pred=r["pred_intent"], p=round(max(r["probabilities"]), 3), style=r["style"])
                for r in full_recs if r["pred_intent"] != r["gold_intent"]],
    )

    # ---------------- summary: streaming early commit
    if stream_recs:
        actionable = [s for s in stream_recs if s["gold"] != "none"]
        stream_summary = {}
        for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            commits, correct, pos, early, saved = 0, 0, [], 0, []
            for s in actionable:
                hit = next((t for t in s["trajectory"] if t["pmax"] >= tau), None)
                if hit is None:
                    continue
                commits += 1
                correct += hit["pred"] == s["gold"]
                pos.append(hit["k"])
                saved.append(s["n_words"] - hit["k"])
                if s["human_commit_word"] > 0 and hit["k"] <= s["human_commit_word"]:
                    early += 1
            stream_summary[f"tau={tau}"] = dict(
                commit_rate=commits / len(actionable), accuracy_at_commit=correct / commits if commits else None,
                mean_commit_word=statistics.mean(pos) if pos else None, mean_words_saved=statistics.mean(saved) if saved else None,
                commit_at_or_before_human=early / commits if commits else None,
            )
        # stable-commit point: first k from which the argmax stays gold to the end
        stable = []
        for s in actionable:
            traj = s["trajectory"]
            k_stable = None
            for t in traj:
                if all(u["pred"] == s["gold"] for u in traj[t["k"] - 1:]):
                    k_stable = t["k"]
                    break
            stable.append(k_stable / s["n_words"] if k_stable else 1.0)
        stream_summary["mean_stable_commit_fraction_of_utterance"] = statistics.mean(stable)
        stream_summary["per_prefix_prompt_ms_p50"] = statistics.median(t["prompt_ms"] for s in stream_recs for t in s["trajectory"])
        summary["stream"] = stream_summary

    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k != "errors"}
    printable["n_errors"] = len(summary["errors"])
    print(json.dumps(printable, indent=2, default=lambda o: round(o, 4) if isinstance(o, float) else o))


if __name__ == "__main__":
    main()
