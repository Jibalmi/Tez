"""Offline analysis of streaming voice trajectories (results/voice_<tag>_stream.jsonl).

Questions answered without re-scoring:
  1. How does accuracy grow with the fraction of the utterance heard?
  2. Naive commit (first prefix with pmax >= tau) vs STABILITY commit (argmax unchanged for the
     last m prefixes AND pmax >= tau) vs a minimum-word gate: commit rate, accuracy at commit,
     mean words saved, and whether commits happen at or before the human commit word.
  3. Compound rows: accuracy when the terminal action ("then") is also accepted.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def commit_point(traj, tau, m=1, min_words=1, skip_none=False):
    """First index k such that pmax>=tau at k, argmax stable over prefixes k-m+1..k, and k>=min_words.
    skip_none: a confident 'none' on a PREFIX means "not enough heard yet", never a commit --
    this single rule is what makes early commit work (see report)."""
    for i, t in enumerate(traj):
        k = t["k"]
        if k < min_words or t["pmax"] < tau:
            continue
        if skip_none and t["pred"] == "none":
            continue
        if i + 1 < m:
            continue
        window = traj[i + 1 - m: i + 1]
        if all(w["pred"] == t["pred"] for w in window):
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--full", required=True, help="matching full-utterance results jsonl")
    ap.add_argument("--out")
    args = ap.parse_args()
    S = load(args.stream)
    F = {r["id"]: r for r in load(args.full)}
    actionable = [s for s in S if s["gold"] != "none"]
    # accepted set per row: gold, plus 'then' for compound, plus 'alt' for ambiguous
    accept = {}
    for s in S:
        f = F[s["id"]]
        acc = {s["gold"]}
        if f.get("then"):
            acc.add(f["then"])
        if f.get("alt"):
            acc.add(f["alt"])
        accept[s["id"]] = acc

    res = {}
    # 1. accuracy vs fraction heard (buckets of 20 %)
    buckets = defaultdict(list)
    for s in actionable:
        for t in s["trajectory"]:
            frac = t["k"] / s["n_words"]
            b = min(4, int(frac * 5))
            buckets[b].append(t["pred"] == s["gold"])
    res["accuracy_by_fraction_heard"] = {f"{b*20}-{(b+1)*20}%": round(statistics.mean(v), 3) for b, v in sorted(buckets.items())}
    # also: what does the model say on 1-word prefixes?
    first = defaultdict(int)
    for s in actionable:
        first[s["trajectory"][0]["pred"]] += 1
    res["first_word_predictions"] = dict(sorted(first.items(), key=lambda kv: -kv[1])[:6])

    # 2. commit policies
    policies = {}
    for skip_none in (False, True):
      for tau in (0.9, 0.99):
        for m in (1, 2, 3):
            for min_words in (1, 2, 3):
                commits, correct, correct_acc, pos, saved, early = 0, 0, 0, [], [], 0
                for s in actionable:
                    t = commit_point(s["trajectory"], tau, m, min_words, skip_none)
                    if t is None:
                        continue
                    commits += 1
                    correct += t["pred"] == s["gold"]
                    correct_acc += t["pred"] in accept[s["id"]]
                    pos.append(t["k"]); saved.append(s["n_words"] - t["k"])
                    if s["human_commit_word"] > 0 and t["k"] <= s["human_commit_word"]:
                        early += 1
                n = len(actionable)
                policies[f"{'nonnone' if skip_none else 'naive'} tau={tau} stable={m} min_words={min_words}"] = dict(
                    commit_rate=round(commits / n, 3),
                    acc_at_commit=round(correct / commits, 3) if commits else None,
                    acc_at_commit_accepting_then_alt=round(correct_acc / commits, 3) if commits else None,
                    mean_commit_word=round(statistics.mean(pos), 2) if pos else None,
                    mean_words_saved=round(statistics.mean(saved), 2) if saved else None,
                    at_or_before_human=round(early / commits, 3) if commits else None,
                )
    res["commit_policies"] = policies
    # "no early commit" reference: full utterance
    full_ok = statistics.mean(F[s["id"]]["pred_intent"] == s["gold"] for s in actionable)
    full_ok_acc = statistics.mean(F[s["id"]]["pred_intent"] in accept[s["id"]] for s in actionable)
    res["full_utterance_accuracy_actionable"] = dict(strict=round(full_ok, 3), accepting_then_alt=round(full_ok_acc, 3))
    all_ok_acc = statistics.mean(F[s["id"]]["pred_intent"] in accept[s["id"]] for s in S)
    res["full_utterance_accuracy_all_accepting_then_alt"] = round(all_ok_acc, 3)

    # oracle stability: first k from which argmax stays in accept-set to the end
    fr = []
    for s in actionable:
        traj = s["trajectory"]
        k0 = next((t["k"] for t in traj if all(u["pred"] in accept[s["id"]] for u in traj[t["k"] - 1:])), None)
        fr.append((k0 or s["n_words"]) / s["n_words"])
    res["oracle_stable_commit_fraction"] = round(statistics.mean(fr), 3)

    print(json.dumps({k: v for k, v in res.items() if k != "commit_policies"}, indent=2))
    print(f"{'policy':36s} {'commit':>6s} {'acc':>6s} {'acc+then':>8s} {'word':>5s} {'saved':>6s} {'<=human':>7s}")
    for k, v in policies.items():
        print(f"{k:36s} {v['commit_rate']:6.3f} {str(v['acc_at_commit']):>6s} {str(v['acc_at_commit_accepting_then_alt']):>8s} {str(v['mean_commit_word']):>5s} {str(v['mean_words_saved']):>6s} {str(v['at_or_before_human']):>7s}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
