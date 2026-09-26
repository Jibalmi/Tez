"""Prefix-labelled training rows for a streaming probe, two folds, as in the speed study (experiments/speed_voice.py).

Every word prefix of every command of the training fold is a row for `tez fit`: labelled `none` before the command's
decisive word (commit_word) and with its intent from that word on; out-of-scope commands are `none` throughout. The
folds are stratified by intent (seed 13, speed_voice.folds_of). For fold f the probe is trained on the other fold's
prefixes and the replay runs on fold f's commands only, so no command is scored by a probe that saw it.

  python experiments/voice_probe_labels.py --set theioma --out C:/temp/voice-fit
writes <out>/<set>/fold<f>/<schema>.yaml (a copy of the set's schema, where `tez fit` puts its artefacts),
labels.jsonl (the other fold's prefixes) and ids.txt (fold f's command ids, for voice_replay.py --ids).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = {"tez220": ROOT / "data" / "voice", "theioma": ROOT / "data" / "voice" / "theioma"}


def folds_of(cmds: list[dict], seed: int = 13) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for c in cmds:
        by[c["intent"]].append(c["id"])
    fold = {}
    for intent in sorted(by):
        ids = list(by[intent])
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            fold[i] = j % 2
    return fold


def prefix_rows(c: dict) -> list[dict]:
    words = c["text"].split()
    cw = c.get("commit_word", -1)
    out = []
    for k in range(1, len(words) + 1):
        label = c["intent"] if c["intent"] != "none" and cw > 0 and k >= cw else "none"
        out.append({"state": " ".join(words[:k]), "labels": {"action": label}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--out", default="C:/temp/voice-fit")
    args = ap.parse_args()
    d = SETS[args.set]
    cmds = [json.loads(x) for x in (d / "commands.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    fold = folds_of(cmds)
    for f in (0, 1):
        o = Path(args.out) / args.set / f"fold{f}"
        o.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / "schema.yaml", o / "schema.yaml")
        rows, seen = [], set()
        for c in cmds:
            if fold[c["id"]] == f:
                continue
            for r in prefix_rows(c):
                if r["state"] in seen:
                    continue          # the same prefix twice ("open ..."): tez fit keeps one label per state
                seen.add(r["state"])
                rows.append(r)
        (o / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (o / "ids.txt").write_text("\n".join(c["id"] for c in cmds if fold[c["id"]] == f) + "\n", encoding="utf-8")
        print(f"{o}: {len(rows)} prefix rows, {sum(1 for c in cmds if fold[c['id']] == f)} held-out commands")


if __name__ == "__main__":
    main()
