"""Compound utterances as two decisions: after the first action fires at the human commit word,
score the RESIDUAL ("... and type buy milk") word by word and check that it routes to the
terminal action (`then`). Offline counterpart of Decider.step's cursor logic in voice_demo.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vf", ROOT / "experiments" / "run_voice_fast.py")
vf = importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)  # type: ignore[union-attr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--variant", default="final")
    ap.add_argument("--prior", action="store_true", help="tell the model which action already fired (as the live loop does)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    intents = json.loads((ROOT / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
    ids = [i["id"] for i in intents]
    cmds = [json.loads(l) for l in (ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    comp = [c for c in cmds if c.get("then")]
    rows = []
    for c in comp:
        words = c["text"].split()
        cut = c.get("commit_word", 0)
        res = words[cut:]
        traj = []
        for k in range(1, len(res) + 1):
            r = vf.score(args.server, vf.build_prompt(" ".join(res[:k]), intents, args.variant, "last", c["intent"] if args.prior else None), len(ids), 200, True)
            p = r["probabilities"]; i = max(range(len(p)), key=lambda j: p[j])
            traj.append(dict(k=k, prefix=" ".join(res[:k]), pred=ids[i], pmax=round(p[i], 3)))
        final = traj[-1]["pred"] if traj else None
        # first residual prefix that reaches `then` at p>=0.9 and stays there
        first_ok = next((t["k"] for t in traj if t["pred"] == c["then"] and t["pmax"] >= 0.9 and all(u["pred"] == c["then"] for u in traj[t["k"] - 1:])), None)
        rows.append(dict(id=c["id"], text=c["text"], first=c["intent"], then=c["then"], residual=" ".join(res), final_pred=final,
                         ok=final == c["then"], first_stable_k=first_ok, residual_words=len(res), traj=traj))
    n_ok = sum(r["ok"] for r in rows)
    summary = dict(n=len(rows), residual_routes_to_then=n_ok, rate=n_ok / len(rows),
                   mean_stable_fraction=sum((r["first_stable_k"] or r["residual_words"]) / r["residual_words"] for r in rows) / len(rows))
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for r in rows:
        if not r["ok"]:
            print("  MISS", r["id"], r["residual"], "->", r["final_pred"], "(wanted", r["then"] + ")")


if __name__ == "__main__":
    main()
