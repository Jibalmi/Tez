"""`tez fit` of a streaming probe on one fold's prefix-labelled rows (experiments/voice_probe_labels.py), in process.

The same as `tez fit --schema <fold>/schema.yaml --labels <fold>/labels.jsonl --no-letters --no-blend --no-feedback`,
with the in-process settings a hybrid model wants on a shared GPU (few sequences, 6 threads; the CLI's defaults
reserve 64 sequences). The probe reads the last-token state of the question's own prompt (the answer position), and
without the blend the stream reads only that prompt per word.

  python experiments/voice_probe_fit.py --fold C:/temp/voice-fit/theioma/fold0 --model <Qwen3.5-4B-Q8_0-L24.gguf>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--lib", default="C:/temp/llamacpp")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    from tez import Tez
    from tez.fit import fit
    d = Path(args.fold)
    t0 = time.perf_counter()
    tez = Tez(backend=f"inproc:{args.model}", template=args.template, schemas=d / "schema.yaml",
              inproc={"lib": args.lib, "n_ctx": 2048, "n_seq_max": 4, "n_threads": args.threads})
    (name,) = list(tez.schemas)
    cal = fit(tez, tez.schemas[name], [d / "labels.jsonl"], feedback=False, letters=False, use_blend=False)
    q = next(iter(cal["questions"].values()))
    print(json.dumps({"fold": str(d), "seconds": round(time.perf_counter() - t0, 1),
                      "probe": {k: q.get("probe", {}).get(k) for k in ("accuracy", "ece", "temperature", "n_fit",
                                                                       "n_held_out")}}, indent=1))
    tez.backend.close()


if __name__ == "__main__":
    main()
