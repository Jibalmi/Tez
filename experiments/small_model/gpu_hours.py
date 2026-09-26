"""GPU-hours the pilot held the shared compute lock, from runs/pilot/logs/lock_watch.log (20 s polling of
C:/temp/gpu-lock.json) plus the hold of 2026-09-25, which ended when the PC crashed (runs/pilot/logs/gpu_session1.log).

    python experiments/small_model/gpu_hours.py      # -> results/pilot/gpu_hours.json
"""
from __future__ import annotations

import datetime as dt

from common import RESULTS, RUNS, dump

# 2026-09-25: lock acquired 16:09:30 (eval embeddings, pool filter, clusters, schema server); the PC shut down hard
# at 16:32 with the hold still open (C:/temp/run_heavy.py's docstring gives 16:32).
EARLIER = [("2026-09-25T16:09:30+02:00", "2026-09-25T16:32:00+02:00", "eval embeddings, filter, clusters, 5 schema clusters (lost in the crash)")]


def main() -> None:
    holds = [{"from": a, "to": b, "note": n} for a, b, n in EARLIER]
    start = None
    for line in (RUNS / "logs" / "lock_watch.log").read_text(encoding="utf-8").splitlines():
        ts, state = line.split()
        if state == "held" and start is None:
            start = ts
        elif state == "free" and start is not None:
            holds.append({"from": start, "to": ts})
            start = None
    if start is not None:
        holds.append({"from": start, "to": None, "note": "still held"})
    total = 0.0
    for h in holds:
        if h["to"]:
            h["hours"] = round((dt.datetime.fromisoformat(h["to"]) - dt.datetime.fromisoformat(h["from"])).total_seconds() / 3600, 3)
            total += h["hours"]
    dump(RESULTS / "gpu_hours.json", {"title": "Compute-lock holds of the small-model pilot (GPU-hours)",
                                      "method": "20 s polling of the lock file; each hold is exclusive use of the GPU",
                                      "total_hours": round(total, 2), "holds": holds})
    print(f"{total:.2f} h in {len(holds)} holds")


if __name__ == "__main__":
    main()
