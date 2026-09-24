"""Record who holds the GPU's memory (Windows per-process dedicated / shared GPU memory counters, Ollama's loaded
models) into results/speed/vram_<tag>.json. A llama.cpp process whose weights do not fit in dedicated VRAM gets part
of them in shared system memory (NVIDIA sysmem fallback) and runs several times slower, so every timing session
records this next to its results.

  python experiments/speed_vram_snapshot.py --tag after_np8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import speed_common as sc


def snapshot() -> dict:
    ps = ("$c = Get-Counter -Counter '\\GPU Process Memory(*)\\Dedicated Usage','\\GPU Process Memory(*)\\Shared Usage' "
          "-ErrorAction SilentlyContinue; $c.CounterSamples | Where-Object { $_.CookedValue -gt 50MB } | "
          "ForEach-Object { '{0}|{1}|{2}' -f $_.InstanceName, $_.Path.Split('\\')[-1], $_.CookedValue }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    procs: dict = {}
    for line in out.splitlines():
        try:
            inst, counter, val = line.strip().split("|")
            pid = int(inst.split("_")[1])
            procs.setdefault(pid, {})[counter.replace(" usage", "")] = round(float(val) / 2**30, 3)
        except Exception:  # noqa: BLE001
            continue
    names = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-Process | ForEach-Object { '{0}|{1}' -f $_.Id, $_.ProcessName }"],
                           capture_output=True, text=True, timeout=60).stdout
    pname = {}
    for line in names.splitlines():
        try:
            i, n = line.strip().split("|")
            pname[int(i)] = n
        except Exception:  # noqa: BLE001
            continue
    for pid in procs:
        procs[pid]["name"] = pname.get(pid, "?")
    try:
        ollama = sc.S.get("http://127.0.0.1:11434/api/ps", timeout=5).json()
    except Exception as exc:  # noqa: BLE001
        ollama = {"error": str(exc)}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "nvidia_smi_memory": smi, "gpu": sc.gpu_now(),
            "per_process_GB": dict(sorted(procs.items(), key=lambda kv: -kv[1].get("dedicated", 0))), "ollama_ps": ollama}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    s = snapshot()
    s["note"] = a.note
    sc.dump(f"vram_{a.tag}.json", s)
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
