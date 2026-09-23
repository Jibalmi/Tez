"""Record real `tez serve` responses for every use-case sample, for the website's replay mode.

Sends each sample in examples/usecases/<id>/samples.jsonl with that use case's questions to a running Tez server
(POST /v1/systemone, wire format per docs/API.md) and stores request, response and client round-trip time.

  tez serve --backend http://127.0.0.1:8091 --template gemma4 --port 8787
  py experiments/record_site_replays.py --server http://127.0.0.1:8787 --out site/data/replays.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def wire_questions(schema: dict) -> dict:
    out = {}
    for qid, q in (schema.get("questions") or {}).items():
        item = {"type": q["type"], "instructions": q.get("instructions", "")}
        crit = q.get("criteria")
        if q["type"] == "noul":
            crit = {str(k).lower(): v for k, v in (crit or {}).items()}
        item["criteria"] = crit
        out[qid] = item
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8787")
    ap.add_argument("--usecases", default=str(ROOT / "examples" / "usecases"))
    ap.add_argument("--out", default=str(ROOT / "site" / "data" / "replays.json"))
    args = ap.parse_args()
    s = requests.Session()
    health = s.get(f"{args.server}/healthz", timeout=30).json()
    models = s.get(f"{args.server}/v1/models", timeout=30).json()
    manifest = json.loads((Path(args.usecases) / "usecases.json").read_text(encoding="utf-8"))
    items = manifest if isinstance(manifest, list) else manifest.get("usecases", [])
    replays, errors = [], []
    for uc in items:
        uid = uc["id"]
        schema = yaml.safe_load((Path(args.usecases) / uid / "schema.yaml").read_text(encoding="utf-8"))
        questions = wire_questions(schema)
        samples = [json.loads(l) for l in (Path(args.usecases) / uid / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, smp in enumerate(samples):
            req = {"model": "tez-latest", "state": smp["state"], "questions": questions}
            t0 = time.perf_counter()
            r = s.post(f"{args.server}/v1/systemone", json=req, timeout=600)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                errors.append(dict(usecase=uid, sample_index=i, status=r.status_code, body=r.text[:400]))
                print("ERROR", uid, i, r.status_code, r.text[:200], flush=True)
                continue
            resp = r.json()
            replays.append(dict(usecase=uid, sample_index=i, state=smp["state"], note=smp.get("note"),
                                request=req, response=resp, client_ms=round(ms, 1)))
            first = next(iter(resp["answers"].values()))
            print(f"{uid:24s} #{i} {ms:7.1f} ms  first answer: {json.dumps(first)[:110]}", flush=True)
    meta = dict(recorded=time.strftime("%Y-%m-%dT%H:%M:%S"), server=args.server, healthz=health, models=models,
                model="Gemma 4 12B Q8_0 via llama.cpp llama-server b11100", hardware="RTX 5080 laptop GPU (16 GB)",
                host=platform.platform(), inputs="synthetic example messages from examples/usecases/*/samples.jsonl",
                note="Real responses from tez serve; replayed in the browser, nothing runs there.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(meta=meta, replays=replays, errors=errors), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}: {len(replays)} replays, {len(errors)} errors")


if __name__ == "__main__":
    main()
