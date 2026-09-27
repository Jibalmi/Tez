"""Record current upstream model revisions and declarations, separate from local measurements."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

MODELS = ["convaiinnovations/laya", "convaiinnovations/laya-multilingual", "convaiinnovations/laya-typed-decisions",
          "mpnikhil/dev-0.4b", "jaredpalmer/kev-0.8b", "Mapika/decider-0.8b",
          "knowledgator/gliclass-x-base", "knowledgator/gliclass-multilang-mini",
          "knowledgator/gliclass-instruct-base-v1.0", "knowledgator/gliclass-instruct-edge-v1.0",
          "Qwen/Qwen3.5-0.8B"]


def fetch(name):
    url = "https://huggingface.co/api/models/" + name
    req = urllib.request.Request(url, headers={"User-Agent": "tez-research/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
        card = data.get("cardData") or {}
        return name, {"source_url": url, "model_card": "https://huggingface.co/" + name,
                      "revision": data.get("sha"), "last_modified": data.get("lastModified"),
                      "license_declared": card.get("license"), "languages_declared": card.get("language"),
                      "base_model_declared": card.get("base_model"), "parameters": data.get("safetensors"),
                      "training_datasets_declared": card.get("datasets"),
                      "files": [r["rfilename"] for r in data.get("siblings", [])]}
    except Exception as exc:
        return name, {"source_url": url, "error": type(exc).__name__ + ": " + str(exc)}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = dict(pool.map(fetch, MODELS))
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Upstream metadata only. Published claims are not measurements on Tez's suites.", "models": rows}
    out = Path(__file__).resolve().parents[2] / "results/pilot-improvement-20260927/competitor_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: row.get(k) for k in ("revision", "license_declared", "parameters", "error")}
                      for name, row in rows.items()}, indent=2))
