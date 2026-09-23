"""Tez quickstart: the Python API.

Uses a llama-server when one answers at TEZ_BACKEND (default http://127.0.0.1:8091); otherwise it falls back to
the offline FakeBackend (keyword matching, not a model) so the script always runs.

    pip install -e .            # or: pip install tez-decisions
    llama-server -m gemma-4-12b-it-Q8_0.gguf -ngl 99 -c 4096 --port 8091 --embeddings --pooling last --swa-full
    python examples/quickstart.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from tez import LlamaCppBackend, Tez
except ImportError:          # running from a source checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tez import LlamaCppBackend, Tez

URL = os.environ.get("TEZ_BACKEND", "http://127.0.0.1:8091")
TEMPLATE = os.environ.get("TEZ_TEMPLATE", "gemma4")          # gemma4 | qwen3
backend = URL if LlamaCppBackend(URL, template=TEMPLATE).health()["ok"] else "fake"
print(f"backend: {backend}")

# 1. Jev-style: one state, many typed questions. The result is the /v1/systemone wire format.
tez = Tez(backend=backend, template=TEMPLATE)
questions = {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "topic": {"type": "choice", "instructions": "What is the message about?",
              "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": None}},
    "anger": {"type": "score", "instructions": "How upset is the writer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
}
res = tez.decide("Help! My payouts have been failing for 3 days.", questions=questions)
print(json.dumps(res, indent=2))
topic = res["answers"]["topic"]
print(f"topic={topic['choice']} (confidence {topic['confidence']:.2f}), "
      f"urgent={res['answers']['is_urgent']['noul']:.2f}, anger={res['answers']['anger']['score']:.2f}, "
      f"{res['tez']['latency_ms']:.0f} ms")

# 2. A reusable schema. `tez fit --schema <file> --labels <jsonl>` later adds probes, temperatures and a gate
#    next to it (in .tez/<name>/); until then every question is read zero-shot.
schema_dir = Path(tempfile.mkdtemp(prefix="tez-schemas-"))
(schema_dir / "support-triage.yaml").write_text("""\
name: support-triage
description: Route and prioritise inbound customer messages.
questions:
  topic:
    type: choice
    instructions: What is the message about?
    criteria:
      billing: Payments, payouts, invoices, refunds
      technical: Something is broken or not working
      account: Login, profile, settings
      sales: Pricing, plans, buying more
  is_urgent:
    type: noul
    instructions: Does this convey urgency?
""", encoding="utf-8")
tez = Tez(backend=backend, template=TEMPLATE, schemas=schema_dir)

# abstain adds an implicit "__none__" option to choice questions ("none of these fits")
res = tez.decide("Do you ship to the Moon?", schema="support-triage", abstain=True)
print(json.dumps(res["answers"]["topic"], indent=2))

# a gate needs a fitted schema: without one every decision is escalated
res = tez.decide("I can't log in since this morning, the app says my password is wrong.", schema="support-triage", alpha=0.05)
print({qid: meta for qid, meta in res["tez"]["questions"].items()})
shutil.rmtree(schema_dir, ignore_errors=True)
