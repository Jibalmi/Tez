from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

DOCS_REQUEST = {
    "model": "tez-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "topic": {
            "type": "choice",
            "instructions": "What is the message about?",
            "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": None},
        },
        "anger": {
            "type": "score",
            "instructions": "How upset is the writer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}

DOCS_SCHEMA_YAML = """\
name: support-triage
description: Route and prioritise inbound customer messages.
state: "One customer message (text)."
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
    criteria:
      "true": Explicitly time-sensitive or blocking
      "false": No urgency expressed
  anger:
    type: score
    instructions: How upset is the writer?
    criteria: [Calm, Frustrated, Very angry]
gate:
  alpha: 0.05          # optional default for tez.gate
examples:              # optional labelled rows (few-shot and probe training)
  - state: "I was charged twice this month."
    labels: {topic: billing, is_urgent: "false", anger: 1}
"""

# A synthetic schema for fitting: the words of a message decide its labels, so the FakeBackend's hashed
# bag-of-words embeddings are class dependent and a probe can learn them.
SYNTH_SCHEMA_YAML = """\
name: synth
description: Synthetic routing for tests.
questions:
  topic:
    type: choice
    instructions: Which team should handle the message?
    criteria:
      billing: money questions
      technical: product faults
      sales: buying interest
  is_urgent:
    type: noul
    instructions: Is the message urgent?
  anger:
    type: score
    instructions: How upset is the writer?
    criteria: [calm, annoyed, furious]
"""

VOCAB = {
    "billing": ["invoice", "refund", "charged", "payment", "receipt", "billing", "overcharge", "card"],
    "technical": ["crash", "error", "broken", "bug", "timeout", "freeze", "login", "outage"],
    "sales": ["pricing", "quote", "upgrade", "seats", "enterprise", "discount", "trial", "plan"],
}
URGENT = ["immediately", "asap", "deadline", "tonight", "critical"]
FILLER = ["hello", "team", "please", "thanks", "regarding", "account", "message", "today", "again", "still",
          "could", "would", "check", "question", "about", "our", "company", "week", "email", "note"]
TOPICS = list(VOCAB)


def synth_state(rng: random.Random, topic: str, urgent: bool) -> str:
    words = rng.sample(VOCAB[topic], 3) + rng.sample(FILLER, 4) + (rng.sample(URGENT, 2) if urgent else [])
    rng.shuffle(words)
    return " ".join(words)


def synth_rows(n: int, seed: int = 0, anger_every: int = 10) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        topic = TOPICS[i % 3]
        urgent = rng.random() < 0.4
        labels = {"topic": topic, "is_urgent": "true" if urgent else "false"}
        if anger_every and i % anger_every == 0:
            labels["anger"] = rng.randrange(3)
        rows.append({"state": synth_state(rng, topic, urgent), "labels": labels})
    return rows


def write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def docs_request() -> dict:
    return copy.deepcopy(DOCS_REQUEST)


@pytest.fixture
def schema_dir(tmp_path: Path) -> Path:
    d = tmp_path / "schemas"
    d.mkdir()
    (d / "support-triage.yaml").write_text(DOCS_SCHEMA_YAML, encoding="utf-8")
    return d


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    d = tmp_path / "synth"
    d.mkdir()
    (d / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    return d
