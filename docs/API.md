# Tez API and schema format (v1)

Tez runs locally and speaks **TypeSafe's `/v1/systemone` wire format**, so a client written for Jev's wire format can
point its base URL at a Tez server (JevBench's own runner included). Everything Tez adds lives in optional fields that
strict clients ignore.

Default address: `http://127.0.0.1:8787`. Any `Authorization: Bearer <token>` (or `x-api-key`) is accepted: no key is
needed locally, and `--api-key` on `tez serve` makes one required (`/healthz` stays open). CORS is open by default and
preflights answer `Access-Control-Allow-Private-Network: true`, so the website playground can call a local server.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/systemone` | Decide: one state, many typed questions (Jev-compatible) |
| `GET` | `/v1/models` | List model aliases (Jev-compatible) |
| `GET` | `/healthz` | Liveness, backend and readout status |
| `GET` | `/v1/schemas` | Tez: schemas loaded from `--schemas` |
| `GET` | `/v1/schemas/{name}` | Tez: one schema (questions, probe status, calibration) |
| `POST` | `/v1/feedback` | Tez: record a correct label for a past decision (feeds `tez fit`) |

## `POST /v1/systemone`

### Request

```json
{
  "model": "tez-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
    },
    "topic": {
      "type": "choice",
      "instructions": "What is the message about?",
      "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": null}
    },
    "anger": {
      "type": "score",
      "instructions": "How upset is the writer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

- `state`: string, object or array (objects are serialised as JSON).
- `questions`: map of id → question. `type` is `noul` (yes/no), `choice` (2–255 options; `criteria` maps label → description or `null`),
  or `score` (2–10 ordered levels; `criteria` is a list, level *i* = index *i*). `instructions` may be a string, object or array.
- `model`: any string; `tez-latest` is the default alias.

**Tez extensions** (all optional):

```json
{
  "schema": "support-triage",
  "tez": {
    "readout": "auto",
    "abstain": false,
    "gate": {"alpha": 0.05}
  }
}
```

- `schema`: name of a loaded schema. Questions with the same id use that schema's trained probe and calibration.
  A request may also omit `questions` entirely when `schema` is given: the schema's questions are used.
- `tez.readout`: `auto` (probe when one is trained for the question, otherwise letters), `letters`, or `probe`.
- `tez.abstain`: adds an implicit `__none__` option to every `choice` question ("none of these fits").
- `tez.gate.alpha`: target error rate among acted decisions (conformal selection, needs a fitted schema); each answer
  then gets `act` or `escalate` in the `tez` block.

### Response

```json
{
  "model": "tez-0.1.0 (gemma-4-12b-q8_0, letters)",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.93},
    "topic": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.91, "technical": 0.08, "sales": 0.01},
      "confidence": 0.865
    },
    "anger": {
      "type": "score",
      "score": 1.12,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.04, "1": 0.8, "2": 0.16},
      "confidence": 0.7
    }
  },
  "usage": {"input_tokens": 612, "output_tokens": 0},
  "tez": {
    "latency_ms": 212.4,
    "questions": {
      "is_urgent": {"readout": "letters", "decision": "escalate"},
      "topic": {"readout": "probe", "decision": "act", "p_correct": 0.94, "calibration_id": "support-triage@2026-09-24"},
      "anger": {"readout": "letters", "decision": "escalate"}
    }
  }
}
```

The three answer shapes are exactly Jev's:

- **noul**: `noul` = P(yes). No `confidence` (Jev's noul answers carry none).
- **choice**: `choice` = argmax label, `probabilities` keyed by label (sum to 1), `confidence` = (k·p_max − 1)/(k − 1).
- **score**: `score` = expected level Σ i·p_i, `legend` keyed by level string, `probabilities` keyed by level string,
  `confidence` as for choice.

The `tez` block: `latency_ms`, and per question the `readout` used (`letters` or `probe`), `decision` (`act` /
`escalate`, present only when a gate applies), `p_correct` and `calibration_id` when a fitted calibration exists.

The gate never acts without evidence: `act` always comes with `p_correct` and `calibration_id`. A question with no
usable fitted calibration, and any `__none__` answer, gets `escalate` whenever a gate is requested (in the example,
`is_urgent` and `anger` were never fitted). `tez.gate: null` or `false` turns a schema's default gate off; `tez.gate: {}`
uses the schema's default alpha. Thresholds are precomputed for alpha 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3 and
the schema default; any other alpha uses the largest precomputed value at or below it.

`tez.readout: "probe"` is strict: a request fails with 422 if a requested question has no usable probe. `auto` falls
back to letters question by question. A fit is used only while the question's prompt (instructions, options,
few-shot examples, template) and the model still match what `tez fit` saw; otherwise the question falls back to
uncalibrated letters and `GET /v1/schemas/{name}` marks it `stale`.

### Errors

Same codes as Jev: `401` missing/invalid key (only with `--api-key`), `422` invalid request (including a prompt longer
than the model's context, with llama.cpp's message), `503` backend unavailable. Also `404` for an unknown route or
schema and `405` for a wrong method. Body: `{"error": {"type": "invalid_request", "message": "..."}}`; the types are
`unauthorized`, `invalid_request`, `not_found`, `method_not_allowed` and `backend_unavailable`.

## `GET /v1/models`

```json
{"models": [{"name": "tez-latest", "description": "Tez local decision engine (gemma-4-12b-q8_0, letters)", "release_date": "2026-09-24"}]}
```

## `GET /healthz`

```json
{"status": "ok", "version": "0.1.0", "backend": "http://127.0.0.1:8091", "template": "gemma4", "backend_status": "ok",
 "model": "gemma-4-12b-q8_0", "embed_backend": null, "schemas": ["support-triage"], "probes": {"support-triage": ["topic"]}}
```

`status` is `degraded` when the backend is unreachable. With a separate embedding backend, `embed_template` and
`embed_backend_status` are added.

## `GET /v1/schemas` and `GET /v1/schemas/{name}`

```json
{"schemas": [{"name": "support-triage", "description": "...", "questions": {"topic": "choice", "is_urgent": "noul"},
              "calibration_id": "support-triage@2026-09-24", "probes": ["topic"]}]}
```

One schema returns its questions in wire form plus `calibration_id`, `probes` (per question: `probe` = `ready`,
`stale` or `none`, `letters_calibrated`, `n_labels`, `note`), `calibration` and `manifest`.

## `POST /v1/feedback`

```json
{"schema": "support-triage", "question": "topic", "state": "...", "label": "billing"}
```

Appends to `<data-dir>/feedback/<schema>.jsonl` (default data dir: `<schemas>/.tez`) and returns
`{"ok": true, "schema": ..., "question": ..., "label": ...}`. `tez fit` trains from these rows plus any labelled file.

## Schema files (`schemas/*.yaml`)

A schema is a named, reusable set of questions plus what Tez has learned about them.

```yaml
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
examples:              # optional labelled rows: the first 4 per question (26 options or fewer) become few-shot
                       # examples in the prompt; the rest are training rows
  - state: "I was charged twice this month."
    labels: {topic: billing, is_urgent: "false", anger: 1}
```

Trained artefacts live next to the schema in `schemas/.tez/<name>/`: `probes.npz` (per-question logistic weights on
the embedding backend's features), `calibration.json` (temperature per question, conformal thresholds per alpha),
and `manifest.json` (backend, template, layer/cut, label counts, date).

## Readouts

| Readout | Needs | Backend call | Measured (typed-decisions) |
|---|---|---|---|
| `letters` | nothing (zero-shot) | `/completion`, one token, top-200 log-probs | Gemma 4 12B Q8: 0.704 |
| `probe` | ≥ 25–50 labels per question (`tez fit`) | `/embedding` (last-token state) | Qwen3.5-4B cut to 24 blocks: 0.793 |
| blend | a few labels | both | 12B letters + 50 typical labels per question: 0.766 |

Prompt layout (letters): instructions and options first, state last, answered with a single option letter;
more than 26 options run as a chunked tournament. Hybrid Qwen3.5 GGUFs need prompt caching off in llama.cpp b11100
(`tez serve` turns it off automatically when the model name matches Qwen3.5; `--no-cache-prompt` forces it).

## Server defaults

`tez serve` listens on `127.0.0.1:8787`. The backend defaults to `http://127.0.0.1:8080` (llama-server's own default)
or the `TEZ_BACKEND` environment variable; `--backend fake` runs an offline demo backend whose answers mean nothing.
`tez fit` needs at least 20 labels per question to train a probe; with fewer it calibrates the letters only and says so.
