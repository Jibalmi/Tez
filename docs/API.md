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
    "gate": {"alpha": 0.05},
    "layout": "auto"
  }
}
```

- `schema`: name of a loaded schema. Questions with the same id use that schema's trained probe and calibration.
  A request may also omit `questions` entirely when `schema` is given: the schema's questions are used.
- `tez.readout`: `auto` (probe when one is trained for the question, otherwise letters), `letters`, or `probe`.
- `tez.abstain`: adds an implicit `__none__` option to every `choice` question ("none of these fits").
- `tez.gate.alpha`: target error rate among acted decisions (conformal selection, needs a fitted schema); each answer
  then gets `act` or `escalate` in the `tez` block.
- `tez.layout`: the prompt layout (see "Prompt layout" below): `auto`, `question_first` or `state_first`. Default: the
  schema's `layout:`, else the server's `--layout`, else `auto`.

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
`escalate`, present only when a gate applies), `p_correct` and `calibration_id` when a fitted calibration exists, and
`layout` when the question was read in another layout than the request's (a fitted question under `auto`).

The gate never acts without evidence: `act` always comes with `p_correct` and `calibration_id`. A question with no
usable fitted calibration, and any `__none__` answer, gets `escalate` whenever a gate is requested (in the example,
`is_urgent` and `anger` were never fitted). `tez.gate: null` or `false` turns a schema's default gate off; `tez.gate: {}`
uses the schema's default alpha. Thresholds are precomputed for alpha 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3 and
the schema default; any other alpha uses the largest precomputed value at or below it.

`tez.readout: "probe"` is strict: a request fails with 422 if a requested question has no usable probe. `auto` falls
back to letters question by question. A fit is used only while the question's prompt (instructions, options,
few-shot examples, template, layout) and the model still match what `tez fit` saw; otherwise the question falls back
to uncalibrated letters and `GET /v1/schemas/{name}` marks it `stale`.

### Prompt layout

Every question is one prompt that ends where the model answers with an option letter. Two layouts:

| Layout | Prompt | What the backend's prompt cache keeps |
|---|---|---|
| `question_first` | instructions, question, options, then `Input:` and the state last | the question's prefix, across states (one question, many states: the voice loop) |
| `state_first` | instructions, `Input:` and the state, then the question and its options | the state, across the questions about it |

`auto` (the default) reads a request with two or more questions `state_first` and a single question `question_first`
(streaming, when it comes, stays `question_first`). Measured on typed-decisions with Gemma 4 12B Q8_0: the same
accuracy (0.7015 state first vs 0.705, McNemar p = 0.74) for about half the evaluated tokens. State-first questions of a
request run back to back, so with llama.cpp prompt caching on one slot each question after the first evaluates only its
own tail. Worked examples (a schema's `examples`) come before the input in both layouts, so a question with examples
shares only the instructions with the other questions about its state.

A fit records its layout (`tez fit --layout`, default the schema's `layout:` or `question_first`). Under `auto` a fitted
question keeps the layout it was fitted under, so a request never makes a fit stale by having more questions; an
explicit layout that differs from the fit's reads the question with uncalibrated letters, and a schema served with such
a layout shows the fit as `stale`. `tez plan` shows the layout, calls and tokens of every question without calling the
model.

llama.cpp settings for Tez: `-np 1`, `--swa-full` for Gemma (sliding-window models otherwise re-read the whole prompt)
and `--cache-ram 0`: with host-memory prompt caching on (the default, 8 GiB) a request that keeps less than half of the
slot's cached tokens first copies the slot to host RAM, which is what a new state does. `tez doctor` checks these.

### Response headers

Every response (errors and CORS preflights included) carries:

- `x-typesafe-request-id`: 32 hex characters. For a decision it is the run id that hooks see (`ctx.run_id`), that a
  `DecisionLog` row records and that `POST /v1/feedback` accepts as `run_id`; for other routes a fresh id.
- `server-timing`: `total;dur=<ms>` for every response; decisions add `tez;dur=<ms>` (the engine) and
  `backend;dur=<ms>` (time spent in model calls), e.g. `tez;dur=212.4, backend;dur=205.1, total;dur=214.0`.

Decisions (`/v1/systemone`, successful or not) also carry `X-Tez-Run-Id` (the same id) and, when they got that far,
`X-Tez-Layout` (`question_first`, `state_first` or `mixed`). Browsers can read all four (CORS `Expose-Headers`).

### Errors

Same codes as Jev: `401` missing/invalid key (only with `--api-key`), `422` invalid request (including a prompt longer
than the model's context, with llama.cpp's message), `503` backend unavailable. Also `404` for an unknown route or
schema, `405` for a wrong method and `500` for an unexpected failure (a hook that raised, for example). Body:
`{"error": {"type": "invalid_request", "message": "..."}}`; the types are `unauthorized`, `invalid_request`,
`not_found`, `method_not_allowed`, `backend_unavailable` and `internal_error`.

## `GET /v1/models`

```json
{"models": [{"name": "tez-latest", "description": "Tez local decision engine (gemma-4-12b-q8_0, letters)", "release_date": "2026-09-24"}]}
```

## `GET /healthz`

```json
{"status": "ok", "version": "0.1.0", "backend": "http://127.0.0.1:8091", "template": "gemma4", "backend_status": "ok",
 "model": "gemma-4-12b-q8_0", "embed_backend": null, "schemas": ["support-triage"], "probes": {"support-triage": ["topic"]},
 "layout": "auto"}
```

`status` is `degraded` when the backend is unreachable. With a separate embedding backend, `embed_template` and
`embed_backend_status` are added. `layout` is the server's default prompt layout (`--layout`).

## `GET /v1/schemas` and `GET /v1/schemas/{name}`

```json
{"schemas": [{"name": "support-triage", "description": "...", "questions": {"topic": "choice", "is_urgent": "noul"},
              "calibration_id": "support-triage@2026-09-24", "probes": ["topic"]}]}
```

One schema returns its questions in wire form plus `layout` (the schema's own, or null), `served_layout` (what a
request that names none gets), `calibration_id`, `probes` (per question: `probe` = `ready`, `stale` or `none`,
`letters_calibrated`, `n_labels`, `note`, and the `layout` it was fitted under), `calibration` and `manifest`.

## `POST /v1/feedback`

```json
{"schema": "support-triage", "question": "topic", "state": "...", "label": "billing", "run_id": "5f0c..."}
```

Appends to `<data-dir>/feedback/<schema>.jsonl` (default data dir: `<schemas>/.tez`) and returns
`{"ok": true, "schema": ..., "question": ..., "label": ...}`. `tez fit` trains from these rows plus any labelled file.
`run_id` (optional) is the decision's `X-Tez-Run-Id`; it is stored with the row. `on_feedback` hooks see the row first
(`Redact` removes personal data from its state; docs/HOOKS.md).

## Hooks

Hooks run in the engine around every decision (docs/HOOKS.md): `tez serve --hook package.module:object` (repeatable),
`--decision-log PATH` (a JSONL row per decision, the format `tez fit` reads once labels are filled in) and
`--hook-errors raise|log` (whether a failing hook fails the request).

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
layout: auto           # optional: auto | question_first | state_first (default: the server's --layout)
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
| `letters` | nothing (zero-shot) | `/completion`, one token, top-`n_probs` log-probs (200) | Gemma 4 12B Q8: 0.704 |
| `probe` | ≥ 25–50 labels per question (`tez fit`) | `/embedding` (last-token state) | Qwen3.5-4B cut to 24 blocks: 0.793 |
| blend | a few labels | both | 12B letters + 50 typical labels per question: 0.766 |

Letters are answered with a single option letter (layouts above); more than 26 options run as a chunked tournament. A
letter outside the top `n_probs` next-token log-probabilities gets a floor (the smallest listed log-probability
minus 2); `--n-probs` changes how many are asked for (default 200, the measured setting). Hybrid Qwen3.5 GGUFs need
prompt caching off in llama.cpp b11100 (`tez serve` turns it off automatically when the model name matches Qwen3.5;
`--no-cache-prompt` forces it).

## Server defaults

`tez serve` listens on `127.0.0.1:8787`. The backend defaults to `http://127.0.0.1:8080` (llama-server's own default)
or the `TEZ_BACKEND` environment variable; `--backend fake` runs an offline demo backend whose answers mean nothing.
`tez fit` needs at least 20 labels per question to train a probe; with fewer it calibrates the letters only and says so.
