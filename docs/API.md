# Tez API and schema format (v1)

Tez runs locally and speaks **TypeSafe's `/v1/systemone` wire format**, so a client written for Jev's wire format can
point its base URL at a Tez server (JevBench's own runner included). Everything Tez adds lives in optional fields that
strict clients ignore.

Default address: `http://127.0.0.1:8787`. Any `Authorization: Bearer <token>` (or `x-api-key`) is accepted: no key is
needed locally, and `--api-key` on `tez serve` makes one required (`/healthz` stays open).

Browsers: CORS allows only listed origins, by default `http://127.0.0.1:*` and `http://localhost:*` (pages served
from this machine, any port) and `https://jibalmi.github.io` (the website playground). `--cors-origins LIST` (or
`TEZ_CORS_ORIGINS`) replaces the list: origins, `host:*` for any port, `https://*.domain`, `null`, or `*` for any
origin; `--no-cors` sends no CORS headers and allows no origin. A preflight from an allowed origin is answered with
`Access-Control-Allow-Private-Network: true` (Chrome's Private Network Access, which the playground needs to reach a
server on your machine); from any other origin it gets `403`. Every `POST` (`/v1/systemone`, `/v1/systemone/batch`,
`/v1/plan`, `/v1/feedback`) from an origin that is not allowed gets `403 forbidden` before anything runs: a web page
elsewhere could not read the answer anyway, but a plain cross-site form or `text/plain` POST would otherwise still use
the model, run the hooks (a `DecisionLog` row) or write training labels. Browsers send `Origin` on every `POST`, so a
browser app behind a same-origin reverse proxy still needs its origin in the list. Requests without an `Origin` header
(curl, SDKs, other servers) are not affected. A page reached through DNS rebinding sends its own origin, so its POSTs
are refused too, but it can read the `GET` routes (schemas, models, health); beyond this machine, run with
`--api-key`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/systemone` | Decide: one state, many typed questions (Jev-compatible) |
| `POST` | `/v1/systemone/batch` | Tez: many states against the same questions, in one request |
| `POST` | `/v1/plan` | Tez: what a `/v1/systemone` request would do, without calling the model |
| `GET` | `/v1/models` | List model aliases (Jev-compatible) |
| `GET` | `/healthz` | Liveness, backend and readout status |
| `GET` | `/v1/schemas` | Tez: schemas loaded from `--schemas` (and `--presets`) |
| `GET` | `/v1/schemas/{name}` | Tez: one schema (questions, probe status, calibration) |
| `POST` | `/v1/feedback` | Tez: record a correct label for a past decision (feeds `tez fit`) |
| `GET` | `/` | Tez: the server's name, version and this list of endpoints |

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

- `state`: string, object or array (objects are serialised as JSON; at most 100 levels of nested objects and arrays).
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
`escalate`, present only when a gate applies), `p_correct` and `calibration_id` when a fitted calibration exists,
`temperature` when an unfitted letters answer was read at a default temperature (below), and `layout` for every question when the request's questions were not all read in the same layout (`X-Tez-Layout: mixed`,
for example fitted and unfitted questions under `auto`); otherwise they were all read in the `X-Tez-Layout` layout.

The gate never acts without evidence: `act` always comes with `p_correct` and `calibration_id`. A question with no
usable fitted calibration, and any `__none__` answer, gets `escalate` whenever a gate is requested (in the example,
`is_urgent` and `anger` were never fitted). `tez.gate: null` or `false` turns a schema's default gate off; `tez.gate: {}`
uses the schema's default alpha. Thresholds are precomputed for alpha 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3 and
the schema default; any other alpha uses the largest precomputed value at or below it.

`tez.readout: "probe"` is strict: a request fails with 422 if a requested question has no usable probe. `auto` falls
back to letters question by question. A fit is used only while the question's prompt (instructions, options,
few-shot examples, template, layout), the model and `--n-probs` still match what `tez fit` saw; otherwise the question
falls back to uncalibrated letters and `GET /v1/schemas/{name}` and `/v1/plan` mark it `stale` with the reason (a fit
made before `n_probs` was recorded counts as the default, 200).

### Default temperature

Letter probabilities read at temperature 1 are overconfident on most tasks. For Gemma 4 12B Q8_0 with the `gemma4`
template, a letters answer that no fit calibrates is read at a default temperature per question type: yes/no 6.01,
choice with up to 10 options shown 4.71, choice with more (a tournament counts its finalists) 2.66, score 5.18. They
were fitted on 13,610 labelled decisions from 19 tasks and checked leave-one-task-out: mean expected calibration error
0.218 to 0.132, NLL 2.16 to 0.92 (`results/calibration/default_temperature.md`).

No argmax changes: the most probable option or level was the same on all 13,610 decisions, and a temperature never
reorders them. The numbers around it do move, since all of them are read from the tempered probabilities:

- a yes/no probability (`noul`) moves towards 0.5 without crossing it;
- a choice's `probabilities` flatten and its `confidence` drops;
- a score's expected level (`score` = Σ i·p_i) moves towards the middle of the scale while its most probable level
  stays the same (as the probabilities flatten; a score whose probabilities have two separate peaks can first move
  outwards).

Where you need an integer level, take the most probable level (the argmax of `probabilities`), which no temperature
changes, or run with `--default-temperature off`. The `tez` block carries `temperature` for every answer read at a default. Easy questions
read under-confident until they are fitted; the gate is unaffected, because it only acts on fitted questions. Any other
model or template is read at temperature 1.
`--default-temperature off` (env `TEZ_DEFAULT_TEMPERATURE`) restores temperature 1, and a number sets one temperature
for every unfitted question. `tez fit` centres its temperature search on the same default, which makes a handful of
labels help rather than hurt.

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
own tail; the in-process backend reads them in one batched decode instead (see "In-process backend"). Worked
examples (a schema's `examples`) come before the input in both layouts, so a question with examples
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
- `server-timing`: `total;dur=<ms>` for every response. A successful `/v1/systemone` decision adds `tez;dur=<ms>`
  (the engine) and `backend;dur=<ms>` (time spent in model calls): `tez;dur=<ms>, backend;dur=<ms>, total;dur=<ms>`.
  A successful batch adds `tez;dur=<ms>` for the whole batch, without `backend`. A decision or batch that fails has
  `total` only.

Decisions (`/v1/systemone` and `/v1/systemone/batch`) that reach the engine, successful or not, also carry
`X-Tez-Run-Id` (the same id; not on a `401`, a `403`, or a body over `--max-body-bytes` or not JSON, which are
answered before). Only a successful `/v1/systemone` response carries `X-Tez-Layout`: the layout its questions were
read in (`question_first` or `state_first`; under `auto`, a fitted question counts in its fit's layout), or `mixed`
when they differ. A decision that fails in the engine (a `tez.readout: "probe"` without a usable probe, a backend
error) has no `X-Tez-Layout`, and a batch never has one (each result's `tez.questions` names the layouts when they
differ). Browsers can read all four (CORS `Expose-Headers`).

### Errors

Same codes as Jev: `401` missing/invalid key (only with `--api-key`), `422` invalid request (including a prompt longer
than the model's context, with llama.cpp's message, and a body that is not strict JSON: `NaN`, `Infinity`, a number
too large for a float such as `1e400`, or a string with a lone surrogate such as `"\ud800"`, refused before any hook
or model call), `503` backend unavailable. Also `404` for an unknown route, or an unknown schema in the path
(`GET /v1/schemas/{name}`), `405` for a wrong method, `403` for a `POST` (or a CORS preflight) from a browser origin that is not allowed,
`413` for a request over the server's limits (see "Limits") and `500` for an unexpected failure (a hook that raised,
for example). Body: `{"error": {"type": "invalid_request", "message": "..."}}`; the types are `unauthorized`,
`forbidden`, `invalid_request`, `not_found`, `method_not_allowed`, `payload_too_large`, `backend_unavailable` and
`internal_error` (any other `4xx` the web framework answers is `invalid_request`, any other `5xx` `internal_error`).
A schema named in a request body that is not loaded (`schema` in a decision, a batch, a plan or feedback) is a field
the server cannot use, so it is `422 invalid_request` with the loaded names; only the path of
`GET /v1/schemas/{name}` gives `404 not_found`.

### Structured extraction (`json_schema`)

In place of `questions`, a request may carry `json_schema`: a JSON schema of the object you want back. Tez turns its
fields into questions, decides them like any others, and returns the object in `tez.values` (the answers are there too,
keyed by the field's dotted path).

```json
{"state": "Charged twice for invoice 4411, please fix it today!",
 "json_schema": {"type": "object", "properties": {
   "department": {"enum": ["billing", "technical", "sales"], "description": "Which team should handle it?"},
   "urgent": {"type": "boolean", "description": "Does it need a reply today?"},
   "priority": {"type": "integer", "minimum": 1, "maximum": 3},
   "customer": {"type": "object", "properties": {"tier": {"enum": [1, 2, 3]}}},
   "refund": {"anyOf": [{"enum": ["full", "partial"]}, {"type": "null"}]}}}}
```

```json
"tez": {"latency_ms": "...", "questions": {...},
        "values": {"department": "billing", "urgent": true, "priority": 3, "customer": {"tier": 1}, "refund": null}}
```

(The shape of the values; which values come back depends on the model.)

| JSON schema | Question | Value |
|---|---|---|
| `enum`, `Literal[...]`, an `Enum` class, `oneOf`/`anyOf` of `const`s | `choice`, 2 to 255 options (a const's `description` describes its option) | the enum value as given (string, number, boolean, null) |
| `"type": "boolean"` | `noul` | `true` / `false` |
| `"type": "integer"` with `minimum` and `maximum` (or the exclusive bounds), 2 to 10 values | `score` | the most likely value |
| one possible value (`const`, a one-value enum or range) | none: filled in | the value |
| an object with `properties` (a nested model) | its fields, ids `parent.child` | a nested object |
| optional (`anyOf` with `{"type": "null"}`, `"type": [..., "null"]`) | a choice or a boolean gets a `null` option ("not stated, or does not apply"); an integer scale always gets a value | `null` |
| an optional object (`Optional[Address]`) | each field inside gets a `null` option (an integer scale there becomes a choice of its values and `null`) | `null` when none of its fields has a value; an optional object with nothing to decide is refused |
| `description` | the question's instructions | |

Local `$ref`s (`#/$defs/...`) and a one-element `allOf` are followed. Refused with a 422 that names the path
(`json_schema.properties.note: a free-text string cannot be decided in one pass; ...`): free strings and numbers
without an enum, arrays, integer ranges over 10 values, more than 255 options, unions of different types, references
outside the document and recursive schemas. `json_schema` and `questions` cannot both be given; `schema` may name a
loaded schema whose identical questions then use its fits. Questions are counted while the schema is expanded, so a
schema whose `$ref`s multiply its fields is refused with 413 at the first question past `--max-questions`; whatever
that limit, its `$ref`s may copy in at most 10,000 references and 1,000,000 characters of schema (413 past either).

Python: `Schema.from_json_schema(js, name)`, `Schema.from_pydantic(Model)` (pydantic is never imported by Tez; the
model's own JSON schema is read), and `Tez.extract(state, schema_or_model, *, alpha=None, return_details=False)`, which
returns a dict or an instance of the pydantic model (a loaded schema's name works too: booleans, labels, level numbers).
When a gate applies (`alpha`, or the `gate:` of a named schema), a field it escalates raises `tez.EscalationRequired`
(hand the case to a person or a larger model), unless `return_details=True` returns an `ExtractResult` (value, values,
decisions, escalated fields, the response).
`RemoteTez.extract` does the same over HTTP.

## `POST /v1/systemone/batch`

Many states against the same questions. The body is a `/v1/systemone` request with `states` (an array) in place of
`state`; `questions` (or `schema`, or `json_schema`), `model` and `tez` apply to every state. With `json_schema`, each
result carries its own object in `tez.values`.

```json
{"model": "tez-latest", "states": ["My invoice is wrong", {"text": "the app crashes"}],
 "schema": "support-triage", "tez": {"gate": {"alpha": 0.05}}}
```

```json
{"model": "tez-0.1.0 (gemma-4-12b-q8_0, letters)",
 "results": [{"model": "...", "answers": {...}, "usage": {...}, "tez": {...}},
             {"error": {"type": "invalid_request", "message": "state must be a string, object or array"}}],
 "usage": {"input_tokens": "<sum over the decided states>", "output_tokens": 0},
 "tez": {"latency_ms": "<ms for the whole batch>", "run_id": "<32 hex characters>"}}
```

- `results` has one entry per state, in input order: the state's `/v1/systemone` response, or `{"error": {"type",
  "message"}}` for a state that failed. One state's error never fails the others.
- A bad shared field (questions, schema, `tez` options) fails the whole batch with 422; too many states, questions or
  a state over the limits with 413 (see "Limits"; a single oversized state is that state's error). When the backend is
  unavailable before any state was decided the batch fails with 503; once some states are decided, the ones not yet
  tried get a `backend_unavailable` error without calling the backend again.
- Each state's questions run back to back (state first with two or more questions), so the backend's prompt cache
  keeps the state between them. States are decided one after another.
- `usage` sums the decided states. `tez.run_id` is the batch's run id (also `X-Tez-Run-Id`); state *i* is decided
  with run id `<run_id>.<i>` (what hooks and a `DecisionLog` see).

Python: `Tez.decide_many(states, questions=None, schema=None, ...)` returns the `results` list,
`Tez.decide_batch(...)` the whole response; `RemoteTez` has both over HTTP. CLI: `tez decide --states-file rows.jsonl
--out out.jsonl` (rows with a `state`, bare JSON values or text lines; one result line per row).

## `POST /v1/plan`

The body of a `/v1/systemone` request (`state` may be left out); nothing is sent to the model. Per question: the
readout it would use, its `fit`, the
number of options, the backend calls, the layout, the prompt hash (`prompt_sha`, what `tez fit` records) and a token
estimate (characters / 4), with the prefix the prompt cache would keep from the question read just before it. The
top-level `layout` is the one the questions share, or `mixed` (what `X-Tez-Layout` would say); `order` is the order
they would be read in (state-first questions first, back to back).

For the schema file below (`support-triage`, with its one worked example) and the state of the first example:

```json
{"backend": "http://127.0.0.1:8091", "template": "gemma4", "model": null, "schema": "support-triage",
 "readout": "auto", "requested_layout": "auto", "layout": "state_first", "state_tokens": 12,
 "order": ["topic", "is_urgent", "anger"],
 "questions": {"topic": {"type": "choice", "options": 4, "layout": "state_first", "readout": "letters",
                         "fit": {"status": "none", "reason": "not fitted (tez fit --schema support-triage)"},
                         "calls": {"letters": 1, "embed": 0}, "prompt_sha": "8e1d7d48b9da936b",
                         "prompt_tokens": 210, "cached_tokens": 0},
               "is_urgent": {"type": "noul", "options": 2, "layout": "state_first", "readout": "letters",
                             "fit": {"status": "none", "reason": "not fitted (tez fit --schema support-triage)"},
                             "calls": {"letters": 1, "embed": 0}, "prompt_sha": "063d9da00ff17974",
                             "prompt_tokens": 165, "cached_tokens": 69},
               "anger": {"...": "..."}},
 "totals": {"calls": {"letters": 3, "embed": 0}, "prompt_tokens": 528, "cached_tokens": 138, "evaluated_tokens": 390},
 "notes": ["state first: 3 questions read back to back, each after the first reusing the prefix it shares with the one before (cached_tokens; needs llama.cpp prompt caching on one slot, --swa-full for Gemma)",
           "model names not checked (the plan does not call the backend): a fit made on another model would not be used"]}
```

Worked examples come before the input, so here the questions share only the instructions; without examples each
question after the first reuses the whole state.

With the in-process backend a question read in a batched call has `"batch"` (`state`: the prompts that share the
state; `rest`: the others), `totals.batches` counts the batched calls, and a note describes each.

A question's `fit` has `status` (`ready`, `stale` or `none`) and `reason` (null when ready). A question with a fit
also has `calibration_id`, `fit_layout` (the layout it was fitted under), `letters_calibrated` (its letters
calibration would be used) and `probe` (its probe would be used):

```json
"fit": {"status": "ready", "reason": null, "calibration_id": "synth@2026-09-24", "fit_layout": "question_first",
        "letters_calibrated": true, "probe": true}
```

A question that would fail (`tez.readout: "probe"` without a usable probe) has `"readout": null` and an `error`
instead of failing the plan. Model names are compared only when the server already knows them (`model` is null
until a decision has asked the backend); a note says so. CLI: `tez plan "text" --schema FILE` (a table; `--json` for
this body).

## `tez doctor`

`tez doctor --backend http://127.0.0.1:8091 --template gemma4` checks a llama-server for what Tez needs and prints a
fix for anything that is off (exit code 1 when a check fails): `/health`; the model, build, context and slots from
`/props`; that the chat template matches `--template`; that `/completion` returns the top `--n-probs` log-probabilities
with the option letters among them; that a repeated prompt is served from the prompt cache and a second question about
the same state reuses it (timings `prompt_n` and `cache_n`; the fix for Gemma is `--swa-full`); whether `/embedding`
works (fitted probes need `--embeddings --pooling last`); one slot (`-np 1`); and the `--cache-ram 0` recommendation.
It sends a few tiny prompts and changes nothing on the server. `--json` prints the checks. With `--backend fake` there
is no server behind it: one `info` check says there is nothing to diagnose, and the exit code is 0.

`tez doctor --backend inproc:model.gguf --llama-lib DIR` checks an in-process model instead, loading it in the
doctor's own process: the library and its build (b11100 is the tested one), that a GPU backend loaded and holds the
layers (and the memory left free), the model, hybrid memory, the chat template, that the option letters are single
tokens holding most of the probability, that two questions about one state read in one batched decode match the
same questions read one by one, and embeddings.

## In-process backend (`inproc:`)

`--backend inproc:PATH.gguf` (Python: `Tez(backend="inproc:PATH.gguf")`, or `tez.InprocBackend(PATH, ...)` for every
setting) runs the model inside the Tez process through llama.cpp's own C library: no llama-server, no HTTP between Tez
and the model.

```
tez doctor --backend inproc:gemma-4-12b-it-Q8_0.gguf --llama-lib /opt/llama.cpp
tez serve  --backend inproc:gemma-4-12b-it-Q8_0.gguf --llama-lib /opt/llama.cpp --template gemma4 --presets
```

- **The library.** Unpack the llama.cpp release archive for your platform and GPU into a directory (Tez was tested with
  release b11100; for CUDA on Windows unpack the release's CUDA runtime (cudart) archive into the same directory) and
  point `--llama-lib` or `TEZ_LLAMA_LIB` at the directory that holds `llama.dll`, `libllama.so` or `libllama.dylib`.
  Without either, Tez looks next to `llama-server` on `PATH`. Tez binds the library with `ctypes`: nothing is compiled
  and there is no new Python dependency.
- **Settings.** `--n-ctx N` (`TEZ_N_CTX`, default 4096: the KV cache, shared by the questions of a request),
  `--n-batch N` (`TEZ_N_BATCH`, default `--n-ctx`: tokens per decode) and `--n-gpu-layers N` (`TEZ_N_GPU_LAYERS`,
  default -1, all layers; 0 runs on the CPU). If no GPU backend loads (on Windows `ggml-cuda.dll` fails with error 126
  when the CUDA runtime DLLs are not beside it, and llama.cpp would carry on on the CPU), Tez stops with the reason
  instead, unless `--n-gpu-layers 0`. `tez serve` loads the model at start-up; the Python API loads it on first use,
  and `tez plan` never loads it.
- **Readouts.** Letters are the log-probabilities of the option letters over the whole vocabulary (nothing is floored,
  so `--n-probs` does not apply; the normaliser sums the tokens within 25 nats of the top one, which moves it by less
  than 4e-6 nats for a 262,144-token vocabulary). Probe features are the last token's hidden state after the output norm,
  what `llama-server --embeddings --pooling last` returns, and the model name is derived the same way as over HTTP
  (`gemma-4-12b-q8_0`): fits made over HTTP apply in process, and the other way round.
- **Many questions in one pass.** A request with two or more questions is read in one batched call: the prompts'
  common prefix (the instructions and the state, state first) is evaluated once and copied to one sequence per
  question (`llama_memory_seq_cp`; the unified KV cache shares the cells rather than copying them), and every
  question's suffix goes into one `llama_decode`, with an output at each suffix end. Questions read question first, or
  with worked examples before the state, form a second batch that shares only the instructions; a question with more
  than 26 options runs its tournament on its own. Each state of `/v1/systemone/batch` is read the same way. Answers,
  temperatures and gate decisions are computed exactly as when questions are read one at a time. `tez plan` marks the
  batched reads (`batch` per question, `totals.batches`); hooks see the batched calls in `ctx.batches`
  ([`HOOKS.md`](HOOKS.md)).
- **One process per GPU.** The model and its KV cache live in the Tez process. A llama-server or a second Tez process
  holding the same model needs its memory again (the production llama-server with Gemma 4 12B Q8_0 held 14.2 GB of the
  laptop's 16 GB; loaded in process, the 12B left 1.1 GB free), so run one Tez process per GPU and stop a llama-server
  serving the same model first. Calls take turns on one context under a lock, as they
  do on llama-server's single slot.
- **Hybrid models** (Qwen3.5) cannot roll back their recurrent state. In process a shared prefix is reused only through
  sequence copies, or when a prompt extends the previous one, so `--no-cache-prompt` is not needed.

When to use which:

| | In process (`inproc:`) | llama-server (`http://...`) |
|---|---|---|
| Many questions about one state | one batched decode for all of them (below) | question after question, each reusing the state from the prompt cache |
| One question | no HTTP hop to the model and no top-200 list: a little faster | the voice loop's measured baseline |
| Memory | the model in the Tez process: one Tez process per GPU; restarting Tez reloads it (47 s for the 12B in the `tez doctor` run, `results/speed/inproc_runtime_doctor.txt`) | the model in llama-server: several Tez processes or clients share it, and Tez restarts without reloading it |
| Setup | a llama.cpp release directory | a running llama-server; also the Docker images (`compose.yaml`), Ollama's GGUFs or a GPU on another machine |

Measured with `tez serve` on the laptop's RTX 5080 (16 GB), Gemma 4 12B Q8_0, llama.cpp b11100: Laya's protocol with
distinct questions (a new ticket per call, 3-option choice and yes/no questions alternating; the speed study's pacing),
p50 / p95 per call in ms, from `experiments/inproc_runtime_bench.py` (`results/speed/inproc_runtime_*laya.json`,
`results/speed/http_runtime_laya.json`):

| `tez serve` backend | 1 question | 5 | 10 | 50 | ms per question at 50 | CPU load from other processes |
|---|---:|---:|---:|---:|---:|---|
| in process, run 1 | 80 / 88 | 182 / 191 | 347 / 436 | 1,652 / 1,805 | 33.0 | not recorded |
| in process, run 2 | 87 / 139 | 231 / 316 | 381 / 675 | 1,807 / 1,905 | 36.1 | not recorded |
| in process, run 3 | 81 / 114 | 180 / 288 | 318 / 770 | 1,832 / 2,912 | 36.6 | 88-100 % |
| llama-server (the production settings above, `-np 1`), state first | 155 / 215 | 627 / 1,132 | 2,234 / 3,373 | 10,832 / 13,593 | 216.7 | 100 % |

Other sessions kept the laptop's CPU at 80-100 % during most of these runs, and llama.cpp's threads meet at barriers
even with every layer on the GPU, so they are slower and noisier than an idle machine gives: in the speed study, on an
idle CPU, the same llama-server answered state-first questions sent straight to `/completion` in 752 / 4,255 ms at 10 /
50 questions, and its in-process batched arm took 1,357 ms at 50 (`results/speed/tables.md`); within this session the
engine called directly and through `tez serve` took the same time (medians 1,821 and 1,617 ms at 10 questions over
llama-server, interleaved). Measured instead in one process on one loaded model, interleaved call by call so the load
hits every arm alike, the runtime adds 1-3 % to the study's batched arm at 50 questions: 1,436 ms p50 for the study's
logic, 1,443-1,486 for the backend's batched read, 1,451-1,470 for the whole engine (`experiments/inproc_ab.py`,
`results/speed/inproc_runtime_ab.json`; the same letters).

The answers match: on the typed-decisions test split (400 rows x 5 questions, one call per row, state first) the
in-process runtime scored 0.702 (choice 0.658, yes/no 0.823, score 0.644) against 0.7015 for llama-server state first,
and gave the same decision on 99.3 % of the 2,000 (McNemar p = 1.0; `results/speed/inproc_runtime_td.json`). The
letters differ by 0.12 nats at the median (0.67 at p95, for letters within 10 nats of the top one): llama.cpp's
numbers depend a little on how a prompt's tokens are split into decodes, which is also why a prompt read from
llama-server's prompt cache differs slightly from the same prompt read fresh. Against a llama-server serving the same
Qwen3.5-4B Q8_0 GGUF the in-process letters gave the same answer on 18 of 18 prompts, letter probabilities within 0.031
(`experiments/inproc_parity.py`, `results/speed/inproc_parity_qwen35_4b.json`).

## `GET /v1/models`

```json
{"models": [{"name": "tez-latest", "description": "Tez local decision engine (gemma-4-12b-q8_0, letters)", "release_date": "2026-09-24"},
            {"name": "jev-latest", "description": "Alias of tez-latest: Tez local decision engine (gemma-4-12b-q8_0, letters)",
             "release_date": "2026-09-24"}]}
```

`jev-latest` is listed so that clients written for TypeSafe's SDK, which look a model up by that name, find it. Every
alias, and any other `model` string, is decided by the same engine.

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
              "calibration_id": "support-triage@2026-09-24", "probes": ["topic"], "builtin": false}]}
```

`builtin` is true for a preset loaded with `tez serve --presets` (see "Presets").

One schema returns the schema in wire form (`name`, `description`, `state`, `questions`, `gate`, `examples`) plus
`builtin` (as in the list), `layout` (the schema's own, or null), `served_layout` (what a request that names none
gets), `calibration_id`, `probes` (per question: `probe` = `ready`, `stale` or `none`, `letters_calibrated`,
`n_labels`, `note`, and the `layout` it was fitted under), `calibration` and `manifest`. An unknown name is
`404 not_found`.

## `POST /v1/feedback`

```json
{"schema": "support-triage", "question": "topic", "state": "...", "label": "billing", "run_id": "5f0c..."}
```

Appends to `<data-dir>/feedback/<schema>.jsonl` (default data dir: the schema file's directory + `/.tez`; for a preset,
the `--schemas` directory's `.tez`, else `./.tez`) and returns `{"ok": true, "schema": ..., "question": ..., "label":
...}`. `tez fit` trains from these rows plus any labelled file; feedback recorded for a preset is found once the preset
is copied into the schema directory to be fitted.
`run_id` (optional) is the decision's `X-Tez-Run-Id`; it is stored with the row. `on_feedback` hooks see the row first
(`Redact` removes personal data from its state; docs/HOOKS.md).

## Hooks

Hooks run in the engine around every decision (docs/HOOKS.md): `tez serve --hook package.module:object` (repeatable),
`--decision-log PATH` (a JSONL row per decision, the format `tez fit` reads once labels are filled in) and
`--hook-errors raise|log` (whether a failing hook fails the request).

## MCP server

`tez-mcp` (install the `mcp` extra: `pip install "tez-decisions[mcp]"`) serves Tez to agents over the Model Context
Protocol on stdio. Tools:

| Tool | Does |
|---|---|
| `tez_decide(state, questions=None, schema=None, readout="auto", abstain=False, alpha=None)` | a `/v1/systemone` decision; returns the response |
| `tez_schemas()` | the loaded schemas (`GET /v1/schemas`) |
| `tez_schema(name)` | one schema's questions and fit status (the detail without calibration tables) |
| `tez_feedback(schema, question, state, label)` | records a correct label (`POST /v1/feedback`) |
| `tez_status()` | version, in-process or remote, backend health, schema names |

The tool descriptions tell the agent that a gate decision of `escalate` means: do not act on that answer yourself, hand
the case to a person or to a larger model. Configuration is by environment: `TEZ_URL` forwards every call to a running
`tez serve` (with `TEZ_API_KEY` as the bearer token and `TEZ_TIMEOUT` seconds per request, default 120); otherwise
Tez runs in the MCP process from `TEZ_BACKEND`, `TEZ_TEMPLATE`, `TEZ_SCHEMAS`, `TEZ_DATA_DIR`, `TEZ_LAYOUT`,
`TEZ_DEFAULT_TEMPERATURE` and `TEZ_PRESETS=1` (each also as `VAR_FILE`), with `tez serve`'s request limits. For example, in an
MCP client's configuration:

```json
{"mcpServers": {"tez": {"command": "tez-mcp",
                        "env": {"TEZ_BACKEND": "http://127.0.0.1:8091", "TEZ_TEMPLATE": "gemma4", "TEZ_PRESETS": "1"}}}}
```

## Limits

`tez serve` refuses oversized requests with `413` and `{"error": {"type": "payload_too_large", "message": "..."}}`:

| Flag | Default | Limit |
|---|---|---|
| `--max-body-bytes` | 2 MiB (2,097,152) | request body; a `Content-Length` over it is refused before reading, a chunked body as soon as it passes it |
| `--max-questions` | 64 | questions per request (and per batch); a `json_schema` is counted as it is expanded |
| `--max-state-chars` | 50,000 | characters per state (an object or array counts as its JSON) |
| `--max-batch` | 64 | states per `/v1/systemone/batch` request |

`0` switches a limit off (`None` too, in Python). The engine's Python API (`handle`, `execute`, `handle_batch`,
`plan`) applies none unless it is given `limits=tez.config.Limits(...)`. `create_app` applies these defaults unless it
is given other limits (`Limits.unlimited()` for none), and so does the MCP server.

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
the embedding backend's features), `calibration.json` (temperature per question, conformal thresholds per alpha,
the layout it was fitted under, the letters' `n_probs`), and `manifest.json` (backend, template, `n_probs`, layer/cut,
label counts, date).

### Presets

The ten worked use cases of `examples/usecases/` ship inside the package as ready-made schemas: `agent-trace-review`,
`intent-router`, `invoice-routing`, `multilingual-intent`, `passage-check`, `prompt-injection-guard`,
`security-triage`, `support-triage`, `topic-news` and `voice-commands`.

- `tez serve --presets` loads all of them next to `--schemas` (a schema of your own with the same name wins); they are
  listed with `"builtin": true`.
- `tez decide "My card was declined" --preset intent-router` decides with one; `tez presets` lists them and
  `tez presets --show NAME` prints one.
- Python: `tez.presets.names()`, `tez.presets.load(name)` (a `Schema`), `Tez.add_presets()`.

A preset is zero-shot: it never loads fitted artefacts and `tez fit` refuses it. To fit one, copy it into your schema
directory first (`tez presets --show support-triage > schemas/support-triage.yaml`).

## Readouts

| Readout | Needs | Backend call | Measured (typed-decisions) |
|---|---|---|---|
| `letters` | nothing (zero-shot) | `/completion`, one token, top-`n_probs` log-probs (200); in process, the letter tokens' logits | Gemma 4 12B Q8: 0.704 |
| `probe` | ≥ 25–50 labels per question (`tez fit`) | `/embedding` (last-token state); in process, the same state | Qwen3.5-4B cut to 24 blocks: 0.793 |
| blend | a few labels | both | 12B letters + 50 typical labels per question: 0.766 |

Letters are answered with a single option letter (layouts above); more than 26 options run as a chunked tournament. A
letter outside the top `n_probs` next-token log-probabilities gets a floor (the smallest listed log-probability
minus 2); `--n-probs` changes how many are asked for (default 200, the measured setting), and a letters calibration
fitted at another `n_probs` is not used (`stale`). Hybrid Qwen3.5 GGUFs need
prompt caching off in llama.cpp b11100 (`tez serve` turns it off automatically when the model name matches Qwen3.5;
`--no-cache-prompt` forces it).

## Server defaults

`tez serve` listens on `127.0.0.1:8787`. The backend defaults to `http://127.0.0.1:8080` (llama-server's own default);
`--backend inproc:PATH.gguf` runs the model in process (see "In-process backend"); `--backend fake` runs an offline
demo backend whose answers mean nothing. `tez fit` needs at least 20 labels per
question to train a probe; with fewer it calibrates the letters only and says so.

The CLI reads its settings from the environment too (a flag wins): `TEZ_BACKEND`, `TEZ_TEMPLATE`, `TEZ_EMBED_BACKEND`
and `TEZ_DEFAULT_TEMPERATURE` for the commands that talk to a model (`TEZ_DATA_DIR` for `tez fit` too), with
`TEZ_LLAMA_LIB`, `TEZ_N_CTX`, `TEZ_N_BATCH` and `TEZ_N_GPU_LAYERS` for an in-process model; for `tez serve` also `TEZ_HOST`, `TEZ_PORT`, `TEZ_SCHEMAS`, `TEZ_DATA_DIR`,
`TEZ_API_KEY`, `TEZ_LOG_LEVEL`, `TEZ_CORS_ORIGINS`, `TEZ_PRESETS` (`1` = `--presets`) and `TEZ_LAYOUT`. Each can be read
from a file instead, for Docker and Compose secrets: `TEZ_API_KEY_FILE=/run/secrets/tez_api_key` (surrounding
whitespace and a UTF-8 byte order mark stripped; setting both `X` and `X_FILE` is an error).
