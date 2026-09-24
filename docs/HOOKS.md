# Hooks

Hooks observe or shape every decision without forking Tez: log decisions for review, redact personal data before the
model reads it, answer repeats from a cache, count, trace. They run in-process around the engine, so they apply to the
Python API, the CLI, `tez serve`, batches and the MCP server alike.

```python
from tez import Tez, DecisionLog, Redact, Cache, Metrics

metrics = Metrics()
tez = Tez(backend="http://127.0.0.1:8091", schemas="schemas/",
          hooks=[Redact(), Cache(maxsize=4096), DecisionLog("decisions.jsonl", sample=0.1), metrics])
tez.decide("Refund to ana@example.com please", schema="support-triage")
metrics.snapshot()
```

```
tez serve --schemas schemas/ --decision-log decisions.jsonl --hook mypkg.hooks:AuditHook --hook mypkg.hooks:make_tracer
```

## Events

A hook is any object with one or more of these methods; subclass `tez.BaseHook` to override only some.

| Event | When | May |
|---|---|---|
| `on_decide_start(ctx)` | the request is valid, before any backend call | replace `ctx.state`; answer with `ctx.skip(response)` |
| `on_question_end(ctx, qid, answer, meta)` | after each question, in the order questions are read (state-first questions first) | read the answer and its `tez` block |
| `on_decide_end(ctx)` | the response is built (`ctx.response`), also after a skip | edit `ctx.response` |
| `on_error(ctx)` | the decision failed (`ctx.error`), including a request that could not be parsed | observe only |
| `on_feedback(row)` | a `POST /v1/feedback` row is about to be written | edit the row in place (Redact does) |

`ctx.skip(response)` in `on_decide_start` makes `response` the decision's answer without calling the backend; its
`tez.latency_ms` is set to the time actually spent. Every `on_decide_start` hook still runs (`ctx.skipped` is then
true), no `on_question_end` fires, and `on_decide_end` runs as usual.

## The context

`tez.DecisionContext` is shared by the hooks of one decision:

| Field | |
|---|---|
| `run_id` | 32 hex characters; the `X-Tez-Run-Id` and `x-typesafe-request-id` headers of the HTTP response. A batch item's is `<batch run id>.<index>`, with `parent_run_id` and `index` set |
| `request` | the wire-format request body |
| `engine` | the `tez.Tez` deciding it |
| `state`, `questions`, `schema` | the parsed state, `{id: tez.schema.Question}` and the named `Schema` (or None) |
| `readout`, `abstain`, `alpha`, `model` | the request's options (alpha after the schema's default gate) |
| `requested_layout`, `layout` | the layout asked for (`auto`, ...) and the one the questions were read in (`question_first`, `state_first`, or `mixed` when they differ; the `X-Tez-Layout` header) |
| `response`, `usage`, `latency_ms` | set when the decision is done |
| `traces` | per question: `readout`, `layout`, `tokens`, `ms` and `calls`, one record per backend call with its `kind` (`letters` or `embed`), `tokens`, wall `ms` and llama.cpp's own `timings` (`prompt_n`, `cache_n`, `prompt_ms`, ...) when the server sends them |
| `error` | the exception, in `on_error` |
| `skipped` | true after `ctx.skip()` |
| `data` | a dict the hooks of this decision can share |
| `hook_errors` | `(hook, event, exception)` for every hook failure swallowed under `hooks_raise=False` |

After a request that could not be parsed only `run_id`, `request` and `error` are set; `on_error` hooks must allow for
that. A batch refused as a whole (no `states`, a bad shared field, over `--max-batch`) reaches `on_error` once, with
the batch's run id and no `index`; a state that fails inside a batch reaches it with its own `<run id>.<index>`.

Hooks see requests that reached the engine. The server answers some before that, without any hook: a body that is not
JSON, is nested too deeply or is over `--max-body-bytes` (`422`, `413`), a missing API key (`401`), a `POST` from a
browser origin that is not allowed (`403`), a wrong route or method (`404`, `405`). These responses still carry
`x-typesafe-request-id`.

## Order

For one call: the process-wide hooks (`tez.set_default_hooks([...])`, read at call time), then the engine's
(`Tez(hooks=[...])`, `tez.add_hook(h)`, `tez.remove_hook(h)`), then the call's own (`decide(..., hooks=[...])`,
`handle(body, hooks=...)`, `decide_many(..., hooks=...)`). Within each list, the order given. Put `Redact` first so
every later hook, the cache key and the model see the redacted state (`ctx.request` keeps the body as it arrived, so a
hook that records anything should read `ctx.state`, not `ctx.request["state"]`).

`tez serve --hook package.module:object` (repeatable) loads a hook by name: a class is instantiated without arguments,
a hook instance is used as it is, any other callable is called and must return a hook. `--decision-log PATH` adds a
`DecisionLog`. Hooks given on the command line run in the order given, the decision log last.

## Failure rules

`Tez(hooks_raise=True)` is the default (`tez serve --hook-errors raise`):

- An exception from `on_decide_start`, `on_question_end` or `on_decide_end` fails the decision. The caller gets that
  exception; `on_error` hooks run first with it in `ctx.error`. Over HTTP a `TezError` keeps its status and type (a
  hook can refuse a request with `InvalidRequest`, 422); anything else is a `500 internal_error`.
- An exception from `on_feedback` fails the feedback: nothing is written.

`Tez(hooks_raise=False)` (`--hook-errors log`) is for hooks that must never fail a request, such as telemetry:

- An exception from any hook is logged (logger `tez`, warning), recorded in `ctx.hook_errors`, and skipped; the
  decision and the other hooks go on. A hook cannot refuse a request in this mode.

In both modes an exception raised by an `on_error` hook is logged and skipped, so it never replaces the original
error.

Hooks run on the server's worker threads, several decisions at a time: the built-in hooks lock their own state, and
yours should too.

## Built-in hooks

- **`DecisionLog(path, sample=1.0, seed=None)`** appends one JSON line per decision (a random `sample` share of
  them): `{"state", "labels": {}, "predicted": {question: label}, "decisions": {question: {confidence, readout,
  decision, p_correct, calibration_id}}, "schema", "run_id", "ts", "model", "cached"}`. It is the row format `tez fit`
  and `tez eval` read. `labels` is empty on purpose: review the rows, fill in the correct labels (the model's are in
  `predicted`) and pass the file to `tez fit --labels`; rows left unlabelled are ignored. Fitting the model's own
  answers would measure the model against itself, and the gate's guarantee would rest on nothing. `tez serve
  --decision-log PATH` installs one.
- **`Redact(patterns=None, replacement=None)`** replaces personal data in the state before the model reads it, and in
  feedback rows before they are written. Built-in patterns: `email`, `iban`, `card` (13 to 19 digits passing the Luhn
  check) and `phone` (9 to 15 digits), all four by default; add your own regular expressions. Built-ins become
  `[EMAIL]`, `[IBAN]`, `[CARD]`, `[PHONE]`, your patterns `[REDACTED]`, or one fixed `replacement`. Strings inside
  object and array states are redacted, keys kept. Decisions are then made on the redacted text.
- **`Cache(maxsize=1024)`** answers a repeated decision from memory (least recently used). The key covers the state,
  every question's definition, the schema and its calibration id, readout, abstain, the gate's alpha, the requested
  layout, the model alias and the backend, so a new fit or another model never serves an old answer. A cached
  response has `tez.cached: true` and zero usage. `stats()`, `clear()`.
- **`Metrics(window=10000)`** counts decisions, errors by type, cached decisions, questions by readout, layout and gate
  decision, input tokens and feedback rows, with latency percentiles over the last `window` decisions:
  `snapshot()` (a dict) and `prometheus()` (Prometheus text format).
- **`OTelHook(tracer=None)`** opens one OpenTelemetry span per decision, `tez.decide`, with attributes `tez.run_id`,
  `tez.questions`, `tez.schema`, `tez.readout`, `tez.layout`, `tez.latency_ms`, `tez.input_tokens`, `tez.model` and an
  event per question; failures are recorded on the span. The state is never recorded. Needs the `otel` extra
  (`pip install "tez-decisions[otel]"`) unless you pass your own tracer.

## Writing one

```python
from tez import BaseHook, InvalidRequest

class RejectLongStates(BaseHook):
    def on_decide_start(self, ctx):
        if len(str(ctx.state)) > 20_000:
            raise InvalidRequest("state too long for this deployment")      # 422 with hooks_raise on

class SlowQuestions(BaseHook):
    def on_question_end(self, ctx, qid, answer, meta):
        trace = ctx.traces[qid]
        if trace["ms"] > 500:
            print(ctx.run_id, qid, trace["layout"], [c.get("timings") for c in trace["calls"]])
```
