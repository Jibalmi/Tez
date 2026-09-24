# tez-client

A dependency-free TypeScript client for [Tez](https://github.com/Jibalmi/Tez)'s decision API: the `/v1/systemone`
wire format (TypeSafe's Jev format, plus Tez's optional extensions) and Tez's own endpoints (batches, plans, schemas,
feedback). It works in Node 18+, browsers, Deno and Bun, ships ES module and CommonJS builds with type declarations,
and types every answer by its question.

## Install

`tez-client` is not published to npm. Install it from a Tez checkout; `dist/` is not in the repository, so build it
first:

```
git clone https://github.com/Jibalmi/Tez
npm --prefix Tez/clients/ts ci                 # the TypeScript compiler, the only dev dependency
npm --prefix Tez/clients/ts run build          # dist/esm and dist/cjs, with declarations
```

Then, in your project:

```
npm install /path/to/Tez/clients/ts            # a link to the checkout: rebuild there after pulling
```

For a copy that does not depend on the checkout staying where it is, pack it and install the tarball:
`cd Tez/clients/ts && npm pack` writes `tez-client-0.1.0.tgz` (building first), then
`npm install /path/to/tez-client-0.1.0.tgz`.

Start a server (`tez serve`, or `docker compose up` in the repository; see its README), which listens on
`http://127.0.0.1:8787`. `tez serve --backend fake` runs an offline stand-in whose answers mean nothing, handy for
trying the client.

## Decide

```ts
import { TezClient } from "tez-client";

const tez = new TezClient();                         // { baseUrl: "http://127.0.0.1:8787", timeoutMs: 30000 }

const res = await tez.decide("Help! My payouts have been failing for 3 days.", {
  questions: {
    is_urgent: { type: "noul", instructions: "Does this convey urgency?" },
    topic: {
      type: "choice",
      instructions: "What is the message about?",
      criteria: { billing: "Payments, payouts, invoices", technical: "Something is broken", sales: null },
    },
    anger: { type: "score", instructions: "How upset is the writer?", criteria: ["Calm", "Frustrated", "Very angry"] },
  },
});

res.answers.is_urgent.noul;          // P(yes), a number
res.answers.topic.choice;            // typed "billing" | "technical" | "sales" | "__none__"
res.answers.topic.probabilities;     // { billing: 0.91, technical: 0.08, sales: 0.01 }
res.answers.topic.confidence;        // (k * p_max - 1) / (k - 1)
res.answers.anger.score;             // expected level, with legend and probabilities keyed "0", "1", "2"
res.tez?.questions.topic;            // Tez's block: { readout, decision?, p_correct?, calibration_id?, temperature?, layout? }
res.meta.runId;                      // the X-Tez-Run-Id header (see "Response headers")
```

The three answer shapes are exactly Jev's (`noul`, `choice`, `score`); `res.tez` is Tez's extension and is absent when
the server only speaks Jev's format. In the `tez` block, per question: `readout` (`letters` or `probe`), `decision`
(`act` or `escalate`) when a gate applies, `p_correct` and `calibration_id` when a fit calibrated the answer,
`temperature` when an unfitted letters answer was read at a default temperature, and `layout` when the questions were
not all read in the same layout.

### Schemas, the gate, abstain, the layout

```ts
// A schema loaded on the server (tez serve --schemas dir/, or --presets): its questions, fitted probes and calibration.
const triage = await tez.decide({ text: "I can't log in since this morning", customer: "Ana" }, {
  schema: "support-triage",
  alpha: 0.05,                       // gate: act only where the fitted error rate among acted decisions is <= 5 %
  abstain: true,                     // add "__none__" ("none of these fits") to every choice question
  layout: "state_first",             // the prompt layout: "auto", "question_first" or "state_first"
});
for (const [id, meta] of Object.entries(triage.tez?.questions ?? {})) {
  if (meta.decision === "escalate") console.log(`${id}: send to a person`);
}
```

Options of `decide(state, options)`: `questions`, `jsonSchema`, `schema`, `readout` (`"auto"`, `"letters"`,
`"probe"`), `abstain`, `alpha`, `gate` (`{}` for the schema's default alpha, `null` or `false` to turn a schema's
default gate off), `layout` (default: the schema's, else the server's `--layout`), `model` and `signal`. Tez's
extensions are sent only when set, so a Jev-only server receives a plain request.

## Structured extraction

Pass a JSON schema of the object you want back, and get the object:

```ts
const ticket = {
  type: "object",
  properties: {
    department: { enum: ["billing", "technical", "sales"], description: "Which team should handle it?" },
    urgent: { type: "boolean", description: "Does it need a reply today?" },
    priority: { type: "integer", minimum: 1, maximum: 3 },
  },
};

const values = await tez.extract<{ department: string; urgent: boolean; priority: number }>(
  "Charged twice for invoice 4411, please fix it today!", ticket);
// for example { department: "billing", urgent: true, priority: 3 }; the values depend on the model
```

Enums become choice questions, booleans yes/no questions, small integer ranges scores, nested objects dotted fields
(the mapping and what is refused: docs/API.md, "Structured extraction"). `extract(state, "support-triage")` uses a
loaded schema instead (booleans, labels, level numbers). With `alpha`, a field the gate escalates makes `extract`
reject with `EscalationRequired` (`fields`, `values`, `response`): the object is not certified, so hand the case to a
person or a larger model. `returnDetails: true` resolves to `{ values, decisions, escalated, response }` instead.
`decide(state, { jsonSchema })` sends the same request and returns the whole response, the object in `tez.values`.

## Many states at once

```ts
const batch = await tez.decideBatch(["My invoice is wrong", { text: "the app crashes" }], {
  schema: "support-triage",
  alpha: 0.05,
});
for (const [i, result] of batch.results.entries()) {
  if ("error" in result) console.log(i, result.error.type, result.error.message);   // one state failed, not the batch
  else console.log(i, result.answers, `run id ${batch.tez.run_id}.${i}`);
}

const results = await tez.decideMany(states, { questions });   // just the results, one per state, in order
```

`decideBatch` takes the same options as `decide` and posts to `/v1/systemone/batch`; the server reads each state's
questions back to back, so its prompt cache keeps the state. A bad shared option, or more states than the server's
`--max-batch`, fails the whole batch with a `TezError`.

## Plan a request

```ts
import { requestBody } from "tez-client";

const plan = await tez.plan(requestBody(undefined, { schema: "support-triage" }));   // the state is optional
plan.layout;                          // "question_first" | "state_first" | "mixed"
plan.questions["topic"]?.fit;         // { status: "ready" | "stale" | "none", reason, ... }
plan.totals;                          // backend calls, prompt, cached and evaluated tokens
```

`plan` posts a `/v1/systemone` body to `/v1/plan`, which says what the request would do without calling the model.
`requestBody(state, options)` builds the body `decide` would send.

## The other endpoints

```ts
await tez.health();                  // GET /healthz: { status: "ok" | "degraded", backend, model, schemas, probes, layout, ... }
await tez.models();                  // GET /v1/models
await tez.schemas();                 // GET /v1/schemas; builtin: true marks a preset
await tez.schema("support-triage");  // GET /v1/schemas/{name}: questions, layout, served_layout, probe status, calibration
await tez.systemone({ state: "...", questions: { /* ... */ } });   // POST a raw wire-format body
await tez.systemoneBatch({ states: ["..."], schema: "support-triage" });

// Record the correct label for a past decision (tez fit learns from these rows), linked to it by its run id.
const res = await tez.decide("Need it by 5pm", { schema: "support-triage" });
await tez.feedback({ schema: "support-triage", question: "urgent", state: "Need it by 5pm", label: true,
  run_id: res.meta.runId ?? undefined });
```

Every method resolves to the response body exactly as the wire format defines it.

## Response headers

Every result has a `meta` property with the response's status and headers, typed:

```ts
res.meta.status;       // 200
res.meta.requestId;    // x-typesafe-request-id: 32 hex characters (a decision's run id; a fresh id elsewhere)
res.meta.runId;        // X-Tez-Run-Id: decisions and batches that reached the engine; null otherwise
res.meta.layout;       // X-Tez-Layout: "question_first" | "state_first" | "mixed" for a single decision; else null
res.meta.timing;       // server-timing in ms: { tez, backend, total }
res.meta.headers;      // all of them: res.meta.headers.get("x-something")
```

`meta` is not one of the body's own keys: `Object.keys(res)`, `JSON.stringify(res)` and `{ ...res }` see the wire
body only. The return types are the body types with `meta` added, so they are still assignable to `DecideResponse`
and the rest. A `TezError` has the same `meta` when a response arrived, so a failed decision's run id is readable.
`parseServerTiming(header)` is exported too. In a browser, `tez serve` exposes these headers through CORS.

## Errors, timeouts and cancellation

```ts
import { TezError } from "tez-client";

try {
  await tez.decide("...", { schema: "support-triage" });
} catch (err) {
  if (err instanceof TezError) {
    err.status;    // HTTP status; 0 when no response arrived
    err.type;      // "invalid_request" | "unauthorized" | "forbidden" | "not_found" | "payload_too_large"
                   // | "backend_unavailable" | "internal_error" | ... (docs/API.md), or the client's own:
                   // "timeout" | "network_error" | "redirect" | "invalid_response" | "http_error"
    err.message;   // the server's message
    err.meta?.runId;
  }
}
```

- Error bodies `{"error": {"type", "message"}}` become `TezError`; other error bodies (a proxy's HTML page) are mapped
  by status: 401 `unauthorized`, 403 `forbidden`, 404 `not_found`, 405 `method_not_allowed`, 413
  `payload_too_large`, 422 `invalid_request`, 500 `internal_error`, and 502, 503 and 504 `backend_unavailable`.
- `forbidden` (403): a browser page on an origin `tez serve --cors-origins` does not allow. `payload_too_large` (413):
  a request over the server's limits (body size, questions, state length, states per batch).
- `timeoutMs` (default 30000) covers the whole request; when it runs out the request is aborted with a `timeout` error.
- Pass `signal` to cancel: the call then rejects with the signal's reason (an `AbortError`), not a `TezError`.
- Redirects are never followed (a `redirect` error instead), so a state or an API key never goes to a host you did not
  name. Point `baseUrl` at the server itself.
- `EscalationRequired` (from `extract` with `alpha`) is not a `TezError`: the request succeeded.

## Options

```ts
new TezClient({
  baseUrl: "https://tez.internal.example",   // a bare host:port gets http://
  apiKey: process.env.TEZ_API_KEY,           // sent as Authorization: Bearer <key> (servers started with --api-key)
  timeoutMs: 10_000,
  fetch: myFetch,                            // any fetch-compatible function (undici with a proxy agent, a test double)
});
```

In a browser, a public page may call a server on the same machine: `tez serve` answers CORS and Private Network Access
preflights for the origins it allows. Do not ship an API key in browser code.

## Types

Everything is exported: `Question`, `NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion`, `Answer`, `NoulAnswer`,
`ChoiceAnswer<L>`, `ScoreAnswer`, `DecideRequest`, `DecideResponse<A>`, `TezBlock`, `TezQuestionMeta`, `TezOptions`,
`Layout`, `LayoutOption`, `ReadLayout`, `JsonSchema`, `ExtractedObject`, `BatchRequest`, `BatchResponse<A>`,
`BatchResult<A>`, `BatchItemError`, `PlanRequest`, `PlanResponse`, `PlanQuestion`, `HealthResponse`, `SchemaDetail`,
`FeedbackRequest`, `ResponseMeta`, `WithMeta<T>`, `TezErrorType` and the rest. `decide`, `decideBatch` and
`decideMany` infer the answer type of each question from the `questions` you pass; with only a `schema` or a
`jsonSchema`, answers are the `Answer` union.

The package has ES module (`import`) and CommonJS (`require`) builds; they are separate copies, so check errors with
`instanceof TezError` from the same module system that made the call.

## Development

```
npm ci
npm run build          # dist/esm and dist/cjs, with declarations
npm test               # type checks, node:test against a stub wire-format server, and end to end against
                       # python -m tez serve --backend fake from the repository root
```

The end-to-end test starts `tez serve --backend fake` on a free port and stops it afterwards; it skips itself when no
Python can import `tez` with its server dependencies (`pip install -e .` at the repository root; `TEZ_PYTHON` picks the
interpreter).

MIT licence.
