# tez-client

A dependency-free TypeScript client for [Tez](https://github.com/Jibalmi/Tez)'s decision API: the `/v1/systemone`
wire format (TypeSafe's Jev format, plus Tez's optional extensions). It works in Node 18+, browsers, Deno and Bun, ships
ES module and CommonJS builds with type declarations, and types every answer by its question.

```
npm install tez-client                 # once published; until then: npm install ./clients/ts from a Tez checkout
```

Start a server first (`tez serve`, or `docker compose up` in the repository; see its README), which listens on
`http://127.0.0.1:8787`.

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
res.tez?.questions.topic;            // Tez's block: { readout: "letters" | "probe", decision?, p_correct?, calibration_id? }
```

The three answer shapes are exactly Jev's (`noul`, `choice`, `score`); `res.tez` is Tez's extension and is absent when
the server only speaks Jev's format.

### Schemas, the gate, abstain

```ts
// A schema loaded on the server (tez serve --schemas dir/): its questions, fitted probes and calibration.
const triage = await tez.decide({ text: "I can't log in since this morning", customer: "Ana" }, {
  schema: "support-triage",
  alpha: 0.05,                       // gate: act only where the fitted error rate among acted decisions is <= 5 %
  abstain: true,                     // add "__none__" ("none of these fits") to every choice question
});
for (const [id, meta] of Object.entries(triage.tez?.questions ?? {})) {
  if (meta.decision === "escalate") console.log(`${id}: send to a person`);
}
```

Options of `decide(state, options)`: `questions`, `schema`, `readout` (`"auto"`, `"letters"`, `"probe"`), `abstain`,
`alpha`, `gate` (`{}` for the schema's default alpha, `null` or `false` to turn a schema's default gate off), `model`
and `signal`. Tez's extensions are sent only when set, so a Jev-only server receives a plain request.

## The other endpoints

```ts
await tez.health();                  // GET /healthz: { status: "ok" | "degraded", backend, model, schemas, probes, ... }
await tez.models();                  // GET /v1/models
await tez.schemas();                 // GET /v1/schemas
await tez.schema("support-triage");  // GET /v1/schemas/{name}: questions, probe status, calibration
await tez.feedback({ schema: "support-triage", question: "topic", state: "...", label: "billing" });   // POST /v1/feedback
await tez.systemone({ state: "...", questions: { /* ... */ } });   // POST a raw wire-format body
```

Every method resolves to the response body exactly as the wire format defines it.

## Errors, timeouts and cancellation

```ts
import { TezError } from "tez-client";

try {
  await tez.decide("...", { schema: "support-triage" });
} catch (err) {
  if (err instanceof TezError) {
    err.status;    // HTTP status; 0 when no response arrived
    err.type;      // "invalid_request" | "unauthorized" | "not_found" | "backend_unavailable" | ... (docs/API.md)
                   // or the client's own: "timeout" | "network_error" | "redirect" | "invalid_response" | "http_error"
    err.message;   // the server's message
  }
}
```

- Error bodies `{"error": {"type", "message"}}` become `TezError`; other error bodies (a proxy's HTML page) are mapped
  by status (502, 503 and 504 are `backend_unavailable`).
- `timeoutMs` (default 30000) covers the whole request; when it runs out the request is aborted with a `timeout` error.
- Pass `signal` to cancel: the call then rejects with the signal's reason (an `AbortError`), not a `TezError`.
- Redirects are never followed (a `redirect` error instead), so a state or an API key never goes to a host you did not
  name. Point `baseUrl` at the server itself.

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
preflights by default. Do not ship an API key in browser code.

## Types

Everything is exported: `Question`, `NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion`, `Answer`, `NoulAnswer`,
`ChoiceAnswer<L>`, `ScoreAnswer`, `DecideRequest`, `DecideResponse<A>`, `TezBlock`, `TezQuestionMeta`,
`HealthResponse`, `SchemaDetail`, `FeedbackRequest`, `TezErrorType` and the rest. `decide` infers the answer type of
each question from the `questions` you pass; with only a `schema`, answers are the `Answer` union.

The package has ES module (`import`) and CommonJS (`require`) builds; they are separate copies, so check errors with
`instanceof TezError` from the same module system that made the call.

## Development

```
npm ci
npm run build          # dist/esm and dist/cjs, with declarations
npm test               # type checks, then node:test against a mock wire-format server
```

MIT licence.
