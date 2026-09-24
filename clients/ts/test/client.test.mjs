// The ES module build, imported through the package's own exports map, against the mock server.
import assert from "node:assert/strict";
import { createServer } from "node:net";
import { after, before, describe, test } from "node:test";

import {
  DEFAULT_BASE_URL,
  EscalationRequired,
  NONE_LABEL,
  parseServerTiming,
  requestBody,
  TezClient,
  TezError,
} from "tez-client";

import { SCHEMA, startMockServer } from "./mock-server.mjs";

const QUESTIONS = {
  is_urgent: { type: "noul", instructions: "Does this convey urgency?", criteria: { true: "Time-sensitive", false: null } },
  topic: { type: "choice", instructions: "What is the message about?", criteria: { billing: "Payments", technical: null, sales: null } },
  anger: { type: "score", instructions: "How upset is the writer?", criteria: ["Calm", "Frustrated", "Very angry"] },
};

async function rejects(promise) {
  try {
    await promise;
  } catch (err) {
    return err;
  }
  assert.fail("expected a rejection");
}

async function freePort() {
  const srv = createServer();
  await new Promise((resolve) => srv.listen(0, "127.0.0.1", resolve));
  const { port } = srv.address();
  await new Promise((resolve) => srv.close(resolve));
  return port;
}

describe("TezClient against a wire-format server", () => {
  let mock;
  let client;
  before(async () => {
    mock = await startMockServer();
    client = new TezClient({ baseUrl: mock.url });
  });
  after(() => mock.close());

  test("decide sends Jev's plain body and returns the three answer shapes", async () => {
    const res = await client.decide("Help! My payouts have been failing for 3 days.", { questions: QUESTIONS });
    const req = mock.requests.at(-1);
    assert.equal(req.method, "POST");
    assert.equal(req.path, "/v1/systemone");
    assert.deepEqual(req.body, { state: "Help! My payouts have been failing for 3 days.", questions: QUESTIONS });
    assert.equal(req.headers["content-type"], "application/json");
    assert.equal(req.headers.accept, "application/json");
    assert.equal(req.headers.authorization, undefined);
    assert.deepEqual(Object.keys(res), ["model", "answers", "usage", "tez"]);
    assert.deepEqual(res.answers.is_urgent, { type: "noul", noul: 0.8 });
    const topic = res.answers.topic;
    assert.deepEqual(Object.keys(topic), ["type", "choice", "probabilities", "confidence"]);
    assert.equal(topic.choice, "billing");
    assert.deepEqual(Object.keys(topic.probabilities), ["billing", "technical", "sales"]);
    const anger = res.answers.anger;
    assert.deepEqual(Object.keys(anger), ["type", "score", "legend", "probabilities", "confidence"]);
    assert.deepEqual(anger.legend, { 0: "Calm", 1: "Frustrated", 2: "Very angry" });
    assert.deepEqual(res.usage, { input_tokens: 42, output_tokens: 0 });
    assert.deepEqual(res.tez.questions.topic, { readout: "letters" });
  });

  test("Tez extensions are sent only when set", async () => {
    await client.decide({ text: "hi", customer: "Ana" }, { schema: "support-triage" });
    assert.deepEqual(mock.requests.at(-1).body, { state: { text: "hi", customer: "Ana" }, schema: "support-triage" });
    const res = await client.decide("hi", { questions: { t: QUESTIONS.topic }, readout: "letters", abstain: true,
      alpha: 0.1, gate: false, model: "tez-latest" });
    assert.deepEqual(mock.requests.at(-1).body, { state: "hi", model: "tez-latest", questions: { t: QUESTIONS.topic },
      tez: { readout: "letters", abstain: true, gate: { alpha: 0.1 } } });                 // alpha wins over gate
    assert.ok(NONE_LABEL in res.answers.t.probabilities);
    assert.deepEqual(res.tez.questions.t, { readout: "letters", decision: "escalate" });
    await client.decide("hi", { schema: "support-triage", gate: null, readout: "auto" });
    assert.deepEqual(mock.requests.at(-1).body, { state: "hi", schema: "support-triage", tez: { gate: null } });
    await client.systemone({ state: "raw", questions: { q: QUESTIONS.is_urgent } });
    assert.deepEqual(mock.requests.at(-1).body, { state: "raw", questions: { q: QUESTIONS.is_urgent } });
  });

  test("the other endpoints return their wire bodies", async () => {
    assert.equal((await client.health()).status, "ok");
    assert.equal((await client.models()).models[0].name, "tez-mock");
    assert.equal((await client.schemas()).schemas[0].name, "support-triage");
    const detail = await client.schema("support-triage");
    assert.deepEqual(detail.questions, SCHEMA.questions);
    assert.equal(detail.probes.topic.probe, "none");
    const fb = await client.feedback({ schema: "support-triage", question: "is_urgent", state: "Need it by 5pm", label: true });
    assert.deepEqual(fb, { ok: true, schema: "support-triage", question: "is_urgent", label: "true" });
    const err = await rejects(client.schema("a b/c"));
    assert.equal(mock.requests.at(-1).rawPath, "/v1/schemas/a%20b%2Fc");                 // one encoded path segment
    assert.ok(err instanceof TezError);
    assert.equal(err.status, 404);
    assert.equal(err.type, "not_found");
    assert.equal(err.message, "unknown schema 'a b/c'");
  });

  test("error bodies become TezError with status, type and message", async () => {
    const err = await rejects(client.decide("s", { questions: { q: { type: "maybe", instructions: "x" } } }));
    assert.ok(err instanceof TezError && err instanceof Error);
    assert.equal(err.name, "TezError");
    assert.equal(err.status, 422);
    assert.equal(err.type, "invalid_request");
    assert.match(err.message, /type must be one of noul, choice, score/);
    assert.deepEqual(err.body, { error: { type: "invalid_request", message: err.message } });
  });
});

describe("failures without a wire-format answer", () => {
  let mock;
  before(async () => {
    mock = await startMockServer({
      routes: {
        "GET /v1/models": ({ send }) => send(503, { error: { type: "backend_unavailable", message: "loading model" } }),
        "GET /healthz": ({ send }) => send(502, "<html><body>Bad Gateway</body></html>"),
        "GET /v1/schemas": ({ send }) => send(200, "not json"),
        "POST /v1/feedback": ({ send }) => send(500, { detail: "Internal Server Error" }),
        "POST /v1/systemone": ({ res }) => setTimeout(() => {                                   // answers too late
          res.writeHead(200, { "content-type": "application/json" });
          res.end("{}");
        }, 2000).unref(),
      },
    });
  });
  after(() => mock.close());

  test("error statuses map to types even without the error body", async () => {
    const client = new TezClient({ baseUrl: mock.url });
    let err = await rejects(client.models());
    assert.deepEqual([err.status, err.type, err.message], [503, "backend_unavailable", "loading model"]);
    err = await rejects(client.health());
    assert.deepEqual([err.status, err.type], [502, "backend_unavailable"]);
    assert.match(err.message, /Bad Gateway/);
    err = await rejects(client.schemas());
    assert.deepEqual([err.status, err.type], [200, "invalid_response"]);
    err = await rejects(client.feedback({ schema: "s", question: "q", state: "x", label: 1 }));
    assert.deepEqual([err.status, err.type, err.message], [500, "internal_error", "Internal Server Error"]);
  });

  test("timeoutMs aborts the request with a timeout TezError", async () => {
    const client = new TezClient({ baseUrl: mock.url, timeoutMs: 150 });
    const t0 = performance.now();
    const err = await rejects(client.decide("s", { questions: { q: { type: "noul", instructions: "x" } } }));
    assert.ok(performance.now() - t0 < 1500);
    assert.ok(err instanceof TezError);
    assert.deepEqual([err.status, err.type], [0, "timeout"]);
    assert.match(err.message, /did not answer within 150 ms/);
  });

  test("your own AbortSignal rejects with its reason, not a TezError", async () => {
    const client = new TezClient({ baseUrl: mock.url });
    const controller = new AbortController();
    const pending = client.decide("s", { questions: { q: { type: "noul", instructions: "x" } }, signal: controller.signal });
    setTimeout(() => controller.abort(), 50);
    const err = await rejects(pending);
    assert.equal(err.name, "AbortError");
    assert.ok(!(err instanceof TezError));
    const already = AbortSignal.abort(new Error("stop"));
    assert.equal((await rejects(client.models({ signal: already }))).message, "stop");
  });

  test("an unreachable server is a network_error", async () => {
    const client = new TezClient({ baseUrl: `127.0.0.1:${await freePort()}` });           // bare host:port gets http://
    assert.match(client.baseUrl, /^http:\/\/127\.0\.0\.1:\d+$/);
    const err = await rejects(client.health());
    assert.deepEqual([err.status, err.type], [0, "network_error"]);
    assert.match(err.message, /cannot reach http:\/\/127\.0\.0\.1:\d+\/healthz/);
  });
});

test("redirects are never followed: the state and the key stay put", async () => {
  const target = await startMockServer();
  const front = await startMockServer({
    routes: { "POST /v1/systemone": ({ send }) => send(307, "", { location: `${target.url}/v1/systemone` }) },
  });
  try {
    const client = new TezClient({ baseUrl: front.url, apiKey: "secret" });
    const err = await rejects(client.decide("private state", { questions: { q: { type: "noul", instructions: "x" } } }));
    assert.deepEqual([err.status, err.type], [307, "redirect"]);
    assert.match(err.message, /does not follow redirects/);
    assert.equal(target.requests.length, 0);
    assert.equal(front.requests[0].headers.authorization, "Bearer secret");
  } finally {
    await front.close();
    await target.close();
  }
});

test("apiKey is sent as a bearer token", async () => {
  const mock = await startMockServer({ apiKey: "s3cret" });
  try {
    const err = await rejects(new TezClient({ baseUrl: mock.url }).models());
    assert.deepEqual([err.status, err.type], [401, "unauthorized"]);
    const ok = await new TezClient({ baseUrl: `${mock.url}/`, apiKey: "s3cret" }).models();          // trailing slash
    assert.equal(ok.models.length, 1);
    assert.equal(mock.requests.at(-1).headers.authorization, "Bearer s3cret");
    assert.equal(mock.requests.at(-1).path, "/v1/models");
  } finally {
    await mock.close();
  }
});

test("a custom fetch is called as a plain function with the request", async () => {
  const calls = [];
  function fakeFetch(url, init) {
    "use strict";
    calls.push({ url, init, self: this });
    return Promise.resolve({ status: 200, statusText: "OK", headers: new Headers(),
      text: async () => JSON.stringify({ models: [] }) });
  }
  const client = new TezClient({ baseUrl: "https://tez.example/prefix", fetch: fakeFetch, timeoutMs: 1000 });
  assert.deepEqual(await client.models(), { models: [] });
  assert.equal(calls[0].url, "https://tez.example/prefix/v1/models");
  assert.equal(calls[0].self, undefined);                     // window.fetch must not be called with another `this`
  assert.equal(calls[0].init.method, "GET");
  assert.equal(calls[0].init.redirect, "manual");
  assert.ok(calls[0].init.signal instanceof AbortSignal);
  assert.equal("body" in calls[0].init, false);
});

test("constructor defaults and validation", () => {
  const client = new TezClient();
  assert.equal(client.baseUrl, DEFAULT_BASE_URL);
  assert.equal(client.timeoutMs, 30000);
  for (const baseUrl of ["ftp://host", "http://", "  "]) assert.throws(() => new TezClient({ baseUrl }), TypeError);
  for (const timeoutMs of [0, -1, Number.NaN, Infinity, "30"]) assert.throws(() => new TezClient({ timeoutMs }), TypeError);
  assert.throws(() => new TezClient({ fetch: "nope" }), TypeError);
});

const RUN_ID = /^[0-9a-f]{32}$/;
const TICKET = {
  type: "object",
  properties: {
    department: { enum: ["billing", "technical", "sales"], description: "Which team should handle it?" },
    urgent: { type: "boolean", description: "Does it need a reply today?" },
    priority: { type: "integer", minimum: 1, maximum: 3 },
    customer: { type: "object", properties: { tier: { enum: [1, 2, 3] } } },
    kind: { const: "ticket" },
  },
};

describe("response meta: status and headers on every result", () => {
  let mock;
  let client;
  before(async () => {
    mock = await startMockServer();
    client = new TezClient({ baseUrl: mock.url });
  });
  after(() => mock.close());

  test("a decision carries its run id, layout and server-timing, outside the body", async () => {
    const res = await client.decide("Help! My payouts have been failing for 3 days.", { questions: QUESTIONS });
    const { meta } = res;
    assert.equal(meta.status, 200);
    assert.match(meta.requestId, RUN_ID);
    assert.equal(meta.runId, meta.requestId);                       // a decision's request id is its run id
    assert.equal(meta.layout, "state_first");
    assert.deepEqual(Object.keys(meta.timing), ["tez", "backend", "total"]);
    assert.equal(meta.timing.tez, 1.5);
    assert.equal(typeof meta.timing.total, "number");
    assert.equal(meta.headers.get("X-Tez-Run-Id"), meta.runId);
    assert.deepEqual(Object.keys(res), ["model", "answers", "usage", "tez"]);   // not one of the body's keys
    assert.equal(JSON.stringify(res).includes("meta"), false);
    assert.equal("meta" in { ...res }, false);
    assert.throws(() => { res.meta = null; }, TypeError);                          // read-only
  });

  test("other routes have a request id but no run id or layout", async () => {
    for (const res of [await client.health(), await client.models(), await client.schemas(),
      await client.feedback({ schema: "support-triage", question: "topic", state: "x", label: "billing" })]) {
      assert.match(res.meta.requestId, RUN_ID);
      assert.equal(res.meta.runId, null);
      assert.equal(res.meta.layout, null);
      assert.deepEqual(Object.keys(res.meta.timing), ["total"]);
    }
  });

  test("errors carry meta: a decision that reached the engine has a run id", async () => {
    const err = await rejects(client.decide("s", { questions: { q: { type: "maybe", instructions: "x" } } }));
    assert.ok(err instanceof TezError);
    assert.equal(err.meta.status, 422);
    assert.match(err.meta.runId, RUN_ID);
    assert.equal(err.meta.layout, null);                             // only a decision that was read has one
    const notFound = await rejects(client.schema("nope"));
    assert.equal(notFound.meta.status, 404);
    assert.equal(notFound.meta.runId, null);
    assert.match(notFound.meta.requestId, RUN_ID);
  });

  test("a response without Tez's headers still has meta, with nulls", async () => {
    const bare = new TezClient({ baseUrl: "http://jev.example", fetch: async () => ({ status: 200, statusText: "OK",
      headers: new Headers({ "content-type": "application/json" }), text: async () => '{"models": []}' }) });
    const res = await bare.models();
    assert.deepEqual({ ...res.meta, headers: undefined },
      { status: 200, requestId: null, runId: null, layout: null, timing: {}, headers: undefined });
  });

  test("parseServerTiming reads durations, skips the rest, keeps the first of a name", () => {
    assert.deepEqual(parseServerTiming("tez;dur=2.3, backend;dur=1.0, total;dur=4.2"), { tez: 2.3, backend: 1, total: 4.2 });
    assert.deepEqual(parseServerTiming('cache;desc="hit, then; miss", db;dur=53, app;dur=47.2;desc="a"'), { db: 53, app: 47.2 });
    assert.deepEqual(parseServerTiming("total;dur=1, total;dur=9, edge;DUR=\"2e1\""), { total: 1, edge: 20 });
    assert.deepEqual(parseServerTiming(null), {});
    assert.deepEqual(parseServerTiming("garbage"), {});
  });
});

describe("the new request options and response fields", () => {
  let mock;
  let client;
  before(async () => {
    mock = await startMockServer({ defaultTemperature: 4.71 });
    client = new TezClient({ baseUrl: mock.url });
  });
  after(() => mock.close());

  test("layout is sent as tez.layout, auto included; X-Tez-Layout follows it", async () => {
    let res = await client.decide("hi", { questions: QUESTIONS, layout: "question_first" });
    assert.deepEqual(mock.requests.at(-1).body.tez, { layout: "question_first" });
    assert.equal(res.meta.layout, "question_first");
    assert.equal(res.tez.questions.topic.layout, undefined);           // all read in the X-Tez-Layout layout
    res = await client.decide("hi", { schema: "support-triage", layout: "auto", gate: false });
    assert.deepEqual(mock.requests.at(-1).body, { state: "hi", schema: "support-triage", tez: { gate: false, layout: "auto" } });
    assert.equal(res.meta.layout, "mixed");                         // the fitted question keeps its fit's layout
    assert.deepEqual(res.tez.questions.topic, { readout: "letters", p_correct: 0.7,
      calibration_id: "support-triage@2026-09-24", layout: "question_first" });
    assert.deepEqual(res.tez.questions.is_urgent, { readout: "letters", temperature: 4.71, layout: "state_first" });
  });

  test("an unfitted letters answer reports its default temperature", async () => {
    const res = await client.decide("hi", { questions: { t: QUESTIONS.topic } });
    assert.deepEqual(res.tez.questions.t, { readout: "letters", temperature: 4.71 });
    assert.equal(res.meta.layout, "question_first");                // one question under auto
  });

  test("jsonSchema is sent as json_schema and the object comes back in tez.values", async () => {
    const res = await client.decide("Charged twice for invoice 4411, please fix it today!", { jsonSchema: TICKET });
    assert.deepEqual(mock.requests.at(-1).body, { state: "Charged twice for invoice 4411, please fix it today!",
      json_schema: TICKET });
    assert.deepEqual(Object.keys(res.answers), ["department", "urgent", "priority", "customer.tier"]);
    assert.deepEqual(res.tez.values, { kind: "ticket", department: "billing", urgent: true, priority: 1, customer: { tier: 1 } });
    const err = await rejects(client.decide("x", { jsonSchema: { type: "object", properties: { note: { type: "string" } } } }));
    assert.deepEqual([err.status, err.type], [422, "invalid_request"]);
    assert.match(err.message, /json_schema\.properties\.note/);
  });

  test("schemas, schema, health and feedback: builtin, layouts and run_id", async () => {
    assert.equal((await client.schemas()).schemas[0].builtin, false);
    const detail = await client.schema("support-triage");
    assert.deepEqual([detail.layout, detail.served_layout, detail.builtin], [null, "auto", false]);
    assert.equal(detail.probes.topic.layout, "question_first");
    assert.equal((await client.health()).layout, "auto");
    const res = await client.decide("Need it by 5pm", { schema: "support-triage" });
    const fb = await client.feedback({ schema: "support-triage", question: "is_urgent", state: "Need it by 5pm", label: true,
      run_id: res.meta.runId });
    assert.equal(mock.requests.at(-1).body.run_id, res.meta.runId);
    assert.deepEqual(fb, { ok: true, schema: "support-triage", question: "is_urgent", label: "true" });
    const err = await rejects(client.feedback({ schema: "support-triage", question: "topic", state: "x", label: "billing",
      run_id: "x".repeat(129) }));
    assert.deepEqual([err.status, err.type], [422, "invalid_request"]);
  });
});

describe("batches", () => {
  let mock;
  let client;
  before(async () => {
    mock = await startMockServer();
    client = new TezClient({ baseUrl: mock.url });
  });
  after(() => mock.close());

  test("decideBatch sends states with the shared options and returns one result per state", async () => {
    const res = await client.decideBatch(["My invoice is wrong", { text: "the app crashes" }, 42],
      { questions: { topic: QUESTIONS.topic }, alpha: 0.05, layout: "state_first" });
    assert.equal(mock.requests.at(-1).path, "/v1/systemone/batch");
    assert.deepEqual(mock.requests.at(-1).body, { states: ["My invoice is wrong", { text: "the app crashes" }, 42],
      questions: { topic: QUESTIONS.topic }, tez: { gate: { alpha: 0.05 }, layout: "state_first" } });
    assert.deepEqual(Object.keys(res), ["model", "results", "usage", "tez"]);
    assert.equal(res.results.length, 3);
    const [first, second, third] = res.results;
    assert.equal(first.answers.topic.choice, "billing");
    assert.deepEqual(second.tez.questions.topic, { readout: "letters", decision: "escalate" });
    assert.deepEqual(third, { error: { type: "invalid_request", message: "state must be a string, object or array" } });
    assert.ok("error" in third && !("error" in first));
    assert.deepEqual(res.usage, { input_tokens: 84, output_tokens: 0 });
    assert.match(res.tez.run_id, RUN_ID);
    assert.equal(res.meta.runId, res.tez.run_id);
    assert.equal(res.meta.layout, null);                            // X-Tez-Layout is for single decisions
    assert.equal(res.meta.timing.tez, 3);
  });

  test("decideMany resolves to the results; systemoneBatch posts a raw body", async () => {
    const results = await client.decideMany(["a", "b"], { schema: "support-triage", gate: null });
    assert.deepEqual(mock.requests.at(-1).body, { states: ["a", "b"], schema: "support-triage", tez: { gate: null } });
    assert.ok(Array.isArray(results));
    assert.deepEqual(results.map((r) => Object.keys(r.answers)), [["topic", "is_urgent"], ["topic", "is_urgent"]]);
    const raw = await client.systemoneBatch({ states: ["x"], questions: { q: QUESTIONS.is_urgent } });
    assert.deepEqual(raw.results[0].answers.q, { type: "noul", noul: 0.8 });
  });

  test("a bad shared field or too many states fails the whole batch", async () => {
    let err = await rejects(client.decideBatch(["a"], { questions: { q: { type: "maybe", instructions: "x" } } }));
    assert.deepEqual([err.status, err.type], [422, "invalid_request"]);
    assert.match(err.meta.runId, RUN_ID);
    err = await rejects(client.decideBatch(["a", "b", "c", "d", "e"], { schema: "support-triage" }));
    assert.deepEqual([err.status, err.type], [413, "payload_too_large"]);
    assert.match(err.message, /too many states/);
    err = await rejects(client.decideBatch("not an array", { schema: "support-triage" }));      // rejects, not throws
    assert.ok(err instanceof TypeError);
  });
});

describe("plan", () => {
  let mock;
  let client;
  before(async () => {
    mock = await startMockServer();
    client = new TezClient({ baseUrl: mock.url });
  });
  after(() => mock.close());

  test("plan posts a request body, state optional; requestBody builds it from decide's options", async () => {
    const body = requestBody(undefined, { schema: "support-triage", readout: "letters", layout: "auto" });
    assert.deepEqual(body, { schema: "support-triage", tez: { readout: "letters", layout: "auto" } });
    const plan = await client.plan(body);
    assert.equal(mock.requests.at(-1).path, "/v1/plan");
    assert.deepEqual(mock.requests.at(-1).body, body);
    assert.equal(plan.state_tokens, null);
    assert.equal(plan.layout, "mixed");
    assert.deepEqual(plan.order, ["is_urgent", "topic"]);
    assert.deepEqual(plan.questions.topic.fit, { status: "ready", reason: null, calibration_id: "support-triage@2026-09-24",
      fit_layout: "question_first", letters_calibrated: true, probe: false });
    assert.equal(plan.questions.is_urgent.fit.status, "none");
    assert.equal(plan.totals.evaluated_tokens, 200);
    assert.equal(plan.meta.runId, null);                             // nothing was decided
    const withState = await client.plan(requestBody("I was charged twice", { questions: QUESTIONS }));
    assert.equal(withState.state_tokens, 12);
    assert.deepEqual(requestBody("s", { questions: QUESTIONS, signal: AbortSignal.timeout(1000), model: "m" }),
      { state: "s", model: "m", questions: QUESTIONS });
  });
});

describe("extract", () => {
  let mock;
  let client;
  before(async () => {
    mock = await startMockServer();
    client = new TezClient({ baseUrl: mock.url });
  });
  after(() => mock.close());

  test("a JSON schema goes out as json_schema and resolves to tez.values", async () => {
    const values = await client.extract("Charged twice for invoice 4411", TICKET, { readout: "letters", layout: "state_first" });
    assert.deepEqual(mock.requests.at(-1).body, { state: "Charged twice for invoice 4411", json_schema: TICKET,
      tez: { readout: "letters", layout: "state_first" } });
    assert.deepEqual(values, { kind: "ticket", department: "billing", urgent: true, priority: 1, customer: { tier: 1 } });
  });

  test("a schema name: answers become values in the client", async () => {
    // (the mock's support-triage has a default gate that escalates is_urgent: returnDetails reads the values anyway)
    const { values } = await client.extract("Need it by 5pm", "support-triage", { returnDetails: true });
    assert.deepEqual(mock.requests.at(-1).body, { state: "Need it by 5pm", schema: "support-triage" });
    assert.deepEqual(values, { topic: "billing", is_urgent: true });
    const scored = new TezClient({ baseUrl: "http://t.example", fetch: async () => ({ status: 200, statusText: "OK",
      headers: new Headers(), text: async () => JSON.stringify({ model: "m", usage: { input_tokens: 1, output_tokens: 0 },
        answers: { anger: { type: "score", score: 1.2, legend: { 0: "a", 1: "b", 2: "c" },
          probabilities: { 0: 0.1, 1: 0.6, 2: 0.3 }, confidence: 0.4 },
          topic: { type: "choice", choice: "__none__", probabilities: { a: 0.2, __none__: 0.8 }, confidence: 0.6 },
          urgent: { type: "noul", noul: 0.3 } } }) }) });
    assert.deepEqual(await scored.extract("x", "s"), { anger: 1, topic: null, urgent: false });
  });

  test("a gate that escalates a field rejects with EscalationRequired, unless returnDetails", async () => {
    const err = await rejects(client.extract("x", TICKET, { alpha: 0.05 }));
    assert.ok(err instanceof EscalationRequired && !(err instanceof TezError));
    assert.deepEqual(err.fields, ["department", "urgent", "priority", "customer.tier"]);
    assert.equal(err.values.department, "billing");
    assert.match(err.response.meta.runId, RUN_ID);
    assert.match(err.message, /the gate escalated department, urgent, priority, customer\.tier/);
    const details = await client.extract("x", "support-triage", { alpha: 0.05, returnDetails: true });
    assert.deepEqual(details.decisions, { topic: "act", is_urgent: "escalate" });
    assert.deepEqual(details.escalated, ["is_urgent"]);
    assert.deepEqual(details.values, { topic: "billing", is_urgent: true });
    assert.equal(details.response.answers.topic.choice, "billing");
    // A schema's own gate escalates without alpha: the object is not certified either.
    const own = await rejects(client.extract("x", "support-triage"));
    assert.ok(own instanceof EscalationRequired);
    assert.deepEqual([own.fields, own.values], [["is_urgent"], { topic: "billing", is_urgent: true }]);
  });

  test("a server that answers json_schema without tez.values is an invalid_response", async () => {
    const jev = new TezClient({ baseUrl: "http://jev.example", fetch: async () => ({ status: 200, statusText: "OK",
      headers: new Headers(), text: async () => JSON.stringify({ model: "m", answers: {}, usage: {} }) }) });
    const err = await rejects(jev.extract("x", TICKET));
    assert.deepEqual([err.status, err.type], [200, "invalid_response"]);
    assert.ok((await rejects(client.extract("x", 42))) instanceof TypeError);
  });
});

describe("the error types forbidden, payload_too_large and internal_error", () => {
  test("from the error body, and from the status alone", async () => {
    const mock = await startMockServer({
      routes: {
        "GET /v1/models": ({ send }) => send(403, "<html>Forbidden</html>"),
        "GET /v1/schemas": ({ send }) => send(413, "<html>Request Entity Too Large</html>"),
        "GET /healthz": ({ send }) => send(500, { error: { type: "internal_error", message: "RuntimeError: hook failed" } }),
      },
    });
    try {
      const origin = (fetchImpl) => (url, init) => fetchImpl(url, { ...init, headers: { ...init.headers, Origin: "https://evil.example" } });
      const browser = new TezClient({ baseUrl: mock.url, fetch: origin(fetch) });
      let err = await rejects(browser.decide("x", { questions: QUESTIONS }));
      assert.deepEqual([err.status, err.type], [403, "forbidden"]);
      assert.match(err.message, /origin https:\/\/evil\.example/);
      const client = new TezClient({ baseUrl: mock.url });
      err = await rejects(client.decide("x".repeat(70 * 1024), { questions: QUESTIONS }));
      assert.deepEqual([err.status, err.type], [413, "payload_too_large"]);
      err = await rejects(client.models());
      assert.deepEqual([err.status, err.type, err.message], [403, "forbidden", "<html>Forbidden</html>"]);
      err = await rejects(client.schemas());
      assert.deepEqual([err.status, err.type], [413, "payload_too_large"]);
      err = await rejects(client.health());
      assert.deepEqual([err.status, err.type, err.message], [500, "internal_error", "RuntimeError: hook failed"]);
      assert.match(err.meta.requestId, RUN_ID);
    } finally {
      await mock.close();
    }
  });
});
