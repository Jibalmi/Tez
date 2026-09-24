// The ES module build, imported through the package's own exports map, against the mock server.
import assert from "node:assert/strict";
import { createServer } from "node:net";
import { after, before, describe, test } from "node:test";

import { DEFAULT_BASE_URL, NONE_LABEL, TezClient, TezError } from "tez-client";

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
