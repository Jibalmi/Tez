// End to end against the real server: `python -m tez serve --backend fake`, started from the repository root. The fake
// backend is Tez's offline stand-in (answers from keyword overlap, so they mean nothing), but everything around it is
// tez serve's own: routes, validation, limits, CORS, headers. Skipped when Python or the tez package with its server
// dependencies cannot be imported; TEZ_PYTHON names the interpreter to use.
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import { EscalationRequired, requestBody, TezClient, TezError } from "tez-client";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));        // clients/ts/test -> the repository root
const RUN_ID = /^[0-9a-f]{32}$/;
const TICKET = {
  type: "object",
  properties: {
    department: { enum: ["billing", "technical", "sales"], description: "Which team should handle it?" },
    urgent: { type: "boolean", description: "Does it need a reply today?" },
    priority: { type: "integer", minimum: 1, maximum: 3 },
    customer: { type: "object", properties: { tier: { enum: [1, 2, 3] } } },
    refund: { anyOf: [{ enum: ["full", "partial"] }, { type: "null" }] },
  },
};
const QUESTIONS = {
  topic: { type: "choice", instructions: "What is the message about?", criteria: { billing: "Invoices", technical: null } },
  urgent: { type: "noul", instructions: "Does it need a reply today?" },
};

/** The environment without TEZ_* settings, so the server starts from its defaults and our flags. */
function childEnv() {
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^TEZ_/i.test(key)));
  return { ...env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" };
}

/** The interpreter (its sys.executable, so no launcher sits between us and the server) or why there is none. */
function findPython() {
  if (!existsSync(join(ROOT, "tez", "server.py"))) return { skip: "not inside a Tez checkout (no tez/server.py)" };
  const candidates = process.env.TEZ_PYTHON ? [process.env.TEZ_PYTHON]
    : process.platform === "win32" ? ["python", "py"] : ["python3", "python"];
  const tried = [];
  for (const cmd of candidates) {
    const r = spawnSync(cmd, ["-c", "import sys, tez.server, uvicorn; print(sys.executable)"],
      { cwd: ROOT, env: childEnv(), encoding: "utf8", timeout: 120_000, windowsHide: true });
    const executable = r.stdout?.trim().split(/\r?\n/).at(-1);
    if (r.status === 0 && executable) return { executable };
    tried.push(`${cmd}: ${r.error ? r.error.code ?? r.error.message : (r.stderr || "").trim().split(/\r?\n/).at(-1)}`);
  }
  return { skip: `Python with tez's server dependencies is not importable (${tried.join("; ")})` };
}

async function freePort() {
  const srv = createServer();
  await new Promise((resolve) => srv.listen(0, "127.0.0.1", resolve));
  const { port } = srv.address();
  await new Promise((resolve) => srv.close(resolve));
  return port;
}

async function rejects(promise) {
  try {
    await promise;
  } catch (err) {
    return err;
  }
  assert.fail("expected a rejection");
}

const python = findPython();

describe("end to end: tez serve --backend fake", { skip: python.skip ?? false }, () => {
  let child;
  let exited;
  let stderr = "";
  let dataDir;
  let client;
  let baseUrl;

  before(async () => {
    const port = await freePort();
    dataDir = mkdtempSync(join(tmpdir(), "tez-e2e-"));
    child = spawn(python.executable, ["-m", "tez", "serve", "--backend", "fake", "--host", "127.0.0.1", "--port", String(port),
      "--presets", "--data-dir", dataDir, "--default-temperature", "2.5", "--log-level", "warning"],
    { cwd: ROOT, env: childEnv(), stdio: ["ignore", "ignore", "pipe"], windowsHide: true });
    exited = new Promise((resolve) => child.once("exit", (code, signal) => resolve({ code, signal })));
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr = (stderr + chunk).slice(-4000); });
    baseUrl = `http://127.0.0.1:${port}`;
    client = new TezClient({ baseUrl, timeoutMs: 10_000 });
    const deadline = Date.now() + 60_000;
    for (let last = "no answer yet"; ; await new Promise((resolve) => setTimeout(resolve, 200))) {
      if (child.exitCode !== null) throw new Error(`tez serve exited with ${child.exitCode}:\n${stderr}`);
      if (Date.now() > deadline) throw new Error(`tez serve did not come up on ${baseUrl} (${last}):\n${stderr}`);
      try {
        const health = await client.health();
        if (health.status === "ok") break;
        last = `health status ${health.status}`;
      } catch (err) {
        if (!(err instanceof TezError) || err.type !== "network_error") throw err;
        last = err.message;
      }
    }
  });

  after(async () => {
    if (child && child.exitCode === null) {
      child.kill();
      await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 10_000).unref())]);
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    }
    if (dataDir) rmSync(dataDir, { recursive: true, force: true });
  });

  test("health, schemas and a preset's detail", async () => {
    const health = await client.health();
    assert.deepEqual([health.status, health.backend, health.layout], ["ok", "fake", "auto"]);
    assert.match(health.meta.requestId, RUN_ID);
    assert.equal(health.meta.runId, null);
    assert.equal(typeof health.meta.timing.total, "number");
    const { schemas } = await client.schemas();
    const triage = schemas.find((s) => s.name === "support-triage");
    assert.equal(triage.builtin, true);                                // loaded with --presets
    const detail = await client.schema("support-triage");
    assert.deepEqual([detail.builtin, detail.layout, detail.served_layout], [true, null, "auto"]);
    assert.deepEqual(Object.keys(detail.questions), ["topic", "urgent", "frustration"]);
    assert.equal(detail.probes.topic.layout, null);                    // presets are never fitted
  });

  test("a decision: answers, run id, layout, server-timing and the default temperature", async () => {
    const res = await client.decide("My invoice is wrong and I need it fixed today", { questions: QUESTIONS });
    assert.deepEqual(Object.keys(res), ["model", "answers", "usage", "tez"]);
    assert.ok(["billing", "technical"].includes(res.answers.topic.choice));
    assert.equal(typeof res.answers.urgent.noul, "number");
    assert.match(res.meta.runId, RUN_ID);
    assert.equal(res.meta.runId, res.meta.requestId);
    assert.equal(res.meta.layout, "state_first");                      // two questions under auto
    assert.deepEqual(Object.keys(res.meta.timing).sort(), ["backend", "tez", "total"]);
    for (const id of ["topic", "urgent"]) {
      assert.deepEqual(res.tez.questions[id], { readout: "letters", temperature: 2.5 });   // --default-temperature 2.5
    }
    const one = await client.decide("hi", { questions: QUESTIONS, layout: "question_first", readout: "letters" });
    assert.equal(one.meta.layout, "question_first");
    assert.notEqual(one.meta.runId, res.meta.runId);
  });

  test("json_schema, and extract by JSON schema and by schema name", async () => {
    const state = "Charged twice for invoice 4411, please fix it today!";
    const res = await client.decide(state, { jsonSchema: TICKET });
    assert.deepEqual(Object.keys(res.answers), ["department", "urgent", "priority", "customer.tier", "refund"]);
    assert.deepEqual(Object.keys(res.tez.values).sort(), ["customer", "department", "priority", "refund", "urgent"]);
    const values = await client.extract(state, TICKET);
    assert.ok(TICKET.properties.department.enum.includes(values.department));
    assert.equal(typeof values.urgent, "boolean");
    assert.ok([1, 2, 3].includes(values.priority));
    assert.ok([1, 2, 3].includes(values.customer.tier));
    assert.ok(["full", "partial", null].includes(values.refund));
    const byName = await client.extract(state, "support-triage");
    assert.deepEqual(Object.keys(byName), ["topic", "urgent", "frustration"]);
    assert.equal(typeof byName.topic, "string");
    assert.equal(typeof byName.urgent, "boolean");
    assert.ok(Number.isInteger(byName.frustration));
    const err = await rejects(client.extract(state, TICKET, { alpha: 0.05 }));      // nothing is fitted: all escalate
    assert.ok(err instanceof EscalationRequired);
    assert.deepEqual(err.fields, ["department", "urgent", "priority", "customer.tier", "refund"]);
    const details = await client.extract(state, "support-triage", { alpha: 0.05, returnDetails: true });
    assert.deepEqual(details.decisions, { topic: "escalate", urgent: "escalate", frustration: "escalate" });
    const bad = await rejects(client.extract(state, { type: "object", properties: { note: { type: "string" } } }));
    assert.deepEqual([bad.status, bad.type], [422, "invalid_request"]);
    assert.match(bad.message, /json_schema\.properties\.note/);
  });

  test("a batch: per-state errors, the batch's run id, decideMany", async () => {
    const batch = await client.decideBatch(["My invoice is wrong", { text: "the app crashes" }, 42],
      { schema: "support-triage", alpha: 0.05 });
    assert.equal(batch.results.length, 3);
    const [first, second, third] = batch.results;
    assert.deepEqual(Object.keys(first.answers), ["topic", "urgent", "frustration"]);
    assert.equal(second.tez.questions.topic.decision, "escalate");
    assert.deepEqual(third, { error: { type: "invalid_request", message: "state must be a string, object or array" } });
    assert.equal(batch.usage.input_tokens, first.usage.input_tokens + second.usage.input_tokens);
    assert.match(batch.tez.run_id, RUN_ID);
    assert.equal(batch.meta.runId, batch.tez.run_id);
    assert.equal(batch.meta.layout, null);
    const results = await client.decideMany(["a", "b"], { questions: QUESTIONS });
    assert.equal(results.length, 2);
    assert.ok(results.every((r) => !("error" in r)));
  });

  test("a plan, with and without a state", async () => {
    const plan = await client.plan(requestBody(undefined, { schema: "support-triage" }));
    assert.equal(plan.backend, "fake");
    assert.equal(plan.state_tokens, null);
    assert.deepEqual([plan.requested_layout, plan.layout], ["auto", "state_first"]);
    assert.deepEqual(plan.order, ["topic", "urgent", "frustration"]);
    assert.deepEqual(plan.questions.topic.fit, { status: "none", reason: "a built-in preset: zero-shot" });
    assert.deepEqual(plan.questions.topic.calls, { letters: 1, embed: 0 });
    assert.equal(plan.totals.evaluated_tokens, plan.totals.prompt_tokens - plan.totals.cached_tokens);
    assert.equal(plan.meta.runId, null);
    const withState = await client.plan(requestBody("My invoice is wrong", { questions: QUESTIONS, layout: "question_first" }));
    assert.equal(typeof withState.state_tokens, "number");
    assert.equal(withState.layout, "question_first");
  });

  test("feedback with the decision's run id is stored with the row", async () => {
    const state = "I was charged twice this month";
    const res = await client.decide(state, { schema: "support-triage" });
    const fb = await client.feedback({ schema: "support-triage", question: "topic", state, label: "billing",
      run_id: res.meta.runId });
    assert.deepEqual(fb, { ok: true, schema: "support-triage", question: "topic", label: "billing" });
    const rows = readFileSync(join(dataDir, "feedback", "support-triage.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    assert.equal(rows.at(-1).run_id, res.meta.runId);
  });

  test("errors: invalid_request, not_found, forbidden, payload_too_large", async () => {
    let err = await rejects(client.decide("s", { questions: { q: { type: "maybe", instructions: "x" } } }));
    assert.deepEqual([err.status, err.type], [422, "invalid_request"]);
    assert.match(err.meta.runId, RUN_ID);                              // it reached the engine
    err = await rejects(client.schema("no-such-schema"));
    assert.deepEqual([err.status, err.type], [404, "not_found"]);
    const browser = new TezClient({ baseUrl, fetch: (url, init) =>
      fetch(url, { ...init, headers: { ...init.headers, Origin: "https://elsewhere.example" } }) });
    err = await rejects(browser.decide("s", { questions: QUESTIONS }));
    assert.deepEqual([err.status, err.type], [403, "forbidden"]);
    assert.equal(err.meta.runId, null);                                // refused before the engine
    err = await rejects(client.decideBatch(Array.from({ length: 65 }, (_, i) => `state ${i}`), { questions: QUESTIONS }));
    assert.deepEqual([err.status, err.type], [413, "payload_too_large"]);
    assert.match(err.message, /too many states/);
    err = await rejects(client.decide("x".repeat(50_001), { questions: QUESTIONS }));
    assert.deepEqual([err.status, err.type], [413, "payload_too_large"]);
  });
});
