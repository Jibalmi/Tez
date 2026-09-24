// A tiny /v1/systemone server for the tests: wire-format answers from fixed numbers (the first option always wins),
// the docs/API.md error bodies, and per-route overrides for redirects, slow answers and broken proxies.
import { createServer } from "node:http";

export const SCHEMA = {
  name: "support-triage",
  description: "Route and prioritise inbound customer messages.",
  state: "One customer message (text).",
  questions: {
    topic: { type: "choice", instructions: "What is the message about?", criteria: { billing: "Payments", technical: null } },
    is_urgent: { type: "noul", instructions: "Does this convey urgency?" },
  },
  gate: { alpha: 0.05 },
  examples: 1,
};

const TYPES = new Set(["noul", "choice", "score"]);

function answer(q, abstain) {
  if (q.type === "noul") return { type: "noul", noul: 0.8 };
  const keys = q.type === "choice" ? Object.keys(q.criteria) : q.criteria.map((_, i) => String(i));
  if (q.type === "choice" && abstain) keys.push("__none__");
  const k = keys.length;
  const probabilities = Object.fromEntries(keys.map((key, i) => [key, i === 0 ? 0.7 : 0.3 / (k - 1)]));
  const confidence = (k * 0.7 - 1) / (k - 1);
  if (q.type === "choice") return { type: "choice", choice: keys[0], probabilities, confidence };
  const legend = Object.fromEntries(q.criteria.map((c, i) => [String(i), c]));
  const score = keys.reduce((s, key, i) => s + i * probabilities[key], 0);
  return { type: "score", score, legend, probabilities, confidence };
}

function decide(body) {
  if (typeof body !== "object" || body === null || Array.isArray(body)) return [422, "the request body must be a JSON object"];
  if (body.state === undefined || body.state === null) return [422, "state is required"];
  let questions = body.questions;
  if (body.schema !== undefined && body.schema !== SCHEMA.name) return [422, `unknown schema '${body.schema}'`];
  if (questions === undefined) {
    if (body.schema === undefined) return [422, "questions is required (or name a loaded schema)"];
    questions = SCHEMA.questions;
  }
  for (const [id, q] of Object.entries(questions)) {
    if (!TYPES.has(q?.type)) return [422, `questions.${id}.type must be one of noul, choice, score; got ${JSON.stringify(q?.type)}`];
  }
  const tez = body.tez ?? {};
  const gated = (tez.gate !== undefined && tez.gate !== null && tez.gate !== false) || (body.schema !== undefined && !("gate" in tez));
  const answers = {};
  const metas = {};
  for (const [id, q] of Object.entries(questions)) {
    answers[id] = answer(q, tez.abstain === true);
    metas[id] = gated ? { readout: "letters", decision: "escalate" } : { readout: "letters" };
  }
  return [200, { model: "tez-mock (fixed numbers, letters)", answers, usage: { input_tokens: 42, output_tokens: 0 },
    tez: { latency_ms: 1.5, questions: metas } }];
}

export async function startMockServer({ apiKey, routes = {} } = {}) {
  const requests = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const raw = Buffer.concat(chunks).toString("utf8");
    let body;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      body = raw;
    }
    const path = new URL(req.url, "http://mock").pathname;
    requests.push({ method: req.method, path, rawPath: req.url, headers: req.headers, body });
    const send = (status, payload, headers = {}) => {
      const text = typeof payload === "string" ? payload : JSON.stringify(payload);
      const type = typeof payload === "string" ? "text/html" : "application/json";
      res.writeHead(status, { "content-type": type, ...headers });
      res.end(text);
    };
    const error = (status, type, message) => send(status, { error: { type, message } });
    const route = routes[`${req.method} ${path}`];
    if (route) return route({ req, res, body, send });
    if (apiKey && path !== "/healthz" && req.headers.authorization !== `Bearer ${apiKey}`) {
      return error(401, "unauthorized", "missing or invalid API key (send Authorization: Bearer <key>)");
    }
    if (req.method === "POST" && path === "/v1/systemone") {
      const [status, out] = decide(body);
      return status === 200 ? send(200, out) : error(status, "invalid_request", out);
    }
    if (req.method === "GET" && path === "/v1/models") {
      return send(200, { models: [{ name: "tez-mock", description: "fixed numbers", release_date: "2026-09-24" }] });
    }
    if (req.method === "GET" && path === "/healthz") {
      return send(200, { status: "ok", version: "0.1.0", backend: "mock", template: "gemma4", backend_status: "ok",
        model: "mock", embed_backend: null, schemas: [SCHEMA.name], probes: { [SCHEMA.name]: [] } });
    }
    if (req.method === "GET" && path === "/v1/schemas") {
      return send(200, { schemas: [{ name: SCHEMA.name, description: SCHEMA.description,
        questions: { topic: "choice", is_urgent: "noul" }, calibration_id: null, probes: [] }] });
    }
    if (req.method === "GET" && path.startsWith("/v1/schemas/")) {
      const name = decodeURIComponent(path.slice("/v1/schemas/".length));
      if (name !== SCHEMA.name) return error(404, "not_found", `unknown schema '${name}'`);
      return send(200, { ...SCHEMA, calibration_id: null, calibration: null, manifest: null,
        probes: { topic: { probe: "none", letters_calibrated: false, n_labels: null, note: null } } });
    }
    if (req.method === "POST" && path === "/v1/feedback") {
      if (!body?.schema || !body?.question || body?.label === undefined) return error(422, "invalid_request", "label is required");
      const label = body.label === true ? "true" : body.label === false ? "false" : body.label;
      return send(200, { ok: true, schema: body.schema, question: body.question, label });
    }
    return error(404, "not_found", `no route ${req.method} ${path}`);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    close: () => new Promise((resolve) => {
      server.closeAllConnections();
      server.close(resolve);
    }),
  };
}
