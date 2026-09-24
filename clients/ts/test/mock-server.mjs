// A tiny /v1/systemone server for the tests: wire-format answers from fixed numbers (the first option always wins),
// the docs/API.md error bodies and response headers, and per-route overrides for redirects, slow answers and broken
// proxies. Like tez serve: the loaded schema's `topic` counts as fitted under question_first, so under `auto` a
// request with it and other questions is read `mixed`; json_schema, batches, plans, 403 for a foreign browser origin
// and 413 past the limits.
import { randomUUID } from "node:crypto";
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
  layout: null,
  builtin: false,
};
const FITTED = { topic: { layout: "question_first", calibration_id: "support-triage@2026-09-24" } };
export const LIMITS = { maxBatch: 4, maxQuestions: 8, maxBodyBytes: 64 * 1024 };

const TYPES = new Set(["noul", "choice", "score"]);
const LAYOUTS = new Set(["auto", "question_first", "state_first"]);

class HttpError extends Error {
  constructor(status, type, message) {
    super(message);
    this.status = status;
    this.type = type;
  }
}
const invalid = (message) => new HttpError(422, "invalid_request", message);

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

// A small part of tez/extract.py: enum -> choice, boolean -> noul, an integer range -> score, const -> filled in,
// nested objects -> dotted ids. Each field records how its answer becomes a value.
function fromJsonSchema(node, prefix, out) {
  if (typeof node !== "object" || node === null || typeof node.properties !== "object") {
    throw invalid(`json_schema${prefix ? `.properties.${prefix}` : ""}: the top level must be an object with properties`);
  }
  for (const [key, sub] of Object.entries(node.properties)) {
    const id = prefix ? `${prefix}.${key}` : key;
    const instructions = sub.description ?? `The value of ${id}`;
    if ("const" in sub) out.consts.push([id, sub.const]);
    else if (Array.isArray(sub.enum)) {
      out.questions[id] = { type: "choice", instructions, criteria: Object.fromEntries(sub.enum.map((v) => [String(v), null])) };
      out.fields[id] = (a) => sub.enum[Object.keys(a.probabilities).indexOf(a.choice)] ?? null;
    } else if (sub.type === "boolean") {
      out.questions[id] = { type: "noul", instructions };
      out.fields[id] = (a) => a.noul >= 0.5;
    } else if (sub.type === "integer" && Number.isInteger(sub.minimum) && Number.isInteger(sub.maximum)) {
      const levels = Array.from({ length: sub.maximum - sub.minimum + 1 }, (_, i) => sub.minimum + i);
      out.questions[id] = { type: "score", instructions, criteria: levels.map((v) => `${id} = ${v}`) };
      out.fields[id] = (a) => levels[Number(Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0])];
    } else if (sub.type === "object") fromJsonSchema(sub, id, out);
    else throw invalid(`json_schema.properties.${id}: a free-text string cannot be decided in one pass`);
  }
  return out;
}

function put(obj, path, value) {
  const keys = path.split(".");
  let d = obj;
  for (const key of keys.slice(0, -1)) d = d[key] ??= {};
  d[keys.at(-1)] = value;
}

// A request, parsed as tez's engine does (enough of it): throws HttpError.
function parse(body, { requireState = true } = {}) {
  if (typeof body !== "object" || body === null || Array.isArray(body)) throw invalid("the request body must be a JSON object");
  if (requireState && (body.state === undefined || body.state === null)) throw invalid("state is required");
  if (body.state !== undefined && body.state !== null && typeof body.state !== "string" && typeof body.state !== "object") {
    throw invalid("state must be a string, object or array");
  }
  if (body.schema !== undefined && body.schema !== SCHEMA.name) throw invalid(`unknown schema '${body.schema}'`);
  const schema = body.schema === undefined ? null : SCHEMA;
  let questions = body.questions;
  let extraction = null;
  if (body.json_schema !== undefined) {
    if (questions !== undefined) throw invalid("give questions or json_schema, not both");
    extraction = fromJsonSchema(body.json_schema, "", { questions: {}, fields: {}, consts: [] });
    questions = extraction.questions;
  } else if (questions === undefined) {
    if (schema === null) throw invalid("questions is required (or json_schema, or name a loaded schema)");
    questions = SCHEMA.questions;
  }
  for (const [id, q] of Object.entries(questions)) {
    if (!TYPES.has(q?.type)) throw invalid(`questions.${id}.type must be one of noul, choice, score; got ${JSON.stringify(q?.type)}`);
  }
  if (Object.keys(questions).length > LIMITS.maxQuestions) {
    throw new HttpError(413, "payload_too_large", `too many questions: ${Object.keys(questions).length}`);
  }
  const tez = body.tez ?? {};
  if (tez.layout !== undefined && !LAYOUTS.has(tez.layout)) {
    throw invalid(`tez.layout must be one of auto, question_first, state_first; got ${JSON.stringify(tez.layout)}`);
  }
  const gated = (tez.gate !== undefined && tez.gate !== null && tez.gate !== false) || (schema !== null && !("gate" in tez));
  const requested = tez.layout ?? schema?.layout ?? "auto";
  const n = Object.keys(questions).length;
  const layouts = {};
  const fitted = {};
  for (const [id, q] of Object.entries(questions)) {
    const fit = schema !== null && JSON.stringify(q) === JSON.stringify(SCHEMA.questions[id]) ? FITTED[id] : undefined;
    const usable = fit !== undefined && (requested === "auto" || requested === fit.layout);
    if (usable) fitted[id] = fit;
    layouts[id] = requested !== "auto" ? requested : usable ? fit.layout : n >= 2 ? "state_first" : "question_first";
  }
  const kinds = new Set(Object.values(layouts));
  const layout = kinds.size === 1 ? [...kinds][0] : "mixed";
  return { questions, extraction, tez, gated, requested, layouts, fitted, layout, schema };
}

function decide(body, options) {
  const req = parse(body);
  const answers = {};
  const metas = {};
  for (const [id, q] of Object.entries(req.questions)) {
    answers[id] = answer(q, req.tez.abstain === true);
    const fit = req.fitted[id];
    const meta = { readout: "letters" };
    if (fit === undefined && options.defaultTemperature !== undefined) meta.temperature = options.defaultTemperature;
    if (req.gated) meta.decision = fit !== undefined ? "act" : "escalate";
    if (fit !== undefined) Object.assign(meta, { p_correct: 0.7, calibration_id: fit.calibration_id });
    if (req.layout === "mixed") meta.layout = req.layouts[id];
    metas[id] = meta;
  }
  const res = { model: "tez-mock (fixed numbers, letters)", answers, usage: { input_tokens: 42, output_tokens: 0 },
    tez: { latency_ms: 1.5, questions: metas } };
  if (req.extraction !== null) {
    const values = {};
    for (const [id, value] of req.extraction.consts) put(values, id, value);
    for (const [id, toValue] of Object.entries(req.extraction.fields)) put(values, id, toValue(answers[id]));
    res.tez.values = values;
  }
  return { res, layout: req.layout };
}

function batch(body, options) {
  if (typeof body !== "object" || body === null || Array.isArray(body)) throw invalid("the request body must be a JSON object");
  if ("state" in body) throw invalid("a batch takes states (an array of states), not state");
  const { states, ...shared } = body;
  if (!Array.isArray(states) || states.length === 0) {
    throw invalid("states must be a non-empty array of states (strings, objects or arrays)");
  }
  if (states.length > LIMITS.maxBatch) {
    throw new HttpError(413, "payload_too_large", `too many states in one batch: ${states.length} (the limit is ${LIMITS.maxBatch})`);
  }
  parse({ ...shared, state: "" });
  const results = [];
  let tokens = 0;
  for (const state of states) {
    try {
      const { res } = decide({ ...shared, state }, options);
      results.push(res);
      tokens += res.usage.input_tokens;
    } catch (err) {
      if (!(err instanceof HttpError)) throw err;
      results.push({ error: { type: err.type, message: err.message } });
    }
  }
  return { model: "tez-mock (fixed numbers, letters)", results, usage: { input_tokens: tokens, output_tokens: 0 },
    tez: { latency_ms: 3, run_id: undefined } };
}

function plan(body) {
  const req = parse(body, { requireState: false });
  const order = [...Object.keys(req.questions).filter((id) => req.layouts[id] === "state_first"),
    ...Object.keys(req.questions).filter((id) => req.layouts[id] !== "state_first")];
  const questions = {};
  for (const [id, q] of Object.entries(req.questions)) {
    const fit = req.fitted[id];
    questions[id] = {
      type: q.type,
      options: q.type === "noul" ? 2 : q.type === "choice" ? Object.keys(q.criteria).length : q.criteria.length,
      layout: req.layouts[id],
      readout: "letters",
      fit: fit === undefined ? { status: "none", reason: "no schema named: questions are read zero-shot" }
        : { status: "ready", reason: null, calibration_id: fit.calibration_id, fit_layout: fit.layout, letters_calibrated: true,
          probe: false },
      calls: { letters: 1, embed: 0 },
      prompt_sha: "0123456789abcdef",
      prompt_tokens: 100,
      cached_tokens: 0,
    };
  }
  const n = order.length;
  return { backend: "mock", template: "gemma4", model: null, schema: req.schema?.name ?? null, readout: req.tez.readout ?? "auto",
    requested_layout: req.requested, layout: req.layout, state_tokens: body.state == null ? null : 12, order, questions,
    totals: { calls: { letters: n, embed: 0 }, prompt_tokens: 100 * n, cached_tokens: 0, evaluated_tokens: 100 * n },
    notes: ["model names not checked (the plan does not call the backend): a fit made on another model would not be used"] };
}

const ALLOWED_ORIGIN = /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/;

/**
 * options.apiKey          require it as a bearer token
 * options.routes          "METHOD /path" -> handler({ req, res, body, send }) that answers instead
 * options.defaultTemperature  add `temperature` to every unfitted letters answer's tez block
 */
export async function startMockServer({ apiKey, routes = {}, defaultTemperature } = {}) {
  const requests = [];
  const server = createServer(async (req, res) => {
    const t0 = performance.now();
    const requestId = randomUUID().replaceAll("-", "");
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const raw = Buffer.concat(chunks).toString("utf8");
    let body;
    let badJson = false;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      body = raw;
      badJson = true;
    }
    const path = new URL(req.url, "http://mock").pathname;
    requests.push({ method: req.method, path, rawPath: req.url, headers: req.headers, body });
    const timing = [];
    let decision = null;                  // { layout } once a decision reached the "engine"
    const send = (status, payload, headers = {}) => {
      const text = typeof payload === "string" ? payload : JSON.stringify(payload);
      const type = typeof payload === "string" ? "text/html" : "application/json";
      const out = { "content-type": type, "x-typesafe-request-id": requestId,
        "server-timing": [...timing, `total;dur=${(performance.now() - t0).toFixed(1)}`].join(", ") };
      if (decision !== null) {
        out["x-tez-run-id"] = requestId;
        if (decision.layout) out["x-tez-layout"] = decision.layout;
      }
      res.writeHead(status, { ...out, ...headers });
      res.end(text);
    };
    const error = (status, type, message) => send(status, { error: { type, message } });
    const route = routes[`${req.method} ${path}`];
    if (route) return route({ req, res, body, send });
    if (req.method === "POST" && req.headers.origin !== undefined && !ALLOWED_ORIGIN.test(req.headers.origin)) {
      return error(403, "forbidden", `origin ${req.headers.origin} may not request decisions`);
    }
    if (apiKey && path !== "/healthz" && req.headers.authorization !== `Bearer ${apiKey}`) {
      return error(401, "unauthorized", "missing or invalid API key (send Authorization: Bearer <key>)");
    }
    if (req.method === "POST" && Buffer.byteLength(raw) > LIMITS.maxBodyBytes) {
      return error(413, "payload_too_large", `the request body is over ${LIMITS.maxBodyBytes} bytes`);
    }
    if (req.method === "POST" && badJson) return error(422, "invalid_request", "the request body is not valid JSON");
    try {
      if (req.method === "POST" && path === "/v1/systemone") {
        decision = { layout: null };
        const out = decide(body, { defaultTemperature });
        decision.layout = out.layout;
        timing.push("tez;dur=1.5", "backend;dur=1.0");
        return send(200, out.res);
      }
      if (req.method === "POST" && path === "/v1/systemone/batch") {
        decision = { layout: null };
        const out = batch(body, { defaultTemperature });
        out.tez.run_id = requestId;
        timing.push("tez;dur=3.0");
        return send(200, out);
      }
      if (req.method === "POST" && path === "/v1/plan") return send(200, plan(body));
    } catch (err) {
      if (err instanceof HttpError) return error(err.status, err.type, err.message);
      throw err;
    }
    if (req.method === "GET" && path === "/v1/models") {
      return send(200, { models: [{ name: "tez-mock", description: "fixed numbers", release_date: "2026-09-24" }] });
    }
    if (req.method === "GET" && path === "/healthz") {
      return send(200, { status: "ok", version: "0.1.0", backend: "mock", template: "gemma4", backend_status: "ok",
        model: "mock", embed_backend: null, schemas: [SCHEMA.name], probes: { [SCHEMA.name]: [] }, layout: "auto" });
    }
    if (req.method === "GET" && path === "/v1/schemas") {
      return send(200, { schemas: [{ name: SCHEMA.name, description: SCHEMA.description,
        questions: { topic: "choice", is_urgent: "noul" }, calibration_id: FITTED.topic.calibration_id, probes: [],
        builtin: false }] });
    }
    if (req.method === "GET" && path.startsWith("/v1/schemas/")) {
      const name = decodeURIComponent(path.slice("/v1/schemas/".length));
      if (name !== SCHEMA.name) return error(404, "not_found", `unknown schema '${name}'`);
      return send(200, { ...SCHEMA, served_layout: "auto", calibration_id: FITTED.topic.calibration_id, calibration: null,
        manifest: null, probes: {
          topic: { probe: "none", letters_calibrated: true, n_labels: 40, note: null, layout: "question_first" },
          is_urgent: { probe: "none", letters_calibrated: false, n_labels: null, note: null, layout: null } } });
    }
    if (req.method === "POST" && path === "/v1/feedback") {
      if (!body?.schema || !body?.question || body?.label === undefined) return error(422, "invalid_request", "label is required");
      if (body.run_id !== undefined && (typeof body.run_id !== "string" || !body.run_id || body.run_id.length > 128)) {
        return error(422, "invalid_request", "run_id must be the decision's run id (a string of at most 128 characters)");
      }
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
