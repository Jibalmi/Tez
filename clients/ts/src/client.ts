import { EscalationRequired, TezError } from "./errors.js";
import type { TezErrorType } from "./errors.js";
import { responseMeta, withMeta } from "./meta.js";
import type { HeadersLike, ResponseMeta, WithMeta } from "./meta.js";
import type {
  Answer,
  AnswersFor,
  BatchRequest,
  BatchResponse,
  BatchResult,
  DecideRequest,
  DecideResponse,
  ExtractedObject,
  ExtractedValue,
  FeedbackRequest,
  FeedbackResponse,
  GateDecision,
  GateOptions,
  HealthResponse,
  JsonSchema,
  LayoutOption,
  ModelsResponse,
  PlanRequest,
  PlanResponse,
  Questions,
  ReadoutOption,
  SchemaDetail,
  SchemasResponse,
  State,
  TezOptions,
} from "./types.js";

export const DEFAULT_BASE_URL = "http://127.0.0.1:8787";
export const DEFAULT_TIMEOUT_MS = 30_000;
/** The implicit "none of these fits" option that `abstain: true` adds to choice questions. */
export const NONE_LABEL = "__none__";

/** The part of a fetch Response the client reads. */
export interface FetchResponseLike {
  readonly status: number;
  readonly statusText: string;
  readonly type?: string;
  readonly headers: HeadersLike;
  text(): Promise<string>;
}

/** The request options the client passes to fetch. */
export interface FetchInitLike {
  method: string;
  headers: Record<string, string>;
  body?: string;
  signal: AbortSignal;
  redirect: "manual";
}

/** A fetch implementation: the global fetch, undici's, or a test double. It is called without a `this`. */
export type FetchLike = (url: string, init: FetchInitLike) => Promise<FetchResponseLike>;

export interface TezClientOptions {
  /** Default http://127.0.0.1:8787. A bare host:port gets http://. */
  baseUrl?: string;
  /** Sent as `Authorization: Bearer <key>`; needed only when the server runs with --api-key. */
  apiKey?: string;
  /** Per request, from sending to the parsed body. Default 30000. */
  timeoutMs?: number;
  /** Default: the global fetch (Node 18+, browsers, Deno, Bun). */
  fetch?: FetchLike;
}

export interface RequestOptions {
  /** Cancel the request; it then rejects with the signal's reason. */
  signal?: AbortSignal;
}

/** What `requestBody` turns into a wire-format body (everything `decide` takes except `signal`). */
export interface BodyOptions<Q extends Questions = Questions> {
  /** Question id -> question. Optional when `schema` names a loaded schema (its questions are used). */
  questions?: Q;
  /** Tez: in place of `questions`, a JSON schema of the object to extract; the object comes back in `tez.values`. */
  jsonSchema?: JsonSchema;
  /** Tez: a loaded schema. Questions with the same id and definition use its probes and calibration. */
  schema?: string;
  /** Tez: auto (default), letters or probe. */
  readout?: ReadoutOption;
  /** Tez: add the implicit "__none__" option to every choice question. */
  abstain?: boolean;
  /** Tez: gate at this target error rate among acted decisions (needs a fitted schema). Wins over `gate`. */
  alpha?: number;
  /** Tez: the raw `tez.gate`; `{}` uses the schema's default alpha, null or false turns a schema's default gate off. */
  gate?: GateOptions | null | false;
  /** Tez: the prompt layout, auto, question_first or state_first. Default: the schema's, else the server's. */
  layout?: LayoutOption;
  /** Any string; the server's default is "tez-latest". Sent only when given. */
  model?: string;
}

export interface DecideOptions<Q extends Questions = Questions> extends BodyOptions<Q>, RequestOptions {}

export interface ExtractOptions extends RequestOptions {
  /** Gate every field at this target error rate (a named schema's own gate applies without it); an escalated field
   * then throws EscalationRequired. */
  alpha?: number;
  readout?: ReadoutOption;
  layout?: LayoutOption;
  /** Resolve to an ExtractResult (values, gate decisions, escalated fields, the response); never EscalationRequired. */
  returnDetails?: boolean;
}

/** `extract(..., { returnDetails: true })`. */
export interface ExtractResult<V = ExtractedObject> {
  /** The extracted object. */
  values: V;
  /** Field -> gate decision, for the fields a gate applied to. */
  decisions: Record<string, GateDecision>;
  /** The fields the gate escalated. */
  escalated: string[];
  /** The whole /v1/systemone response. */
  response: WithMeta<DecideResponse>;
}

const STATUS_TYPES: Record<number, TezErrorType> = {
  401: "unauthorized",
  403: "forbidden",
  404: "not_found",
  405: "method_not_allowed",
  413: "payload_too_large",
  422: "invalid_request",
  500: "internal_error",
  502: "backend_unavailable",
  503: "backend_unavailable",
  504: "backend_unavailable",
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function describe(err: unknown): string {
  if (err instanceof Error) {
    const cause = (err as { cause?: unknown }).cause;
    return cause instanceof Error ? `${err.message} (${cause.message})` : err.message;
  }
  return String(err);
}

/** The fields a decision and a batch share, Tez's extensions only when set. */
function sharedFields(options: BodyOptions<Questions>): Omit<DecideRequest, "state"> {
  const body: Omit<DecideRequest, "state"> = {};
  if (options.model !== undefined) body.model = options.model;
  if (options.questions !== undefined) body.questions = options.questions;
  if (options.jsonSchema !== undefined) body.json_schema = options.jsonSchema;
  if (options.schema !== undefined) body.schema = options.schema;
  const tez: TezOptions = {};
  if (options.readout !== undefined && options.readout !== "auto") tez.readout = options.readout;
  if (options.abstain) tez.abstain = true;
  if (options.alpha !== undefined) tez.gate = { alpha: options.alpha };
  else if (options.gate !== undefined) tez.gate = options.gate;
  if (options.layout !== undefined) tez.layout = options.layout;
  if (Object.keys(tez).length > 0) body.tez = tez;
  return body;
}

/**
 * The /v1/systemone body that `decide(state, options)` sends: Tez's extensions only when set, so a Jev-only server
 * sees a plain request. Useful with `plan`, which takes the same body (`state` may be left out there).
 */
export function requestBody(state: State, options?: BodyOptions<Questions>): DecideRequest;
export function requestBody(state: State | undefined, options?: BodyOptions<Questions>): PlanRequest;
export function requestBody(state: State | undefined, options: BodyOptions<Questions> = {}): PlanRequest {
  const body: PlanRequest = {};
  if (state !== undefined) body.state = state;
  return Object.assign(body, sharedFields(options));
}

/** A value per answer, as the Python API's extract does for a schema not made from a JSON schema. */
function answerValue(answer: Answer): ExtractedValue {
  if (answer.type === "noul") return answer.noul >= 0.5;
  if (answer.type === "choice") return answer.choice === NONE_LABEL ? null : answer.choice;
  let best: string | undefined;
  for (const [level, p] of Object.entries(answer.probabilities)) {
    if (best === undefined || p > (answer.probabilities[best] ?? -Infinity)) best = level;
  }
  return best !== undefined ? Number(best) : Math.round(answer.score);
}

/**
 * A client for a Tez server, or any server that speaks the /v1/systemone wire format.
 *
 *     const tez = new TezClient();                       // http://127.0.0.1:8787
 *     const res = await tez.decide("My payouts have failed for 3 days", {
 *       questions: { topic: { type: "choice", instructions: "What is it about?", criteria: { billing: null, technical: null } } },
 *     });
 *     res.answers.topic.choice;                          // "billing" | "technical" | "__none__"
 *     res.meta.runId;                                    // the X-Tez-Run-Id header
 *
 * Every method resolves to the response body exactly as the wire format defines it, with the response's status and
 * headers in a non-enumerable `meta` property. Tez's extensions (`schema`, `json_schema` and the `tez` options) are
 * sent only when set, so a Jev-only server sees a plain request. Redirects are never followed.
 */
export class TezClient {
  readonly baseUrl: string;
  readonly timeoutMs: number;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: FetchLike | undefined;

  constructor(options: TezClientOptions = {}) {
    let base = (options.baseUrl ?? DEFAULT_BASE_URL).trim();
    if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(base)) base = `http://${base}`;
    if (!/^https?:\/\/[^/]/i.test(base)) {
      throw new TypeError(`baseUrl must be an http:// or https:// URL, got ${JSON.stringify(options.baseUrl)}`);
    }
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new TypeError(`timeoutMs must be a positive number of milliseconds, got ${String(options.timeoutMs)}`);
    }
    if (options.fetch !== undefined && typeof options.fetch !== "function") {
      throw new TypeError("fetch must be a function");
    }
    this.baseUrl = base.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
    this.apiKey = options.apiKey || undefined;
    this.fetchImpl = options.fetch;
  }

  /** Decide one state against typed questions (POST /v1/systemone). Answers are typed per question. */
  decide<Q extends Questions = Questions>(
    state: State,
    options: DecideOptions<Q> = {},
  ): Promise<WithMeta<DecideResponse<AnswersFor<Q>>>> {
    return this.request("POST", "/v1/systemone", requestBody(state, options), options.signal);
  }

  /** POST a raw wire-format body to /v1/systemone. */
  systemone(body: DecideRequest, options: RequestOptions = {}): Promise<WithMeta<DecideResponse>> {
    return this.request("POST", "/v1/systemone", body, options.signal);
  }

  /**
   * Decide many states against the same questions in one request (POST /v1/systemone/batch). `results` has one entry
   * per state, in order: the state's response, or `{ error: { type, message } }` for a state that failed (the others
   * are still decided). A bad shared option fails the whole batch with a TezError.
   */
  async decideBatch<Q extends Questions = Questions>(
    states: readonly State[],
    options: DecideOptions<Q> = {},
  ): Promise<WithMeta<BatchResponse<AnswersFor<Q>>>> {
    if (!Array.isArray(states)) throw new TypeError("states must be an array of states");
    const body: BatchRequest = { states: [...states], ...sharedFields(options) };
    return this.request("POST", "/v1/systemone/batch", body, options.signal);
  }

  /** The `results` of `decideBatch`: one per state, in order, each a response or `{ error: { type, message } }`. */
  async decideMany<Q extends Questions = Questions>(
    states: readonly State[],
    options: DecideOptions<Q> = {},
  ): Promise<BatchResult<AnswersFor<Q>>[]> {
    const res = await this.decideBatch(states, options);
    if (!Array.isArray(res.results)) {
      const message = `${this.baseUrl}/v1/systemone/batch answered without results`;
      throw new TezError(res.meta.status, "invalid_response", message, { body: res, meta: res.meta });
    }
    return res.results;
  }

  /** POST a raw wire-format batch body to /v1/systemone/batch. */
  systemoneBatch(body: BatchRequest, options: RequestOptions = {}): Promise<WithMeta<BatchResponse>> {
    return this.request("POST", "/v1/systemone/batch", body, options.signal);
  }

  /**
   * What a /v1/systemone request would do, without calling the model (POST /v1/plan): per question the readout, the
   * fit, the layout, the backend calls and a token estimate. `requestBody(state, options)` builds the body from
   * decide's options; `state` may be left out.
   */
  plan(body: PlanRequest, options: RequestOptions = {}): Promise<WithMeta<PlanResponse>> {
    return this.request("POST", "/v1/plan", body, options.signal);
  }

  /**
   * Decide a state against a JSON schema, or a loaded schema's name, and resolve to the answers as one object
   * (docs/API.md, "Structured extraction"). A JSON schema is sent as `json_schema` and the server's `tez.values`
   * come back; for a schema name, answers become values here: noul -> boolean, choice -> its label (null for
   * "__none__"), score -> the most likely level.
   *
   * When a gate applies (`alpha`, or a named schema's own `gate:`), a field it escalates makes it reject with
   * EscalationRequired (the object is not certified), unless `returnDetails: true`, which resolves to an ExtractResult
   * instead. Give `V` to type the object.
   */
  extract<V = ExtractedObject>(
    state: State,
    source: JsonSchema | string,
    options?: ExtractOptions & { returnDetails?: false },
  ): Promise<V>;
  extract<V = ExtractedObject>(
    state: State,
    source: JsonSchema | string,
    options: ExtractOptions & { returnDetails: true },
  ): Promise<ExtractResult<V>>;
  extract<V = ExtractedObject>(
    state: State,
    source: JsonSchema | string,
    options?: ExtractOptions,
  ): Promise<V | ExtractResult<V>>;
  async extract<V = ExtractedObject>(
    state: State,
    source: JsonSchema | string,
    options: ExtractOptions = {},
  ): Promise<V | ExtractResult<V>> {
    const byName = typeof source === "string";
    if (!byName && !isObject(source)) {
      throw new TypeError("extract takes a JSON schema (an object) or the name of a loaded schema");
    }
    const { readout, layout, alpha } = options;
    const bodyOptions: BodyOptions<Questions> = { readout, layout, alpha };
    if (byName) bodyOptions.schema = source;
    else bodyOptions.jsonSchema = source;
    const response: WithMeta<DecideResponse> = await this.request(
      "POST",
      "/v1/systemone",
      requestBody(state, bodyOptions),
      options.signal,
    );
    let values: ExtractedObject;
    if (byName) {
      values = {};
      for (const [id, answer] of Object.entries(response.answers ?? {})) values[id] = answerValue(answer);
    } else if (isObject(response.tez) && isObject(response.tez.values)) {
      values = response.tez.values;
    } else {
      throw new TezError(
        response.meta.status,
        "invalid_response",
        `${this.baseUrl}/v1/systemone answered a json_schema request without tez.values (a server without extraction?)`,
        { body: response, meta: response.meta },
      );
    }
    const decisions: Record<string, GateDecision> = {};
    for (const [id, meta] of Object.entries(response.tez?.questions ?? {})) {
      if (isObject(meta) && (meta.decision === "act" || meta.decision === "escalate")) decisions[id] = meta.decision;
    }
    const escalated = Object.keys(decisions).filter((id) => decisions[id] === "escalate");
    if (options.returnDetails) return { values: values as V, decisions, escalated, response };
    if (escalated.length > 0) throw new EscalationRequired(escalated, values as V, response);
    return values as V;
  }

  /** The schemas the server loaded (GET /v1/schemas); `builtin` marks the presets. */
  schemas(options: RequestOptions = {}): Promise<WithMeta<SchemasResponse>> {
    return this.request("GET", "/v1/schemas", undefined, options.signal);
  }

  /** One schema: questions, layout, probe status, calibration (GET /v1/schemas/{name}). */
  schema(name: string, options: RequestOptions = {}): Promise<WithMeta<SchemaDetail>> {
    return this.request("GET", `/v1/schemas/${encodeURIComponent(name)}`, undefined, options.signal);
  }

  /**
   * Record the correct label for a past decision (POST /v1/feedback); `tez fit` learns from these rows. Pass the
   * decision's `meta.runId` as `run_id` to link the row to it.
   */
  feedback(body: FeedbackRequest, options: RequestOptions = {}): Promise<WithMeta<FeedbackResponse>> {
    return this.request("POST", "/v1/feedback", body, options.signal);
  }

  /** Liveness, backend, readout and layout status (GET /healthz; never needs the API key). */
  health(options: RequestOptions = {}): Promise<WithMeta<HealthResponse>> {
    return this.request("GET", "/healthz", undefined, options.signal);
  }

  /** Model aliases (GET /v1/models). */
  models(options: RequestOptions = {}): Promise<WithMeta<ModelsResponse>> {
    return this.request("GET", "/v1/models", undefined, options.signal);
  }

  private async request<T extends object>(
    method: "GET" | "POST",
    path: string,
    body: unknown,
    signal?: AbortSignal,
  ): Promise<WithMeta<T>> {
    const fetchImpl = this.fetchImpl ?? (globalThis as { fetch?: FetchLike }).fetch;
    if (typeof fetchImpl !== "function") {
      throw new TypeError("no fetch implementation available: pass options.fetch");
    }
    const url = this.baseUrl + path;
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    let payload: string | undefined;
    if (body !== undefined) {
      payload = JSON.stringify(body);
      headers["Content-Type"] = "application/json";
    }
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    const onAbort = (): void => controller.abort();
    if (signal?.aborted) controller.abort();
    else signal?.addEventListener("abort", onAbort, { once: true });
    const fail = (err: unknown): unknown => {
      if (timedOut) return new TezError(0, "timeout", `${url} did not answer within ${this.timeoutMs} ms`, { cause: err });
      if (signal?.aborted) return signal.reason ?? err;
      return new TezError(0, "network_error", `cannot reach ${url}: ${describe(err)}`, { cause: err });
    };
    try {
      let response: FetchResponseLike;
      let text: string;
      try {
        const init: FetchInitLike = { method, headers, signal: controller.signal, redirect: "manual" };
        if (payload !== undefined) init.body = payload;
        response = await fetchImpl(url, init);
        text = await response.text();
      } catch (err) {
        throw fail(err);
      }
      const meta = responseMeta(response.status, response.headers);
      if (response.type === "opaqueredirect" || (response.status >= 300 && response.status < 400)) {
        const location = response.headers.get("location");
        throw new TezError(
          response.status,
          "redirect",
          `${url} answered with a redirect${location ? ` to ${location}` : ""}; the client does not follow ` +
            "redirects, so set baseUrl to the server itself",
          { meta },
        );
      }
      let data: unknown;
      try {
        data = text ? JSON.parse(text) : undefined;
      } catch {
        data = undefined;
      }
      if (response.status >= 400) throw errorFromResponse(response, data, text, meta);
      if (!isObject(data)) {
        throw new TezError(response.status, "invalid_response", `${url} did not answer with a JSON object`, { meta });
      }
      return withMeta(data as T, meta);
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
  }
}

function errorFromResponse(response: FetchResponseLike, data: unknown, text: string, meta: ResponseMeta): TezError {
  let type: string | undefined;
  let message: string | undefined;
  if (isObject(data)) {
    const err = data.error;
    if (isObject(err)) {
      if (typeof err.type === "string") type = err.type;
      if (typeof err.message === "string") message = err.message;
    } else if (typeof err === "string") {
      message = err;
    } else if (typeof data.detail === "string") {
      message = data.detail;
    }
  }
  message = message || text.trim().slice(0, 300) || response.statusText || `HTTP ${response.status}`;
  return new TezError(response.status, type ?? STATUS_TYPES[response.status] ?? "http_error", message, {
    body: data ?? (text || undefined),
    meta,
  });
}
