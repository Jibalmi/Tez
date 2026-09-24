import { TezError } from "./errors.js";
import type { TezErrorType } from "./errors.js";
import { responseMeta, withMeta } from "./meta.js";
import type { HeadersLike, ResponseMeta, WithMeta } from "./meta.js";
import type {
  AnswersFor,
  DecideRequest,
  DecideResponse,
  FeedbackRequest,
  FeedbackResponse,
  GateOptions,
  HealthResponse,
  ModelsResponse,
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

export interface DecideOptions<Q extends Questions = Questions> extends RequestOptions {
  /** Question id -> question. Optional when `schema` names a loaded schema (its questions are used). */
  questions?: Q;
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
  /** Any string; the server's default is "tez-latest". Sent only when given. */
  model?: string;
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

function decideBody(state: State, options: DecideOptions<Questions>): DecideRequest {
  const body: DecideRequest = { state };
  if (options.model !== undefined) body.model = options.model;
  if (options.questions !== undefined) body.questions = options.questions;
  if (options.schema !== undefined) body.schema = options.schema;
  const tez: TezOptions = {};
  if (options.readout !== undefined && options.readout !== "auto") tez.readout = options.readout;
  if (options.abstain) tez.abstain = true;
  if (options.alpha !== undefined) tez.gate = { alpha: options.alpha };
  else if (options.gate !== undefined) tez.gate = options.gate;
  if (Object.keys(tez).length > 0) body.tez = tez;
  return body;
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
 * headers in a non-enumerable `meta` property. Tez's extensions (`schema` and the `tez` options) are sent only when
 * set, so a Jev-only server sees a plain request. Redirects are never followed.
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
    return this.request("POST", "/v1/systemone", decideBody(state, options), options.signal);
  }

  /** POST a raw wire-format body to /v1/systemone. */
  systemone(body: DecideRequest, options: RequestOptions = {}): Promise<WithMeta<DecideResponse>> {
    return this.request("POST", "/v1/systemone", body, options.signal);
  }

  /** The schemas the server loaded (GET /v1/schemas). */
  schemas(options: RequestOptions = {}): Promise<WithMeta<SchemasResponse>> {
    return this.request("GET", "/v1/schemas", undefined, options.signal);
  }

  /** One schema: questions, probe status, calibration (GET /v1/schemas/{name}). */
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

  /** Liveness, backend and readout status (GET /healthz; never needs the API key). */
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
