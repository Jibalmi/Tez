/**
 * What a response says besides its body: the status and the headers Tez sets on every response (docs/API.md,
 * "Response headers"). Every result the client returns carries it as a non-enumerable `meta` property, so the body
 * itself (its keys, JSON.stringify, a spread) stays exactly the wire format's.
 */
import type { ReadLayout } from "./types.js";

/** The part of a Headers object the client reads. Names are matched without regard to case. */
export interface HeadersLike {
  get(name: string): string | null;
}

/** `server-timing`, parsed: metric name -> duration in milliseconds. */
export interface ServerTiming {
  /** The whole request inside the server (every response). */
  total?: number;
  /** The engine (decisions). */
  tez?: number;
  /** Time spent in model calls (a single decision that called the backend). */
  backend?: number;
  /** Any other metric with a duration, from a proxy for example. */
  [metric: string]: number | undefined;
}

export interface ResponseMeta {
  /** The HTTP status. */
  readonly status: number;
  /**
   * `x-typesafe-request-id`: 32 hex characters on every Tez response. For a decision it is the run id; for other
   * routes a fresh id. Null when the server sent none.
   */
  readonly requestId: string | null;
  /**
   * `X-Tez-Run-Id`: the decision's run id, what hooks see, a DecisionLog row records and `feedback({ run_id })`
   * accepts. Sent for decisions and batches that reached the engine, successful or not; null otherwise. Item i of a
   * batch was decided with run id `<runId>.<i>`.
   */
  readonly runId: string | null;
  /**
   * `X-Tez-Layout`: the layout a single decision's questions were read in, or "mixed" (each question's `tez` block
   * then names its own). Null for anything but a decision that was read.
   */
  readonly layout: ReadLayout | null;
  /** `server-timing`, parsed (empty when the header is missing). */
  readonly timing: ServerTiming;
  /** All the response's headers. */
  readonly headers: HeadersLike;
}

/** A response body with its ResponseMeta, as every client method returns it. */
export type WithMeta<T> = T & { readonly meta: ResponseMeta };

/** Parse a `server-timing` header: `tez;dur=2.3, backend;dur=1.0, total;dur=4.2` -> { tez: 2.3, backend: 1, total: 4.2 }. */
export function parseServerTiming(header: string | null | undefined): ServerTiming {
  const out: ServerTiming = {};
  if (!header) return out;
  // Split on commas outside quoted strings (a metric's desc="..." may contain commas or semicolons).
  for (const entry of header.match(/(?:[^,"]|"(?:[^"\\]|\\.)*")+/g) ?? []) {
    const params = entry.match(/(?:[^;"]|"(?:[^"\\]|\\.)*")+/g) ?? [];
    const name = params[0]?.trim();
    if (!name || Object.prototype.hasOwnProperty.call(out, name)) continue;
    for (const param of params.slice(1)) {
      const m = /^\s*dur\s*=\s*"?([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)"?\s*$/i.exec(param);
      if (m?.[1] !== undefined) {
        out[name] = Number(m[1]);
        break;
      }
    }
  }
  return out;
}

const LAYOUTS: readonly string[] = ["question_first", "state_first", "mixed"];

/** The ResponseMeta of a response. */
export function responseMeta(status: number, headers: HeadersLike): ResponseMeta {
  const header = (name: string): string | null => {
    const value = headers.get(name);
    return value === null || value === undefined || value.trim() === "" ? null : value.trim();
  };
  const layout = header("x-tez-layout");
  return {
    status,
    requestId: header("x-typesafe-request-id"),
    runId: header("x-tez-run-id"),
    layout: layout !== null && LAYOUTS.includes(layout) ? (layout as ReadLayout) : null,
    timing: parseServerTiming(headers.get("server-timing")),
    headers,
  };
}

/** Attach `meta` to a parsed body without making it one of the body's own keys. */
export function withMeta<T extends object>(body: T, meta: ResponseMeta): WithMeta<T> {
  Object.defineProperty(body, "meta", { value: meta, enumerable: false, writable: false, configurable: true });
  return body as WithMeta<T>;
}
