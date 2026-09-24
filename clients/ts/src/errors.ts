/**
 * Error types: the server's (docs/API.md, "Errors") and the client's own, for failures without a usable response.
 */
import type { ResponseMeta, WithMeta } from "./meta.js";
import type { DecideResponse } from "./types.js";

export type TezErrorType =
  | "unauthorized"
  /** A POST (or a CORS preflight) from a browser origin the server does not allow (403). */
  | "forbidden"
  | "invalid_request"
  | "not_found"
  | "method_not_allowed"
  /** A request over the server's limits: body size, questions, state length, batch size (413). */
  | "payload_too_large"
  | "backend_unavailable"
  /** An unexpected failure on the server, such as a hook that raised (500). */
  | "internal_error"
  /** No answer within timeoutMs (status 0). */
  | "timeout"
  /** The server could not be reached (status 0). */
  | "network_error"
  /** The server answered with a redirect, which the client never follows. */
  | "redirect"
  /** A success status without the JSON object body the endpoint defines. */
  | "invalid_response"
  /** An error status without the wire format's error body. */
  | "http_error"
  | (string & {});

/**
 * Every failure the client reports, except a cancellation through your own AbortSignal (that rejects with the
 * signal's reason, an AbortError).
 *
 * `status` is the HTTP status, or 0 when no response arrived. `type` and `message` come from the error body
 * `{"error": {"type": ..., "message": ...}}` when there is one; `body` holds the parsed body. `meta` holds the
 * response's status and headers (the request id, and the run id of a decision that reached the engine) when a
 * response arrived.
 */
export class TezError extends Error {
  readonly status: number;
  readonly type: TezErrorType;
  readonly body?: unknown;
  readonly meta?: ResponseMeta;
  cause?: unknown;

  constructor(
    status: number,
    type: TezErrorType,
    message: string,
    options: { cause?: unknown; body?: unknown; meta?: ResponseMeta } = {},
  ) {
    super(message);
    this.name = "TezError";
    this.status = status;
    this.type = type;
    if (options.body !== undefined) this.body = options.body;
    if (options.meta !== undefined) this.meta = options.meta;
    if (options.cause !== undefined) this.cause = options.cause;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * `extract` without `returnDetails`, when a gate applied (`alpha`, or a named schema's own gate) and escalated at least
 * one field: the extracted object is not certified. The request itself succeeded, so this is not a TezError. `fields` names the escalated fields,
 * `values` holds the model's best guess and `response` the whole decision (with its `meta`). Hand the case to a
 * person or a larger model, or pass `returnDetails: true` to handle it yourself.
 */
export class EscalationRequired<V = unknown> extends Error {
  readonly fields: string[];
  readonly values: V;
  readonly response: WithMeta<DecideResponse>;

  constructor(fields: readonly string[], values: V, response: WithMeta<DecideResponse>) {
    super(
      `the gate escalated ${fields.join(", ")}: no certified value (hand the case to a person or a larger model, or ` +
        "use returnDetails: true)",
    );
    this.name = "EscalationRequired";
    this.fields = [...fields];
    this.values = values;
    this.response = response;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
