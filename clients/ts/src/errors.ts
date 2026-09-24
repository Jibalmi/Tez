/**
 * Error types: the server's (docs/API.md, "Errors") and the client's own, for failures without a usable response.
 */
export type TezErrorType =
  | "unauthorized"
  | "invalid_request"
  | "not_found"
  | "method_not_allowed"
  | "backend_unavailable"
  | "internal_error"
  /** No answer within timeoutMs (status 0). */
  | "timeout"
  /** The server could not be reached (status 0). */
  | "network_error"
  /** The server answered with a redirect, which the client never follows. */
  | "redirect"
  /** A success status without a JSON object body. */
  | "invalid_response"
  /** An error status without the wire format's error body. */
  | "http_error"
  | (string & {});

/**
 * Every failure the client reports, except a cancellation through your own AbortSignal (that rejects with the
 * signal's reason, an AbortError).
 *
 * `status` is the HTTP status, or 0 when no response arrived. `type` and `message` come from the error body
 * `{"error": {"type": ..., "message": ...}}` when there is one; `body` holds the parsed body.
 */
export class TezError extends Error {
  readonly status: number;
  readonly type: TezErrorType;
  readonly body?: unknown;
  cause?: unknown;

  constructor(status: number, type: TezErrorType, message: string, options: { cause?: unknown; body?: unknown } = {}) {
    super(message);
    this.name = "TezError";
    this.status = status;
    this.type = type;
    if (options.body !== undefined) this.body = options.body;
    if (options.cause !== undefined) this.cause = options.cause;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
