/**
 * Types of the /v1/systemone wire format (docs/API.md in the Tez repository): TypeSafe's Jev format, plus Tez's
 * optional extensions (`schema`, `json_schema`, the request's `tez` options and the response's `tez` block) and Tez's
 * own endpoints (batch, plan, schemas, feedback).
 */
import type { TezErrorType } from "./errors.js";

/** What the model decides on: a string, or an object or array (the server serialises it as JSON). */
export type State = string | object;

/** What to decide, as text, or as an object or array (serialised as JSON). */
export type Instructions = string | object;

export type QuestionType = "noul" | "choice" | "score";

/** Yes/no. The answer's `noul` is P(yes). */
export interface NoulQuestion {
  type: "noul";
  instructions: Instructions;
  /** Optional descriptions of the two answers. */
  criteria?: { true?: string | null; false?: string | null } | null;
}

/** One of 2 to 255 options: `criteria` maps each label to a description, or to null. */
export interface ChoiceQuestion<L extends string = string> {
  type: "choice";
  instructions: Instructions;
  criteria: Record<L, string | null>;
}

/** An ordered scale of 2 to 10 levels: `criteria[i]` describes level i. */
export interface ScoreQuestion {
  type: "score";
  instructions: Instructions;
  criteria: readonly string[];
}

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

/** Question id -> question. */
export type Questions = Record<string, Question>;

/** The implicit option `tez.abstain` adds to every choice question: "none of these fits". */
export type NoneLabel = "__none__";

/** noul: P(yes). Jev's noul answers carry no confidence. */
export interface NoulAnswer {
  type: "noul";
  noul: number;
}

/** choice: the most likely label, every label's probability (they sum to 1) and (k * p_max - 1) / (k - 1). */
export interface ChoiceAnswer<L extends string = string> {
  type: "choice";
  choice: L | NoneLabel;
  probabilities: Record<L, number> & { __none__?: number };
  confidence: number;
}

/** score: the expected level sum(i * p_i), the legend and probabilities keyed by level ("0", "1", ...), confidence. */
export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

/** The answer type for one question type; choice answers keep the question's labels. */
export type AnswerFor<Q> = Q extends { type: "noul" }
  ? NoulAnswer
  : Q extends { type: "choice"; criteria: infer C }
    ? ChoiceAnswer<Extract<keyof C, string>>
    : Q extends { type: "score" }
      ? ScoreAnswer
      : Answer;

/** Question id -> answer, for a set of questions. */
export type AnswersFor<Q extends Questions> = { [K in keyof Q]: AnswerFor<Q[K]> };

/** How a question was read. */
export type Readout = "letters" | "probe";

/** `tez.readout`: auto uses a trained probe where one is current, letters otherwise; probe is strict (422 without one). */
export type ReadoutOption = "auto" | Readout;

export type GateDecision = "act" | "escalate";

/** A prompt layout (docs/API.md, "Prompt layout"): the question before the state, or the state before the question. */
export type Layout = "question_first" | "state_first";

/**
 * `tez.layout`: auto reads two or more questions state_first and a single one question_first, and a fitted question
 * keeps the layout it was fitted under; the other two apply to every question. Default: the schema's `layout:`, else
 * the server's `--layout`, else auto.
 */
export type LayoutOption = "auto" | Layout;

/** The layout a request's questions were read in: the one they share, or "mixed" (`X-Tez-Layout`, a plan's `layout`). */
export type ReadLayout = Layout | "mixed";

/**
 * A JSON schema of the object to extract (docs/API.md, "Structured extraction"): an object with `properties`, whose
 * fields are enums, booleans, small integer ranges or nested objects.
 */
export type JsonSchema = object;

/** One extracted value: an enum value as given, a boolean, an integer, null, or a nested object. */
export type ExtractedValue = string | number | boolean | null | ExtractedObject;

/** The object extracted for a `json_schema` request (`tez.values`). */
export interface ExtractedObject {
  [field: string]: ExtractedValue;
}

/** `tez.gate`: {alpha} targets that error rate among acted decisions; {} uses the schema's default; null/false: off. */
export interface GateOptions {
  alpha?: number;
}

/** Tez's per-question block in a response. */
export interface TezQuestionMeta {
  readout: Readout;
  /** Present only when a gate applies. `act` always comes with p_correct and calibration_id. */
  decision?: GateDecision;
  /** Present when a fitted calibration exists. */
  p_correct?: number;
  calibration_id?: string;
  /**
   * The default temperature an unfitted letters answer was read at (docs/API.md, "Default temperature"). Absent when
   * a fit calibrated the answer, for a probe answer, and at temperature 1.
   */
  temperature?: number;
  /**
   * The layout this question was read in. Present only when the request's questions were not all read in the same
   * layout (`X-Tez-Layout: mixed`); otherwise they were all read in the `X-Tez-Layout` layout.
   */
  layout?: Layout;
}

/** Tez's response block. A server that speaks only Jev's format leaves it out. */
export interface TezBlock {
  latency_ms: number;
  /** Question id -> Tez's block for it; for a `json_schema` request the ids are the fields' dotted paths. */
  questions: Record<string, TezQuestionMeta>;
  /** The extracted object, for a request with `json_schema`. */
  values?: ExtractedObject;
  /** True when a Cache hook answered a repeated decision from memory (usage is then zero). */
  cached?: boolean;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

/** The request's `tez` options. */
export interface TezOptions {
  readout?: ReadoutOption;
  abstain?: boolean;
  gate?: GateOptions | null | false;
  layout?: LayoutOption;
}

/** The body of POST /v1/systemone. */
export interface DecideRequest {
  state: State;
  questions?: Questions;
  /** Tez: in place of `questions`, a JSON schema of the object to extract (returned in `tez.values`). */
  json_schema?: JsonSchema;
  /** Any string; the server's default alias is "tez-latest". */
  model?: string;
  /** Tez: a loaded schema; its questions are used when neither `questions` nor `json_schema` is given. */
  schema?: string;
  /** Tez options. */
  tez?: TezOptions;
}

/** The response of POST /v1/systemone. */
export interface DecideResponse<A extends Record<string, Answer> = Record<string, Answer>> {
  model: string;
  answers: A;
  usage: Usage;
  tez?: TezBlock;
}

/** The body of POST /v1/systemone/batch: a /v1/systemone request with `states` in place of `state`. */
export interface BatchRequest {
  states: State[];
  questions?: Questions;
  json_schema?: JsonSchema;
  model?: string;
  schema?: string;
  tez?: TezOptions;
}

/** A batch result for a state that failed. One state's error never fails the others. */
export interface BatchItemError {
  error: { type: TezErrorType; message: string };
}

/** One state's result in a batch: its /v1/systemone response, or its error. Narrow with `"error" in result`. */
export type BatchResult<A extends Record<string, Answer> = Record<string, Answer>> = DecideResponse<A> | BatchItemError;

/** The response of POST /v1/systemone/batch. */
export interface BatchResponse<A extends Record<string, Answer> = Record<string, Answer>> {
  model: string;
  /** One entry per state, in input order. */
  results: BatchResult<A>[];
  /** Summed over the decided states. */
  usage: Usage;
  /** `run_id` is the batch's (also `X-Tez-Run-Id`); state i was decided with run id `<run_id>.<i>`. */
  tez: { latency_ms: number; run_id: string };
}

/** The body of POST /v1/plan: a /v1/systemone request whose `state` may be left out. */
export type PlanRequest = Omit<DecideRequest, "state"> & { state?: State };

/** Backend calls a question would make. */
export interface PlanCalls {
  letters: number;
  embed: number;
}

/** A question's fit in a plan: `ready` or `stale` (with the fit's details), or `none`. */
export interface PlanFit {
  status: "ready" | "stale" | "none";
  /** Why it is stale or not fitted; null when ready. */
  reason: string | null;
  /** The fit's details, present when the question was fitted. */
  calibration_id?: string;
  fit_layout?: Layout;
  letters_calibrated?: boolean;
  probe?: boolean;
}

/** What one question would do. */
export interface PlanQuestion {
  type: QuestionType;
  /** Options shown, `__none__` included. */
  options: number;
  layout: Layout;
  fit: PlanFit;
  /** Null when the question would fail (`tez.readout: "probe"` without a usable probe); `error` then says why. */
  readout: Readout | null;
  error?: string;
  calls: PlanCalls;
  /** The prompt hash `tez fit` records. */
  prompt_sha: string;
  /** Token estimate (characters / 4). */
  prompt_tokens: number;
  /** The prefix the prompt cache would keep from the question read just before it. */
  cached_tokens: number;
}

/** The response of POST /v1/plan: what a /v1/systemone request would do, without calling the model. */
export interface PlanResponse {
  backend: string;
  template: string;
  /** Null until a decision has asked the backend for its model name. */
  model: string | null;
  schema: string | null;
  readout: ReadoutOption;
  requested_layout: LayoutOption;
  /** The layout the questions share, or "mixed" (what `X-Tez-Layout` would say). */
  layout: ReadLayout;
  /** Null when the request has no state. */
  state_tokens: number | null;
  /** The order the questions would be read in (state-first questions first, back to back). */
  order: string[];
  questions: Record<string, PlanQuestion>;
  totals: { calls: PlanCalls; prompt_tokens: number; cached_tokens: number; evaluated_tokens: number };
  notes: string[];
}

export interface ModelInfo {
  name: string;
  description: string;
  release_date: string;
}

/** GET /v1/models */
export interface ModelsResponse {
  models: ModelInfo[];
}

/** GET /healthz. `status` is "degraded" when the model backend cannot be reached. */
export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  backend: string;
  template: string;
  backend_status: string | null;
  model: string | null;
  embed_backend: string | null;
  schemas: string[];
  /** Schema -> questions with a trained, current probe. */
  probes: Record<string, string[]>;
  embed_template?: string;
  embed_backend_status?: string | null;
  /** The server's default prompt layout (`--layout`). */
  layout: LayoutOption;
}

export interface SchemaSummary {
  name: string;
  description: string;
  questions: Record<string, QuestionType>;
  calibration_id: string | null;
  probes: string[];
  /** True for a preset loaded with `tez serve --presets`. */
  builtin: boolean;
}

/** GET /v1/schemas */
export interface SchemasResponse {
  schemas: SchemaSummary[];
}

export interface ProbeStatus {
  probe: "ready" | "stale" | "none";
  letters_calibrated: boolean;
  n_labels: number | null;
  note: string | null;
  /** The layout the question was fitted under; null when it was not fitted. */
  layout: Layout | null;
}

/** GET /v1/schemas/{name} */
export interface SchemaDetail {
  name: string;
  description: string;
  state: string;
  questions: Questions;
  gate: { alpha: number } | null;
  /** Number of labelled examples in the schema file. */
  examples: number;
  /** The schema's own `layout:`, or null. */
  layout: LayoutOption | null;
  /** The layout a request that names none gets: the schema's, else the server's `--layout`. */
  served_layout: LayoutOption;
  /** True for a preset loaded with `tez serve --presets`. */
  builtin: boolean;
  calibration_id: string | null;
  probes: Record<string, ProbeStatus>;
  calibration: Record<string, unknown> | null;
  manifest: Record<string, unknown> | null;
}

/** A label: true/false (noul), a level number (score), or an option label or "__none__" (choice). */
export type Label = string | number | boolean;

/** POST /v1/feedback */
export interface FeedbackRequest {
  schema: string;
  question: string;
  state: State;
  label: Label;
  /** The decision's run id (`meta.runId`, the `X-Tez-Run-Id` header), stored with the row. At most 128 characters. */
  run_id?: string;
}

/** The label comes back in its stored form: "true"/"false", a level number, or the option label (or "__none__"). */
export interface FeedbackResponse {
  ok: true;
  schema: string;
  question: string;
  label: string | number;
}
