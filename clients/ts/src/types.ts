/**
 * Types of the /v1/systemone wire format (docs/API.md in the Tez repository): TypeSafe's Jev format, plus Tez's
 * optional extensions (`schema`, the request's `tez` options and the response's `tez` block).
 */

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
}

/** Tez's response block. A server that speaks only Jev's format leaves it out. */
export interface TezBlock {
  latency_ms: number;
  questions: Record<string, TezQuestionMeta>;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

/** The body of POST /v1/systemone. */
export interface DecideRequest {
  state: State;
  questions?: Questions;
  /** Any string; the server's default alias is "tez-latest". */
  model?: string;
  /** Tez: a loaded schema; its questions are used when `questions` is left out. */
  schema?: string;
  /** Tez options. */
  tez?: {
    readout?: ReadoutOption;
    abstain?: boolean;
    gate?: GateOptions | null | false;
  };
}

/** The response of POST /v1/systemone. */
export interface DecideResponse<A extends Record<string, Answer> = Record<string, Answer>> {
  model: string;
  answers: A;
  usage: Usage;
  tez?: TezBlock;
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
}

export interface SchemaSummary {
  name: string;
  description: string;
  questions: Record<string, QuestionType>;
  calibration_id: string | null;
  probes: string[];
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
}

/** The label comes back in its stored form: "true"/"false", a level number, or the option label. */
export interface FeedbackResponse {
  ok: true;
  schema: string;
  question: string;
  label: string | number;
}
