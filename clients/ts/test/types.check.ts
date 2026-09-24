// Compile-time checks of the public types (`npm run typecheck`); nothing here runs.
import { EscalationRequired, requestBody, TezClient, TezError } from "../src/index.js";
import type {
  Answer,
  BatchResult,
  ChoiceAnswer,
  DecideRequest,
  DecideResponse,
  ExtractedObject,
  ExtractResult,
  GateDecision,
  HealthResponse,
  Layout,
  LayoutOption,
  NoulAnswer,
  PlanRequest,
  PlanResponse,
  Questions,
  ReadLayout,
  ResponseMeta,
  ScoreAnswer,
  TezErrorType,
} from "../src/index.js";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;

export async function answersAreTypedPerQuestion(client: TezClient): Promise<void> {
  const res = await client.decide("Help! My payouts have been failing for 3 days.", {
    questions: {
      urgent: { type: "noul", instructions: "Does this convey urgency?" },
      topic: { type: "choice", instructions: "What is it about?", criteria: { billing: "Payments", technical: null } },
      anger: { type: "score", instructions: "How upset is the writer?", criteria: ["Calm", "Angry"] },
    },
  });
  const urgent: NoulAnswer = res.answers.urgent;
  const anger: ScoreAnswer = res.answers.anger;
  const topic: ChoiceAnswer<"billing" | "technical"> = res.answers.topic;
  const labels: Equal<typeof topic.choice, "billing" | "technical" | "__none__"> = true;
  const p: number = res.answers.topic.probabilities.billing;
  const none: number | undefined = res.answers.topic.probabilities.__none__;
  const latency: number | undefined = res.tez?.latency_ms;
  const decision: "act" | "escalate" | undefined = res.tez?.questions["topic"]?.decision;
  // @ts-expect-error: there is no such question
  void res.answers.missing;
  // @ts-expect-error: "sales" is not one of the question's labels
  const wrong: typeof topic.choice = "sales";
  void [urgent, anger, labels, p, none, latency, decision, wrong];
}

export async function schemaOnlyAnswersAreTheUnion(client: TezClient): Promise<void> {
  const res = await client.decide({ text: "hi", customer: "Ana" }, { schema: "support-triage", alpha: 0.05 });
  const a = res.answers["topic"];
  const union: Equal<typeof a, Answer | undefined> = true;
  if (a?.type === "choice") {
    const label: string = a.choice;
    void label;
  } else if (a?.type === "noul") {
    const pYes: number = a.noul;
    void pYes;
  }
  void union;
}

export function questionsAreChecked(): void {
  // @ts-expect-error: a choice question needs criteria
  const noCriteria: Questions = { t: { type: "choice", instructions: "Pick one" } };
  // @ts-expect-error: the type must be noul, choice or score
  const badType: Questions = { t: { type: "maybe", instructions: "x" } };
  void [noCriteria, badType];
}

export async function statesAndOptionsAreChecked(client: TezClient): Promise<void> {
  // @ts-expect-error: a state is a string, an object or an array
  await client.decide(42, { schema: "support-triage" });
  // @ts-expect-error: readout is auto, letters or probe
  await client.decide("x", { schema: "s", readout: "magic" });
  await client.decide(["a", "b"], { schema: "s", gate: false, readout: "probe", abstain: true });
}

export function errorsCarryStatusAndType(e: unknown): void {
  if (e instanceof TezError) {
    const status: number = e.status;
    const type: TezErrorType = e.type;
    const known: boolean = e.type === "timeout" || e.type === "backend_unavailable";
    void [status, type, known];
  }
}

export async function resultsKeepTheirOldTypesAndGainMeta(client: TezClient): Promise<void> {
  const res = await client.decide("x", { questions: { t: { type: "noul", instructions: "x" } } });
  const plain: DecideResponse = res;                                // the old return types still accept the results
  const health: HealthResponse = await client.health();
  const meta: ResponseMeta = res.meta;
  const runId: string | null = res.meta.runId;
  const layout: ReadLayout | null = res.meta.layout;
  const tezMs: number | undefined = res.meta.timing.tez;
  const other: number | undefined = res.meta.timing["cdn"];
  const temperature: number | undefined = res.tez?.questions["t"]?.temperature;
  const readIn: Layout | undefined = res.tez?.questions["t"]?.layout;
  const values: ExtractedObject | undefined = res.tez?.values;
  const cached: boolean | undefined = res.tez?.cached;
  const serverLayout: LayoutOption = health.layout;
  // @ts-expect-error: meta is read-only
  res.meta = meta;
  void [plain, runId, layout, tezMs, other, temperature, readIn, values, cached, serverLayout];
}

export async function batchesAreTypedPerQuestion(client: TezClient): Promise<void> {
  const questions = { urgent: { type: "noul", instructions: "Urgent?" } } as const;
  const batch = await client.decideBatch(["a", { text: "b" }], { questions, layout: "state_first" });
  const runId: string = batch.tez.run_id;
  for (const result of batch.results) {
    if ("error" in result) {
      const type: TezErrorType = result.error.type;
      void type;
    } else {
      const urgent: NoulAnswer = result.answers.urgent;
      void urgent;
    }
  }
  const results: BatchResult<{ urgent: NoulAnswer }>[] = await client.decideMany(["a"], { questions });
  // @ts-expect-error: a batch takes an array of states, not one state
  await client.decideBatch("one state", { questions });
  // @ts-expect-error: layout is auto, question_first or state_first
  await client.decide("x", { questions, layout: "sideways" });
  void [runId, results];
}

export async function plansAndBodies(client: TezClient): Promise<void> {
  const body: DecideRequest = requestBody("x", { schema: "s", layout: "auto", jsonSchema: { type: "object" } });
  const planBody: PlanRequest = requestBody(undefined, { schema: "s" });
  // @ts-expect-error: without a state it is a plan body, not a decision body
  const noState: DecideRequest = requestBody(undefined, { schema: "s" });
  const plan = await client.plan(planBody);
  const plainPlan: PlanResponse = plan;
  const fit: "ready" | "stale" | "none" | undefined = plan.questions["q"]?.fit.status;
  const read: ReadLayout = plan.layout;
  const fb = await client.feedback({ schema: "s", question: "q", state: "x", label: "a", run_id: plan.meta.requestId ?? "" });
  void [body, noState, plainPlan, fit, read, fb.label];
}

export async function extractIsTypedByItsOptions(client: TezClient): Promise<void> {
  const ticket = { type: "object", properties: { urgent: { type: "boolean" } } };
  const values: ExtractedObject = await client.extract("x", ticket);
  const typed: { urgent: boolean } = await client.extract<{ urgent: boolean }>("x", ticket, { alpha: 0.05 });
  const details: ExtractResult<{ urgent: boolean }> = await client.extract<{ urgent: boolean }>("x", "support-triage", {
    returnDetails: true,
  });
  const escalated: string[] = details.escalated;
  const decision: GateDecision | undefined = details.decisions["urgent"];
  // @ts-expect-error: with returnDetails the result is an ExtractResult, not the values
  const wrong: { urgent: boolean } = await client.extract<{ urgent: boolean }>("x", ticket, { returnDetails: true });
  void [values, typed, escalated, decision, wrong];
}

export function newErrorsCarryMeta(e: unknown): void {
  if (e instanceof TezError) {
    const runId: string | null | undefined = e.meta?.runId;
    void runId;
  } else if (e instanceof EscalationRequired) {
    const fields: string[] = e.fields;
    const runId: string | null = e.response.meta.runId;
    void [fields, runId];
  }
}
