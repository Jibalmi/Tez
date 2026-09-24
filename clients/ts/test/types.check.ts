// Compile-time checks of the public types (`npm run typecheck`); nothing here runs.
import { TezClient, TezError } from "../src/index.js";
import type { Answer, ChoiceAnswer, NoulAnswer, Questions, ScoreAnswer, TezErrorType } from "../src/index.js";

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
