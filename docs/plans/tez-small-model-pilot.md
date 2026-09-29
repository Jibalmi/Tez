# The small-model pilot: G1 decision

Written 2026-09-29 from the three rounds run 27–28 September. The plan is [tez-small-model.md](tez-small-model.md);
this is what happened when it met data. Every number is from a committed results directory; the paired intervals are
whole-row bootstraps (2,000 resamples, seed 13) over identical rows.

**Decision: G1 is not met, by any arm. Option A (the encoder) is dead: it sits near chance on typed decisions.
Option C (a Qwen3.5 student) is the direction, and at 2B it matches Decider 0.8B on typed decisions and comes
within noise of it on multilingual intent, but it fails the speed line and is 2.8 points short of G1's accuracy
floor. M2 as planned (20,000 schemas, 1M decisions for the encoder) is not justified. What is justified is narrower
and cheaper, and is listed at the end.**

## 1. The arms

| Arm | Student | Training decisions | Where | Cost |
|---|---|---:|---|---:|
| A-base-10k / 30k | mmBERT-base, option-span head | 10,000 / 29,990 | H200 | in $6.25 |
| A-small-30k | mmBERT-small | 29,990 | H200 | " |
| C-0.8B-30k | Qwen3.5-0.8B, LoRA r 16 | 29,990 | H200 | " |
| C-0.8B-cont | the same, 150 more registered updates on 2,400 decisions | 29,990 | H100 + laptop | $3.86 |
| C-0.8B-v2 | Qwen3.5-0.8B, LoRA r 16 | 155,835 (5.2×) | H100 | ~$21 incl. two lost Pods |
| **C-2B-v2** | **Qwen3.5-2B, LoRA r 16** | 155,835 | H100 | $11.16 |

All students distil the same Gemma 4 12B Q8 teacher (letter distributions, both option orders, T from §5c).
Validation is the same 278 states / 1,390 decisions throughout; the added v2 states are training-only.
Evaluation is the frozen suite of the 30k run, scored identically, with the 12B, Laya and Decider on the same rows.

## 2. G1, criterion by criterion

The plan's G1 is for A-base. It is applied here to every arm so the decision is about the pilot, not one model.

| G1 criterion | threshold | A-base-30k | C-0.8B-30k | C-0.8B-v2 | **C-2B-v2** | Decider 0.8B |
|---|---|---:|---:|---:|---:|---:|
| Typed-decisions accuracy | ≥ 0.563 | 0.360 | 0.502 | 0.522 | **0.536** | 0.534 |
| Typed-decisions soft accuracy | ≥ 0.46 | 0.339 | 0.387 | 0.383 | **0.404** | 0.425 |
| Above best Laya on held-out workflows (0.659 / 0.805 / 0.535) | 2 of 3 | 2 of 3 | 3 of 3 | 3 of 3 | **3 of 3** | 3 of 3 |
| MASSIVE-11 macro | ≥ 0.70 | 0.527 | 0.641 | 0.629 | **0.727** | 0.745 |
| Khmer (unseen) above 3× random (0.15) | yes | 0.23 | 0.32 | 0.26 | 0.27 | 0.41 |
| Retention: 30k ≥ 10k + 3 points | yes | +4.9 (0.462→0.511) | n/a | +2.9 at 5.2× data | +1.9 over 0.8B-v2 | — |
| Speed: ≤ 30 ms / 1 q, ≤ 2.8 ms / q at 50 | yes | 12.5 / 0.28 (H200) | 60.4 / 5.5 (H200) | not timed | not timed; larger | 30.5 / 1.65 (H200, eager) |
| ECE-15 | ≤ 0.15 | 0.126 | 0.061 | 0.070 | **0.055** | 0.134 |
| **Passes** | all 8 | 4 | 5 | 4 | **6** | — |

Speed lines are H200 figures; the plan's gate is on the laptop GPU, where every timing attempt hit the thermal
requirement first (a heat-soaked shared machine, [gpu notes](../../results/pilot-laptop-20260927/README.md)).

## 3. What the three rounds established

**Capacity beat data.** Same recipe, same evaluation:

| Lever | Typed | MASSIVE-11 | Validation agreement with teacher |
|---|---:|---:|---:|
| 5.2× data (0.8B, 30k → 155k) | +2.05 (−0.15 to +4.30) | −1.2 (pooled interval spans 0) | 0.7835 → 0.8158 |
| 2.5× parameters (0.8B → 2B, 155k) | +1.35 (−0.70 to +3.60) | **+9.82 (+7.27 to +12.36)** | 0.8158 → 0.8345 |

Both levers improve the distillation objective; only capacity moved the multilingual public task, and neither
moved typed decisions past noise. The 150-step continuation on the original data was a clean negative under a
preregistered selector (mean KL rose at every checkpoint; step 0 retained).

**Against Decider 0.8B**, the 2B student is at parity on typed decisions (+0.15, −2.35 to +2.65), 1.7 points behind
on MASSIVE-11 (−4.09 to +0.64), ahead on routing (89.2 vs 84.5) and SemIf authored (77.8 vs 77.1), and clearly
behind only on the content-safety and retrieval applications: toxicity −11.8, spam −5.5, phishing −13.3, RAG −11.
Those are task families, not a scale effect, and they point at Decider's training exposure. The student is the
better-calibrated system on every task (typed ECE 0.055 against 0.134).

**Zero-shot transfer exists.** The plan's open question was whether any small student answers on schemas it has
never seen. The 2B keeps 0.761 of the 12B's typed-decision accuracy on entirely held-out workflows, Decider 0.759.
Laya's best is 0.362 (retention 0.51). So the question is answered yes; the remaining question is how much.

**The cascade already works where the student is strong.** Simulated on the same rows (student answers when its
top probability ≥ τ, else the 12B; τ chosen on the test rows, so exploratory, not a G4 result):

| Suite | 12B alone | 2B + 12B at τ 0.8 | student-answered |
|---|---:|---:|---:|
| MASSIVE-11 | 88.5 | 88.1 | 54 % |
| applications (7) | 77.6 | 76.9 | 50 % |
| typed decisions | 70.4 | 70.2 | 10 % |

That is G4's shape (within 1 point, student answering ≥ 50 %) on the two suites where the student is within
about 10 points of the teacher, and nowhere near it on typed decisions, where the student's confident answers are
few. A fitted threshold on the validation split, and a laptop timing of the cascade, are the missing measurements.

## 4. Why the plan's option A failed

The plan chose the encoder for its speed (12 ms per question, 0.27 ms per question at 50 on the H200; the only
arm inside the speed line) and its multilingual pretraining. It is as fast as promised and as accurate as Laya:
0.360 on typed decisions against Laya's 0.362, 0.527 on MASSIVE against laya-multilingual's 0.524. Ten thousand
more teacher-labelled decisions moved it 3.5 points. Distilling a decoder's letter distributions into an
option-span head over packed questions does not transfer to unseen schemas at this size; the decoder that shares
the teacher's prompt format does. The plan's §3 verdict is reversed by the data.

## 5. Costs

RunPod, all rounds: about $38 of GPU time, of which roughly $7 was lost to a credit-exhausted stop, an idle
Pod after an interrupted session, and two Pod restarts to retrieve results. Laptop GPU time was dominated by
thermal stops; the registered continuation lost three attempts to them before moving to the cloud. See
[runpod-usage notes](pilot-cloud-compute.md).

## 6. What would justify spending again

In the order of expected return per dollar, given what is measured:

1. **Fit and time the cascade** ($0 to $2). Fit τ per question type on the validation split, then measure the
   2B + 12B cascade end to end on the laptop under the idle runner. This turns the simulated G4 rows into a result
   and is the only path that ships something: the 12B's accuracy at about half its calls on intent and applications.
2. **Targeted data for the four weak families** (about $15). Decider's lead is confined to toxicity, spam,
   phishing and RAG relevance. Sampling states from those domains (the pool's domain filter currently removes
   them as too near the evaluation centroids, which is the leakage guard working) needs a separate, held-out-checked
   pool and teacher labels. Expect the application gap to close; expect nothing on typed decisions.
3. **A 4B student** (about $25). Capacity gave +9.8 multilingual for 2.5× parameters. Qwen3.5-4B, pinned 851bf6e8,
   is the next step on that curve, but it moves the student further from the speed line, and the 4B probe already
   measured 19.6 ms per word in the voice loop. Only worth it if the cascade, not the student, is the product.
4. **Not worth it:** more generic web-state data at 0.8B; further registered continuation on the 2,400-decision
   set; the encoder at any data size measured here.

The G3 speed gate is untouched for the Qwen family. The optimised-inference work (CUDA graphs, prefix cache)
failed parity on the 0.8B by about as much as batch shape alone moves BF16 probabilities
([native-single control](../../results/pilot-qwen-native-single-20260927/README.md)); that suggests loosening the
mean-total-variation check to what batch shape already produces, then re-testing the graph path, before any
speed number is claimed.
