# Are the students competitive?

The Qwen student is competitive on routing and its raw probability calibration, but **Decider 0.8B
is the stronger broad small-model reference in this pilot**. Neither has demonstrated all of our
registered release requirements. The encoder's recent rule-training improvement does not close
the public quality gap. Keep the current results experimental.

## Measured quality

Percentages, with each column scored on identical cases across models. These are separate tasks,
not a pooled leaderboard score. Native rendering is retained for released competitors.

| Model | Typed decisions | Intent, 11 languages | Routing | Jailbreak | Toxicity | Fresh rule test |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 12B teacher | 70.40 | 88.45 | 96.99 | 86.50 | 71.25 | Not measured |
| Decider 0.8B | 53.40 | 74.45 | 84.46 | 92.50 | 75.25 | 76.33 |
| Tev1 0.8B experimental | 51.45 | 63.91 | 78.45 | 86.75 | 69.75 | Not measured |
| Our Qwen 0.8B student | 50.15 | 64.09 | 89.22 | 85.75 | 60.25 | 70.50 |
| Kev 0.8B | 49.10 | 61.27 | 84.96 | 89.00 | 64.25 | Not measured |
| GLiNER2.5-Decide, English | 40.30 | 41.55 | 60.90 | 76.00 | 45.75 | Not measured |
| GLiNER2.5-multi-Decide | 39.00 | 48.27 | 50.63 | 74.75 | 54.25 | Not measured |
| Our encoder, 12 epochs | 39.05 | 51.73 | 58.65 | 85.50 | 53.75 | 35.61 |
| Our encoder, rule continuation | 38.80 | 53.18 | 56.89 | 85.50 | 53.50 | 54.67 |

Sources: [full public comparison, including all 16 configurations](COMPARISON.md),
[encoder continuation](../pilot-reasoning-round2-20260927/RESULTS.md), and
[fresh rule comparison](../pilot-reasoning-competitors-20260927/RESULTS.md), and
[complete native GLiNER comparison](../pilot-gliner2-comparison-20260927/RESULTS.md), and
[complete native Tev1 comparison](../pilot-tev1-eval-20260927/RESULTS.md).
Each source links measurement receipts and archived raw predictions.

Decider leads our Qwen student on six of seven application tasks; Qwen leads on routing.
Decider's typed-decision advantage is 3.25 percentage points, with an exploratory paired
state-bootstrap 95% interval of +1.20 to +5.20. Qwen's raw typed ECE is lower: 6.12 points versus
13.39 for Decider. This is calibration on that evaluation set, not a reliability guarantee for
arbitrary inputs. Both systems produce option probability vectors, not just answer labels.

The fresh rule test has 360 states and 1,800 decisions. Its domains and states are new, but logic
families recur in the encoder's added training. The competitor extension was registered after
the encoder confirmation result was observed. It is a useful independent set of states, not a
new public benchmark or evidence of general reasoning mastery.

A [shared-error audit](OVERLAP.md) found that Qwen and Decider both miss 709 of the 2,000 typed
decisions. Even a selector that knew the correct answer could reach only 64.55% by choosing
between their predictions, below the 12B teacher's 70.40%. That oracle uses test labels and is
not deployable. The earlier fixed equal-probability ensemble scored 53.00%, versus Decider's
53.40%. Combining the two may have room to help, but it does not remove the need to improve
their shared errors or use a stronger fallback.

## Speed and release status

On the H200, native eager Decider measured 30.53 ms for one fresh question and 82.61 ms for 50,
compared with Qwen's 43.42 ms and 200.12 ms. These are the recorded implementations on that
hardware; optimized competitor servers were not tested. H200 results cannot satisfy the laptop
latency gate. The latest [laptop full/restricted-readout runs](../pilot-laptop-qwen-readout-20260927/RESULTS.md)
completed with all raw requests retained, but exceeded the registered temperature ceiling and
foreign GPU-memory limit. They establish no clean speedup or laptop gate pass. Earlier cooling
failures also remain preserved.

The encoder's first quality gates include 56.3% typed accuracy and 70% multilingual intent.
The plan separately allows Qwen (arm C) to advance if it exceeds the encoder by at least five
points of teacher-accuracy retention and meets the latency requirement. Its measured quality
lead supports that alternative route, but valid laptop latency remains unverified. The encoder
continuation improved the authored rules but did not improve public typed accuracy. No candidate
is promoted from these results. The full [predeclared gates](../../docs/plans/tez-small-model.md),
including the later clean-student quality requirements, remain unchanged.

## Comparison boundaries and next work

Laya and GLiClass variants are included in the full comparison. Laya-td's 76.65% typed score uses
training on that benchmark's training split; our students excluded the entire dataset. Released
competitors have different training exposure, so utility comparisons do not isolate architecture
or training-method effects. Decider's inspected training registry includes English MASSIVE intent
and toxicity sources; exact released-checkpoint membership and overlap are not reconstructed.
See the [training-exposure audit](../pilot-reasoning-round2-20260927/EXPOSURE.md).

Focus further work on the Qwen/Decider family: preserve complete probability outputs, verify
inference changes against the frozen validation reference, then measure speed under the original
laptop protocol. The [restricted output projection](../pilot-restricted-qwen-20260927/README.md)
passed its validation probability check. The [FLA kernel retry](../pilot-fla-qwen-attempt2-20260927/RESULTS.md)
completed but failed the unchanged maximum-probability and mean-total-variation limits. Keep
the original attention kernels. The [full batch-16 graph trial](../pilot-qwen-graph-validation-20260927/batch-16/RESULTS.md)
also fails its mean-total-variation limit: 0.00623822 versus the required 0.005. Its graph outputs
exactly match the identical uncaptured path, so capture itself adds no observed drift; surrounding
padding and batching changes still need isolation. The failed candidate is not timed or promoted.
Its [single-question arm](../pilot-qwen-graph-validation-20260927/batch-01/README.md) reached the
registered soft deadline after 1,348 of 1,390 decisions; the partial outputs are archived and
cannot establish full parity. The separately registered
[native-prefix batch-16 candidate](../pilot-qwen-prefix-validation-20260927/batch-16/RESULTS.md)
now passes all 1,390 validation decisions: argmax agreement 99.7122%, maximum probability change
0.03446219, and mean total variation 0.00404363. Its captured and uncaptured outputs match exactly.
The single-question arm is separately queued with a longer predeclared resource limit; speed
and release eligibility still require their own measurements.
None of these inference experiments establishes a measured speedup. Larger Decider releases may be useful quality references,
but have not been measured here and would be a different size/cost class. Published JevBench
or DecisionIndex scores use different protocols and cannot be inserted into this table.

Both GLiNER checkpoints are verified on all 6,151 public decisions and 1,100 option-order checks each.
Their native FP32, batch-one runs trail our Qwen student on the five public columns above. They
are not uniformly worse: both score 53.50% on RAG relevance versus Qwen's 51.50%, and English
GLiNER scores 54.92% on the official SemIf perturbation metric versus Qwen's 50.12%.
Multilingual GLiNER improves its English counterpart's mean intent accuracy, but its 17.73%
option-order flip rate still exceeds the later 10% requirement; English GLiNER flips 38.73%.
These results do not measure either GLiNER model's speed or its published server implementation.
The [complete comparison and archived raw predictions](../pilot-gliner2-comparison-20260927/RESULTS.md)
retain every task, including those favoring the competitors. Earlier input-guard failures,
the English memory-driven cancellation and interrupted multilingual runs remain archived.
Windows recorded a hypervisor bugcheck during the second multilingual attempt; causality is
unresolved. The separate batch-one recovery with conservative thermal pacing completed.
[Together's Tev1 0.8B](../pilot-tev1-eval-20260927/RESULTS.md) is now verified on all 6,151 original
decisions and 1,100 option-order checks. Its typed accuracy is 51.45%, versus Qwen's 50.15% and
Decider's 53.40%. The [exploratory paired state intervals](../pilot-tev1-paired-20260927/RESULTS.md)
are wide: Tev1 minus Qwen is +1.30 points, 95% interval [-1.30, +3.95]; Tev1 minus Decider is
-1.95 points, interval [-4.30, +0.35]. These intervals have no multiplicity correction;
neither Tev1 pairwise interval excludes zero.

Tev1 beats Qwen on jailbreak, toxicity and RAG relevance, while Qwen leads on the other four
application tasks. Decider leads Tev1 on all seven. Tev1 is the stronger measured reference of
these three on the official SemIf family-balanced metrics: 78.92% authored and 76.05%
perturbations, versus Decider's 75.61% and 69.75%, and Qwen's 70.52% and 50.12%.
These small, already-observed reasoning sets do not establish broad reasoning superiority.
Tev1's typed ECE is 8.64 points, and its 34.55% intent option-order flip rate exceeds the later
10% limit. No Tev1 latency or fresh rule-test score was measured. The inspected weight card
said its fine-tuned license was still being finalized; evaluation does not establish deployment
readiness or equivalence to a hosted endpoint. Native formats and all unfavorable results remain
in the archived comparison. No model is promoted from this addition.

All cloud pods from the trial are terminated. The elapsed-rate estimate is $9.1452 of the
authorized $10 combined cap; this is not a final provider invoice. These follow-up comparisons
used the local laptop and did not create another paid resource.

The [next continuation comparison](../pilot-qwen-continuation-20260927/public_evaluation.md)
is registered before selected-checkpoint public evaluation. It includes a fresh unchanged
Qwen control on the same laptop/backend, so training effects can be distinguished from the
earlier H200 runtime. The complete input audit passed; trained checkpoint selection and reload
must finish before inference. No new quality or speed result is claimed from this preparation.
