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
| Our Qwen 0.8B student | 50.15 | 64.09 | 89.22 | 85.75 | 60.25 | 70.50 |
| Kev 0.8B | 49.10 | 61.27 | 84.96 | 89.00 | 64.25 | Not measured |
| Our encoder, 12 epochs | 39.05 | 51.73 | 58.65 | 85.50 | 53.75 | 35.61 |
| Our encoder, rule continuation | 38.80 | 53.18 | 56.89 | 85.50 | 53.50 | 54.67 |

Sources: [full public comparison, including all 16 configurations](COMPARISON.md),
[encoder continuation](../pilot-reasoning-round2-20260927/RESULTS.md), and
[fresh rule comparison](../pilot-reasoning-competitors-20260927/RESULTS.md).
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
latency gate. Valid laptop timings remain outstanding because the prior attempts did not meet
the registered cooling requirements.

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
the original attention kernels. Neither experiment establishes a measured speedup. Larger Decider releases may be useful quality references,
but have not been measured here and would be a different size/cost class. Published JevBench
or DecisionIndex scores use different protocols and cannot be inserted into this table.

The next matched candidates are [GLiNER's English and multilingual decision checkpoints](../pilot-gliner2-eval-attempt2-20260927/README.md)
and [Together's Tev1 0.8B](../pilot-tev1-discovery-20260927/README.md). They have no completed
public score in this report yet. GLiNER's initial batch smoke exposed an input-guard mismatch
with native punctuation handling; that failure is archived and the corrected check is registered.
Tev1's pinned weights are staged and its native probability check is registered. Availability
and published development scores are not treated as measured wins.

All cloud pods from the trial are terminated. The elapsed-rate estimate is $9.1452 of the
authorized $10 combined cap; this is not a final provider invoice. These follow-up comparisons
used the local laptop and did not create another paid resource.
