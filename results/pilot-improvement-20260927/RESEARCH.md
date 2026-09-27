# Continuing student research, 2026-09-27

Goal: improve quality and speed, and compare the result with available alternatives before choosing
what to build or reuse. The original $10 combined RunPod cap still applies. Prior estimated spend
is $6.25; the next Pod is limited to 40 minutes at at most $4.60/hour, with export time reserved.
Laptop work uses the shared compute queue.

## What the existing measurements establish

The [original pilot](../pilot-student-h200-20260927/README.md) measures all four students and GLiClass
on the same 6,151 decisions, with recorded reference rows for the 12B and three Laya checkpoints.
Our Qwen student beats the general Laya checkpoints on typed decisions and the multilingual mean,
but Laya leads on several application tasks. Laya's typed-decisions checkpoint reaches 76.6% on
typed decisions after training on that benchmark's training split; our Qwen reaches 50.15% while
holding out the full dataset. These training regimes must be shown next to the scores.
The base encoder's 35.95% is not competitive with the strongest available checkpoint on that task.
Neither H200 latency nor published timings on other hardware prove a laptop speed advantage.

The [descriptive diagnostics](diagnostics.json) locate a pronounced error: the base encoder predicts
the second yes/no option on 574/600 typed-decision questions, reaching 48.83% accuracy. Its teacher
training targets instead favor the first option on 7,296/10,428 yes/no questions. A simple matching
class imbalance does not explain the test behavior. Training-state and question-format transfer,
packing, and insufficient learning remain hypotheses; none is established as the cause.

## Current alternatives

Upstream model revisions and declared licenses, parameter counts, and datasets are saved in
[competitor_inventory.json](competitor_inventory.json). Declarations do not establish absence of
benchmark contamination. Parameter counts of adapter-only repositories exclude their base weights.

| Alternative | Why it belongs in the comparison | Evidence status |
|---|---|---|
| [Laya family](https://github.com/NandhaKishorM/laya) | Same typed-decision interface; English, multilingual, and specialized checkpoints | Existing matched accuracy rows; current revision inventory; laptop speed comparison pending |
| [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) | Direct size-class competitor with a pointer head and typed probability outputs | Matched native-library evaluation and H200 eager timings complete; publisher's results use different suites |
| [Decider-0.8B](https://huggingface.co/Mapika/decider-0.8b) | Same size class with a trained letter readout and native typed API | Matched public and fresh-rule evaluations complete; training exposure is qualified in the later audit |
| [GLiClass Multilang Mini](https://huggingface.co/knowledgator/gliclass-multilang-mini) | Newer multilingual classifier supporting task prompts and label descriptions | Matched evaluation complete; separately measured from the older x-base checkpoint |
| [GLiClass Instruct Base](https://huggingface.co/knowledgator/gliclass-instruct-base-v1.0) and [Edge](https://huggingface.co/knowledgator/gliclass-instruct-edge-v1.0) | Smaller alternatives with instruction-guided classification | Matched evaluation complete; generic label classification mapped explicitly to each typed question |
| [Dev-0.4B](https://huggingface.co/mpnikhil/dev-0.4b) | English encoder with dynamic choices, yes/no, and scores | Current metadata checked; previous repository investigation found reproducibility concerns; no new matched score claimed |
| [Unmodified Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Determines whether our LoRA improves on its actual starting checkpoint | Matched evaluation complete with exactly the same Tez letter prompt and evaluation rows |

A later [discovery audit](../pilot-gliner2-discovery-20260927/README.md) adds GLiNER2.5-Decide to
the candidates for measurement and distinguishes the LocalJev bridge and Ollaya runtime from
trained checkpoint alternatives. No unmeasured candidate is assigned a score or rank.
Both native GLiNER checkpoints now have a [complete matched comparison](../pilot-gliner2-comparison-20260927/RESULTS.md).
The [current assessment](COMPETITIVENESS.md) includes those results and the
[complete native Tev1 comparison](../pilot-tev1-eval-20260927/RESULTS.md).
Tev1 does better than Qwen on the measured SemIf reasoning tasks, while Decider remains stronger
across the seven application tasks. Its small typed-accuracy differences have wide exploratory
paired state intervals. No latency, hosted-service parity or release promotion follows from that run.

[JevBench](https://github.com/fstandhartinger/jevbench) is a useful external check and discovery source.
Its composite rankings are not directly comparable with typed-decision accuracy or these latency
measurements. Broader or paid API comparisons require their own recorded runs and available access.

## Registered follow-up

The executable [round specification](../../experiments/small_model/improvement_round1.json) fixes
the candidates, compute limits, selection policy, and merge-parity thresholds before running:

1. Benchmark frozen Qwen and current GLiClass/Kev releases on the frozen original cases. Use native
   rendering for each competitor, preserve every option, and record model and code revisions.
2. Train the same encoder for up to 12 epochs. Choose the epoch with the lowest teacher-validation
   KL, then evaluate that checkpoint once. The original three-epoch validation curve was still
   improving; this motivates the test but does not predict its outcome.
3. Merge Qwen's LoRA and disable its unused inference cache. First compare all 1,390 validation
   decisions with the original, then measure speed. PEFT documents merging as a way to remove
   adapter overhead; the actual benefit and numerical change must be measured here.
   [PEFT merging documentation](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#merge-lora-weights-into-the-base-model).
4. Complete queued laptop timings of the downloaded original models. Keep GPU contamination and
   thermal checks; do not claim that a failed or unavailable measurement passed a speed gate.

The original public test results have now informed the research direction. Follow-up gains are
exploratory: they need fresh independent confirmation before a release or broad generalization
claim. Test labels are not training targets, calibration labels, or checkpoint-selection inputs.
Published competitor claims remain separate from our own measurements, including negative results.

Success means meeting the predeclared quality/speed gates and demonstrating useful performance
against the best relevant measured alternatives, not merely improving over our first attempt.
If a released competitor performs better, reusing it is a valid outcome of the research.

## Follow-up execution notes

The [matched comparison](COMPARISON.md) is generated from recorded metrics and raw prediction rows.
It includes paired state-cluster bootstrap intervals: all five questions about a typed-decision state
stay together when resampling. These exploratory intervals do not adjust for multiple comparisons.

The initial encoder attempt failed before training because the pinned mmBERT revision contains
`pytorch_model.bin`, which the download allowlist omitted. The recovery adds that file extension
and retains the original seed, data, hyperparameters, and validation-only checkpoint selection.
Its source-weight hash and execution are recorded separately; the failed attempt remains in the log.

Merging the Qwen adapter into bf16 weights failed the registered probability check: argmax agreement
was 99.568%, but maximum probability change was 0.04956 (limit 0.04) and mean total variation was
0.006599 (limit 0.005). Its timing is diagnostic; this candidate is not promoted. The subsequent
[cache-only registration](../../experiments/small_model/cacheoff_registration.json) keeps the unmerged
weights and applies the same thresholds before any speed-based promotion.

A [Decider addition](../../experiments/small_model/decider_registration.json) registers its pinned
native public API, raw temperature, and original released score-level setting. This measurement
and native eager timings for Kev/Decider are bounded by the existing Pod deadline. API probability
rounding is recorded, and missing or timed-out evaluations are not treated as completed scores.
These eager timings do not stand in for the competitors' optimized serving implementations.

A final [batch-size registration](../../experiments/small_model/batch64_registration.json) tests
64 decisions per Qwen forward instead of 16. The 50-question protocol otherwise requires four
sequential forwards despite unused GPU memory. This keeps the original unmerged weights and cache
setting, validates all teacher-validation probabilities, and remains within the same Pod deadline.
Any speed benefit is specific to measured cloud throughput until laptop measurements establish it.

Evaluation now writes progress metadata after each completed task. This preserves partial work
without admitting it to the completed-evaluation table. Source snapshots and amendments accompany
the exported measurements.

Laptop timing revealed that Windows labels dedicated GPU memory with lowercase text; the earlier
case-sensitive check could miss competing processes. The check now handles both Windows and Linux
labels, and missing thermal telemetry cannot report a clean run. One warm GPU attempt was stopped
before timing. CPU timing runs first on retry; GPU cooling is bounded, with the original temperature
requirement preserved. The machine-wide compute lock remains mandatory.

Two read-only audits refine the next training hypothesis. The [input-length audit](encoder_input_audit.json)
found no held-out instruction, option, or state clipping under the encoder's actual evaluation limits.
The [state-shape audit](state_structure_audit.json) found zero nested states among all 5,998 training
examples, versus nested dictionaries/lists in all 400 typed-decision test states. This identifies
a coverage gap worth testing with newly authored structured training states; it does not prove
that nesting causes the accuracy gap. Increasing context length would not address a measured
truncation issue in these suites, because none was found.

The separate [Qwen training-context audit](../pilot-qwen-context-audit-20260927/README.md) found
historical clipping in 20 of 29,990 training decisions and five validation decisions. The
[continuation preparation](../pilot-qwen-continuation-20260927/README.md) retains complete inputs;
the limited incidence does not establish a cause of the broad quality gap.

An [additional kernel audit](../pilot-kernel-discovery-20260927/README.md) records FlashQLA's
current SM120 path and the difference between its released and source dependency requirements.
This metadata is separate from measured performance. The completed
[FLA precision probe](../pilot-fla-precision-probe-20260927/RESULTS.md) used the existing isolated
FLA/Triton files. Promoting attention inputs to FP32 still failed the unchanged mean-TV limit
on four non-benchmark cases, so that candidate does not advance. Its native FLA arm passing
this small check does not supersede the earlier failed full-validation result.

The [Qwen continuation](../pilot-qwen-continuation-20260927/training/README.md) is registered
after the one-step laptop trainability probe passed. It tests 150 updates from original
training data, retaining step zero and selecting solely from the original validation split.
Step-zero probability verification precedes training; a queued run is not a quality improvement.
