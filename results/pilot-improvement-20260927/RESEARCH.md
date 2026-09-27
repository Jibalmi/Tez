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
| [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) | Direct size-class competitor with a pointer head and typed probability outputs | Current released revision selected for a native-library matched evaluation; publisher's results use different suites |
| [Decider-0.8B](https://huggingface.co/Mapika/decider-0.8b) | Same size class with a trained letter readout and native typed API | Publisher reports a much larger training mixture; matched evaluation not yet run |
| [GLiClass Multilang Mini](https://huggingface.co/knowledgator/gliclass-multilang-mini) | Newer multilingual classifier supporting task prompts and label descriptions | Registered for matched evaluation; the older x-base result cannot stand in for it |
| [GLiClass Instruct Base](https://huggingface.co/knowledgator/gliclass-instruct-base-v1.0) and [Edge](https://huggingface.co/knowledgator/gliclass-instruct-edge-v1.0) | Smaller alternatives with instruction-guided classification | Registered for matched evaluation; generic label classification mapped explicitly to each typed question |
| [Dev-0.4B](https://huggingface.co/mpnikhil/dev-0.4b) | English encoder with dynamic choices, yes/no, and scores | Current metadata checked; previous repository investigation found reproducibility concerns; no new matched score claimed |
| [Unmodified Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Determines whether our LoRA improves on its actual starting checkpoint | Registered with exactly the same Tez letter prompt and evaluation rows |

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
