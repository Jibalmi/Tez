# Qwen continuation preparation

**Outcome (2026-09-27): no continuation gain.** All 150 registered updates completed and all four
validations are verified. Mean KL rose at every step (0.1401 at step 0; 0.1477, 0.1482 and 0.1485 at
steps 50, 100 and 150), and step 150 also lost 0.0216 teacher agreement, beyond the 0.01 allowance. The
frozen [selector](selection/selection.json) retained the unchanged step-zero adapter; its export is
byte-identical to the original (SHA-256 `fee3b3fe...`, 372 tensors). As registered, public scores cannot
choose another checkpoint, so the continuation's public comparison would pit the unchanged model against
itself and is not run. Steps 101-150 were trained on a RunPod H100 under a
[recorded deviation](training/README.md#recorded-deviation-steps-101-150-trained-on-a-runpod-gpu); every
validation ran on the laptop.

The CPU-only [preparation](preparation.json) verifies the 35 frozen teacher files and creates
a deterministic schedule of 150 full batches, 16 decisions each. Its 2,400 decisions come
from 2,017 original training states: 873 choice, 848 yes/no and 679 score decisions. No
validation or public-test labels select these examples. The schedule stays under ignored `runs/`.

Every scheduled prompt and all 1,390 validation prompts were tokenized without truncation.
The maxima are 1,650 training tokens and 1,607 validation tokens, within the revised 2,048-token
bound. Three scheduled training prompts and five validation prompts exceed the historical
1,536-token limit. [The full context audit](../pilot-qwen-context-audit-20260927/README.md)
quantifies this defect without attributing the broad accuracy gap to it.

The first preparation rejected the old context limit before training. Its source and failure
record remain in `preparation_attempt1/`; the second preparation preserves complete inputs.

The [trainability probe](trainability/README.md) passed one LoRA backward/update on the longest
scheduled training prompt, under the shared compute lock and conservative 80/76 C pacing.
Loss and every LoRA gradient were finite, and 372 adapter tensors changed. Probe weights were
discarded; the probe cannot serve as a candidate or a training throughput measurement.

The [registered continuation](training/README.md) retains the unchanged checkpoint as step zero
and selects only from original validation data. Its optimizer, checkpoint selection, input
reconstruction and resumable driver are implemented. Complete step-zero validation is now
[independently verified](training/attempt-02-baseline-000/verification.json): all 1,390 predictions
match the frozen reference, and all 372 saved adapter tensors equal the originals. Two bounded
training segments are now [complete and verified through step 50](training/attempt-04-train-050/README.md):
800 original training decisions, with successful resumption from the first segment's saved optimizer
checkpoint. The [step-50 validation](training/attempt-05-validate-050/README.md) is complete and
independently verified. Teacher agreement declined from 78.9928% to 78.1295%, and mean KL rose
from 0.14013744 to 0.14770741. This checkpoint does not improve the registered metrics. Two more
bounded segments reached a [verified step-100 checkpoint](training/attempt-07-train-100/README.md),
whose [validation](training/attempt-09-validate-100/README.md) recovered agreement (0.7906) but
raised mean KL to 0.1482. Step zero (KL 0.1401) still leads under the registered selector; the
step-150 segment and validation follow. No release gate is claimed.

The [descriptive validation breakdown](validation_diagnostics/step-050/README.md) shows worse
teacher agreement and KL across choice, yes/no and score questions. Mean confidence declines
slightly. Only 21 validation decisions have original gold labels, so these teacher-distribution
metrics cannot establish broad human-label accuracy. The diagnosis changes no training or
selection setting.

The next [public comparison is registered](public_evaluation.md): a fresh unchanged-model control
and the frozen selected checkpoint on the same laptop/backend, all original public decisions
and reversed-option checks, with archived historical Decider and Tev1 results as context.
The complete input preflight and five CPU integrity tests passed. Public inference waits for
all four validations, selection, export and strict reload parity; no new accuracy result exists.

A [checkpoint helper](../../experiments/small_model/qwen_checkpoint.py) supports
that driver. It saves only named trainable LoRA tensors, optimizer state, progress and
random state; it refuses overwrite and verifies the registration, content hash and parameter
layout on resume. A one-thread CPU toy test reproduces three subsequent AdamW updates and
the random sequence exactly after restoring a checkpoint. Separate checks reject wrong hashes,
registrations, optimizer parameter order and non-finite adapter values. This tests the persistence
mechanism's numerical replay on CPU. Qwen's second segment has now successfully resumed a trained
GPU checkpoint; comparison with an uninterrupted GPU trajectory remains unmeasured.

The [single-decision loss helper](../../experiments/small_model/qwen_continuation_loss.py)
preserves the original teacher KL plus 0.5 gold CE. Each accumulated decision must be divided
by the full effective batch size, including rows without gold. A one-thread CPU test compares
the accumulated loss and shared-parameter gradients with the original padded batch expression,
using variable option counts, zero-probability targets and missing gold. Other tests verify
option-permutation invariance and reject malformed targets or gold indices. All three passed.
This checks loss arithmetic; it does not establish full-model GPU gradient equivalence,
successful continuation, improved quality or training speed. The completed trainability probe
and its registration are unchanged.
