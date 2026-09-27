# Qwen continuation preparation

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
match the frozen reference, and all 372 saved adapter tensors equal the originals. The first
bounded training segment is queued. No training improvement or release gate is claimed yet.

A [checkpoint helper](../../experiments/small_model/qwen_checkpoint.py) supports
that driver. It saves only named trainable LoRA tensors, optimizer state, progress and
random state; it refuses overwrite and verifies the registration, content hash and parameter
layout on resume. A one-thread CPU toy test reproduces three subsequent AdamW updates and
the random sequence exactly after restoring a checkpoint. Separate checks reject wrong hashes,
registrations, optimizer parameter order and non-finite adapter values. This tests the persistence
mechanism only; Qwen GPU training and interruption recovery have not yet been verified.

The [single-decision loss helper](../../experiments/small_model/qwen_continuation_loss.py)
preserves the original teacher KL plus 0.5 gold CE. Each accumulated decision must be divided
by the full effective batch size, including rows without gold. A one-thread CPU test compares
the accumulated loss and shared-parameter gradients with the original padded batch expression,
using variable option counts, zero-probability targets and missing gold. Other tests verify
option-permutation invariance and reject malformed targets or gold indices. All three passed.
This checks loss arithmetic; it does not establish full-model GPU gradient equivalence,
successful continuation, improved quality or training speed. The completed trainability probe
and its registration are unchanged.
