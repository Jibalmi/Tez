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

Before implementing/running the full continuation, the [trainability probe](../../experiments/small_model/qwen_trainability_registration.json)
will test one LoRA backward/update on the longest scheduled training prompt, under the shared
compute lock and conservative 80/76 C pacing. It requires finite loss and gradients and a
finite nonzero adapter update. It saves no weights and cannot serve as a candidate.

The subsequent continuation should retain the unchanged checkpoint as step zero and select
only from original validation data. Its full optimizer, checkpoint selection and resumable
training driver remain to be registered and implemented. No training improvement or release
gate is claimed from this preparation.
