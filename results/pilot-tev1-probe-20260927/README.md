# Tev1 native probability probe

The [registered interface check](../../experiments/small_model/tev1_probe_registration.json)
uses four fixed non-benchmark decisions. Their state, question and options pass through the
publisher's pinned request helper and the checkpoint's own chat template, with thinking off.
The model is loaded locally in BF16 with eager attention, one case at a time.

For every case, the worker checks unique single-token letters, reads all option logits, and
compares their winner with one-token greedy generation constrained to those same letters.
It also records unrestricted vocabulary mass on the options. The resulting full distribution
is conditional on the supplied options; this check alone cannot establish calibration or
equivalence with Together's hosted endpoint. No test labels or accuracy scores are involved.

The worker verifies every staged checkpoint file before loading. It retains the shared local
compute lock, four threads and original thermal rules, with a five-minute deadline. It makes
no paid endpoint call and introduces no cloud resource or runtime change.
