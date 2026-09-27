# Restricted option-letter projection trial

Completed: [all 1,390 validation decisions passed the registered probability check](RESULTS.md).
Argmax agreement was 100%; maximum probability change was 0.00364906 and mean total variation
was 0.0000026252. This establishes numerical parity on the validation set, not a measured speedup.

The Qwen student currently computes final-position scores for the full vocabulary before selecting
the option letters. This registered experiment projects the same frozen hidden state directly onto
those letter rows. It keeps the original unmerged adapter, prompt, attention mask, padding, cache
setting and batch size. This is an inference experiment, not another trained model.

[Registration](../../experiments/small_model/restricted_qwen_registration.json) fixes the original
probability-parity thresholds before any GPU execution: at least 99% argmax agreement, maximum
probability change at most 0.04, and mean total variation at most 0.005 across all 1,390 original
teacher-validation decisions. The frozen base and adapter hashes and every teacher data file
must match the saved manifests. Public and authored-confirmation labels are not used.

CPU contract tests check the masked backbone path, option mapping, output bias and rejection of
wrapped output heads. They do not establish GPU numerical parity or speed. GPU execution remains
subject to the shared laptop queue, an eight-minute worker limit and the existing thermal rules.
No cloud resource is created.

Passing parity alone cannot promote the candidate. It must also improve measured latency under
the original target-laptop temperature and contamination requirements. A failed parity check is
preserved, and the thresholds are not relaxed. No speedup is claimed.
