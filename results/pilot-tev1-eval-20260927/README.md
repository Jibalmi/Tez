# Tev1 matched public comparison

The [frozen protocol](../../experiments/small_model/tev1_registration.json) evaluates the pinned
Tev1 0.8B experimental checkpoint on the same 6,151 public decisions as Qwen and Decider, plus
1,100 reversed-option intent cases. Execution uses the shared local compute queue; a registration
or running receipt is not a completed result.

The [four-case native probe](../pilot-tev1-probe-20260927/README.md) passed. The CPU-only
[input audit](input_audit.json) confirms the original suite fits without truncation: its longest
native prompt is 1,827 tokens. The evaluator retains the full BF16 checkpoint, eager attention,
single-case scoring, publisher system/JSON format, and all option probabilities at temperature one.
The context safety bound remains 4,096 tokens; exceeding it stops the run rather than dropping text.

No calibration fitting, prompt selection or training uses these public cases. Training exposures
differ between competitors, so this is a utility comparison and not a controlled architecture test.
Model loading, thermal pauses and audit work make total job duration unsuitable as a latency score.
The 45-minute worker bound, original thermal rules and all release gates remain in force.
No hosted endpoint, new paid resource or runtime promotion is involved.
