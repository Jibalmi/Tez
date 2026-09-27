# Registered Qwen continuation

The [frozen protocol](../../../experiments/small_model/qwen_continuation_registration.json)
tests 150 updates from the original adapter, using the already prepared 2,400 original training
decisions. A new AdamW optimizer uses constant learning rate 0.00005, zero weight decay and
gradient clipping at 1.0. Microbatch one accumulates 16 decisions per update. Training retains
the original KL plus 0.5 gold CE, with complete inputs, gradient checkpointing and cache off.
The original optimizer state is unavailable; this is explicitly a new continuation optimizer.

Validation uses all 1,390 original held-out decisions at steps 0, 50, 100 and 150. It keeps the
original sorted batch-16 inference order, cache on, full vocabulary head, native attention and
temperature one. Step zero must first pass the unchanged reference probability checks. Every
training interval requires the preceding validation to be complete. Selection minimizes mean
validation KL while allowing at most 0.01 teacher-agreement loss versus step zero; exact ties
prefer the earlier checkpoint. Step zero stays eligible. Public tests and authored rule/choice
data do not select checkpoints and are excluded from training.

The shared compute wrapper permits four threads, a 23-minute lease and an 84 C outer ceiling.
The worker checks 80/76 C thermal pacing, checkpoints after at most 25 updates or an 18-minute
soft deadline, and has a 21-minute hard bound. At most 20 distinct attempts are allowed.
Checkpoints are immutable and preserve adapter, optimizer, progress and CPU/CUDA random state.
They remain under ignored `runs/`; reports record their hashes, without committing weights.

The [input preflight](../driver_preflight.json) reconstructs all scheduled training prompts
and validation prompts without truncation. CPU checks verify loss/gradient accumulation,
checkpoint replay and validation selection. The one-step [GPU feasibility probe](../trainability/README.md)
passed, but full GPU continuation and checkpoint replay have not yet been exercised. A
registration or queued step-zero evaluation is not a completed training result or a release.

After all four validations, independently verify the saved rows and selected checkpoint before
evaluating the chosen candidate. Existing public tests have informed the research direction;
later public gains remain exploratory. All original quality/speed gates remain unchanged.
