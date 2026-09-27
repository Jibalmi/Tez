# Registered Qwen continuation

Step-zero validation is [complete and independently verified](attempt-02-baseline-000/verification.json).
All 1,390 argmax predictions match the frozen reference; maximum probability difference is
4.44e-16. Teacher agreement is 78.9928% and mean KL is 0.14013744. All 372 saved adapter tensors
equal the original checkpoint, and the optimizer has no prior update state. The evidence archive
retains raw held-out predictions and inputs. The first [bounded training segment](attempt-03-train-050/README.md)
completed and independently verified 25 updates. Its checkpoint now feeds the queued second segment
toward step 50; no trained-candidate validation or public-quality result is available yet.

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
passed, and the first bounded GPU segment has completed. The full continuation and resumption of
a trained GPU checkpoint remain unfinished. A partial training run is not a quality result or release.

The [first baseline attempt](attempt-01-baseline-000/README.md) failed before saving any
prediction or checkpoint because the validation mask omitted the per-row broadcast dimension.
The amended registration adds a tested readout helper; tests compare mixed-size batches with
independent unpadded decisions, including the equal-dimension case that could silently mask
the wrong axis. Attempt two uses the corrected code with all experimental settings unchanged.
The original failed source, protocol and preflight are archived separately.

After all four validations, independently verify the saved rows and selected checkpoint before
evaluating the chosen candidate. Existing public tests have informed the research direction;
later public gains remain exploratory. All original quality/speed gates remain unchanged.

The [registered independent verifier](../../../experiments/small_model/qwen_continuation_analysis_registration.json)
checks source and input hashes, checkpoint tensors and optimizer progress, validation history,
every held-out probability vector and step-zero identity. CPU tests reject changed initial
weights, optimizer drift, malformed moments and inconsistent progress. Verification remains
pending until an operation completes; these tests do not establish GPU checkpoint replay.
