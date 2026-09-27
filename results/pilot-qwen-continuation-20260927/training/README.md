# Registered Qwen continuation

Step-zero validation is [complete and independently verified](attempt-02-baseline-000/verification.json).
All 1,390 argmax predictions match the frozen reference; maximum probability difference is
4.44e-16. Teacher agreement is 78.9928% and mean KL is 0.14013744. All 372 saved adapter tensors
equal the original checkpoint, and the optimizer has no prior update state. The evidence archive
retains raw held-out predictions and inputs. Four bounded training segments are now
[complete and independently verified through step 100](attempt-07-train-100/README.md), including
successful checkpoint resumption. The [step-50 validation](attempt-05-validate-050/README.md)
regressed on both registered metrics. The [step-100 validation](attempt-09-validate-100/README.md)
recovered agreement (0.7906) but has the highest mean KL (0.1482); step zero (0.1401) still leads.
The first segment toward step 150 stopped on the thermal guard after nine updates and is
[preserved](attempt-10-train-150/README.md); its retry waits for a cool, quiet machine. No
selected-checkpoint public-quality result is available yet.

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
passed, and three bounded GPU segments have completed. Resumption from a trained GPU checkpoint
succeeded, but equivalence to an uninterrupted GPU trajectory is unmeasured. The full continuation
remains unfinished. A partial training run is not a quality result or release.

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

The [selection and export implementation](../../../experiments/small_model/export_qwen_continuation.py)
is now [registered](../../../experiments/small_model/qwen_continuation_export_registration.json)
before any trained checkpoint validation. It requires all four verified validation archives,
recomputes their metrics, checks the checkpoint validation histories, and uses the frozen selector.
It exports only the chosen unmerged adapter under ignored `runs/`. The original adapter layout
and config must be preserved, and every serialized tensor must reload exactly. New metadata uses
temperature one and records provenance without copying historical timings or calibration values.

Six one-thread CPU tests passed, including a real tiny PEFT model whose tensors and forward outputs
match exactly after export and reload. Other cases reject incomplete or changed tensor layouts,
non-finite values, modified validation inputs and incomplete selection inputs. The
[integration preflight](../export_preflight.json) also reread the existing unchanged-baseline archive
and reproduced its validation metrics. No trained checkpoint was selected or exported. A selected
Qwen export must still pass its own GPU reload check before public evaluation. The format follows
the [PEFT checkpoint documentation](https://huggingface.co/docs/peft/developer_guides/checkpoint).

The selected model's [reload worker](../../../experiments/small_model/qwen_continuation_reload.py)
and [protocol](../../../experiments/small_model/qwen_continuation_reload_registration.json) are
also ready. Before loading a model, the worker binds the chosen adapter, config, metadata,
checkpoint, selected validation predictions and selection record by content hash. It reuses the
unchanged validation routine on all 1,390 held-out decisions and requires exact argmax agreement,
maximum probability difference at most 0.000001 and mean total variation at most 0.0000001.
The independent report recomputes both probability parity and the saved validation metrics.
Three CPU tests verify that it rejects a different chosen step, changed registration, missing
validation, or an inconsistent unchanged-model flag. GPU reload is not launched until selection
and export finish; these tests do not establish model parity. Completed failures will be retained.

The [first trained validation](attempt-05-validate-050/README.md) is now complete and verified:
all 1,390 decisions, teacher agreement 0.781294964028777 and mean KL 0.14770740977013652.
Both are worse than step zero (agreement 0.7899280575539568 and KL 0.1401374435902039).
Its checkpoint supplied attempt six, which completed another 25 updates. Attempt seven resumed
from verified step 75 with the same validation history, optimizer and random states and
[reached step 100](attempt-07-train-100/README.md). Attempt eight's validation stopped on the
worker's thermal guard after 1,296 of 1,390 decisions and is [preserved](attempt-08-validate-100/README.md);
attempt nine completed it from the same checkpoint.
The registered schedule and selection rule continue unchanged; this intermediate regression
is retained. The [descriptive diagnostics](../validation_diagnostics/step-050/diagnostics.json)
break down the regression by question type without changing training or checkpoint selection.

## Recorded deviation: steps 101-150 trained on a RunPod GPU

The registration's resources name the shared laptop and no cloud resources. On 2026-09-27 the
laptop was heat-soaked by other shared work and two registered operations stopped on the worker's
own thermal guard ([attempt 8](attempt-08-validate-100/README.md), [attempt 10](attempt-10-train-150/README.md)).
The owner then authorized the existing RunPod account for this work. The two remaining training
segments (steps 101-125 and 126-150, from the verified, validated step-100 checkpoint) therefore
run on one rented GPU, recorded here before launch.

Unchanged: the registration file and its hash, sources, data, schedule, optimizer, checkpoint chain,
segment bounds, attempt budget, thermal pacing, selection rule and probability limits. The Pod uses
the laptop's Python line and pinned packages (torch 2.10.0+cu130, transformers 5.17.0, peft 0.21.0,
tokenizers 0.23.2, huggingface-hub 1.33.0), without the optional FLA or causal-conv1d kernels; it
checks every bundled file against a laptop manifest, requires the three registered library files
to be byte-identical to its own install, and verifies the pinned base-weight hash before any
update ([bootstrap](../../../experiments/small_model/continuation_pod_bootstrap.sh),
[bundle](../../../experiments/small_model/continuation_pod_bundle.py)).

Different: the GPU model. Steps 101-150 follow a different floating-point path than laptop training
would have; no laptop trajectory exists for those steps to compare against. The step-150 validation
runs on the laptop, like steps 0, 50 and 100, so every validation that feeds the registered selector
comes from the same hardware and software. The rented GPU is used for training only, with a
bounded allocation and an automatic provider stop; the checkpoints return to the ignored `runs/`
tree and are verified locally by the unchanged CPU verifier.
