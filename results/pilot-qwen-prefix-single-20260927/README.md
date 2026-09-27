# Single-question native-prefix graph validation

The [separate registration](../../experiments/small_model/qwen_prefix_single_registration.json)
retains the unchanged native-prefix implementation and complete batch-one input plan. It
increases the unexecuted arm's allowance from 12 to 18 minutes, with a 20-minute hard child
timeout and a 22-minute shared compute lease. Four threads, 80/76 C pacing, the 84 C wrapper,
memory limits, source hashes and all probability limits remain fixed. No cloud resource is used.

This resource change was registered before any native-prefix batch-one inference. The earlier,
different left-padding candidate reached its 12-minute deadline after 1,348 of 1,390 decisions.
Its incomplete evidence remains archived; none of those predictions are reused here. The
native-prefix [batch-16 arm](../pilot-qwen-prefix-validation-20260927/batch-16/RESULTS.md) has now
passed all 1,390 decisions. Passing that arm does not substitute for the single-question check.

The new driver calls the same pinned native-prefix worker with a different registration receipt
and output directory. It preserves the original parent registration and never edits the worker,
graph implementation or verifier. A separate three-test CPU suite checks that the resource
derivation preserves all inference inputs and numerical limits, leaves the parent untouched,
and requires both a complete passing batch-16 result and the actual incomplete terminal
deadline evidence. Those tests and Ruff passed; they do not establish batch-one GPU parity.

The original [full-input preflight](preflight.json) is copied byte-for-byte and pinned. All 1,390
single-decision native prefixes must be checked against the same frozen full-head batch-16
reference, with argmax agreement at least 0.99, maximum probability difference at most 0.04,
and mean total variation at most 0.005. Captured versus identical uncaptured maximum difference
must be at most 0.00001. The reference comparison includes the effect of changing its batch size.
No temperature, reference, threshold or input is chosen using a single-question result.

The registered job is queued under the shared compute lock. After its actual session exits,
run the existing independent verifier on `batch-01`; a passing result is still not a speed,
public-quality, export or runtime-promotion claim. Partial or failing evidence will be retained.

```powershell
.venv-pilot/Scripts/python.exe experiments/small_model/verify_qwen_prefix_validation.py results/pilot-qwen-prefix-single-20260927/batch-01
```
