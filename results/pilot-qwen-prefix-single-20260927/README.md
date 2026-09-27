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

The registered job completed all 1,390 decisions and its [independent verification](batch-01/verification.json)
confirms a **failure** of the unchanged mean-total-variation limit: 0.006645812989476119 versus
0.005. Argmax agreement is 0.9942446043165467 and maximum probability difference is
0.031232999811175166. Graph and identical uncaptured outputs match exactly on every decision.
The [1,386,691-byte archive](batch-01/artifact_receipt.json) contains 45 verified files, including
all predictions, inputs, execution receipts and frozen sources. No timing or promotion follows.

Five captures and all 1,390 real single-question replays completed under the longer resource
plan. The worker recorded 1,006.4496441 operational seconds, including 299 thermal pauses
totaling 483.5501875 seconds. These are execution accounting, not latency. The failure combines
native batch-size differences versus the batch-16 reference with the surrounding graph-input
changes. Exact captured/uncaptured agreement does not isolate those other causes. A separate
[native full-head batch-one control](../pilot-qwen-native-single-20260927/README.md) is now
registered and queued before selecting another inference change.

The proposed multi-graph request cache remains unregistered and unexecuted because it requires
both full-validation arms to pass. Its CPU preflight does not override this failed requirement.

```powershell
# Already completed; the verifier refuses to overwrite its immutable evidence archive.
.venv-pilot/Scripts/python.exe experiments/small_model/verify_qwen_prefix_validation.py results/pilot-qwen-prefix-single-20260927/batch-01
```
