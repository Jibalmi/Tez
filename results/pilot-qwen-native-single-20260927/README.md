# Native single-question diagnostic control

The completed [single-question graph validation](../pilot-qwen-prefix-single-20260927/README.md)
failed mean total variation against the frozen native batch-16 reference, while its captured
and identical uncaptured outputs matched exactly. This [separate registration](../../experiments/small_model/qwen_native_single_registration.json)
isolates the native batch-size comparison before another inference candidate is chosen.

**Result.** The control completed and its [independent report](control/verification.json)
checked all 1,390 decisions. Native batch-one against the frozen native batch-16 reference:
argmax agreement 0.99712, maximum probability change 0.03618, mean total variation 0.00651.
The failed graph against this native batch-one control: 0.99568, 0.03618 and 0.00670. Both
comparisons fail the retained 0.005 mean-total-variation check and pass the other two. Batch
shape alone therefore moves the native model's probabilities by about as much as the graph
path does; the graph's gap to either native reference is of the same size as the gap between
the two native references. This is consistent with BF16 reduction-order sensitivity to batch
shape, but it does not identify a cause, and it neither accepts the graph nor changes any limit.
The [evidence archive](control/artifact_receipt.json) holds 45 verified files in 1,318,720 bytes,
SHA-256 `e04385b006ca87a6bcb3f592788d04c0475971d0935889f916f97d54cf2b5021`.

It evaluates all 278 original validation states and 1,390 decisions with the original full
vocabulary head, cache enabled, native SDPA/PyTorch recurrence, unmerged FP32 LoRA over BF16
weights, eval mode, no-grad and BF16 autocast. Complete prompts retain their original option
order and stable character-length sort. Each native forward now has one question; there is
no graph capture or appended graph padding. All prompt/token IDs must reproduce the frozen
input audit before inference begins.

Two comparisons are reported independently: the original native batch-16 reference versus
this fresh native batch-one control, and this control versus the saved failed batch-one graph.
The original limits are retained as descriptive checks: argmax agreement at least 0.99,
maximum probability change at most 0.04, mean total variation at most 0.005. **The control
does not replace the original reference or change the graph's failed acceptance outcome.**
Its comparison with the graph includes the combined remaining padding, readout/cache and
capture differences; it does not isolate every individual cause.

Three one-thread CPU tests pass, covering full membership and native order, missing/duplicate
rows, changed input provenance, malformed log probabilities and probability reconstruction.
Ruff passes. Those tests do not establish GPU parity. The worker checks all source, weight,
package and input hashes and retains partial output on failure. The independent report checks
every raw vector and actual settings before archiving both comparison references and executed
sources. No training data, checkpoint choice or public evaluation is changed.

Compute remains four threads under a 22-minute shared lease, with an 18-minute soft deadline,
a 20-minute hard child timeout, 80/76 C thermal pacing and an 84 C wrapper ceiling. Execution
duration is operational accounting, not latency. No new paid resource is used.

After the actual job exits successfully:

```powershell
.venv-pilot/Scripts/python.exe experiments/small_model/qwen_native_single_control.py --report
```
