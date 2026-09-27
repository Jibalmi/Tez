# Matched public evaluation protocol

The [registered workflow](../../experiments/small_model/qwen_public_registration.json) compares
the unchanged Qwen adapter with the checkpoint selected by the frozen original-validation rule.
Both use the same RTX 5080 Laptop, native Torch/SDPA path, complete prompts, unmerged LoRA,
full vocabulary head, cache-on batch-16 inference and temperature one. Neither arm can start
until all four validation checkpoints, selection, export and strict GPU reload verification finish.
No selected model or new public score exists yet.

The one-thread [input preflight](public_preflight.json) checks all 6,151 primary decisions and
all 1,100 reversed-option decisions across the 21 frozen tasks. The longest prompt is 1,803
tokens, so no input reaches the existing 4,096-token public inference bound. Every prompt,
complete token sequence, case and option permutation has a content hash. The private audit
is pinned in the registration and will be included in each completed evidence archive.

[Five CPU tests](../../experiments/small_model/test_qwen_public.py) passed. They check correct
reverse-option mapping while retaining original labels; reject changed prompts, labels,
soft targets, logits and probabilities; reject duplicate, missing or foreign rows; preserve
typed-state grouping; exclude labels from predictor inputs; and reject overlong or empty
inputs. Ruff also passes for the four new scripts. These are integrity checks, not GPU
inference, accuracy, speed or release results.

Each arm has four independently bounded phases: typed decisions, seven application tasks,
11-language intent with both option orders, and the two SemIf suites. This releases the shared
compute lock between phases. Each phase has an 18-minute soft deadline, 20-minute hard child
limit and 22-minute compute lease, with the same 80/76 C pacing and 84 C wrapper bound used
for recent quality experiments. These are limits, not runtime forecasts. Completed task outputs
survive a failed phase, and incomplete phases cannot enter the final comparison.

The [verifier](../../experiments/small_model/report_qwen_public.py) re-tokenizes complete inputs,
checks every probability against its saved readout, verifies both option coordinate systems,
and recomputes raw temperature-one scores. It uses semantic labels for application/intent F1,
the official SemIf evaluator, and whole-state paired bootstrap intervals for typed accuracy
(2,000 resamples, seed 13, without multiplicity correction). No temperature is fitted.

Historical Qwen and Decider H200 predictions and the completed native laptop Tev1 predictions
are pinned as context. Only the new unchanged-versus-selected laptop pair isolates this
continuation under matching hardware and backend. Different external training exposures and
historical execution environments remain explicit. These already-observed public benchmarks
make the comparison exploratory. If the selector retains step zero, report no selected training
gain; public scores cannot choose another checkpoint. All original accuracy and latency gates
remain unchanged, and this workflow makes no latency claim or runtime promotion.

After selection and verified reload, launch each arm/phase through the shared wrapper, then
run its one-thread independent report. The final comparison requires all eight verified phases:

```powershell
python C:/temp/run_heavy.py --owner tez-qwen-public-baseline-typed --purpose 'Matched local unchanged Qwen public quality control' --minutes 22 --threads 4 --max-gpu-c 84 -- .venv-pilot/Scripts/python.exe experiments/small_model/qwen_public_evaluation.py --arm baseline --phase typed
.venv-pilot/Scripts/python.exe experiments/small_model/report_qwen_public.py --arm baseline --phase typed
# Repeat with the registered baseline/selected arms and typed/apps/intent/semif phases.
.venv-pilot/Scripts/python.exe experiments/small_model/report_qwen_public.py --compare
```
