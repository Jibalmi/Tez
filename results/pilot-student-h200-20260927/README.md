# H200 student pilot, 2026-09-27

Four students were trained and all four plus the off-the-shelf GLiClass baseline were evaluated. The downloaded model files and all prediction rows passed the integrity checks. The Pod is terminated.

**Pilot decision:** pending the required laptop latency measurements.
The base encoder misses the predeclared typed-decision and multilingual quality targets. Cloud inference timings below are supplementary and do not certify the laptop speed gate.

Qwen3.5-0.8B leads the trained students on typed decisions, multilingual intent, and routing,
but its single-question H200 p50 is 60.40 ms versus 12.45 ms for the base encoder. Neither is
ready to replace the 12B across these tasks. Qwen's 50.15% typed-decision accuracy clears the
predeclared 49.3% stop floor, so the rule to stop both general students does not apply.
The next gate is a laptop latency measurement with these exact downloaded weights; no arm is
promoted to M2 on the current evidence.

## Held-out results

Each of the four students and GLiClass was evaluated on 6,151 decisions: 2,000 typed decisions, 2,799 application cases, 1,100 multilingual intent cases, and 252 SemIf cases. The teacher and Laya figures reuse the recorded reference rows; they are not new H200 teacher measurements. Student probabilities are scored as read at T=1.

| System | Typed decisions | MASSIVE-11 mean | Routing | Jailbreak | Toxicity |
|---|---:|---:|---:|---:|---:|
| Recorded 12B baseline | 70.40% | 88.45% | 96.99% | 86.50% | 71.25% |
| Laya multilingual | 35.20% | 52.36% | 44.11% | 80.50% | 53.50% |
| a-base-10k | 32.50% | 46.27% | 57.89% | 78.75% | 53.00% |
| a-base-30k | 35.95% | 52.73% | 59.65% | 89.25% | 56.25% |
| a-small-30k | 36.10% | 44.45% | 53.63% | 83.00% | 62.25% |
| c-qwen08-30k | 50.15% | 64.09% | 89.22% | 85.75% | 60.25% |
| gliclass-x-base | 27.35% | 29.91% | 26.57% | 67.00% | 51.50% |

The frozen test states have zero exact or normalized matches in training or validation. The base 30k model loses 2.3 accuracy points when five questions are packed together, exceeding the 1.5-point packing allowance. See [tables.md](tables.md) for every language, calibration metric, and SemIf result.

## Training and compute

All students use the separate, consistent H200 Gemma 4 12B Q8 teacher export. The complete training split contains 5,998 states / 29,990 decisions; validation has 278 states / 1,390 decisions. The learning-curve subset contains 2,000 states / 10,000 decisions. No Codex labels were mixed in.

| Arm | Training stage wall time | Evaluation stage wall time |
|---|---:|---:|
| a-base-10k | 76.4 s | 34.2 s |
| a-base-30k | 186.8 s | 35.7 s |
| a-small-30k | 133.8 s | 32.7 s |
| c-qwen08-30k | 974.0 s | 77.1 s |
| gliclass-x-base | Not trained | 52.9 s |

Stage wall times include process startup, model loading, training or evaluation, and saving. Some model downloads and artifact compression overlapped training; these are operational timings from one run. The encoders use three epochs, a 16,384-token batch budget, and a 2,048-token target context. Qwen3.5-0.8B uses one full epoch, LoRA rank 16, and batches of 16 decisions. These different training configurations are the planned pilot arms, not a controlled architecture-only comparison.

The student Pod was allocated for **40.20 minutes**, at a verified GPU quote of $4.59/hour. Its estimated cost including running storage is **$3.08**. Including the prior teacher trial, the estimated total is **$6.25**, within the authorized $10 cap. These are elapsed-time cost estimates, not a settled provider invoice.

## H200 inference timings

These measurements use the published fresh-state/1, 5, 10, and 50-question protocol, with five warmups, 40 timed calls at 1/5/10 questions, and 15 at 50. They ran after training and evaluation, with GPU telemetry recorded. The [laptop follow-up](../pilot-laptop-20260927/README.md) subsequently measured the small encoder on four CPU threads: 136.80 ms for one question and 1644.39 ms for a 50-question request. All three laptop GPU attempts failed the cooling requirement before timing, so the GPU gate remains pending. H200 results are supplementary.

| Arm | 1 question p50 | 50 questions p50 | ms/question at 50 | Contaminated | Throttled |
|---|---:|---:|---:|---|---|
| a-base-30k | 12.45 ms | 13.74 ms | 0.275 | False | False |
| a-small-30k | 11.59 ms | 13.36 ms | 0.267 | False | False |
| c-qwen08-30k | 60.40 ms | 276.79 ms | 5.536 | False | False |

## Corrections and reproducibility

- Before training, the ordinal score loss was corrected to restore semantic level order after option shuffling. A regression test checks both the loss and gradients under permutation and padding.
- Encoder checkpoints now preserve CUDA RNG state and load the CPU RNG state onto the CPU when resuming. Periodic checkpoints preserve progress without double-counting elapsed time.
- The first Qwen attempt exited at FLA's safety check for Triton 3.4.0 on Hopper. TileLang 0.1.14 supplied the alternate backward kernel, and the control restarted from its original seed. The failed attempt and source-manifest amendment are retained; its time is included in rental cost.
- GLiClass 0.1.20 exposes input preparation on its underlying architecture pipeline, not its public wrapper. The evaluator was corrected to use that pipeline and preserve the complete option-logit vector; the failed baseline attempt is retained.
- The report can apply the predeclared quality stop when both evaluated A and C arms miss the accuracy floor, even if speed measurements are pending. Missing speed evidence still prevents a speed-based promotion.
- Validation: 46 offline pilot tests, Ruff on the changed Python files, bootstrap shell syntax, all downloaded file hashes, every evaluation ID/gold/probability/accuracy, and evaluated-versus-downloaded weight hashes.

The source snapshots and prediction rows are retained under `runs/student-artifacts-h200-20260927/`; model files are under its `runs/student-data/models/` directory. The original laptop teacher outputs remain untouched. Model binaries and training data are not committed.

The [measurement archive](measurement-artifacts.tar.gz) contains the raw prediction rows, logs,
executed source snapshots, input manifests, and environment record. Its download receipt is
[artifact_download.json](artifact_download.json); the per-file hashes are in
[artifact_manifest.json](artifact_manifest.json). Extract it into a new directory to inspect the
executed files without overwriting the working tree. The source used for training is preserved
even where the current scripts subsequently gained metadata or reporting improvements.

Records: [pilot.json](pilot.json), [results_validation.json](results_validation.json), [input_validation.json](input_validation.json), [artifact_manifest.json](artifact_manifest.json), [compute_receipt.json](compute_receipt.json), [cloud latency](cloud-latency/latency.json).
