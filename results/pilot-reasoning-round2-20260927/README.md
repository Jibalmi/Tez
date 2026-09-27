# Bounded continuation with exact-rule supervision

Two registered laptop training jobs are queued under the shared compute lock. No improvement or
completed training is claimed yet. Both start from the identical longer-run encoder checkpoint.

- **Replay control:** original 29,990 training decisions plus 5,400 replayed decisions, sampled
  deterministically from training states only.
- **Rule supervision:** original 29,990 decisions plus 5,400 independently authored decisions that
  passed the existing leakage guards, including overlap checks against the new held-out sets.

Each receives 150 optimizer steps, the same token budget, learning rates and seed. Step zero is
eligible for selection. Validation checks at steps 50, 100 and 150 may replace it only while original
teacher-validation KL remains within +0.02 and agreement within -0.01 of that arm's measured baseline.
Among eligible checkpoints, selection minimizes fixed-weight validation KL. The new-data weight is
5,400 / 35,390. No public, diagnostic or confirmation labels enter selection. These are bounded
continuations, not a convergence study; the intervention changes both content and label certainty.

The [specification](../../experiments/small_model/reasoning_round2.json) fixes the settings before
training. [Data preparation](../pilot-reasoning-data-20260927/README.md) records the filtering outcome.
The confirmation evaluator refuses to open its labels until both training receipts show completion
and unchanged data manifests. It then compares the frozen selected models and their starting model
on fresh authored states/domains. Logical families recur, so this is not an unseen-family benchmark.

Any promising model still needs the original public regressions, calibration, packing, competitive
comparison and valid target-laptop timing. There is no automatic production promotion. Failed or
time-limited workers preserve partial checkpoints for an explicit, reproducible continuation.
No new RunPod resource or spending is authorized by this registration; this round is laptop-only.
