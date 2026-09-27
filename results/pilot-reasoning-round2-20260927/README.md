# Bounded continuation with exact-rule supervision

The completed rule-supervision continuation improves fresh authored rule accuracy from 35.61% to
54.67%, but does not improve public typed decisions: 39.05% to 38.80%. It is not promoted.
All three models completed the original 6,151-decision public suite and the 1,800-decision fresh
confirmation set. All 23,853 primary probability rows were independently validated.
[Full results](RESULTS.md), [machine-readable summary](summary.json),
[competitive research](RESEARCH.md), [verified artifact receipt](artifact_receipt.json).

The confirmation gain is +19.06 points, with a state-bootstrap interval of [+16.44, +21.67].
The typed change is -0.25 points, interval [-1.00, +0.45]. MASSIVE rises from 51.73% to 53.18%,
while routing and both SemIf scores fall. The original quality and valid laptop speed gates remain
unmet; these fresh authored examples demonstrate transfer within recurring rule families only.

Both registered laptop training jobs completed 150 steps under the shared compute lock, starting
from the identical longer-run encoder checkpoint. Rule supervision selected step 150; replay
selected step zero. The replay and starting models agree on every public argmax; small bf16
probability differences are retained in the parity report.

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
The confirmation evaluator refused to open its labels until both training receipts showed completion
and unchanged data manifests. It compared the frozen selected models and their starting model
on fresh authored states/domains. Logical families recur, so this is not an unseen-family benchmark.

Any promising model still needs the original public regressions, calibration, packing, competitive
comparison and valid target-laptop timing. There is no automatic production promotion. Failed or
time-limited workers preserve partial checkpoints for an explicit, reproducible continuation.
No new RunPod resource was launched and this round added no cloud charge. The previous combined
elapsed-rate estimate remains $9.1452 of the authorized $10 cap; this is not a provider invoice.

The inherited public scorer computes evaluation-label temperature diagnostics, which are excluded
from model selection and primary comparisons. See the [scope correction](scorer_scope_correction.json).
New rule targets are exact oracle labels. Compatibility metadata naming their self-agreement
`teacher_order_agreement` does not denote an LLM teacher measurement.
