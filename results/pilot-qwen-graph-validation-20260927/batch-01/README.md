# Single-question graph validation: incomplete at deadline

The original left-padding candidate reached its registered 720-second soft limit and raised
`TimeoutError`. The child exited with code 1 and released the shared compute lock. This is a
terminal failed attempt, not an observation timeout or a completed parity result.

Both raw output files contain 1,348 unique decisions, matching the completed batch receipts;
42 of the planned 1,390 decisions remain unmeasured. Four shapes were captured: 1x128, 1x256,
1x512 and 1x1024. The 1x2048 shape was not reached. The complete-validation verifier therefore
cannot accept this run, and no aggregate parity pass or speed claim is made from the subset.

The worker recorded 720.2531 operational seconds, including 189 thermal pauses totaling
230.2894 seconds, with a maximum sampled temperature of 83 C. These timings include validation
work and thermal pacing; they are not an inference latency or throughput benchmark.

The original protocol, executed sources, complete inputs, partial graph and uncaptured outputs,
traceback and terminal receipts are retained. The failure archive checks source provenance and
every archived file hash. It does not certify full probability parity. No source, probability
limit, deadline or completed artifact is changed to turn this result into a pass.

The separate batch-16 arm of this candidate completed but failed mean total variation. A different
[native-prefix candidate](../../pilot-qwen-prefix-validation-20260927/README.md) is registered and
queued at batch 16; its GPU behavior is still unproven. This incomplete attempt is not silently
restarted, resumed or mixed into that candidate's evidence.
