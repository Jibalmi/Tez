# Competitive follow-up, 2026-09-27

The local [matched comparison](../pilot-improvement-20260927/COMPARISON.md) remains the evidence for
our scores. Sixteen model configurations have recorded quality results; a win over an earlier Tez
checkpoint does not establish competitiveness with the strongest relevant alternative.

The latest source review supports two comparison tiers:

- **Small-model substitutes:** Decider-0.8B, Kev-0.8B, Laya and GLiClass. These already have matched
  public results. The next useful comparison is the unchanged Decider and Qwen checkpoints on the
  fresh authored rules, to test whether improving the encoder closes a gap or solves a shared weakness.
- **Larger quality references:** current Decider-2B/4B releases and hosted decision engines. These
  need their own runs and latency/cost conditions; published leaderboard positions are not matched
  accuracy measurements in this project.

The Decider-0.8B publisher describes one epoch over 1.47 million examples and 455 million tokens,
and explicitly warns that instructions containing rules are unreliable. This is substantially more
training coverage than our pilot, but still does not establish competence on our new rule cases.
Those are publisher statements, not our measurements.
[Model card](https://huggingface.co/Mapika/decider-0.8b).

The current Decider repository also reports newer 2B/4B hard-decision adaptations and records their
regressions and failed release gates. This strengthens the case for measuring both new capability
and old-task retention before adopting a continuation. It does not prove that copying its recipe
will work for this encoder.
[Release notes](https://github.com/Mapika/decider#whats-new).

Two external checks can broaden later evaluation. JevBench combines capability, calibration, speed
and cost and includes sealed cases. Its composite is not typed-decision accuracy.
[JevBench](https://github.com/fstandhartinger/jevbench).
Decision Index 0.2 supplies a reproducible suite and native scorers, with coverage and chance
corrections. Its full rebuild has substantial data/storage requirements and source-specific terms;
it has not been run here. A sample would be a diagnostic, not an official complete index.
[Decision Index kit](https://github.com/apolinario/decision-index).

No extra cloud resources were launched for this review. The combined $10 cap and the original
quality, calibration and valid laptop latency gates remain unchanged.
