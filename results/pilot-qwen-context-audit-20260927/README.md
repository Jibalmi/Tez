# Context audit and its limits

The [completed reconstruction](RESULTS.md) finds 20 of 29,990 original training decisions
(0.0667%) and five of 1,390 training-validation decisions (0.3597%) exceeded the original
1,536-token context. Actual tokenizer output confirms right truncation lost the complete
assistant footer on those rows. No example exceeds 4,096 tokens.

The [historical source check](historical_source_check.json) verifies that the archived H200
training code has structurally identical prompt, truncation, item-shuffle and epoch-batch
functions. Its archived launcher retains seed 13. This is a reproducible source/data
reconstruction rather than a log of historical token tensors.

This is a real prompt-integrity defect, but the affected fraction is small. It does not
establish the cause or size of the broad quality gap. Existing model weights and benchmark
scores are unchanged. The prepared continuation audits complete prompts under a 2,048-token
bound; its checkpoint-selection and full training implementation remain separate work.

Both archives have verified manifests. They contain code and token-length/hash metadata,
not training prompt text, targets or model weights.
