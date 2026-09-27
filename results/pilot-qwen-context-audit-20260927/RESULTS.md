# Qwen training-context audit

Reconstructed from the frozen teacher data, original option-shuffle seeds and pinned Qwen tokenizer.
The original training helper used right truncation at 1,536 tokens, including when fitting its validation temperatures.
Later inference and the frozen local validation reference allow 4,096 tokens.

| Input protocol | Decisions | Over 1,536 | Complete assistant footer lost | Maximum tokens |
|---|---:|---:|---:|---:|
| training_epoch0 | 29990 | 20 | 20 | 1987 |
| training_validation | 1390 | 5 | 5 | 1607 |
| current_validation_original_order | 1390 | 5 | 5 | 1607 |

Right truncation was verified against the tokenizer's actual output for every row.
Losing the complete assistant footer means affected examples train or validate a decision at a different prompt ending.
This audit does not prove how much accuracy that cost. Existing trained checkpoints and benchmark results remain unchanged.
A continuation should audit complete inputs before training and preserve the answer-position suffix.
[Full audit](audit.json). The archive includes source code and per-decision length/hash metadata; no training text, targets or weights.
