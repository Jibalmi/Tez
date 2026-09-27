# Qwen length-bucket graph parity, batch 16

The complete 1,390-decision validation **failed** the unchanged numerical limits. The independent reader checked both sets of predictions and all 89 batch receipts.

Reference argmax agreement: 0.99568345; maximum probability change: 0.03596536; mean total variation: 0.00623822. Graph versus identical uncaptured maximum change: 0.00000000.

These are the original frozen weights, with unmerged LoRA, native attention, cache disabled and a 26-letter head. Every input is complete and left-padded to a strictly larger registered length bucket. Final incomplete batches duplicate their last row; dummy outputs are excluded. Each length has one capture reused across its batches.

This run measures probability parity only. Capture, padding, native-mask construction and buffer copying must be accounted for in any future speed measurement. A pass does not establish concurrent graph caching, public accuracy, speed, or a release gate. A failure remains preserved without relaxing its limits.
