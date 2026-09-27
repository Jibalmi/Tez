# Qwen native-prefix graph parity, nominal batch 16

The complete 1,390-decision validation **passed** the unchanged numerical limits. The independent reader checked both sets of predictions and all 87 batch receipts.

Reference argmax agreement: 0.99712230; maximum probability change: 0.03446219; mean total variation: 0.00404363. Graph versus identical uncaptured maximum change: 0.00000000.

These are the original frozen weights, with unmerged LoRA, native attention, cache disabled and a 26-letter head. Each original character-sorted batch remains an unchanged left-padded prefix of a larger graph input. Only masked trailing tokens are added; readout selects the original final-token column. The short final batch keeps its original size, with no dummy rows. Whole batches are grouped for capture reuse without changing their composition or real-token positions.

This run measures probability parity only. Capture, padding, native-mask construction and buffer copying must be accounted for in any future speed measurement. A pass does not establish concurrent graph caching, public accuracy, speed, or a release gate. A failure remains preserved without relaxing its limits.
