# CUDA graph compatibility result

The four fixed non-benchmark cases passed in original, reversed and restored batch order. An independent reader checked all 36 saved probability vectors and their case membership.

Full-reference argmax agreement: 1.000000; maximum probability change: 0.00000000; mean total variation: 0.00000000. Captured versus identical uncaptured maximum change: 0.00000000.

This is compatibility evidence for four cases at one tensor shape, not a speed measurement. Full validation and the actual serving shapes remain to be tested before latency or promotion. The first metadata-cache failure remains preserved separately.

[Raw probabilities](probe.json), [independent analysis](analysis.json), [registration](registration.json).
