# Qwen LoRA feasibility result

The [registered probe](registration.json) completed one update on the longest scheduled
training input, 1,650 tokens, without truncation. The [execution receipt](probe.json) records
finite loss and all LoRA gradients, a positive pre-clipping gradient norm, and 372 changed
adapter tensors. Peak allocated GPU memory was 1.9953 GiB; maximum sampled temperature was
72 C. The job used four threads and the shared compute lock on the RTX 5080 Laptop GPU.

Probe weights were discarded. This establishes a successful one-step feasibility check, not
training throughput, sustained thermal behavior, checkpoint replay or improved model quality.
The archive verifies the executed sources and receipts; it does not independently replay the
discarded gradient tensors. Full continuation requires its own frozen protocol and step-zero
validation against the original probability reference.
