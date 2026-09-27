# Qwen laptop timing: completed, ineligible

Both executions completed, but neither is eligible for the unchanged laptop latency gate. No clean speedup is established.

Frozen Qwen 0.8B LoRA, BF16 autocast, original eager kernels on RTX 5080 Laptop GPU.
Same fresh-ticket protocol, 5 warmups per size; 40 timed requests at 1/5/10 questions and 15 at 50.
Raw requests, response vectors, source hashes and timing aggregates were independently checked.

| Diagnostic p50, ms per request | Full readout | Restricted readout |
|---|---:|---:|
| 1 questions | 88.22 | 90.10 |
| 5 questions | 112.67 | 117.46 |
| 10 questions | 238.54 | 234.23 |
| 50 questions | 1433.40 | 1373.36 |

These are diagnostic measurements under ineligible conditions, not a clean speed comparison.
Full readout reached 73 C and restricted readout 76 C, above the registered 70 C ceiling.
Neither reported hardware/software thermal slowdown at the recorded samples; the legacy `throttled` flag also covers temperature-limit violations.
Windows Desktop Window Manager held 667 MB of dedicated GPU memory before both runs, above the unchanged 300 MB foreign-process limit.
No unrelated processes were changed. No threshold was relaxed.

The latency gates remain 30 ms for one question and 2.8 ms/question for 50. Both diagnostic timings exceed them.
Across 1,720 decisions including warmups, readout argmax agreement was 100.0000%,
maximum probability difference 0.00256149, and mean total variation 0.00000745.
This repeated-ticket diagnostic does not replace the broader validation parity test.

model_info.kernels describes original H200 training, not this laptop's inference. Logs record current PyTorch fallback dispatch. The worker did not separately persist inference package versions or actual Torch thread count.

[Analysis](analysis.json), [raw timing metadata](latency.json), [protocol](registration.json).
