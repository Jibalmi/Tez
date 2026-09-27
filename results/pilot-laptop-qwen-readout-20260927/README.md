# Laptop Qwen output-readout timing

The [registration](../../experiments/small_model/qwen_readout_latency_registration.json)
preserves the original fresh-state 1/5/10/50-question protocol, five warmups, 40 timed calls
at the first three sizes and 15 at 50. It compares the frozen full-vocabulary Qwen readout with
the restricted projection that passed all 1,390 original validation decisions.

Original starting-temperature, peak recorded temperature, throttling, dedicated-VRAM and
shared-lock requirements remain in force. Each request's timing and log probabilities are
retained; writing them happens outside the timed interval. Both before and after contamination
snapshots must be clean. Execution success alone cannot certify a latency gate.

This is a new bounded attempt after the laptop was observed at 57–61 C and the older job
handles and matching OS processes were absent. Earlier cooling failures and interrupted
quality comparisons remain preserved. No cloud resource or unrelated process is changed.
