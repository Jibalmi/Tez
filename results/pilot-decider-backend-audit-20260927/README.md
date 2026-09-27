# Decider speed-comparison boundaries

The existing pilot uses `PredictDecider(... use_graphs=False)`: native BF16 eager inference,
temperature one, native prompt layout and option probabilities. Those recorded measurements
remain useful as an eager comparison, but they do not establish Decider's best available speed.

The [static source audit](audit.json) inspected the unchanged clean upstream checkout at
`a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f`. The following are source defaults, **not laptop results**:

| Interface | Graphs on CUDA | Compilation | FP8 | Convolution patch | Capture policy |
|---|---|---|---|---|---|
| `Decider` Python library | Yes by default | On | Off | On | First use of a shape |
| `decider.serve` / `EngineV2` | Yes by default | Off | Off | Off when compilation is off | Startup warmup, then sealed |
| Pilot comparison adapter | Disabled explicitly | Eager | Off | Unpatched | No captures |

The HTTP server also enables shared-prefix scoring by default, subject to a 768-token minimum.
Whether a particular workload takes that path must be measured; enabling it does not prove a
cache hit or speed gain. The library and HTTP server have different default configurations,
so “default optimized Decider” is not a sufficiently precise benchmark label.

The server's bucket grid and token budget are configurable. Any workload-specific grid used
to fit the laptop must be disclosed as such. Initialization, graph capture and warm-request
latency need separate reporting. Comparing a new optimized Qwen laptop path only with old
H200 eager Decider timings would mix hardware and implementation changes. A fair speed claim
requires new matched hardware, input, output, eligibility and timing scope, with the native
optimized competitor's probabilities verified before timing. HTTP and in-process overhead
must be distinguished. Upstream CUDA tests have not been run here and are not our evidence.

The [35,208-byte evidence archive](artifact_receipt.json) retains the audit, current pilot
adapter and seven upstream files, including the license. All nine members were reread and
checked against the hash/size manifest. The local checkout was read only; neither it nor the
frozen pilot adapter was modified. Attempts to retrieve these exact upstream files through
the web reader returned cache misses, so this audit relies on the inspected local checkout.
No model, server, compiler, GPU experiment or paid resource was launched.
