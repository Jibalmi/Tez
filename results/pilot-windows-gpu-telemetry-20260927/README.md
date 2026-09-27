# Windows GPU identity and eligibility telemetry

The new read-only logger associates Windows memory counters with the CUDA device's LUID
and physical node, and NVML readings with its PCI address and UUID. It records its own PID,
counter status, exact bytes, full counter paths, temperatures and thermal event flags.
It rejects missing or ambiguous data instead of treating unavailable telemetry as clean.
No model was loaded and no GPU workload, settings or desktop applications were changed.

[Snapshot two](snapshot-02/probe.json) completed on the RTX 5080 Laptop GPU at
2026-09-27T12:17:32.962082+00:00. Its current CUDA context was absent. Both temperature
samples were 76 C, with neither thermal event flag active. DWM, TradingView and Chrome
each exceeded the existing 300 MiB per-process dedicated-memory limit on this same GPU;
another Python GPU workload was also present. It is **not eligible for a latency claim**.
The snapshot was taken while graph validation held the shared compute lease. It diagnoses
the reader and current conditions, not an inference benchmark or future machine state.

Windows per-process memory can include allocations shared across processes, so these rows
must not be summed into physical GPU usage or interpreted as proof of active compute
contention. The adapter total is recorded separately. This interpretation follows
[Microsoft's GPU memory explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).
The pre-existing 300 MiB eligibility rule remains unchanged, alongside starting temperature
at most 65 C and measured temperatures at most 70 C with no thermal slowdown flags.

[Snapshot one](snapshot-01/probe.json) failed closed because two Windows instances on
another adapter shared an `InstanceName`. The raw readings and original sources are retained.
Windows distinguishes such instances using an index in the full path, as documented by
[Microsoft's counter-path structure](https://learn.microsoft.com/en-us/windows/win32/api/pdh/ns-pdh-pdh_counter_path_elements_w).
The revised logger preserves that index and excludes other adapters from the target's memory
assessment. Ambiguous indexed readings on the target adapter still make the assessment
unavailable; they are never silently combined or dropped.

Six focused CPU tests cover adapter/node attribution, own-PID exclusion, the exact existing
threshold, shared-allocation accounting, invalid/missing/duplicate counters and thermal
eligibility. Those tests and Ruff pass. Each snapshot includes its raw readings, executed
sources and a SHA-256/size manifest verified by rereading every listed file. No graph source,
live registration, previous latency evidence or global compute-lock helper was changed.

```powershell
python experiments/small_model/probe_windows_gpu_telemetry.py --output NEW_DIRECTORY
```

This records a single bounded sensor check and refuses to overwrite an existing directory.
It is not an admission decision for a running model: future timing code must use its own
worker PID and collect telemetry around the actual timed requests.
