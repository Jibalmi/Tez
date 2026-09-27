# Laptop follow-up, 2026-09-27

The small encoder completed CPU timing. All three GPU attempts failed the predeclared cooling
requirement before timing began, leaving the GPU speed gate pending.

CPU: Intel Core Ultra 9 275HX, four worker threads, fp32, original verified `a-small-30k` weights.
The shared compute lock was held. System CPU load before loading was 25.8%; this was an interactive
laptop, not an otherwise idle laboratory host. Model load took 10.49 seconds and final process RSS
was 1.42 GiB. Five warmups precede each size; 40 timed requests at 1/5/10 questions and 15 at 50.
Each request uses a fresh state from the original Laya-compatible protocol.

| Questions per request | Median request, ms | p95 request, ms | Median ms/question |
|---:|---:|---:|---:|
| 1 | 136.80 | 166.60 | 136.796 |
| 5 | 251.23 | 268.93 | 50.246 |
| 10 | 400.92 | 414.43 | 40.092 |
| 50 | 1644.39 | 1795.94 | 32.888 |

The RTX 5080 Laptop GPU remained above the 65 C starting requirement after each 90-second cooling
wait. These failed attempts contain no usable GPU latency measurement. An earlier attempt was
stopped during cooling; a prior launcher failed before model execution. Those failures are retained.
Other desktop processes held GPU memory. They were not stopped or reconfigured.

The Windows contamination check now accepts lowercase `dedicated` as well as Linux's label.
Missing thermal readings cannot report an unthrottled run. CPU timings do not satisfy GPU gates,
and H200 timings remain in a separate report.

Records: [latency.json](latency.json), [execution.json](execution.json),
[weight/protocol validation](validation.json), [CPU inventory](cpu.json).
The [measurement archive](measurement-artifacts.tar.gz) also preserves the failed launch and cooling
logs; its [receipt](artifact_download.json) and [manifest](artifact_manifest.json) verify the saved files.
