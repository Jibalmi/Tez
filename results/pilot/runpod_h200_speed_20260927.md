# H200 teacher trial, 2026-09-27

Status: **complete; Pod terminated and its absence verified**. All 6,276 states / 31,380 decisions were exported locally. Production took **18.52 minutes**. Allocation through termination took 41.28 minutes, including setup and diagnostics. Estimated total cost: **$3.17**, including storage, within the $10 cap. Provider billing records were not yet available at the final check; this is a rate-times-duration estimate, not a settled invoice.

The teacher remains Gemma 4 12B Q8_0. We regenerated all 6,276 states / 31,380 decisions as a separate dataset. The 3,170 saved laptop states are retained.

## Worker comparison

100 states / 500 decisions per trial. Times include worker startup, model loading, inference, checkpoints, and exit. Earlier diagnostics had warmed the runtime.

| Workers | Seconds | States/second | Scores versus fresh single-worker reference |
|---:|---:|---:|---|
| 1 | 25.006 | 3.999 | Reference |
| 2 | 22.009 | 4.544 | Identical |
| 4 | 24.015 | 4.164 | Identical |
| 6 | 24.022 | 4.163 | Identical |

Two workers were selected from this single trial per configuration. Four and six workers fit in VRAM but did not finish this sample faster. This does not establish an optimum for every workload or batch size.

## Probability checks

The original resume attempt failed the predefined laptop/cloud probability gate: maximum probability delta 0.055705, mean total variation 0.005549, and 99.4% argmax agreement across 500 decisions. No cloud rows from that attempt were merged with the laptop dataset.

A diagnostic rerun with a different subset also changed scores on the same H200. Setting `cache_prompt=False` removes reuse across states while preserving prefix sharing within each state. With that setting, the 100-state one-worker and two-worker runs, and a 20-state subset rerun, produced exactly the same stored scores. The four-worker and six-worker checks also matched.

The new cloud reference still differs from the old laptop reference: maximum probability delta 0.074381, mean total variation 0.005588, and 99.4% argmax agreement. This is a distinct execution protocol; the original compatibility check has not passed. Full regeneration avoids mixing those targets. The numerical source of the old cross-platform differences has not been completely isolated.

## Outputs and validation

Production outputs, full logs, manifests, and diagnostic results were downloaded and all 400 archived files passed SHA-256 verification before the Pod was terminated. The 100-state reference subset in production matched the cloud reference exactly: zero probability delta across 500 decisions. The exporter validates complete IDs, question metadata, finite logits, and production/reference consistency. It also writes explicit temperature-adjusted option probabilities. These are not probabilities that an answer is correct.

42 offline pilot tests, lint for the changed Python files, and the Bash syntax check passed. The real training reader loaded 5,998 training states (29,990 decisions), a 2,000-state learning-curve subset (10,000 decisions), and 278 validation states (1,390 decisions), with disjoint training/validation IDs. Student training and evaluation have not started.

Raw measurement record: [runpod_h200_speed_20260927.json](runpod_h200_speed_20260927.json).

## Matched comparison with Codex

Same 160 states / 800 assigned decisions. One recorded trial per configuration; Codex results are the earlier measurements. GPU outputs include full two-order option distributions; Codex outputs selected labels. Prompts, execution environments, and output products differ.

| Backend | Workers | Total seconds | Gold answers correct |
|---|---:|---:|---:|
| H200 Gemma 4 12B Q8 | 2 | 72.015 | 68/80 |
| GPT-6 Astra, low effort | 4 | 50.026 | 75/80 |
| GPT-6 Astra, low effort | 8 | 35.855 | 74/80 |
| GPT-6 Astra, low effort | 20 | 35.003 | 75/80 |

The H200 matched test took 72.015 seconds including a measured **42.0-second model load per concurrent worker**. After loading, the workers spent 25.6 and 28.2 seconds processing their partitions. These processing-only times exclude startup and must not be compared directly with Codex end-to-end time. All 800 repeated H200 probability vectors matched the full production dataset exactly. Codex was faster on this measured total-time sample; the long GPU run amortized model loading over the full corpus. The small gold sample does not establish a general quality ranking.

## Saved dataset

- [Dataset manifest](../../runs/pilot-cloud-h200-20260927/manifest.json)
- [Explicit option probabilities](../../runs/pilot-cloud-h200-20260927/probability-targets.jsonl)
- Raw teacher shards: `runs/pilot-cloud-h200-20260927/teacher/`
- Raw artifacts: `runs/pilot/runpod-results-20260927/`
- Final archive SHA-256: `e7a9236993c9b15ef9bb64c6b4fec3d93e7a99f497bb6fb628ecca4315917ace`

For later student training, set `TEZ_PILOT_RUNS` to the exported dataset directory and `TEZ_PILOT_RESULTS` to a new results directory. The original `runs/pilot/teacher/` and its 3,170 states are unchanged. The queued laptop driver was stopped before it could duplicate this work. No Pod or Pod volume from this trial remains rented.
