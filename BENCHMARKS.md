# Tez benchmarks

Every number here was measured on one machine (RTX 5080 Laptop 16 GB, llama.cpp b11100, Gemma 4 12B Q8_0 unless stated), frozen, zero training. Laya's three checkpoints answered **byte-identical rows on the same GPU**; SemIf's published figures are on SemIf's own public fixtures; **Jev was never run here** (closed API) — its figures are third-party published and marked as such. Raw rows and manifests: `results/`. Figures: `docs/figures/`, all-in-one: `docs/figures/tez_dashboard.png`.

![dashboard](docs/figures/tez_dashboard.png)

---

## 1. Same rows, same machine: Tez vs Laya (Laya's own benchmarks)

`experiments/bench_h2h.py`. MASSIVE follows Laya's protocol (20 options = gold + 19 sampled, seed 13, first 100 test rows per language). Banking77 uses a chunked tournament for Tez (77 > 26 letters; four chunks of 20 + "none", winners meet in a final). Laya runs in-process with its shipped temperatures.

| task (n) | **Tez zero-shot** | laya | laya-multilingual | laya-typed-decisions | Jev (published) |
|---|---:|---:|---:|---:|---:|
| AG News (400; in Laya's training mix) | 0.880 | **0.932** | 0.948 | 0.932 | 0.910 |
| DAIR Emotion (400) | 0.540 | **0.550** | 0.487 | 0.547 | 0.480 |
| Banking77 (400, 77 options) | **0.713** | 0.395 | 0.357 | 0.388 | 0.870 |
| SST-5 (400, ordinal score) | **0.512** | 0.362 | 0.280 | 0.460 | — |
| BoolQ (400, noul; in Laya's mix) | **0.850** | 0.843 | 0.782 | 0.835 | — |
| prompt-injections (116, noul, held out) | **0.759** | 0.672 | 0.569 | 0.647 | — |
| MASSIVE intent, English (100, 20 options) | **0.900** | 0.830 | 0.710 | 0.820 | — |
| XNLI, English (100) | 0.730 | 0.900 | 0.860 | **0.920** | — |
| typed-decisions (400 cases, 2,000 decisions) | **0.704** (4-shot in cached prefix: **0.725**) | 0.362 | 0.352 | 0.766 † | 0.727 |
| … soft accuracy vs teacher distribution | **0.575** (4-shot **0.583**) | 0.331 | 0.328 | 0.471 | 0.580 |
| … Brier vs teacher distribution | 0.355 | 0.316 | 0.463 | **0.061** | 0.148 |
| … score MAE | 0.482 | 0.694 | 0.761 | **0.242** | 0.391 |

† fine-tuned on this benchmark's own training split (Laya's own caveat). Laya's published numbers reproduce in this harness (base 0.362 / 0.362, fine-tuned 0.766 / 0.766, Khmer 0.000 / 0.000), which validates the comparison.

**typed-decisions by workflow (Tez / laya-td):** agent-trace observability 0.592 / 0.732 · customer service 0.768 / 0.764 · invoice processing 0.768 / 0.804 · security incidents 0.688 / 0.766. **By primitive (Tez):** noul 0.81 · choice 0.68 · score 0.64.

### Languages — MASSIVE intent, 20 options, 100 cases each

| lang | **Tez** | laya | laya-multilingual | laya-td |
|---|---:|---:|---:|---:|
| en | **0.900** | 0.830 | 0.710 | 0.820 |
| de | **0.870** | 0.420 | 0.500 | 0.390 |
| fr | **0.870** | 0.590 | 0.600 | 0.590 |
| es | **0.890** | 0.510 | 0.580 | 0.460 |
| ja | **0.930** | 0.520 | 0.640 | 0.550 |
| zh-CN | **0.910** | 0.620 | 0.650 | 0.590 |
| ar | **0.870** | 0.120 | 0.460 | 0.080 |
| hi | **0.900** | 0.090 | 0.470 | 0.110 |
| th | **0.890** | 0.080 | 0.470 | 0.100 |
| ko | **0.910** | 0.110 | 0.470 | 0.060 |
| km | **0.790** | 0.000 | 0.210 | 0.050 |
| **macro** | **0.885** | 0.354 | 0.524 | 0.345 |
| languages > 3× random | **11/11** | 6/11 | 11/11 | 6/11 |
| macro ECE (shipped) | **0.099** | 0.487 | 0.299 | 0.338 |

### XNLI (100 per language) — the one family where Laya's trained encoder is ahead

| lang | Tez | laya | **laya-multilingual** |
|---|---:|---:|---:|
| en | 0.730 | 0.900 | 0.860 |
| de | 0.680 | 0.680 | 0.800 |
| fr | 0.650 | 0.760 | 0.850 |
| es | 0.740 | 0.800 | 0.800 |
| ar | 0.680 | 0.430 | 0.780 |
| hi | 0.710 | 0.460 | 0.710 |
| th | 0.700 | 0.390 | 0.760 |
| zh | 0.680 | 0.710 | 0.790 |
| ru | 0.710 | 0.560 | 0.780 |
| tr | 0.710 | 0.400 | 0.810 |
| **macro** | 0.699 | 0.609 | **0.794** |

### Calibration, order robustness, latency

| | Tez | laya | laya-ml | laya-td | Jev (published) |
|---|---:|---:|---:|---:|---:|
| mean ECE-15 as shipped | **0.212** | 0.323 | 0.253 | 0.226 | 0.246 |
| mean ECE-15 after one temperature per task (2-fold OOF) | 0.078 | 0.071 | 0.103 | **0.068** | — |
| option-order flip, MASSIVE-en (20 options) | **0.07** | 0.18 | 0.20 | 0.15 | 0.13 |
| option-order flip, Emotion / AG News | 0.07 / 0.03 | 0.06 / 0.00 | 0.16 / 0.00 | 0.06 / 0.00 | — |
| ms per decision, p50 (this GPU) | 57–250 | 25–50 | 23–35 | 36–52 | 236–276 |

Tez's per-decision time on these tasks is dominated by evaluating the **state** (it is the variable suffix); on constant-option / short-suffix workloads (the voice loop below) it is 30 ms.

---

## 2. SemIf's benchmark (authored144 · perturbations108), frozen models

`experiments/run_direct.py`, `metrics.py`, `calibrate.py`, `perm_analysis.py`, `run_hf.py`. Mean-family balanced accuracy, classes keyed by option id, 95 % CI by source-group bootstrap.

| Model | authored144 mfba (95 % CI) | perturbations | NLL raw → best | Brier raw → best | reversal flips / 36 | p50 |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma 4 12B Q8_0** | **0.943** (0.897–0.981) | **0.992** | 0.478 → **0.158** | 0.094 → **0.075** | **0** (≤ 9.6 %) | 110 ms |
| Gemma 4 12B Q4_K_M | 0.918 (0.871–0.958) | 0.981 | 0.549 → 0.184 | 0.130 → 0.092 | 1 | 100 ms |
| Qwen3.5-4B BF16 (SemIf's model, in-process) | 0.813 (0.755–0.865) | — | 0.427 | 0.248 | — | 224 ms |
| Qwen3.5-4B + 6-permutation batch | **0.912** | — | — | — | — | 321 ms |
| Qwen3-4B BF16 (in-process) / + 6-perm | 0.740 / 0.832 | — | — | — | — | 75 / 204 ms |
| Qwen3.5-9B Q8_0 (llama.cpp, `qwen3` template) | 0.912 (0.863–0.955) | 0.868 | — | — | 3 (8 %) | 92 ms |
| Gemma 3 4B Q4_K_M | 0.646 (0.570–0.728) | 0.729 | 3.433 → 0.696 | 0.661 → 0.397 | 13 (36 %) | 40 ms |
| Llama 3.2 3B Q4_K_M | 0.358 (0.319–0.399) ≈ chance | 0.496 | 1.673 → 1.043 | 0.827 → 0.627 | 10 (28 %) | 25 ms |
| *SemIf published: Qwen3.5-4B / EXL3 27B* | *0.813 / 0.958* | *0.766 / —* | | | | |

Paired McNemar on the same 144 rows: 12B Q8 vs Qwen3.5-4B p = 5e-5; 12B vs Qwen3.5-4B **with** 6-permutation averaging p = 0.18 (not significant); Q8 vs Q4 p = 0.25. SemIf's 0.813 reproduces exactly (0.8132).

**Calibration recipes (Gemma 4 12B Q8, 144 rows, OOF temperature):** raw NLL 0.478 / Brier 0.094 / ECE 0.047 · temperature 0.193 / 0.083 / 0.027 · 6-permutation mean 0.281 / 0.083 / 0.038 (best error-detection AUROC 0.933) · **6-perm + temperature 0.158 / 0.075 / 0.037** · Zhao contextual calibration accuracy **0.951 → 0.778** (harmful: empty evidence legitimately selects "insufficient"; the 3-placeholder average does not rescue it). Letter prior A/B/C = 0.331/0.336/0.333 (PriDe is a no-op). Three Q8 runs byte-identical.

---

### Backbone check: Qwen3.5-9B Q8_0 vs Gemma 4 12B Q8_0 (same rows, same harness)

| task | Qwen3.5-9B | **Gemma 4 12B** |
|---|---:|---:|
| SemIf authored144 / perturbations | 0.912 / 0.868 (8 % reversal flips) | **0.943 / 0.992** (0 %) |
| typed-decisions (2,000) / soft acc | 0.622 / 0.498 | **0.704 / 0.575** |
| MASSIVE en / ja / ar / km | 0.88 / 0.87 / 0.70 / 0.69 | **0.90 / 0.93 / 0.87 / 0.79** |
| Banking77 / SST-5 / BoolQ / prompt-injections | 0.642 / 0.430 / 0.812 / 0.621 | **0.713 / 0.512 / 0.850 / 0.759** |
| order flip, MASSIVE-en / ja | 0.09 / 0.19 | **0.07 / —** |
| JevBench public original / easy (intelligence) | 94.0 / 100 | see §5 |

The 9B saves ~3 GB and nothing else; the 12B stays the backbone.

## 3. Voice → instant action (220 commands, 16 actions)

`experiments/run_voice_fast.py`, `stream_analysis.py`, `stream_policy.py`, `residual_check.py`, `asr_bench.py`, `voice_demo.py`.

| | transcript last (used) | transcript first (control) | "still speaking" wording |
|---|---:|---:|---:|
| intent accuracy, strict / accepting then·alt | **0.905 / 0.982** | 0.877 / 0.955 | 0.868 / 0.968 |
| out-of-scope → none, recall / precision | 1.000 / **0.917** | 0.909 / 0.769 | 0.909 / 0.833 |
| per streamed word: compute / HTTP round-trip (p50) | **30 / 45 ms** | 148 / 221 ms | 30 / 45 ms |
| tokens evaluated per word | **11** | 358 | 11 |

Before `--swa-full` the server re-evaluated all 394 tokens every word (205 ms) — Gemma's sliding-window cache cannot be rolled back without it.

**Streaming (198 actionable utterances):** naive commit at p ≥ 0.9 fires `none` at word 1 (17 % accurate). Rule: a confident `none` on a prefix never commits. Non-none commit: 98.5 % commit at 0.918 accuracy, mean word 2.5 of 5.4, 93 % at/before the human commit word. **Class-aware policy** (opens fire on one partial; media/play need two; type/close at end of utterance; refinement graph open_x → play_x): **harmful actions 1/198**, final action consistent 0.985, first action at mean word 3.6, out-of-scope false actions 2/22. Compound commands as two decisions on the residual with a `prior`: 22/22.

**ASR (faster-whisper, synthetic SAPI speech, 200 ms chunks):** base.en partial p50 41 ms, WER 7.8 %, intent-from-ASR agrees with intent-from-text 97.3 %; small.en 120 ms, 7.3 %, 97.7 % — base.en is the pick. **Live from audio:** "open notes and type hello world" → Notepad launched at **+652 ms** (1.3 s before speech ends), text typed at end of speech. Per-word budget ≈ 40 ASR + 30 decide + 15 HTTP + <1 policy ≈ 85 ms.

---

## 4. Abstention, debiasing, cascades

- **Conformal sets, typed-decisions (2,000 decisions, Mondrian by question type, calibrated on half the cases):** at 90 % target coverage Tez covers 90.1 %, acts (set size 1) on **52 %** of decisions at **0.853** accuracy, escalates the rest, zero fallbacks; at 95 %: acts on 37 % at 0.912. Laya fine-tuned: 63 % at 0.876; Laya base: 5 %. (`experiments/conformal_td.py`)
- **Batch Calibration (Zhou et al. 2023), out-of-fold:** NLL improves on 30/33 sets (prompt-injections 2.89 → 1.03, XNLI-th 2.18 → 1.26, SST-5 4.04 → 3.26, typed-decisions 1.75 → 1.42); accuracy +1–4 pts on XNLI and prompt-injections, unchanged on most tasks, −4 on Khmer, −1 on typed-decisions. (`experiments/batch_calibration.py`)
- **Small→large cascade (Gemma 3 4B → 12B, escalate on ordering disagreement): negative result** — 0.854 at 117 ms vs the 12B alone 0.951 at 110 ms. (`experiments/cascade_analysis.py`)

---

## 4b. Ablations on the 12B (typed-decisions unless stated)

| lever | result |
|---|---:|
| 4 worked examples of the same question in the cached prefix (train split, no training) | 0.704 → **0.725** (p = 0.053), soft 0.583, ECE-refit 0.038 |
| thinking budget before the readout, 32 / 128 tokens (s1-style seeded thought) | 0.687 → 0.657 (n = 300, p = 0.12) / 0.753 → 0.753 (n = 150); 2.0 s / 4.3 s per decision — **null/negative** |
| thinking budget 64 tokens on the JevBench hard tier (111) | 0.703 → 0.658 (p = 0.27); temporal_numeric 0.13 → 0.27, long_policy 0.79 → 0.58; 2.6 s per decision — **negative** |
| answer symbols: digits / lowercase / uppercase letters (SemIf rows) | 0.947 / 0.943 / 0.943 — **no effect at 12B** |
| ordinal expected-value readout instead of argmax (SST-5, score questions) | no change for Tez; hurts Laya-td — **null** |
| Batch Calibration (out-of-fold) | NLL down on 30/33 sets; accuracy ±1–4 pts task-dependent |
| conformal sets (90 % coverage) | act on 52 % of decisions at 0.853 accuracy |
| 4B → 12B cascade | 0.854 @ 117 ms vs 12B alone 0.951 @ 110 ms — **negative** |

### Hidden-state probe (Hidden Calibration, `experiments/hidden_probe.py`) — frozen Qwen3.5-4B, typed-decisions

| readout | test accuracy (2,000) | NLL |
|---|---:|---:|
| letter logits (the model's own one-pass answer) | 0.490 | 1.18 |
| logreg probe on final hidden state (layer −1) | 0.772 | 0.65 |
| logreg probe, layer −4 / −8 | 0.791 / 0.792 | 0.52 / 0.51 |
| **logreg probe, layer −12** | **0.794** | **0.51** |
| nearest-centroid, layer −12 | 0.715 | — |
| Gemma 4 12B Q8, final-layer last-token state via llama-server `--embeddings --pooling last` (L2-normalised), logreg / centroid | 0.735 / 0.724 | 0.67 |

**Full layer sweep (Qwen3.5-4B, 33 layers, `hidden_probe_sweep.py`):** 0.483 (layer 0) · 0.574 (12) · 0.661 (14) · 0.737 (16) · 0.777 (18) · **0.790–0.793 (20–28, best 26)** · 0.787 (30) · 0.772 (32). **Data efficiency at layer 26 (rows per question, 3 seeds):** 10 → 0.669 · 25 → 0.707 · 50 → 0.741 · 100 → 0.760 · 200 → 0.788 · all 300 → 0.793. **Ensemble** probe ⊕ letter logits (w = 0.25): 0.799. **Conformal on the probe, 90 % coverage:** act on 66.6 % at 0.884 accuracy (12B letters: 52 % @ 0.853; laya-td: 63 % @ 0.876).

**Early readout (`early_exit_bench.py`, Qwen3.5-4B truncated to its first N layers, 200 typed-decisions prompts, 227 tokens mean, eager PyTorch bf16, CUDA-synchronised):**

| layers kept | prefill p50 | speed-up | probe accuracy at that layer |
|---|---|---|---|
| 32 (full) | 387 ms | 1.00× | 0.772 (final) |
| 28 | 351 ms | 1.10× | 0.787 |
| 24 | 296 ms | 1.30× | 0.793 |
| 20 | 250 ms | 1.54× | 0.790 |
| 16 | 200 ms | 1.94× | 0.737 |

Reading the decision at layer 20–24 keeps the probe's full accuracy and removes a third to a half of the prefill. Absolute ms are the eager PyTorch path (the GGUF server does the same prompt in ~30 ms); the ratios are what transfer.

One probe per question schema, fitted on the 6,000 train decisions (same data access as laya-typed-decisions, 0.766). Above Jev's 0.727 and our 12B's 0.704 / 0.725, from a 4B with untouched weights, reading 12 layers below the top.

## 5. JevBench public tiers (leaderboard formulas, `experiments/bench_jevbench.py`)

| tier (n) | Gemma 4 12B Q8: accuracy / intelligence / ECE / median s | Qwen3.5-9B Q8 |
|---|---:|---:|
| original (72) | **0.944 / 91.9 / 0.053 / 0.05 s** (paraphrase-pair consistency 0.944) | 0.958 / 94.0 / 0.069 / 0.12 s |
| easy (48) | **1.000 / 100 / 0.000 / 0.06 s** | 1.000 / 100 / 0.011 / 0.13 s |
| hard (111; states up to 3.9k tokens, 8k ctx) | **0.703 / 55.2 / 0.251 / 0.46 s** (temporal_numeric 0.13, routing_hard 1.0, trap 0.88, ambiguous 0.86, adversarial 0.83) | — |

Weighted over the three public tiers with the leaderboard's weights (hard 30 %, easy 14 %, standard 28 %; no judge tier available), Tez's intelligence is **78.2**. The leaderboard's headline uses a held-out tier plus a judge tier (Jev 85.7, SemIf 79.0, Laya 45.8), so these public numbers are indicative only. The hard tier's one collapsed family is `temporal_numeric` (2/15): date and quantity arithmetic that a single forward pass cannot do.

## 6. Audit trail

The first draft's balanced accuracy was keyed on gold *position* instead of option *id* (inflated by 1–3 points; all numbers here are corrected), and its "~110 ms floor / no prefix reuse" was the sliding-window cache (fixed with `--swa-full`). Runs that hit the wrong model are quarantined under `results/INVALID_wrongmodel_*`. Every run writes a manifest with data SHA-256, model path and settings.
