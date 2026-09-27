# Native GLiNER comparison

Complete, matched inputs: 6,151 original public decisions per model.
GLiNER additionally has 1,100 reversed-option intent decisions per model. Raw T=1 probabilities; no calibration fitting.
The GLiNER arms use original FP32 native eager inference; Qwen and Decider use their previously frozen implementations.
Both GLiNER arms use batch one; multilingual uses conservative 80/76 C thermal pacing after a preserved host reboot.
These are utility comparisons across different training exposures, not controlled architecture or new held-out confirmation tests.

| Accuracy, % | GLiNER English | GLiNER multilingual | Qwen student | Decider 0.8B |
|---|---:|---:|---:|---:|
| typed_decisions | 40.30 | 39.00 | 50.15 | 53.40 |
| app_model_routing_domain | 60.90 | 50.63 | 89.22 | 84.46 |
| app_guardrails_jailbreak | 76.00 | 74.75 | 85.75 | 92.50 |
| app_moderation_toxicity | 45.75 | 54.25 | 60.25 | 75.25 |
| app_email_spam | 32.25 | 49.75 | 87.25 | 95.50 |
| app_phishing | 70.75 | 69.50 | 74.25 | 92.50 |
| app_rag_relevance | 53.50 | 53.50 | 51.50 | 65.00 |
| app_support_triage | 42.75 | 35.00 | 43.50 | 44.75 |
| massive_en | 80.00 | 71.00 | 83.00 | 89.00 |
| massive_de | 64.00 | 60.00 | 68.00 | 83.00 |
| massive_fr | 64.00 | 58.00 | 69.00 | 83.00 |
| massive_es | 60.00 | 58.00 | 68.00 | 86.00 |
| massive_ja | 65.00 | 66.00 | 76.00 | 89.00 |
| massive_zh-CN | 54.00 | 63.00 | 80.00 | 90.00 |
| massive_ar | 20.00 | 53.00 | 51.00 | 61.00 |
| massive_hi | 11.00 | 6.00 | 51.00 | 52.00 |
| massive_th | 1.00 | 35.00 | 51.00 | 58.00 |
| massive_ko | 37.00 | 59.00 | 76.00 | 87.00 |
| massive_km | 1.00 | 2.00 | 32.00 | 41.00 |
| semif_authored144 | 67.36 | 57.64 | 68.75 | 77.08 |
| semif_perturbations108 | 55.56 | 45.37 | 50.00 | 71.30 |
| Intent, mean of 11 languages | 41.55 | 48.27 | 64.09 | 74.45 |

SemIf rows above show ordinary accuracy; the official mean family balanced metric is separate below.

| Raw typed probability metric | GLiNER English | GLiNER multilingual | Qwen student | Decider 0.8B |
|---|---:|---:|---:|---:|
| ece | 0.1351 | 0.1504 | 0.0612 | 0.1339 |
| nll | 1.1454 | 1.2044 | 1.0436 | 1.1094 |
| brier | 0.6670 | 0.7008 | 0.5945 | 0.6301 |
| soft_accuracy | 0.3649 | 0.3546 | 0.3867 | 0.4251 |

| GLiNER arm | Intent option-order flip | SemIf authored balanced | SemIf perturbations balanced |
|---|---:|---:|---:|
| english | 0.3873 | 0.6783 | 0.5492 |
| multilingual | 0.1773 | 0.5733 | 0.4651 |

Job elapsed time includes loading, input audits and thermal pauses; it is not a latency result.
The worker's option-index F1 is retained as a diagnostic only. Supplemental app/intent macro-F1 uses semantic option keys;
these metrics are not used for fitting, checkpoint selection or release gates. Typed tasks mix label meanings; SemIf keeps its official evaluator.
No gates were relaxed and no runtime was changed. Competitor training membership is unknown.
[Full measurements and hashes](comparison.json), [frozen protocol](../../experiments/small_model/gliner2_comparison_serial_registration.json),
[broader comparison and release requirements](../pilot-improvement-20260927/COMPETITIVENESS.md).
