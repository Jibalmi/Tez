# Native Tev1 comparison

All 6,151 original public decisions verified per model; Tev1 also has 1,100 reversed-option intent cases.
Raw temperature-one probabilities. Tev1 uses the pinned full checkpoint in BF16, eager attention, native JSON/chat format and batch size one.
These are already-observed public tasks with differing and incompletely reconstructed training exposures.

| Accuracy, % | Tev1 0.8B | Qwen student | Decider 0.8B |
|---|---:|---:|---:|
| typed_decisions | 51.45 | 50.15 | 53.40 |
| app_model_routing_domain | 78.45 | 89.22 | 84.46 |
| app_guardrails_jailbreak | 86.75 | 85.75 | 92.50 |
| app_moderation_toxicity | 69.75 | 60.25 | 75.25 |
| app_email_spam | 63.50 | 87.25 | 95.50 |
| app_phishing | 64.50 | 74.25 | 92.50 |
| app_rag_relevance | 58.25 | 51.50 | 65.00 |
| app_support_triage | 43.00 | 43.50 | 44.75 |
| massive_en | 80.00 | 83.00 | 89.00 |
| massive_de | 70.00 | 68.00 | 83.00 |
| massive_fr | 65.00 | 69.00 | 83.00 |
| massive_es | 67.00 | 68.00 | 86.00 |
| massive_ja | 83.00 | 76.00 | 89.00 |
| massive_zh-CN | 83.00 | 80.00 | 90.00 |
| massive_ar | 56.00 | 51.00 | 61.00 |
| massive_hi | 48.00 | 51.00 | 52.00 |
| massive_th | 38.00 | 51.00 | 58.00 |
| massive_ko | 79.00 | 76.00 | 87.00 |
| massive_km | 34.00 | 32.00 | 41.00 |
| semif_authored144 | 79.86 | 68.75 | 77.08 |
| semif_perturbations108 | 74.07 | 50.00 | 71.30 |
| Intent, 11-language mean | 63.91 | 64.09 | 74.45 |

| Raw typed probability metric | Tev1 | Qwen | Decider |
|---|---:|---:|---:|
| ece | 0.0864 | 0.0612 | 0.1339 |
| nll | 1.1158 | 1.0436 | 1.1094 |
| brier | 0.6132 | 0.5945 | 0.6301 |
| soft_accuracy | 0.4165 | 0.3867 | 0.4251 |

Semantic app/intent macro-F1 and the official SemIf family metrics are in the full analysis.
Typed rows combine different label meanings; option-index F1 remains diagnostic only.
Execution time includes loading and thermal pauses and is not a latency result. No release gates changed.
[Full analysis](comparison.json), [registered protocol](registration.json), [broader comparison](../pilot-improvement-20260927/COMPETITIVENESS.md).
