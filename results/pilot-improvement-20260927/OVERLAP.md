# Shared errors: Qwen and Decider

These counts use the complete frozen public cases and validate all raw probability rows against them.
The oracle chooses a correct answer whenever either model is correct. It uses test labels, so it cannot be deployed.
It is an upper bound on choosing between these two answers, not a measured ensemble or a quality gate result.

| Task | Qwen accuracy | Decider accuracy | Oracle ceiling | Qwen-only correct | Decider-only correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| typed_decisions | 50.15% | 53.40% | 64.55% | 223 | 288 | 709 |
| app_model_routing_domain | 89.22% | 84.46% | 95.74% | 45 | 26 | 17 |
| app_guardrails_jailbreak | 85.75% | 92.50% | 93.75% | 5 | 32 | 25 |
| app_moderation_toxicity | 60.25% | 75.25% | 78.00% | 11 | 71 | 88 |
| app_email_spam | 87.25% | 95.50% | 96.00% | 2 | 35 | 16 |
| app_phishing | 74.25% | 92.50% | 93.00% | 2 | 75 | 28 |
| app_rag_relevance | 51.50% | 65.00% | 72.50% | 30 | 84 | 110 |
| app_support_triage | 43.50% | 44.75% | 54.00% | 37 | 42 | 184 |
| massive_en | 83.00% | 89.00% | 94.00% | 5 | 11 | 6 |
| massive_de | 68.00% | 83.00% | 88.00% | 5 | 20 | 12 |
| massive_fr | 69.00% | 83.00% | 87.00% | 4 | 18 | 13 |
| massive_es | 68.00% | 86.00% | 91.00% | 5 | 23 | 9 |
| massive_ja | 76.00% | 89.00% | 91.00% | 2 | 15 | 9 |
| massive_zh-CN | 80.00% | 90.00% | 94.00% | 4 | 14 | 6 |
| massive_ar | 51.00% | 61.00% | 72.00% | 11 | 21 | 28 |
| massive_hi | 51.00% | 52.00% | 63.00% | 11 | 12 | 37 |
| massive_th | 51.00% | 58.00% | 62.00% | 4 | 11 | 38 |
| massive_ko | 76.00% | 87.00% | 93.00% | 6 | 17 | 7 |
| massive_km | 32.00% | 41.00% | 47.00% | 6 | 15 | 53 |
| semif_authored144 | 68.75% | 77.08% | 88.89% | 17 | 29 | 16 |
| semif_perturbations108 | 50.00% | 71.30% | 78.70% | 8 | 31 | 23 |

All columns here use ordinary accuracy. SemIf's official mean family balanced accuracy remains in the matched comparison;
it is a different metric. The counts include every decision rather than filtering to disagreements.
No confidence thresholds, task routing rules, model weights or temperatures were selected from this diagnostic.
The earlier [fixed equal-probability ensemble](ENSEMBLE.md) remains a separate measured result.
[Machine-readable counts and source hashes](overlap.json).
