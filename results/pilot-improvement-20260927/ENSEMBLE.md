# Fixed Qwen/Decider probability blend

This is exploratory analysis of already observed public evaluations. It averages the two raw
probability vectors with fixed equal weights; no mixture search or test-label calibration is used.
No inference was run and no model weights or runtime were changed. Combining models would require
both inference paths; this analysis makes no latency, cost-efficiency or promotion claim.

| Task | Qwen | Decider | Equal blend |
|---|---:|---:|---:|
| typed_decisions | 50.15% | 53.40% | 53.00% |
| app_model_routing_domain | 89.22% | 84.46% | 89.22% |
| app_guardrails_jailbreak | 85.75% | 92.50% | 91.75% |
| app_moderation_toxicity | 60.25% | 75.25% | 68.00% |
| app_email_spam | 87.25% | 95.50% | 95.75% |
| app_phishing | 74.25% | 92.50% | 89.75% |
| app_rag_relevance | 51.50% | 65.00% | 56.50% |
| app_support_triage | 43.50% | 44.75% | 47.00% |
| massive_en | 83.00% | 89.00% | 90.00% |
| massive_de | 68.00% | 83.00% | 84.00% |
| massive_fr | 69.00% | 83.00% | 83.00% |
| massive_es | 68.00% | 86.00% | 83.00% |
| massive_ja | 76.00% | 89.00% | 90.00% |
| massive_zh-CN | 80.00% | 90.00% | 89.00% |
| massive_ar | 51.00% | 61.00% | 65.00% |
| massive_hi | 51.00% | 52.00% | 58.00% |
| massive_th | 51.00% | 58.00% | 55.00% |
| massive_ko | 76.00% | 87.00% | 87.00% |
| massive_km | 32.00% | 41.00% | 40.00% |
| semif_authored144 | 68.75% | 77.08% | 81.94% |
| semif_perturbations108 | 50.00% | 71.30% | 71.30% |
| MASSIVE-11 macro | 64.09% | 74.45% | 74.91% |

On typed decisions, blend minus Decider is -0.40 points;
the exploratory state-cluster 95% interval is [-1.85, +0.95]. Whole states stay together
during 2,000 bootstrap resamples; these intervals have no multiplicity correction.

Complementarity counts and a hindsight oracle are preserved in the JSON. The oracle chooses
the correct model using the true answer and is not an implementable routing system.

Sources: [registration](ensemble_registration.json), [validated measurements](ensemble_diagnostic.json).
