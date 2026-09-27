# Qwen teacher-validation audit

This reuses all 278 original validation states and 1,390 frozen Qwen probability vectors.
The 12B targets use the original fixed type temperatures and two-order average. No inference or fitting was performed.
Teacher agreement is not gold accuracy; most questions here have no independent gold label.

| Group | Decisions | Teacher agreement | Mean KL | Student confidence | Teacher confidence |
|---|---:|---:|---:|---:|---:|
| all: all | 1390 | 78.99% | 0.1401 | 0.6696 | 0.7290 |
| qtype: choice | 526 | 76.05% | 0.2183 | 0.6529 | 0.7290 |
| qtype: noul | 491 | 93.08% | 0.0471 | 0.8137 | 0.8424 |
| qtype: score | 373 | 64.61% | 0.1524 | 0.5036 | 0.5797 |
| source: aya | 130 | 77.69% | 0.1586 | 0.6799 | 0.7376 |
| source: clinc | 30 | 80.00% | 0.3285 | 0.7651 | 0.8135 |
| source: csqa | 25 | 68.00% | 0.1343 | 0.5688 | 0.6027 |
| source: fineweb | 655 | 83.36% | 0.1182 | 0.6860 | 0.7406 |
| source: fineweb2 | 500 | 75.60% | 0.1475 | 0.6558 | 0.7189 |
| source: gmmlu | 50 | 64.00% | 0.1957 | 0.5607 | 0.6678 |
| lang: ar | 80 | 76.25% | 0.1713 | 0.6491 | 0.7104 |
| lang: arb | 5 | 100.00% | 0.0594 | 0.5855 | 0.7017 |
| lang: ary | 5 | 60.00% | 0.3621 | 0.7758 | 0.8560 |
| lang: de | 65 | 69.23% | 0.1701 | 0.6600 | 0.7477 |
| lang: en | 715 | 82.66% | 0.1271 | 0.6847 | 0.7382 |
| lang: es | 70 | 77.14% | 0.1440 | 0.6536 | 0.7091 |
| lang: fr | 35 | 74.29% | 0.1332 | 0.6776 | 0.6968 |
| lang: fra | 10 | 90.00% | 0.0404 | 0.7559 | 0.8194 |
| lang: guj | 5 | 80.00% | 0.1593 | 0.7639 | 0.7640 |
| lang: hi | 65 | 66.15% | 0.1744 | 0.6022 | 0.7016 |
| lang: hin | 5 | 80.00% | 0.0484 | 0.6053 | 0.6762 |
| lang: ibo | 5 | 100.00% | 0.0361 | 0.7410 | 0.7744 |
| lang: ja | 50 | 74.00% | 0.1077 | 0.6458 | 0.7046 |
| lang: kir | 10 | 70.00% | 0.2328 | 0.6242 | 0.7328 |
| lang: ko | 45 | 80.00% | 0.1237 | 0.7240 | 0.7940 |
| lang: mar | 5 | 80.00% | 0.0605 | 0.6832 | 0.7458 |
| lang: pan | 5 | 40.00% | 0.2150 | 0.6078 | 0.5563 |
| lang: pbt | 5 | 80.00% | 0.2021 | 0.6962 | 0.6880 |
| lang: por | 5 | 60.00% | 0.4543 | 0.6594 | 0.7916 |
| lang: sin | 5 | 100.00% | 0.1664 | 0.6700 | 0.7744 |
| lang: som | 5 | 80.00% | 0.2152 | 0.5983 | 0.5817 |
| lang: tam | 10 | 60.00% | 0.3454 | 0.6219 | 0.6891 |
| lang: tel | 10 | 70.00% | 0.1595 | 0.5998 | 0.6860 |
| lang: th | 60 | 78.33% | 0.1642 | 0.5895 | 0.6851 |
| lang: vie | 5 | 60.00% | 0.1824 | 0.6942 | 0.7790 |
| lang: yor | 5 | 60.00% | 0.1334 | 0.7015 | 0.7202 |
| lang: zh | 75 | 76.00% | 0.1557 | 0.6558 | 0.6992 |
| lang: zho | 5 | 100.00% | 0.1379 | 0.6628 | 0.8368 |
| lang: zsm | 20 | 90.00% | 0.0339 | 0.7571 | 0.7690 |

Small language/source subgroups are descriptive, not stable rankings. Different option counts also affect KL and confidence.
These results may guide a registered training continuation using the original training split and validation-only selection.
They do not justify changing a public-test prompt, fitting to test labels, weakening a gate or promoting a checkpoint.
[Full metrics and provenance](audit.json). Raw teacher targets and frozen student probabilities are in the measurement archive.
