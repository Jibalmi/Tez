# v2: five times the teacher data for the Qwen student

**Outcome (2026-09-28): a small typed-decision gain, no multilingual gain; G1 still not met.** Training the
unchanged Qwen3.5-0.8B control recipe on 155,840 teacher-labelled decisions instead of 29,990 raised held-out
typed-decision accuracy from 50.15% to 52.20% (+2.05 points; paired 95% bootstrap interval -0.15 to +4.30). The
MASSIVE-11 mean moved from 64.09% to 62.91%, and the application tasks moved both ways. Decider 0.8B's typed lead
shrank from 3.25 to 1.20 points (interval -1.10 to +3.50) and is no longer clearly separated, but Decider still leads
by 11.5 points on MASSIVE-11 and on most application tasks. More data of this kind is a weak lever for this student.

## What was run

| Step | Result |
|---|---|
| Pool | 25,169 new states, 4x the original per-source targets, sampled with a new seed and disjoint from the original 6,276 by id, source row and normalised text ([manifest](pool_manifest.json)). The unchanged leakage guard dropped 6 hash matches, 1 MinHash match, 87 at cosine >= 0.95 and 1,391 nearer an evaluation-domain centroid. |
| Schemas | Gemma 4 12B Q8 wrote 5,000 schemas for 500 language-homogeneous clusters; 2,740 survived the unchanged validity, dedupe and evaluation-option-overlap filters ([manifest](schemas_manifest.json)). Gold-dataset question wording was reused from the original pool. |
| Teacher | 125,845 new decisions in both option orders, two isolated workers with fresh state caches (as the original H200 export), 327 ms per state per worker, no errors. |
| Student | `train_c.py` defaults (LoRA r 16, lr 2e-4, batch 16, one epoch) on the original 29,990 plus the new 125,845 training decisions: 9,740 steps in 3,940 s ([record](train_c-qwen08-v2.json)). New states are training-only (`TEZ_PILOT_TRAIN_ONLY_PREFIX=v2-`), so validation stays the original 278 states / 1,390 decisions; teacher agreement there rose from 0.7835 to 0.8158. |
| Evaluation | The 30k run's frozen evaluation suites, same scorer ([results](eval_c-qwen08-v2.json), [rows](rows/)). |

## Public comparison (accuracy, %; identical cases per column)

| Task | Qwen 30k (current) | Qwen v2 (155k) | Decider 0.8B |
|---|---:|---:|---:|
| Typed decisions | 50.15 | 52.20 | 53.40 |
| MASSIVE-11 mean | 64.09 | 62.91 | 74.45 |
| Routing | 89.22 | 87.22 | 84.46 |
| Jailbreak | 85.75 | 87.75 | 92.50 |
| Toxicity | 60.25 | 57.25 | 75.25 |
| Email spam | 87.25 | 86.75 | 95.50 |
| Phishing | 74.25 | 76.50 | 92.50 |
| RAG relevance | 51.50 | 50.00 | 65.00 |
| Support triage | 43.50 | 44.25 | 44.75 |
| SemIf authored144 | 68.75 | 68.75 | 77.08 |
| SemIf perturbations108 | 50.00 | 44.44 | 71.30 |

Typed-decision ECE: 0.061 (30k), 0.070 (v2), 0.134 (Decider). Paired over the same 2,000 typed decisions:
v2 minus 30k +2.05 points (285 gained, 244 lost; 95% bootstrap -0.15 to +4.30); Decider minus v2 +1.20 points
(-1.10 to +3.50). Bootstrap: 2,000 resamples, seed 13, rows as units. These public benchmarks were already observed,
so the comparison is exploratory; G1 (typed >= 56.3%, MASSIVE-11 >= 70%) is not met by any student.

## Execution and deviations

- The new teacher labels and the v2 student ran on one RunPod H100 SXM; the original export and the 30k student ran on
  an H200. The software stack is the same scripts and pinned transformers/peft/FLA versions as the 30k student run.
- The schema-assignment step needed the `datasets` package and, because newer `huggingface_hub` releases reject the
  legacy repository ids of three evaluation datasets, the evaluation option sets precomputed on the laptop
  (`schemas.py option-sets`, 3,687 sets). The filtering rule is unchanged.
- Earlier attempts lost two Pods before any result (RunPod credit ran out; an idle stop), and this Pod's final download
  used a CPU restart after its machine's GPUs were taken. Cloud cost for the round is about $21.
- The trained adapter is under ignored `runs/student-data-v2/models/c-qwen08-v2` (43 MB), not in Git.
