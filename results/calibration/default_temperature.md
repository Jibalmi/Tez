# A default letter temperature per question type

Tez tempers a question's letter logits only after `tez fit`; an unfitted question is read at T = 1, and those answers are over-confident. Over the 13,610 labelled decisions used here, the mean confidence is 0.977 against an accuracy of 0.768, and the mean ECE-15 over the 19 tasks is 0.218.

This note fits one default temperature per question type on every labelled decision the default model (Gemma 4 12B Q8_0) left in `results/`. It then measures the defaults leave-one-task-out (LOTO): each task is scored with a temperature fitted without it.

Script: `experiments/default_temperature.py` (CPU only, saved rows only; it regenerates the tables below). Everything else is in `default_temperature.json`: per-task, per-language and per-workflow metrics, every fitted value, the inputs with SHA-256, and the protocol.

## Recommendation

**Ship per-type defaults for unfitted letter readouts, with choice questions split at 10 options.** The values are pooled NLL fits on all 19 tasks:

| question | default T | tasks behind it | LOTO fits (min–max) |
|---|---:|---:|---:|
| noul | **6.01** | 9 | 4.99–6.53 |
| choice, up to 10 options | **4.71** | 9 | 4.36–5.10 |
| choice, 11–26 options | **2.66** | 2 | 2.65–3.28 |
| score | **5.18** | 3 | 4.48–6.25 |

- **How to pick the value.** Count the options the model is shown, including `__none__`. A chunked tournament (more than 26 options) goes by the options of its final round, because the tempered softmax spans only those: Banking77's 4 finalists get 4.71.
- **Choice with 2 options: not measured.** No task in the pool has one. The rule gives it 4.71.
- **Expected change, leave-one-task-out.** Mean over the 19 tasks:
  - ECE-15 0.218 → 0.132 (the in-sample per-task oracle is 0.065);
  - NLL 2.157 → 0.919;
  - Brier 0.441 → 0.348.
  - BENCHMARKS.md's mean over its 28 head-to-head entries falls from 0.212 → 0.140; its one-temperature-per-entry refit reaches 0.078.
  - typed-decisions falls 0.262 → 0.051 as a whole task, level with its own out-of-fold refit (0.048).
  - Accuracy does not move: the argmax is identical on all 13,610 decisions under every setting.
- **Where it hurts: the most accurate tasks, which the default makes under-confident.** Every other task improves on ECE, NLL and Brier.
  - `model_routing_domain` (choice, 6 options, accuracy 0.970) is the one real casualty. Mean confidence falls 0.993 → 0.645; ECE 0.030 → 0.325, NLL 0.194 → 0.505, Brier 0.060 → 0.204. Its own best T is 1.67.
  - `email_spam` (noul, 0.968): ECE 0.031 → 0.108, NLL 0.177 → 0.180, Brier 0.060 → 0.074.
  - SemIf (0.951): ECE 0.047 → 0.067, while NLL and Brier improve.
  - JevBench's 18 score items: ECE 0.182 → 0.194, while NLL and Brier improve. At n = 18 that is noise.
  - A few labels repair these cases, if the prior change below ships with the defaults. With 5 labels, routing goes to NLL 0.178 / ECE 0.053 and email spam to 0.113 / 0.039.
- **Do not ship a single choice value.** The pooled fit is 3.15, but MASSIVE is 55 % of the choice decisions and sets it. Held out, MASSIVE gets 4.54 and its ECE doubles (0.153 → 0.292), and the 28-entry mean rises to 0.241, worse than T = 1.
  - The option count matters mechanically. A high T moves probability onto implausible options, and a question with many options has many of them.
  - At T = 4.71, MASSIVE (20 options) puts 0.34 of its probability outside each decision's four best letters; at T = 2.66, 0.09.
- **Why one split at 10, not the requested buckets.** The 3–5 and 6–10 buckets fit the same value (4.59 and 4.85).
  - Kept separate, the 6–10 value rests on four tasks, including the two hardest choice tasks (support triage 0.41, emotion 0.54). Held out, routing then gets 6.16 and ECE 0.403.
  - Merged, the rule scores best of every variant run on the same 19 tasks: ECE 0.132, NLL 0.919, Brier 0.348, 28-entry mean 0.140.
  - The split was chosen after seeing the bucket fits, so read its margin over the requested buckets (ECE 0.139) as small.
- **Score questions: keep `score = Σ i·p_i` over the returned, tempered probabilities.** Tempering does make the expected level worse against the integer gold level. Over all 1,218 score decisions:
  - MAE 0.439 → 0.535; within one level 0.918 → 0.831 (SST-5: 0.890 → 0.740).
  - Taking the score from untempered probabilities while returning tempered probabilities and confidence restores those two numbers exactly.
  - But the same change loses on squared error (0.506 untempered vs 0.456 tempered).
  - It also loses badly on the MAE against the gold *expected* score, the score metric typed-decisions itself reports: 0.482 untempered vs 0.309 tempered. Jev's published figure there is 0.391.
  - It would also make `score` disagree with the `probabilities` returned beside it.
  - A client that needs an integer level should read the median of `probabilities`: MAE 0.429, the lowest of every readout, and almost unchanged by T.
- **The prior of `fit_temperature_logits` should centre on the question's default, not on 1.** The same default should also replace the `T = 1.0` it returns below `min_rows`. Otherwise a question fitted from 1–4 labels stores T = 1 and loses its default.
  - The fold medians of the pooled fits are noul 6.21, choice (up to 10) 4.76, choice (11–26) 2.96, score 5.21.
  - The per-task optimum temperatures are lower and spread wide: medians 4.90 / 3.53 / 4.51, standard deviation of log T 0.63 / 0.41 / 0.33.
  - With the prior's median at 1, fitting on 5 labels is worse than not fitting at all: held-out NLL 1.153 against 0.900 for the default alone. The prior pulls T back towards 1, and a few labels are mostly correct answers.
  - With the median at the default, 5 labels help (0.891), 10 labels give 0.875 against today's 0.989, and the two meet by 50 labels (0.844 vs 0.850).
  - Tightening the sd from 1 to the observed spread gains at most 0.017 in NLL, so keep sd = 1.
  - Keep median 1 for probe temperatures (`fit_temperature_probs`). A probe is fitted with a proper scoring rule and comes out nearly calibrated.

**Engine notes.**
- Where the default applies: `decide_question` reads `softmax(z, fq.letters.temperature if letters_ok else 1.0)`, and the default replaces that `1.0`.
  - A fitted schema is unaffected, including its probe blend, which was calibrated with the schema's own letters T.
- The gate is unaffected. Unfitted questions already escalate under `alpha`.
- Re-derive anything tuned on T = 1 confidences. The voice loop's commit threshold (τ 0.9, `experiments/stream_policy.py`) and the site's recorded strips were chosen at T = 1.
- The values belong to Gemma 4 12B Q8_0 with the `gemma4` letter prompt. Key them to model and template, and keep T = 1 for any model they were not fitted on.
- Consider marking a default temperature in the answer meta, so a client can tell it from a fitted one.

## Data

- **Model and settings.** Every row is Gemma 4 12B Q8_0 (Ollama blob `sha256-047dae1d…`) reading option letters zero-shot, with llama-server's top-200 log-probabilities.
- **Missing letters.** A letter missing from the top 200 was floored at (lowest listed − 2), exactly as `tez/backends.py` does. The saved logits therefore already carry the engine's handling of the truncated mass. Voice had a floored letter in 9 of 220 rows and SemIf in none; no logits file shows two letters at the floor.
- **Probability-only files are tempered exactly.** Voice and the JevBench runtime rows keep only probabilities. They were computed as softmax at T = 1, so tempering `log p` gives p^(1/T) renormalised exactly; none holds a 0 or a 1.
- **Tournaments.** Options that left a tournament are −∞, as in `engine.letter_logits`. In 72 of Banking77's 400 rows the gold was eliminated before the final, and those rows have probability 0 at every T.

Tasks (the unit of leave-one-task-out):
- typed-decisions (2,000: 600 noul, 600 choice, 800 score)
- Laya's public suite from the head-to-head runs:
  - AG News, Emotion, Banking77 (77 options, tournament), SST-5 (score), BoolQ and prompt-injections (noul)
  - XNLI, 10 languages as one task
  - MASSIVE, 51 languages as one task (`vs_laya/massive51_rows`; its 11 head-to-head languages are byte-identical to `h2h/rows_massive_*`)
- Laya's seven application workflows: five noul, support triage and model routing choice
- JevBench's three public tiers through the runtime (`tez serve`), one task
- SemIf authored144 (choice, 3 options)
- the voice router (choice, 16 options, transcript last)

The runtime's prompt layout is behind everything except SemIf and voice, which keep their own prompts. Pooled without those two, the values move by at most 0.03 (up to 10 options 4.74, 11–26 options 2.65).

Excluded, each with its reason in the JSON:
- other models and quantisations, Laya's checkpoints, and the quarantined wrong-model runs;
- bit-identical repeats, and permuted or perturbed copies of the same items;
- content-free inputs, verbaliser, thinking and few-shot variants, and non-default voice layouts;
- per-word voice prefixes (the gold of a prefix is undefined);
- the typed-decisions train split (same task).

**No Jev or TypeSafe output is used.** Jev was never run here, and the JevBench rows are Tez's own answers. The runtime manifest shows `http://127.0.0.1:8787`, `tez-0.1.0 (gemma-4-12b-q8_0, letters)`, checked when loading. JevBench's labels are its authored gold, and SemIf's labels are project-authored (checked per row). SemIf's `typesafe` source is not in `data/`.

One caveat: typed-decisions' gold is the mean of three samples from an unnamed ~4B "teacher endpoint". The dataset card says the benchmark is not affiliated with TypeSafe and lists Jev only as a separately measured leaderboard row. The teacher itself is not identified.

## Protocol

**Pooled fit.** The minimum over T in [0.05, 20] of the mean NLL of every decision in a cell. Every decision weighs the same, p is clipped at 1e-12 as in `tez.readout.nll`, and the search is the engine's 120-point grid refined by golden section. A task-balanced pool is a variant.

**Leave-one-task-out.** For each cell and each task in it, T is fitted on the other tasks and measured on the held-out one. A cell with no other task falls back to its type.

**Settings compared:**
- **(a)** T = 1;
- **(b)** one pooled T per type;
- **(b′)** as (b), with choice in the requested buckets 2 / 3–5 / 6–10 / 11–26 (no task has 2-option choice questions);
- **(r)** as (b), with choice split at 10 options;
- **(c)** the per-task in-sample oracle.

**Metrics.** These are the engine's own definitions from `tez/readout.py`:
- ECE-15: maximum probability against correctness, 15 equal-width bins;
- NLL;
- multi-class Brier;
- accuracy.

Averages are unweighted over tasks. A pooled-decision ECE is shown too, but over- and under-confident tasks cancel inside its bins, so it is not used to decide.

**Checks, rerun by the script.**
- At T = 1 it reproduces BENCHMARKS.md's 0.212 and the 0.078 refit.
- Its T = 1 metrics match the published summaries of the head-to-head runs, the apps and the JevBench tiers.
- scipy finds the same pooled noul T.
- Another session re-read typed-decisions through llama-server with the same layout; it agrees on 98.4 % of argmaxes.

## Caveats

- **The pool mixes easy and hard tasks, and NLL follows the hard ones.** Accuracies run from 0.41 to 0.97, and NLL is dominated by confident mistakes. The defaults therefore suit questions of benchmark difficulty. If real traffic is mostly easy questions, like routing, spam or intent, they will read under-confident until fitted.
- **The 11–26 value is thin.** It rests on MASSIVE (96 % of that cell) and voice. MASSIVE's held-out check uses voice alone: dropping the 33 voice rows that accept alternative answers moves the 28-entry mean from 0.140 to 0.110.
- **The score value is thin.** It rests on three tasks, one of them 18 decisions.
- **Label noise.** typed-decisions' gold is a teacher's argmax, and its noise raises the fitted T.
- **Small tasks.** ECE-15 is noisy below a few hundred decisions: JevBench's cells, prompt-injections at 116, SemIf at 144.
- **Few-shot prompts.** They were not pooled, but the defaults held out from typed-decisions also calibrate its 4-shot rows (ECE 0.245 → 0.041).

## Tables

The script regenerates everything between the markers.

<!-- tables:start -->
### Data: decisions per task and type (options = option count: decisions)

| task | prompt | noul | choice | score | decisions | options |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| massive | engine |  | 5100 |  | 5100 | 20:5100 |
| typed_decisions | engine | 600 | 600 | 800 | 2000 | 2:600, 4:1100, 5:300 |
| xnli | engine |  | 1000 |  | 1000 | 3:1000 |
| ag_news | engine |  | 400 |  | 400 | 4:400 |
| banking77 | engine |  | 400 |  | 400 | 77:400 |
| boolq | engine | 400 |  |  | 400 | 2:400 |
| email_spam | engine | 400 |  |  | 400 | 2:400 |
| emotion | engine |  | 400 |  | 400 | 6:400 |
| guardrails_jailbreak | engine | 400 |  |  | 400 | 2:400 |
| moderation_toxicity | engine | 400 |  |  | 400 | 2:400 |
| phishing | engine | 400 |  |  | 400 | 2:400 |
| rag_relevance | engine | 400 |  |  | 400 | 2:400 |
| sst5 | engine |  |  | 400 | 400 | 5:400 |
| support_triage | engine |  | 400 |  | 400 | 10:400 |
| model_routing_domain | engine |  | 399 |  | 399 | 6:399 |
| jevbench | engine | 74 | 139 | 18 | 231 | 2:74, 3:15, 4:70, 5:56, 6:16 |
| voice | voice |  | 220 |  | 220 | 16:220 |
| semif | semif |  | 144 |  | 144 | 3:144 |
| prompt_injections | engine | 116 |  |  | 116 | 2:116 |
| **all (19 tasks)** |  | **3190** (9 tasks) | **9202** (11 tasks) | **1218** (3 tasks) | **13610** |  |

### Fitted temperatures (pooled NLL; LOTO = the same fit with each task left out in turn)

| cell | T pooled on all tasks | LOTO folds: median (min–max) | T task-balanced | per-task oracle: median (min–max) | sd of log T across tasks | tasks | decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| noul (b, r) | **6.01** | 6.21 (4.99–6.53) | 6.37 | 4.90 (2.74–20.00) | 0.63 | 9 | 3190 |
| choice (b) | **3.15** | 3.14 (3.00–4.54) | 3.92 | 3.53 (1.67–6.48) | 0.41 | 11 | 9202 |
| score (b, r) | **5.18** | 5.21 (4.48–6.25) | 4.88 | 4.51 (3.28–6.38) | 0.33 | 3 | 1218 |
| choice, 11-26 options (b′) | **2.66** | 2.96 (2.65–3.28) | 2.91 |  |  | 2 | 5320 |
| choice, 3-5 options (b′) | **4.59** | 4.68 (3.74–4.82) | 4.01 |  |  | 6 | 2667 |
| choice, 6-10 options (b′) | **4.85** | 4.68 (3.89–6.16) | 4.51 |  |  | 4 | 1215 |
| choice, 11-26 options (r) | **2.66** | 2.96 (2.65–3.28) | 2.91 |  |  | 2 | 5320 |
| choice, 2-10 options (r) | **4.71** | 4.76 (4.36–5.10) | 4.41 |  |  | 9 | 3882 |

### Leave-one-task-out, per task and type

(a) T = 1; (b) one pooled T per type, fitted without the task; (b′) as (b) with choice split into the buckets 2 / 3–5 / 6–10 / 11–26 options (a tournament by its finalists); (c) per-task oracle.

| task | type | n | accuracy | ECE-15 a / b / b′ / c | NLL a / b / b′ / c | Brier a / b / b′ / c | T (b) | T (b′) | T (c) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ag_news | choice | 400 | 0.880 | 0.114 / 0.071 / 0.040 / 0.041 | 1.133 / 0.424 / 0.401 / 0.395 | 0.230 / 0.198 / 0.188 / 0.188 | 3.13 | 4.69 | 4.12 |
| banking77 | choice | 400 | 0.713 | 0.269 / 0.157 / 0.066 / 0.157 | 5.647 / 5.289 / 5.337 / 5.289 | 0.547 / 0.463 / 0.443 / 0.462 | 3.15 | 4.82 | 3.16 |
| boolq | noul | 400 | 0.850 | 0.133 / 0.122 / 0.122 / 0.056 | 0.665 / 0.388 / 0.388 / 0.335 | 0.260 / 0.229 / 0.229 / 0.207 | 6.37 | 6.37 | 3.39 |
| email_spam | noul | 400 | 0.968 | 0.031 / 0.108 / 0.108 / 0.011 | 0.177 / 0.180 / 0.180 / 0.092 | 0.060 / 0.074 / 0.074 / 0.051 | 6.53 | 6.53 | 2.74 |
| emotion | choice | 400 | 0.540 | 0.433 / 0.277 / 0.144 / 0.056 | 4.111 / 1.554 / 1.327 / 1.289 | 0.875 / 0.718 / 0.644 / 0.621 | 3.07 | 4.49 | 5.89 |
| guardrails_jailbreak | noul | 400 | 0.865 | 0.132 / 0.105 / 0.105 / 0.077 | 0.803 / 0.304 / 0.304 / 0.287 | 0.259 / 0.180 / 0.180 / 0.182 | 6.26 | 6.26 | 4.54 |
| jevbench | noul | 74 | 0.838 | 0.156 / 0.073 / 0.073 / 0.079 | 0.989 / 0.373 / 0.373 / 0.367 | 0.300 / 0.232 / 0.232 / 0.232 | 6.04 | 6.04 | 4.90 |
| jevbench | choice | 139 | 0.856 | 0.123 / 0.051 / 0.122 / 0.048 | 0.677 / 0.377 / 0.433 / 0.377 | 0.247 / 0.190 / 0.199 / 0.191 | 3.15 | 4.67/4.87 | 3.00 |
| jevbench | score | 18 | 0.778 | 0.182 / 0.194 / 0.194 / 0.143 | 0.740 / 0.497 / 0.497 / 0.444 | 0.301 / 0.245 / 0.245 / 0.245 | 5.21 | 5.21 | 3.28 |
| massive | choice | 5100 | 0.816 | 0.153 / 0.292 / 0.104 / 0.057 | 1.473 / 1.037 / 0.819 / 0.777 | 0.332 / 0.366 / 0.276 / 0.266 | 4.54 | 3.28 | 2.65 |
| model_routing_domain | choice | 399 | 0.970 | 0.030 / 0.140 / 0.403 / 0.019 | 0.194 / 0.261 / 0.630 / 0.148 | 0.060 / 0.094 / 0.269 / 0.058 | 3.18 | 6.16 | 1.67 |
| moderation_toxicity | noul | 400 | 0.713 | 0.273 / 0.120 / 0.120 / 0.127 | 2.059 / 0.567 / 0.567 / 0.543 | 0.550 / 0.393 / 0.393 / 0.370 | 5.72 | 5.72 | 8.56 |
| phishing | noul | 400 | 0.897 | 0.101 / 0.056 / 0.056 / 0.039 | 0.756 / 0.290 / 0.290 / 0.269 | 0.200 / 0.164 / 0.164 / 0.161 | 6.28 | 6.28 | 4.40 |
| prompt_injections | noul | 116 | 0.759 | 0.240 / 0.167 / 0.167 / 0.070 | 2.894 / 0.619 / 0.619 / 0.535 | 0.483 / 0.396 / 0.396 / 0.355 | 5.82 | 5.82 | 11.27 |
| rag_relevance | noul | 400 | 0.610 | 0.384 / 0.262 / 0.262 / 0.059 | 3.808 / 0.908 / 0.908 / 0.657 | 0.766 / 0.600 / 0.600 / 0.465 | 4.99 | 4.99 | 20.00 |
| semif | choice | 144 | 0.951 | 0.047 / 0.040 / 0.057 / 0.023 | 0.478 / 0.188 / 0.203 / 0.185 | 0.094 / 0.081 / 0.084 / 0.081 | 3.15 | 4.64 | 3.53 |
| sst5 | score | 400 | 0.512 | 0.468 / 0.213 / 0.213 / 0.089 | 4.036 / 1.241 / 1.241 / 1.183 | 0.947 / 0.693 / 0.693 / 0.639 | 4.48 | 4.48 | 6.38 |
| support_triage | choice | 400 | 0.407 | 0.567 / 0.415 / 0.301 / 0.092 | 6.494 / 2.366 / 2.029 / 1.826 | 1.152 / 0.961 / 0.861 / 0.757 | 3.00 | 3.89 | 6.48 |
| typed_decisions | noul | 600 | 0.810 | 0.173 / 0.045 / 0.045 / 0.037 | 1.149 / 0.436 / 0.436 / 0.429 | 0.356 / 0.276 / 0.276 / 0.275 | 6.21 | 6.21 | 5.05 |
| typed_decisions | choice | 600 | 0.680 | 0.288 / 0.117 / 0.053 / 0.067 | 1.815 / 0.826 / 0.816 / 0.804 | 0.592 / 0.451 / 0.431 / 0.433 | 3.12 | 4.79 | 3.96 |
| typed_decisions | score | 800 | 0.642 | 0.313 / 0.073 / 0.073 / 0.086 | 2.145 / 0.904 / 0.904 / 0.872 | 0.655 / 0.489 / 0.489 / 0.486 | 6.25 | 6.25 | 4.51 |
| voice | choice | 220 | 0.905 | 0.092 / 0.053 / 0.065 / 0.070 | 1.206 / 0.474 / 0.504 / 0.472 | 0.179 / 0.165 / 0.168 / 0.164 | 3.14 | 2.65 | 3.28 |
| xnli | choice | 1000 | 0.699 | 0.286 / 0.205 / 0.164 / 0.083 | 2.512 / 0.938 / 0.826 / 0.730 | 0.577 / 0.489 / 0.460 / 0.415 | 3.01 | 3.74 | 6.32 |

### The recommended rule (r), leave-one-task-out: noul, score, and choice with up to 10 / 11–26 options

| task | type | n | ECE-15 a → r | NLL a → r | Brier a → r | T (r) | ECE-15 (c) | T (c) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ag_news | choice | 400 | 0.114 → 0.051 | 1.133 → 0.403 | 0.230 → 0.188 | 4.77 | 0.041 | 4.12 |
| banking77 | choice | 400 | 0.269 → 0.073 | 5.647 → 5.337 | 0.547 → 0.443 | 4.84 | 0.157 | 3.16 |
| boolq | noul | 400 | 0.133 → 0.122 | 0.665 → 0.388 | 0.260 → 0.229 | 6.37 | 0.056 | 3.39 |
| email_spam | noul | 400 | 0.031 → 0.108 | 0.177 → 0.180 | 0.060 → 0.074 | 6.53 | 0.011 | 2.74 |
| emotion | choice | 400 | 0.433 → 0.139 | 4.111 → 1.323 | 0.875 → 0.642 | 4.55 | 0.056 | 5.89 |
| guardrails_jailbreak | noul | 400 | 0.132 → 0.105 | 0.803 → 0.304 | 0.259 → 0.180 | 6.26 | 0.077 | 4.54 |
| jevbench | noul | 74 | 0.156 → 0.073 | 0.989 → 0.373 | 0.300 → 0.232 | 6.04 | 0.079 | 4.90 |
| jevbench | choice | 139 | 0.123 → 0.122 | 0.677 → 0.436 | 0.247 → 0.200 | 4.76 | 0.048 | 3.00 |
| jevbench | score | 18 | 0.182 → 0.194 | 0.740 → 0.497 | 0.301 → 0.245 | 5.21 | 0.143 | 3.28 |
| massive | choice | 5100 | 0.153 → 0.104 | 1.473 → 0.819 | 0.332 → 0.276 | 3.28 | 0.057 | 2.65 |
| model_routing_domain | choice | 399 | 0.030 → 0.325 | 0.194 → 0.505 | 0.060 → 0.204 | 5.10 | 0.019 | 1.67 |
| moderation_toxicity | noul | 400 | 0.273 → 0.120 | 2.059 → 0.567 | 0.550 → 0.393 | 5.72 | 0.127 | 8.56 |
| phishing | noul | 400 | 0.101 → 0.056 | 0.756 → 0.290 | 0.200 → 0.164 | 6.28 | 0.039 | 4.40 |
| prompt_injections | noul | 116 | 0.240 → 0.167 | 2.894 → 0.619 | 0.483 → 0.396 | 5.82 | 0.070 | 11.27 |
| rag_relevance | noul | 400 | 0.384 → 0.262 | 3.808 → 0.908 | 0.766 → 0.600 | 4.99 | 0.059 | 20.00 |
| semif | choice | 144 | 0.047 → 0.067 | 0.478 → 0.205 | 0.094 → 0.085 | 4.74 | 0.023 | 3.53 |
| sst5 | score | 400 | 0.468 → 0.213 | 4.036 → 1.241 | 0.947 → 0.693 | 4.48 | 0.089 | 6.38 |
| support_triage | choice | 400 | 0.567 → 0.251 | 6.494 → 1.939 | 1.152 → 0.823 | 4.36 | 0.092 | 6.48 |
| typed_decisions | noul | 600 | 0.173 → 0.045 | 1.149 → 0.436 | 0.356 → 0.276 | 6.21 | 0.037 | 5.05 |
| typed_decisions | choice | 600 | 0.288 → 0.051 | 1.815 → 0.817 | 0.592 → 0.431 | 4.82 | 0.067 | 3.96 |
| typed_decisions | score | 800 | 0.313 → 0.073 | 2.145 → 0.904 | 0.655 → 0.489 | 6.25 | 0.086 | 4.51 |
| voice | choice | 220 | 0.092 → 0.065 | 1.206 → 0.504 | 0.179 → 0.168 | 2.65 | 0.070 | 3.28 |
| xnli | choice | 1000 | 0.286 → 0.134 | 2.512 → 0.775 | 0.577 → 0.441 | 4.36 | 0.083 | 6.32 |

### Summary (a / b / b′ / r / c)

|  | ECE-15 a / b / b′ / r / c | NLL a / b / b′ / r / c | Brier a / b / b′ / r / c | accuracy (every setting) |
| --- | ---: | ---: | ---: | ---: |
| mean over the 19 tasks | 0.218 / 0.153 / 0.139 / 0.132 / 0.065 | 2.157 / 0.955 / 0.933 / 0.919 / 0.848 | 0.441 / 0.362 / 0.355 / 0.348 / 0.319 | 0.769 |
| mean over the 23 task×type cells | 0.217 / 0.146 / 0.133 / 0.127 / 0.069 | 1.998 / 0.889 / 0.871 / 0.860 / 0.796 | 0.436 / 0.354 / 0.348 / 0.342 / 0.318 | 0.768 |
| noul: mean over its tasks | 0.180 / 0.118 / 0.118 / 0.118 / 0.062 | 1.478 / 0.452 / 0.452 / 0.452 / 0.390 | 0.359 / 0.283 / 0.283 / 0.283 / 0.255 | 0.814 |
| choice: mean over its tasks | 0.218 / 0.165 / 0.138 / 0.126 / 0.065 | 2.340 / 1.249 / 1.211 / 1.188 / 1.117 | 0.444 / 0.380 / 0.366 / 0.355 / 0.331 | 0.774 |
| score: mean over its tasks | 0.321 / 0.160 / 0.160 / 0.160 / 0.106 | 2.307 / 0.881 / 0.881 / 0.881 / 0.833 | 0.634 / 0.476 / 0.476 / 0.476 / 0.457 | 0.602 |
| all decisions pooled (over- and under-confidence cancel in the bins) | 0.208 / 0.122 / 0.023 / 0.022 / 0.045 | 1.934 / 0.993 / 0.898 / 0.888 / 0.833 | 0.433 / 0.384 / 0.347 / 0.342 / 0.322 | 0.768 |

- (b) highest ECE: **support_triage** (choice) 0.567 → 0.415; largest gain: sst5 (score) 0.468 → 0.213
  - ECE worse than T = 1: massive (choice, n 5100) 0.153 → 0.292, model_routing_domain (choice, n 399) 0.030 → 0.140, email_spam (noul, n 400) 0.031 → 0.108, jevbench (score, n 18) 0.182 → 0.194
  - NLL worse than T = 1: model_routing_domain (choice, n 399) 0.194 → 0.261, email_spam (noul, n 400) 0.177 → 0.180
  - Brier worse than T = 1: massive (choice, n 5100) 0.332 → 0.366, model_routing_domain (choice, n 399) 0.060 → 0.094, email_spam (noul, n 400) 0.060 → 0.074
- (b′) highest ECE: **model_routing_domain** (choice) 0.030 → 0.403; largest gain: emotion (choice) 0.433 → 0.144
  - ECE worse than T = 1: model_routing_domain (choice, n 399) 0.030 → 0.403, email_spam (noul, n 400) 0.031 → 0.108, jevbench (score, n 18) 0.182 → 0.194, semif (choice, n 144) 0.047 → 0.057
  - NLL worse than T = 1: model_routing_domain (choice, n 399) 0.194 → 0.630, email_spam (noul, n 400) 0.177 → 0.180
  - Brier worse than T = 1: model_routing_domain (choice, n 399) 0.060 → 0.269, email_spam (noul, n 400) 0.060 → 0.074
- (r) highest ECE: **model_routing_domain** (choice) 0.030 → 0.325; largest gain: support_triage (choice) 0.567 → 0.251
  - ECE worse than T = 1: model_routing_domain (choice, n 399) 0.030 → 0.325, email_spam (noul, n 400) 0.031 → 0.108, semif (choice, n 144) 0.047 → 0.067, jevbench (score, n 18) 0.182 → 0.194
  - NLL worse than T = 1: model_routing_domain (choice, n 399) 0.194 → 0.505, email_spam (noul, n 400) 0.177 → 0.180
  - Brier worse than T = 1: model_routing_domain (choice, n 399) 0.060 → 0.204, email_spam (noul, n 400) 0.060 → 0.074
- argmax changes against T = 1: raw 0, default_loto 0, bucket_loto 0, rule_loto 0, oracle 0

### Against BENCHMARKS.md: mean ECE-15 over its 28 head-to-head entries (7 tasks, MASSIVE × 11, XNLI × 10)

| setting | mean ECE-15 |
| --- | ---: |
| (a) as shipped, T = 1 (BENCHMARKS.md: 0.212) | 0.212 |
| (b) held-out pooled default per type | 0.241 |
| (b′) as (b), choice by option-count bucket | 0.149 |
| (r) as (b), choice up to 10 / 11–26 options | 0.140 |
| one T per entry, 2-fold out-of-fold, bench_h2h (BENCHMARKS.md: 0.078) | 0.078 |
| (c) per-task oracle per type | 0.090 |

### Score questions: the expected level sum(i·p_i) at T = 1 / (b = r) / (c)

| task | n | MAE vs gold level | squared error vs gold level | within 1 level | MAE vs the benchmark's gold expected score | MAE of the median level | MAE of the argmax level (any T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jevbench | 18 | 0.220 / 0.287 / 0.220 | 0.290 / 0.233 / 0.244 | 0.944 / 0.944 / 0.944 | – / – / – | 0.222 / 0.222 / 0.222 | 0.278 |
| sst5 | 400 | 0.522 / 0.607 / 0.707 | 0.574 / 0.579 / 0.702 | 0.890 / 0.740 / 0.698 | 0.522 / 0.607 / 0.707 | 0.517 / 0.522 / 0.517 | 0.517 |
| typed_decisions | 800 | 0.402 / 0.505 / 0.452 | 0.478 / 0.399 / 0.362 | 0.931 / 0.874 / 0.896 | 0.482 / 0.309 / 0.324 | 0.403 / 0.388 / 0.395 | 0.405 |
| all score decisions | 1218 | 0.439 / 0.535 / 0.532 | 0.506 / 0.456 / 0.472 | 0.918 / 0.831 / 0.832 |  | 0.438 / 0.429 / 0.433 | 0.440 |

### Variants, each leave-one-task-out (mean over the variant's tasks; T = 1 gives ECE 0.218, NLL 2.157, Brier 0.441)

| variant | T fitted on all tasks | tasks | ECE-15 | NLL | Brier | 28-entry mean ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| (b) one pooled T per type | choice 3.15, noul 6.01, score 5.18 | 19 | 0.153 | 0.955 | 0.362 | 0.241 |
| (b′) per type, choice by option count, tournaments by finalists | 11-26 2.66, 3-5 4.59, 6-10 4.85, noul 6.01, score 5.18 | 19 | 0.139 | 0.933 | 0.355 | 0.149 |
| (r) per type, choice up to 10 options / 11-26 options | 11-26 2.66, 2-10 4.71, noul 6.01, score 5.18 | 19 | 0.132 | 0.919 | 0.348 | 0.140 |
| (b) with task-balanced pools | choice 3.92, noul 6.37, score 4.88 | 19 | 0.149 | 0.933 | 0.356 | 0.202 |
| (b′) with task-balanced pools | 11-26 2.91, 3-5 4.01, 6-10 4.51, noul 6.37, score 4.88 | 19 | 0.142 | 0.931 | 0.355 | 0.154 |
| (r) with task-balanced pools | 11-26 2.91, 2-10 4.41, noul 6.37, score 4.88 | 19 | 0.137 | 0.922 | 0.350 | 0.144 |
| one T for every type and option count | all 3.45 | 19 | 0.159 | 0.983 | 0.373 | 0.257 |
| (b′) with tournaments as their own bucket (> 26) | 11-26 2.66, 3-5 4.82, 6-10 4.85, >26 3.16, noul 6.01, score 5.18 | 19 | 0.145 | 0.931 | 0.356 | 0.150 |
| (r) pooled on the runtime prompt layout only (no SemIf, no voice) | 11-26 2.65, 2-10 4.74, noul 6.01, score 5.18 | 19 | 0.143 | 0.933 | 0.354 | 0.219 |
| (r) without the 33 voice rows that accept alternative answers | 11-26 2.64, 2-10 4.71, noul 6.01, score 5.18 | 19 | 0.128 | 0.896 | 0.341 | 0.110 |
| (r) with typed-decisions as its 4 workflows (held out one at a time) | 11-26 2.66, 2-10 4.71, noul 6.01, score 5.18 | 22 | 0.126 | 0.893 | 0.356 | – |

### A question fitted from n labels: prior centred at 1 vs at the default (NLL / ECE-15 on the cell's other rows; mean over 21 task×cell combinations with ≥ 100 decisions, 100 draws each; default = (r), held out; spread sd = noul 0.63, choice 0.41, score 0.33)

| method | n = 5 | n = 10 | n = 20 | n = 50 |
| --- | ---: | ---: | ---: | ---: |
| T = 1, no labels (engine today) | 2.106 / 0.221 | 2.107 / 0.222 | 2.106 / 0.221 | 2.108 / 0.222 |
| default, no labels | 0.900 / 0.126 | 0.900 / 0.127 | 0.900 / 0.127 | 0.900 / 0.128 |
| fit, prior median 1, sd 1 (engine today) | 1.153 / 0.135 | 0.989 / 0.113 | 0.892 / 0.092 | 0.850 / 0.080 |
| fit, prior median = default, sd 1 | 0.891 / 0.106 | 0.875 / 0.095 | 0.856 / 0.083 | 0.844 / 0.077 |
| fit, prior median = default, sd = spread across tasks | 0.874 / 0.102 | 0.861 / 0.093 | 0.850 / 0.084 | 0.842 / 0.077 |

### Side check: typed-decisions with 4 worked examples in the prefix (not pooled), T = 1 → (r) held out from typed-decisions

|  | ECE-15 | NLL | Brier |
| --- | ---: | ---: | ---: |
| all | 0.245 → 0.041 | 1.754 → 0.708 | 0.507 → 0.390 |
| noul | 0.141 → 0.048 | 1.027 → 0.378 | 0.288 → 0.234 |
| choice | 0.325 → 0.087 | 2.452 → 0.865 | 0.658 → 0.469 |
| score | 0.266 → 0.112 | 1.775 → 0.838 | 0.558 → 0.449 |

### Why many options need a lower T: mean probability outside each decision's four best letters

| task | options | n | T = 1 | T = 2.66 (11-26 default) | T = 4.71 (2-10 default) |
| --- | --- | ---: | ---: | ---: | ---: |
| emotion | 6 | 400 | 0.000 | 0.021 | 0.086 |
| jevbench | 5, 6 | 71 | 0.001 | 0.013 | 0.041 |
| massive | 20 | 5100 | 0.002 | 0.087 | 0.343 |
| model_routing_domain | 6 | 399 | 0.000 | 0.019 | 0.082 |
| support_triage | 10 | 400 | 0.000 | 0.027 | 0.167 |
| typed_decisions | 5 | 200 | 0.000 | 0.016 | 0.052 |
| voice | 16 | 220 | 0.000 | 0.013 | 0.160 |

### Self-checks

- T = 1 metrics against h2h/summary.json (28 entries): largest absolute difference 0.0e+00
- T = 1 metrics against vs_laya/apps.json (7 apps): largest absolute difference 5.0e-05
- T = 1 metrics against JevBench runtime summaries (3 tiers): largest absolute difference 1.7e-16
- pooled noul T with scipy (log_softmax, bounded Brent): 6.0131; this script: 6.0131
- typed-decisions re-read through llama-server by another session (results/speed/td_rows_prod_today_http.jsonl): 2000 decisions matched, argmax agreement 0.984, median largest |Δp| 1.4e-04
<!-- tables:end -->
