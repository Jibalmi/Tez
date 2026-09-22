# Survey: raising accuracy / calibration / order-robustness of the single-pass readout (22 Sept 2026)

Commissioned literature survey (web-verified arXiv ids). "Gain" is an estimate for Tez (Gemma 4 12B, letter-logit readout). Training-free unless marked.

| # | Method (paper) | Expected gain, on which Tez task | Cost | Risk |
|---|---|---|---|---|
| 1 | **Hidden Calibration** – nearest-centroid / linear head on the last-token hidden state instead of label-token logits (2406.16535, NAACL'25); token/layer-selective probes (2601.13288) | Largest single lever: +20–50 % relative over token-based ICL on 10 classification sets. typed-decisions 0.704 → likely 0.75+; Banking77, MASSIVE, Emotion, SST-5. Needs ~50–200 labelled rows per schema | Low: one hidden-state dump per pass; fit centroids/logreg per schema (probe only) | Per-schema labelled data; drift if the template changes; tune the layer |
| 2 | **Order-invariant attention** – PINE (2407.01100), Set-Based Prompting (2406.06581), **RoToR** (2502.08662, ACL'25): position IDs + mask make the options a set; one pass, provably order-invariant | Replaces 6-permutation averaging (6× cost) with 1 pass; +8–10 pts reported on MCQ/judge for 7–70B; RoToR handles letter "indexing bias" | Medium: custom attention mask/pos-ids in HF/vLLM; RoToR has code | Train/inference mismatch; Gemma 4 local-attention layers need checking |
| 3 | **PriDe** (2309.03882) / **CalibraEval** NOA (2410.15393): estimate the option-ID prior from a few permuted samples and divide it out | +2–4 pts, halves recall std across positions; typed-decisions, Banking77 chunks | Very low: ~5 % of rows permuted once | Interacts with Batch Calibration — apply PriDe before BC |
| 4 | **Option elimination** PoE_ID^log (2501.15175), POE (2310.15575) | Gains grow with option count: Banking77 (77), MASSIVE (20) +2–5 pts vs the chunked tournament | Low: 2 passes | Eliminating the right answer early; tune threshold held-out |
| 5 | **Retrieval-narrowed label space + kNN demos** (2309.10954; RAC 2501.12332; confusion-aware retrieval 2609.01564) | Banking77 with retrieved top-k labels + 5 retrieved demos: LLaMA-2-70B 0.890; Tez likely 0.80–0.85 | Medium: embedder + index; demos break the prefix cache unless bucketed | Retrieval miss = guaranteed error; k≈10–15 for recall > 97 % |
| 6 | **Many-shot ICL in the cached prefix** (2404.11018; 2405.00200) | Banking77/MASSIVE keep improving to 1000+ shots; shuffle (grouping same-label examples hurts) | Prefix caching makes it cheap; long context memory | Check long-context ICL degradation (2404.02060) |
| 7 | **Surprise Calibration** (2506.12796, EMNLP'25) – per-query class prior from demonstration "surprise" | Beats BC/DC/CC on standard sets; NLI, Emotion in the few-shot config | Low | Needs demos in context |
| 8 | **Linear Probe Calibration LinC** (2401.12406) – affine map on logits from 5–20 labelled samples | Up to +21 % avg on text classification, lower ECE and template variance | Trivial | ("Supervised Calibration" 2505.23783 withdrawn — ignore) |
| 9 | **Content scoring + PMI_DC** (2104.08315; 2210.12353; 2403.00998; 2404.08382; 2601.03914) | Ensemble member for yes/no and small ordinal schemas; probes at option-boundary states sidestep binding failures | Medium: N extra passes | Worse than symbols on strong 12B models alone |
| 10 | **Verbalizer / label-word ensembles** (PET 2001.07676; LM-BFF 2012.15723; kNN-Prompt 2205.13792; 2410.06173; Semantic Anchors 2511.21038; In-Context Fixation 2605.08295; ABCD 2602.17445) | Emotion / SST-5 / typed-decisions: letter + natural-language label, average over 3–5 synonym verbalizers; ABCD unordered labels reduce position bias; +1–3 pts | Low | Multi-token labels need length normalisation |
| 11 | **Expected-value / risk-averse ordinal readout** (2503.03064; E-Score 2508.03550; 2505.19334) + cumulative-link / CORAL-CORN head (1901.07884; 2111.08851; DLOM 2603.14891) | SST-5: mean-of-distribution beats mode (+6.5 % in judge settings); CORN head on E[score] | Trivial to light | Mean is a regression; report MAE/QWK |
| 12 | **Conformal with class-conditional / clustered guarantees** (2306.09335; 2508.05544; 2508.10022; 2509.24095) | Selective prediction; clustered conformal fixes small-class coverage on Banking77/MASSIVE | Low | Per-schema calibration set |
| 13 | **Abstention as a separate head** (NOTA 2503.01550; AUGRC 2407.01032; 2609.04582; 2608.26121) | Treat "none of the above" as a binary head (margin + probe), evaluate with AUGRC; explains the contextual-calibration failure (CC forces a uniform prior incl. NOTA) | Low | Two thresholds |
| 14 | **Thermometer** (2403.08819, ICML'24) – auxiliary net predicts a per-task temperature | Calibration on unseen schemas without labels | Light training once | NLL/ECE only |
| 15 | **Unsupervised LoRA + KV-cached majority vote BaQCKV** (2511.21709) | Cheap permutation voting; LoRA removes selection bias | Light LoRA | Base-model change |

Superseded / negative: generative calibration (2310.10266) needs sampling; domain-context calibration (2305.19148) is BC-like; NTL (2411.02083) needs pretraining-style loss.

## Top 6 to try first (experiment designs)

1. **Hidden-state centroid/probe.** Dump the last-token hidden state (layers L−4..L); fit per-schema nearest-centroid and logreg on 100 out-of-fold labelled rows for typed-decisions and Banking77; compare with letter logits and with ModernBERT's 0.766; ensemble probe + letter logits.
2. **RoToR order-invariant single pass** on the option block only; measure accuracy and per-position recall std on authored144 / typed-decisions vs 1-perm and 6-perm at 1× cost.
3. **PriDe → BC.** Option-ID prior from cyclic permutations of 5 % of each schema; ablate order (PriDe→BC vs BC) on NLI and typed-decisions.
4. **Retrieval-narrowed Banking77.** bge-m3 index over label descriptions + train utterances; top-12 labels + 5 nearest demos (bucketed for the prefix cache); then PoE_ID^log on the survivors; target > 0.83.
5. **Ordinal expected-value readout + CORN head** on SST-5: E[score] vs argmax, MAE/QWK/accuracy; then a 1-D cumulative-link head, out-of-fold.
6. **NOTA as a separate selective head with AUGRC** on typed-decisions; re-test contextual calibration with NOTA excluded from the prior.
