# Survey: abstract, cross-disciplinary methods for the one-pass decision readout (23 Sept 2026)

This is a commissioned literature survey. The question: which ideas from other fields could carry over to "one forward pass of a frozen open LLM → typed decision", especially ideas nobody has yet applied to LLM decision readout. Every arXiv id and DOI below was opened during this survey, through the arXiv export API, arxiv.org or Crossref. None is cited from memory. Gains marked *est.* are estimates for Tez, not measurements.

**Baselines used below** (from `BENCHMARKS.md`), all on typed-decisions unless stated:
- Gemma 4 12B letter readout: 0.704 (0.725 with 4 examples in the prefix).
- Qwen3.5-4B letter readout: 0.490.
- 4B layer-26 probe: 0.793. With 10 / 25 / 50 labelled rows per question it reaches 0.669 / 0.707 / 0.741.
- Probe ⊕ letters: 0.799.
- Conformal sets at 90 % coverage: the probe acts on 66.6 % of decisions at 0.884 accuracy.
- Banking77: the chunked tournament gets 0.713 in 5 passes; the MiniLM top-20 shortlist gets 0.738.

**Already covered by `experiments/probe_lab.py` (23 Sept).** It queues or has run: probes trained on pseudo-labels (hard, soft, confidence-filtered, two-teacher agreement), self-training, cluster-then-label, per-question shrinkage LDA, kNN and MLP probes, layer concatenation, a code-size sweep, a last-token universal probe, an escalation loop, logit lens and DoLa, and a depth-wise SPRT gate. First results:
- Adding up log-odds over depth ties with a fixed exit at layer 20 but does not beat it: 0.788 at mean depth 19.8, against 0.7905 at layer 20 (`results/probe_lab_anytime_qwen35-4b.json`).
- The logit lens at layer 29 beats the 4B's own final letter readout: 0.583 against 0.486.
- DoLa peaks at 0.523 (`results/probe_lab_lens_qwen35-4b.json`).

The ideas below are chosen to complement that queue.

## Ranked top 12

| # | Idea | Source field | What to compute | Expected gain (est.) | Cost | Novelty for LLM decision readout |
|---|---|---|---|---|---|---|
| 1 | Label model over several zero-shot readouts → a probe trained with no human labels | Epidemiology (latent-class analysis) / weak supervision | Dawid–Skene EM per question over 12B letters, 4B letters, 4B lens and an embedding similarity; fit the layer-26 probe on its posteriors | 0.73–0.76 with zero labels | offline EM (seconds); 0 extra passes at inference | partial |
| 2 | Pooled LDA with a covariance learned without labels | Neuroscience (BCI population decoding) | Ledoit–Wolf covariance from all unlabelled states; class means from a few labels | +3–6 pts at 10–25 labels per question | one 2,560² inverse; < 0.1 ms | partial |
| 3 | Universal option-boundary scorer, layer chosen by cross-condition generalisation | Neuroscience (abstraction geometry) + representation engineering | one "this option is correct" probe, shared by all questions, on each option's end-of-line state | ≥ 0.70 on unseen workflows with no new labels; no symbol binding | one sweep with options after the input; 0 extra passes | partial |
| 4 | Conformal selection: bound the error rate among acted decisions | Multiple testing / sequential statistics | conformal p-values against calibration errors, then Benjamini–Hochberg; adaptive conformal inference (ACI) online | guarantee "≤ α of acted decisions are wrong" | µs | done-before (selective labelling), not for typed decisions |
| 5 | Collapsing-bound evidence accumulation for streaming commit | Neuroscience (drift-diffusion, confidence) | commit when margin > b(t) = b₀·e^(−t/τ) per action class; confidence that depends on time; a change-of-mind bound | same harm (1/198) with earlier commits or fewer hand-set rules | 0 passes; fitted on stored stream logs | none-found |
| 6 | Select-and-copy QK readout | Associative memory (attention as retrieval) | query of the final token · key at each option's end, in a few mid-layer heads | 4B zero-shot 0.49 → 0.58–0.65 | same pass + a hook | done-before (English multiple-choice QA) |
| 7 | Stacked, extremized log-pooling of readouts | Forecast aggregation | out-of-fold log-linear pool of probes, lens and 4B/12B letters, with an exponent a > 1 | +0.5–2 pts over 0.799; lower NLL | trivial (+1 pass for the 12B) | partial |
| 8 | Typicality-first cold start + a random audit slice for calibration | Active learning | TypiClust / ProbCover choose the first labels; conformal is calibrated on a random slice | +2–4 pts at 10 labels per question; valid coverage | trivial | partial |
| 9 | Error-correcting tournament with Plackett–Luce decoding | Coding theory + social choice | every label in ≥ 2 random chunks; MM fit of Luce weights | Banking77 zero-shot 0.713 → 0.73–0.75 | ~2× the tournament's passes | none-found |
| 10 | MDL codelength + intrinsic dimension to choose the layer without labels | Information theory | prequential codelength per layer and probe family; TwoNN intrinsic dimension on unlabelled states | right layer without a labelled dev set; forecast of the label budget | minutes offline | partial |
| 11 | Hopfield context enrichment + energy-based abstention | Associative memory / physics | β-softmax retrieval from a bank of unlabelled states; log-sum-exp energy as an out-of-distribution score | +1–3 pts at ≤ 25 labels; better out-of-scope escalation | one matmul (< 1 ms) | none-found |
| 12 | Distance-aware probe uncertainty (Laplace / SNGP) | Bayesian ML / kernels | last-layer Laplace on a 256-d code; moderated probabilities | escalation that fires on unseen workflows and languages | trivial | partial |

---

## 1. Label model over zero-shot readouts (Dawid–Skene → probe)

**Mechanism.** Dawid & Skene estimate each annotator's confusion matrix, and the posterior over the true label, by EM from unlabelled agreement patterns alone ([doi:10.2307/2346806](https://doi.org/10.2307/2346806)). It is epidemiology's answer to "several imperfect tests, no gold standard". Data programming and Snorkel extend it to correlated labelling functions ([1605.07723](https://arxiv.org/abs/1605.07723), [1711.10160](https://arxiv.org/abs/1711.10160)).

**Plug-in.** Use four sources on the 6,000 train decisions:
- 12B letters (the teacher sweep is running);
- 4B final letters;
- the 4B logit lens averaged over layers 24–30 (single layers there score 0.54–0.58);
- one independent view: MiniLM cosine between the state and each option's description.

Run EM per question (at most 5 classes, 300 rows each). Train probe_lab's soft-label probe at layer 26 on the resulting posteriors. At inference nothing changes: one 4B pass, then the probe. Co-training between the 12B view and the 4B-state view is the iterative variant ([doi:10.1145/279943.279962](https://doi.org/10.1145/279943.279962)).

**Gain / risk.** *Est.* 0.73–0.76 with no human labels, which is above the 12B teacher. Burns et al. ran weak-to-strong with linear probes on frozen features and report trends qualitatively like fine-tuning ([2312.09390](https://arxiv.org/abs/2312.09390)). The risk: the lens and letters of the same 4B are correlated, which breaks conditional independence and makes the posteriors over-confident. Keep one source per model, or model the dependency.

**Prior work.** AMA aggregates prompt outputs with a label model: +10.2 % over few-shot, and GPT-J-6B beats few-shot GPT-3-175B on 15 of 20 benchmarks ([2210.02441](https://arxiv.org/abs/2210.02441)). Also Smith et al. ([2205.02318](https://arxiv.org/abs/2205.02318)), and ICM, which picks labels by mutual predictability ([2506.10139](https://arxiv.org/abs/2506.10139)). None of them trains a hidden-state decision probe from a label model over layer and model readouts.

## 2. Pooled LDA with a label-free covariance (BCI decoding)

**Mechanism.** Neuroscience reads out intended movements and choices from linear combinations of many noisy neurons ([doi:10.1126/science.3749885](https://doi.org/10.1126/science.3749885)). Brain–computer interfaces with few trials rely on LDA with Ledoit–Wolf shrinkage ([doi:10.1016/j.neuroimage.2010.06.048](https://doi.org/10.1016/j.neuroimage.2010.06.048), [doi:10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X%2803%2900096-4)), and adapt its statistics without labels ([doi:10.1109/TBME.2010.2093133](https://doi.org/10.1109/TBME.2010.2093133)). The class means need labels; the noise covariance does not.

**Plug-in.**
1. At layers 20–28, centre each question's states on that question's unlabelled mean.
2. Pool the centred states across all 20 questions (plus live traffic) and estimate Σ with Ledoit–Wolf.
3. With n labelled rows per class, score δ_c(x) = xᵀΣ⁻¹μ_c − ½μ_cᵀΣ⁻¹μ_c + log π_c.

This amounts to whitening and then taking the nearest centroid, as in SimpleShot ([1911.04623](https://arxiv.org/abs/1911.04623)). Park et al. argue that linear concepts live under a whitened "causal" inner product ([2311.03658](https://arxiv.org/abs/2311.03658)). The cost is one inverse per backbone plus one matrix-vector product per decision.

**Gain / risk.** *Est.* +3–6 pts at 10–25 rows per question, where logreg gets 0.669 / 0.707. probe_lab's per-question LDA estimates Σ from the labelled rows only, so at 10 rows it is almost pure shrinkage. The risks: the pooled Σ absorbs between-class structure, and the tied-Gaussian model can fit badly.

**Prior work.** Class-conditional Gaussians and centroids are used for detecting out-of-distribution inputs ([1807.03888](https://arxiv.org/abs/1807.03888)) and in Hidden Calibration ([2406.16535](https://arxiv.org/abs/2406.16535)). I found no use of a label-free covariance for few-shot decision probes.

## 3. Universal option-boundary scorer, layer chosen by cross-condition generalisation

**Mechanism.** Bernardi et al. define abstraction operationally: a decoder trained on some task conditions must work on held-out conditions (cross-condition generalisation performance, CCGP) ([doi:10.1016/j.cell.2020.09.031](https://doi.org/10.1016/j.cell.2020.09.031)). dPCA separates task variables from condition identity ([doi:10.7554/eLife.10989](https://doi.org/10.7554/eLife.10989)). Wong et al. find that a model first picks the winner in content space, and that the residual state at each option boundary carries linearly decodable per-option correctness ([2601.03914](https://arxiv.org/abs/2601.03914)). Truth directions generalise better across datasets as the training data gets more diverse ([2407.08582](https://arxiv.org/abs/2407.08582), [2407.12831](https://arxiv.org/abs/2407.12831)).

**Plug-in.** Put the input before the options. Tez currently renders the Input last, so the option states never see it; the alternative is to repeat a compact option list after the input.
- In one pass, read the residual state at each option's end-of-line token, at layers 16–28.
- Subtract the per-question mean (demixing).
- Apply one logistic "this option is correct" probe, shared by every question.
- Take a softmax over the options' logits.

Choose the layer by leave-one-workflow-out accuracy (CCGP), not in-distribution accuracy. The scorer works for any number of options and any symbols.

**Gain / risk.** New schemas need no labels. *Est.* ≥ 0.70 on held-out workflows. The scorer also sidesteps the 4B's binding failure (0.49). The risks:
- Causal attention means early options never see later ones, so the scores are absolute rather than relative.
- Probes can latch onto format ([2606.02907](https://arxiv.org/abs/2606.02907)).
- The layout gives up option-prefix caching.

**Prior work.** Partial. Wong et al. show decodability but test no transfer across schemas; probe_lab's universal probe reads only the last token.

## 4. Conformal selection with FDR control

**Mechanism.** Jin & Candès select the test units whose unseen outcome is "good": they compute conformal p-values and run Benjamini–Hochberg, which bounds the false-selection rate ([2210.01408](https://arxiv.org/abs/2210.01408)). e-values make this robust to dependence and usable online ([1912.06116](https://arxiv.org/abs/1912.06116), [2009.02824](https://arxiv.org/abs/2009.02824)). Adaptive conformal inference tracks drift ([2106.00170](https://arxiv.org/abs/2106.00170)).

**Plug-in.** The score V is the probe's maximum probability. For a new decision, p = (1 + #{calibration errors with V_i ≥ V}) / (n + 1). Run BH over a batch, or e-BH online, at level α. Act on the selected decisions and escalate the rest. Run it separately per question type (Mondrian).

**Gain / risk.** The guarantee becomes "at most α of the acted decisions are wrong", which is the contract automation needs. The current rule guarantees coverage, and leaves 11.6 % errors among acted decisions. *Est.* a similar act rate at α = 0.12, and 45–55 % acted at α = 0.05. The risks: BH needs batches; the guarantee is marginal unless done per question type; drift breaks exchangeability, which ACI handles.

**Prior work.** Done before as selective labelling: Conformal Labeling ([2510.14581](https://arxiv.org/abs/2510.14581)) and Conformal Alignment ([2405.10301](https://arxiv.org/abs/2405.10301)). Not yet applied to typed-decision escalation.

## 5. Collapsing-bound accumulation for streaming commit

**Mechanism.** From the decision-making and psychology literature:
- The drift-diffusion model adds up noisy evidence until it hits a bound ([doi:10.1162/neco.2008.12-06-420](https://doi.org/10.1162/neco.2008.12-06-420), [doi:10.1146/annurev.neuro.29.051605.113038](https://doi.org/10.1146/annurev.neuro.29.051605.113038)), and SPRT is its optimal form ([doi:10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118), [doi:10.1037/0033-295X.113.4.700](https://doi.org/10.1037/0033-295X.113.4.700)).
- When time has a cost, the optimal bound shrinks over time ([doi:10.1523/JNEUROSCI.4010-11.2012](https://doi.org/10.1523/JNEUROSCI.4010-11.2012)).
- Confidence depends on both the evidence and the time elapsed ([doi:10.1126/science.1169405](https://doi.org/10.1126/science.1169405)).
- Late evidence can reverse a choice already made ([doi:10.1038/nature08275](https://doi.org/10.1038/nature08275)).
- With many options, the choice becomes a race between competing accumulators ([doi:10.1037/0033-295X.108.3.550](https://doi.org/10.1037/0033-295X.108.3.550)).

**Plug-in.** The voice loop already stores a distribution for every streamed word (`results/voicefast_*_stream.jsonl`). Replace the hand-built class-aware policy with three pieces:
- Commit when the top-1 minus top-2 margin reaches b_c(t) = b₀_c·e^(−t/τ), with t counted in words.
- Report confidence through an isotonic map p(correct | margin, t).
- Allow one revision along the refinement graph once a second bound is crossed.

**Gain / risk.** *Est.* the same ≤ 1/198 harmful actions with fewer rules, or a first action before word 3.6. The risk is overfitting: there are only 198 actionable utterances, all synthetic speech. The depth-wise SPRT came out null because consecutive layers are not independent evidence. Across time, each new word does add evidence.

**Prior work.** None found for LLM streaming decisions. The nearest analogue is CALM's exit threshold, which decays over the generated sequence and is calibrated with Learn-then-Test; that is a collapsing bound for generation, not for commit ([2207.07061](https://arxiv.org/abs/2207.07061)). Also close: sentence-wise 2-D early exit ([2604.18592](https://arxiv.org/abs/2604.18592)), streaming moderation probes ([2606.10487](https://arxiv.org/abs/2606.10487)), and SPRT on time series ([2006.05587](https://arxiv.org/abs/2006.05587), [2501.18059](https://arxiv.org/abs/2501.18059)).

## 6. Select-and-copy QK readout

**Mechanism.** An attention step works like a Hopfield retrieval ([2008.02217](https://arxiv.org/abs/2008.02217)), and some heads "select and copy" the option the model has chosen. Tulchinskii et al. score option i by q_N·k_{t_i}: the final token's query against the key at the end-of-line after option i. They pick the heads on a 5 % labelled validation split; the best heads sit in the middle layers. Reported gains are up to 16 % on LLaMA2-7B and up to 10 % on larger models ([2410.02343](https://arxiv.org/abs/2410.02343)). ICR applies the same idea to reranking ([2410.02642](https://arxiv.org/abs/2410.02642)).

**Plug-in.** Use the same pass and hook Q/K, only on the full-attention layers:
- Qwen3.5-4B: decoder layers 3, 7, …, 31 (0-indexed). The other layers are linear-attention (Gated DeltaNet) blocks with no softmax attention matrix.
- Gemma 4 12B: global layers 5, 11, …, 47. Its 1,024-token sliding window can hide the options from the local layers.

The score is a softmax over the mean QK of the top-h heads. Choose the heads on SemIf authored144 so that typed-decisions stays zero-label. In llama.cpp this needs the eval callback that imatrix uses.

**Gain / risk.** *Est.* 4B zero-shot goes from 0.49 to 0.58–0.65: the lens already reaches 0.583, so the answer exists before the last layers garble it. The risk: the head choice may not carry over across languages. For the 12B the gain is small.

**Prior work.** Done before on English multiple-choice QA; a mechanistic account is in [2407.15018](https://arxiv.org/abs/2407.15018).

## 7. Stacked, extremized log-pooling ("wisdom of layers")

**Mechanism.** Forecasters learned that averaging forecasts built on partly different information is under-confident. Pooling log-odds and then extremizing (a > 1) corrects this ([doi:10.1287/deca.2014.0293](https://doi.org/10.1287/deca.2014.0293), [doi:10.1016/j.ijforecast.2013.09.009](https://doi.org/10.1016/j.ijforecast.2013.09.009)). Stacking fits the weights out-of-fold ([doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080%2805%2980023-1), [1704.02030](https://arxiv.org/abs/1704.02030)), and proper scoring rules keep the fit honest ([doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)).

**Plug-in.** Pool p ∝ exp(a Σ_s w_s log p_s) over probes at layers 20, 24 and 28, the lens mean, the 4B letters, and the 12B letters. Fit w and a by log loss, 5-fold, separately per workflow ("competence gating", [2609.12101](https://arxiv.org/abs/2609.12101)).

**Gain / risk.** *Est.* +0.5–2 pts over 0.799, with lower NLL. The risks: highly correlated layers make the weights unstable, and the pooling rule has to match the dependence between sources ([2608.11275](https://arxiv.org/abs/2608.11275)).

**Prior work.** Partial. Multi-layer probe ensembles for deception detection report +29 % AUROC ([2604.13386](https://arxiv.org/abs/2604.13386)). Paraphrase-and-aggregate cuts intent errors by 22.7 % on CLINC and 15.1 % on Banking ([2406.17163](https://arxiv.org/abs/2406.17163)). Psychology calls averaging one person's repeated guesses "the crowd within" ([doi:10.1111/j.1467-9280.2008.02136.x](https://doi.org/10.1111/j.1467-9280.2008.02136.x)).

## 8. Typicality-first cold start + audit calibration

**Mechanism.** On tiny budgets, labelling typical points (dense cluster centres) beats uncertainty sampling, which only wins later ([2202.02794](https://arxiv.org/abs/2202.02794), [2205.11320](https://arxiv.org/abs/2205.11320); the classic loop is [cmp-lg/9407020](https://arxiv.org/abs/cmp-lg/9407020)). When the model decides which items get labelled, the calibration set stops being exchangeable ([2202.03613](https://arxiv.org/abs/2202.03613)). Importance weighting restores unbiasedness ([0812.4952](https://arxiv.org/abs/0812.4952)).

**Plug-in.**
1. Per question, run k-means on the unlabelled layer-26 states, with k equal to the label budget.
2. Label the densest point in each cluster, stratified by the teacher's predicted class.
3. After about 5 labels per class, switch to margin sampling (probe_lab's online loop).
4. Calibrate conformal on a uniformly random audit slice. Use escalated labels for training only.

**Gain / risk.** *Est.* +2–4 pts at 10 labels per question; the edge shrinks beyond 50. The risk: skewed class priors can leave a class with no typical point at all.

**Prior work.** Partial. TypiClust on frozen LLM embeddings gives an early head start ([2506.01992](https://arxiv.org/abs/2506.01992)), and active learning has been used to pick in-context examples ([2305.14264](https://arxiv.org/abs/2305.14264)). Neither covers decision probes with escalation.

## 9. Error-correcting tournament with Plackett–Luce decoding

**Mechanism.** Error-correcting output codes give each class a codeword spread over many learners, so single errors are corrected ([cs/9501101](https://arxiv.org/abs/cs/9501101)). Error-correcting tournaments do the same for elimination brackets ([0902.3176](https://arxiv.org/abs/0902.3176)). The Plackett–Luce model ([doi:10.2307/2346567](https://doi.org/10.2307/2346567)), fitted by MM ([doi:10.1214/aos/1079120141](https://doi.org/10.1214/aos/1079120141)), merges choices made from different subsets into one score per label.

**Plug-in.** For tasks with more than 26 options:
1. Draw random chunks of 20 labels plus "none", so that every label appears in 2–3 chunks (8–12 passes instead of 5).
2. Treat each chunk's softmax as fractional Luce choices.
3. Fit the label weights by MM and take the argmax.

A sequential-design variant re-chunks the current top 20 ([doi:10.1214/aoms/1177706205](https://doi.org/10.1214/aoms/1177706205)). The MiniLM shortlist can serve as round one.

**Gain / risk.** *Est.* 0.713 → 0.73–0.75 zero-shot, at about twice the passes. The risk: LLM choices violate independence of irrelevant alternatives (attraction effects, [2409.15299](https://arxiv.org/abs/2409.15299); divisive normalisation, [doi:10.1073/pnas.1217854110](https://doi.org/10.1073/pnas.1217854110)). Randomising the chunks mitigates this.

**Prior work.** None found for classification. Closest are setwise and pairwise ranking ([2310.09497](https://arxiv.org/abs/2310.09497), [2306.17563](https://arxiv.org/abs/2306.17563)), permutation self-consistency ([2310.07712](https://arxiv.org/abs/2310.07712)), and multi-LLM query planning ([2603.24617](https://arxiv.org/abs/2603.24617)).

## 10. MDL codelength and intrinsic dimension to choose the layer without labels

**Mechanism.** MDL probing counts the bits a probe needs to transmit labels online, which folds accuracy and learnability into one number ([2003.12298](https://arxiv.org/abs/2003.12298); Pareto probing, [2010.02180](https://arxiv.org/abs/2010.02180)). Without any labels, intrinsic dimension (TwoNN, [doi:10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y)) peaks in an intermediate "abstraction phase" whose representations transfer best ([2405.15471](https://arxiv.org/abs/2405.15471), [2302.00294](https://arxiv.org/abs/2302.00294)). Label-free metrics track how good each layer's representation is ([2502.02013](https://arxiv.org/abs/2502.02013)), and effective rank predicts when extra labels stop helping (AUC 0.787, [2606.24903](https://arxiv.org/abs/2606.24903)).

**Plug-in.** Compute the prequential codelength for every layer and probe family at budgets of 10–300 rows. Compute TwoNN intrinsic dimension per layer. Check whether the dimension peak lines up with the layer 20–28 plateau and with Gemma's layer 34 of 48.

**Gain / risk.** No direct accuracy gain. A new backbone or domain can pick its layer and budget without a labelled dev set. The risk: the peak may be broad. The question of how many dimensions a decision needs is already covered by probe_lab's code-size sweep; neural collapse predicts C−1 ([2008.08186](https://arxiv.org/abs/2008.08186)), and an information-bottleneck head ([1612.00410](https://arxiv.org/abs/1612.00410), [2106.05469](https://arxiv.org/abs/2106.05469)) is the regularised form.

## 11. Hopfield context enrichment + energy abstention

**Mechanism.** Modern Hopfield networks store patterns and retrieve with ξ′ = X softmax(βXᵀξ) ([2008.02217](https://arxiv.org/abs/2008.02217), [1606.01164](https://arxiv.org/abs/1606.01164)). MHNfs enriches few-shot queries with retrievals from a large unlabelled reference set, and its ablation shows this enrichment is the key step ([2305.09481](https://arxiv.org/abs/2305.09481)). Log-sum-exp energy works as an out-of-distribution score ([2010.03759](https://arxiv.org/abs/2010.03759)).

**Plug-in.** The bank is the unlabelled layer-26 states. Feed [ξ, ξ′] to the probe or LDA. Use the energy −β⁻¹ log Σ exp(β x_iᵀξ) as an escalation feature, and −T·logsumexp(letter logits) for out-of-scope voice commands.

**Gain / risk.** *Est.* +1–3 pts at ≤ 25 labels, and better out-of-scope precision (0.917 today). The risk: it may add nothing on top of idea 2's whitening.

**Prior work.** kNN Prompting is the closest ([2303.13824](https://arxiv.org/abs/2303.13824)).

## 12. Distance-aware probe uncertainty (Laplace / SNGP)

**Mechanism.** A Laplace approximation turns a linear head into a Gaussian posterior, which damps confidence away from the training data ([2106.14806](https://arxiv.org/abs/2106.14806)). SNGP uses a random-feature Gaussian-process head, so uncertainty grows with distance ([2006.10108](https://arxiv.org/abs/2006.10108)). Chow's rule then decides when to reject ([doi:10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406)).

**Plug-in.** Fit a last-layer Laplace approximation on a PCA-256 code, apply a probit-moderated softmax, and escalate on posterior variance as well as on the margin.

**Gain / risk.** Escalation that fires on unseen workflows and languages, where the softmax stays over-confident. Accuracy is unchanged. The risk: the variance may just re-express the margin.

**Prior work.** Partial: Bayesian linear probes per layer ([2510.04108](https://arxiv.org/abs/2510.04108)).

---

## Also considered

| Method (field) | How it plugs in / expected effect | Precedent, verdict, citations |
|---|---|---|
| SPRT / e-processes over extra passes (sequential statistics) | Run permutations only until the likelihood ratio or e-value crosses 1/α. Helps low-binding backbones (4B on SemIf rows: 0.813 → 0.912 with 6 permutations); adds nothing for the 12B, which has 0 reversal flips | Adaptive-Consistency [2305.11860](https://arxiv.org/abs/2305.11860), ESC [2401.10480](https://arxiv.org/abs/2401.10480); [doi:10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118), [2210.01948](https://arxiv.org/abs/2210.01948), [1810.08240](https://arxiv.org/abs/1810.08240), [2001.05989](https://arxiv.org/abs/2001.05989), [2503.13050](https://arxiv.org/abs/2503.13050) |
| Patience / SPRT over depth | Done in probe_lab: no gain over a fixed exit at layer 20 | PABEE [2006.04152](https://arxiv.org/abs/2006.04152); Learn-then-Test thresholds [2207.07061](https://arxiv.org/abs/2207.07061), [2110.01052](https://arxiv.org/abs/2110.01052), [2510.02480](https://arxiv.org/abs/2510.02480) |
| Tuned lens, calibration direction (representation engineering) | Next step after the logit lens and DoLa results | [2303.08112](https://arxiv.org/abs/2303.08112), [2511.00280](https://arxiv.org/abs/2511.00280), [2309.03883](https://arxiv.org/abs/2309.03883) |
| Function / task / in-context vectors, label-word anchors | Inject the 4-shot task vector into a zero-shot pass. The cached 4-shot prefix is already cheap, so this only pays for many-shot compression | [2310.15213](https://arxiv.org/abs/2310.15213), [2310.15916](https://arxiv.org/abs/2310.15916), [2311.06668](https://arxiv.org/abs/2311.06668), [2305.14160](https://arxiv.org/abs/2305.14160) |
| Consensus game (game theory) | Equilibrium ranking between the letter readout and the idea-3 scorer; LLaMA-7B sometimes beat 65B/540B | [2310.09139](https://arxiv.org/abs/2310.09139) |
| Surprisingly popular / truth serum (economics) | Skip: for LLM crowds, majority voting beats SP ([2510.01499](https://arxiv.org/abs/2510.01499)), and contextual calibration already failed here | [doi:10.1038/nature21054](https://doi.org/10.1038/nature21054), [2311.07692](https://arxiv.org/abs/2311.07692) |
| Group testing / compressed sensing (coding theory) | Multi-label rule checks: pooled yes/no questions on a cached state, about k·log₂(n/k) passes instead of n. None found for LLMs | [doi:10.1214/aoms/1177731363](https://doi.org/10.1214/aoms/1177731363), [1902.06002](https://arxiv.org/abs/1902.06002), [0902.1284](https://arxiv.org/abs/0902.1284) |
| Optimal transport / re-estimating the class prior | Sinkhorn matching to label marginals (OTTER reports +4.8 % and +15.9 % average); EM prior adaptation; rational-inattention logit. Batch Calibration already covers most of this | [2404.08461](https://arxiv.org/abs/2404.08461), [1306.0895](https://arxiv.org/abs/1306.0895), [doi:10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446), [doi:10.1257/aer.20130047](https://doi.org/10.1257/aer.20130047) |
| CCS (unsupervised directions) | Yes/no questions only; known pitfalls | [2212.03827](https://arxiv.org/abs/2212.03827), [2312.10029](https://arxiv.org/abs/2312.10029) |
| Causal subspaces (DAS) | Would make probes robust to format; research-grade cost | [2303.02536](https://arxiv.org/abs/2303.02536) |
| Test-time training; kernel / eNTK; sparse-autoencoder probes | Backprop per decision breaks the latency budget; kernels bring no expected gain over linear probes; sparse-autoencoder probes did not beat linear baselines | [2305.18466](https://arxiv.org/abs/2305.18466), [2210.05643](https://arxiv.org/abs/2210.05643), [2502.16681](https://arxiv.org/abs/2502.16681) |
| meta-d′ and Murphy diagrams (evaluation) | Choose escalation signals by metacognitive efficiency; rank readouts over every act/escalate cost ratio | [doi:10.1016/j.concog.2011.09.021](https://doi.org/10.1016/j.concog.2011.09.021), [2512.10451](https://arxiv.org/abs/2512.10451), [2603.25112](https://arxiv.org/abs/2603.25112), [1503.08195](https://arxiv.org/abs/1503.08195) |

---

## Experiments to run first

**E1 — Zero-label probe from a label model (idea 1).**
- Inputs: `probe_cache_qwen35-4b.npz` and the 12B teacher labels, once that sweep finishes.
- Sources per train decision: 12B letters; 4B letters; the 4B lens mean over layers 24–30; MiniLM cosine between the state and each option.
- Fit Dawid–Skene per question with no gold labels: full confusion matrices when there are ≤ 3 classes, one-coin otherwise. Controls: majority vote and the 12B alone.
- Train the layer-26 probe three ways: on hard labels, on soft posteriors, and on posteriors ≥ 0.8 only.
- Report accuracy, NLL and the conformal act rate. Success: ≥ 0.73 with zero human labels.

**E2 — Pooled-covariance LDA with few labels (idea 2).**
- Layers 20, 24, 26 and 28.
- Estimate Σ with Ledoit–Wolf on train states after removing each question's mean.
- Budgets of 1, 2, 5 and 10 labels per class; 5 seeds; stratified sampling.
- Compare against logreg, per-question sklearn LDA, cosine centroid and SimpleShot. Success: ≥ +3 pts over logreg at ≤ 25 rows per question.

**E3 — One 4B sweep for option-boundary states and QK scores (ideas 3 and 6).**
- Re-render the 8,000 prompts with the options last.
- Record the last-token states, the option end-of-line states at every second layer from 12 to 28 (about 1.3 GB in fp16), q_N·k_{t_i} for all heads of the 8 full-attention layers, and the letter logits.
- Offline, test three things: the universal option scorer with leave-one-workflow-out per layer; the QK readout with heads chosen on SemIf; and the new layout's letter baseline.
- Success: ≥ 0.70 on held-out workflows for the scorer, and ≥ 0.58 zero-label for QK.

**E4 — Conformal selection on the cached probe (idea 4).**
- 20 random 50/50 splits of test cases into calibration and evaluation, as in `conformal_td.py`.
- Compute conformal p-values from the probe's maximum probability; run BH at α ∈ {0.05, 0.10, 0.15}, per question type.
- Report the act rate and the realised false-action rate, against the set-size-1 rule (66.6 % acted at 0.884). Repeat with the idea-7 pool.
- Success: realised FDR ≤ α, and an act rate within 3 pts of the current rule at the same error.

**E5 — Collapsing-bound streaming commit (idea 5).**
- Data: `voicefast_partial_last_gemma4-12b-q8_0_stream.jsonl`, 220 utterances.
- Commit a non-none action when the margin reaches b₀·e^(−t/τ); never commit on a confident none; allow one refinement.
- Fit b₀ per class group (open, media, type/close) and τ by 5-fold cross-validation over utterances.
- Compare against the current policy: 1/198 harmful, first action at word 3.6, 2/22 false actions on out-of-scope commands, 0.985 consistency. Also measure the ECE of p(correct | margin, word).
