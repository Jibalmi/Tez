# Product

<!-- impeccable:product-schema 1 -->

The owner confirmed the stack, the primary audience and the visual stance on 2026-09-23. Facts still marked
*(inferred)* come from the repository and the original request, not from a confirmed answer.

## Platform

web

## Stack

Static HTML, CSS and JavaScript with no build step, served from `site/` by GitHub Pages through a GitHub Actions
workflow (confirmed). The playground calls a local Tez server from the browser; CORS is open on `tez serve`.

## Users

- **Builders of automated pipelines** (primary audience, confirmed): developers and ML engineers wiring decisions into agents, support
  queues, routers, moderation, document workflows and voice interfaces. They today call a hosted LLM (slow, paid,
  uncalibrated, data leaves the building) or train a classifier per task (needs labels and upkeep). They evaluate a
  tool by its numbers, its install path and whether it fits the client code they already have *(inferred)*.
- **Researchers and benchmark followers** *(inferred)*: people tracking "System One" decision models (Jev, Laya,
  decider, SemIf, the JevBench leaderboard) who read methods, ablations and negative results.
- **Privacy-bound teams** *(inferred)*: organisations that cannot send records to a hosted API and need the decision
  to run on their own hardware.

## Product Purpose

Tez is an open, local "System One" decision layer. A caller sends one state (a message, a document, a transcript) and
typed questions: yes/no (`noul`), pick one option (`choice`, up to 255 options), or an ordered level (`score`). Tez
answers every question with a probability distribution, read from one forward pass of a frozen open model, with a
calibrated confidence and an explicit abstain or escalate path. No tokens are generated.

Success *(inferred)*: a builder installs it, points an existing Jev-compatible client at `http://127.0.0.1:8787`,
gets correct, calibrated decisions in tens to hundreds of milliseconds on their own GPU, and trusts it because every
claim traces to a script and a result file.

## Positioning

- Reads the decision out of the model instead of asking it to write: the option letter's probability at the moment
  before it would speak (letters), or a linear probe on a middle layer's hidden state (probe).
- Frozen open weights, no fine-tuning, built from the published technique, never from Jev's outputs (Jev's terms forbid
  that).
- Speaks TypeSafe's `/v1/systemone` wire format, so clients written for Jev work unchanged (`docs/API.md`).
- Measured head to head against Laya, the open encoder-based competitor, on Laya's own datasets and protocol, with the
  places Laya wins stated.
- Knows when not to decide: calibrated confidence, an abstain option, and a conformal gate that bounds the error rate
  among acted decisions.

## Operating Context

- Runs on the builder's machine: llama.cpp `llama-server` hosting a GGUF model, `tez serve` in front of it. Reference
  hardware for every published number: an RTX 5080 laptop GPU (16 GB).
- Default models: Gemma 4 12B Q8_0 for zero-shot letters; Qwen3.5-4B cut to 24 of 32 blocks for probes.
- Integration points: Jev SDKs and anything speaking the wire format; Python API; CLI (`tez serve`, `decide`, `fit`,
  `suggest`, `eval`, `truncate`); schema YAML files; a feedback log that feeds `tez fit`.
- Evaluation rituals the audience knows: JevBench (four axes: intelligence, calibration, speed, cost), accuracy tables,
  reliability diagrams, latency percentiles, Hugging Face model cards, arXiv-style reports.

## Capabilities and Constraints

- Measured (`BENCHMARKS.md`): typed-decisions zero-shot 0.704 (12B letters), 0.793 with a probe on the frozen 4B,
  0.766 with the 12B prior plus 50 typical labels per question; Banking77 0.713 zero-shot, 0.860 with a probe; MASSIVE
  0.885 macro over 11 languages; voice commands 0.905 intent accuracy at about 30 ms per streamed word.
- Constraints: the model runs locally, so the public website cannot run it; the playground needs a local server, and
  anything shown without one is a recorded replay and must be labelled as such. Probes need 25 to 50 labels per
  question. On unseen schemas without labels, fine-tuned competitors can beat zero-shot letters. The 12B model costs
  more time and memory than small encoders.
- Licences: Tez code MIT; Gemma 4 12B weights Apache-2.0 (`docs/REPORT.md` licence audit); Qwen3.5 Apache-2.0.
- Measured weights: Gemma 4 12B Q8_0 is Ollama's `gemma4:12b-it-q8_0` GGUF (blob `sha256-047dae1d…`); the equivalent
  public file is `unsloth/gemma-4-12b-it-GGUF` / `gemma-4-12b-it-Q8_0.gguf`, not separately measured.
- Undecided: package name on PyPI (`tez-decisions` proposed); Hugging Face repository names; JevBench score (not yet
  submitted: never state a leaderboard rank).

## Brand Commitments

- Name: **Tez** (renamed from "onepass").
- Voice, evidenced by `README.md`, `BENCHMARKS.md` and `docs/REPORT.md`: plain, specific and measured; negative results
  are published and labelled as negative; every number carries its conditions (model, hardware, dataset, n).
- Visual stance (owner's choice, 2026-09-23): the category standard for open local-model tools, executed at full
  craft, without irony or smuggled quirk. Craft bar: Ollama, LM Studio and Hugging Face. Standing preference: new
  surfaces sit alongside those products rather than departing from them.

## Evidence on Hand

- Numbers: `BENCHMARKS.md`, `results/*.json`; head-to-head tables in `README.md`.
- Research: `docs/REPORT.md`, `docs/Tez_research_report.pdf` (35 pages), `docs/paper/tez-paper.tex`.
- Figures: `docs/figures/*.png` (dashboard, calibration, languages, order flip, speed, typed decisions, probe lab, Tez vs
  Laya vs Jev).
- Data: `data/voice/` (220 spoken commands), `data/jevbench/` (public tiers), `data/semif/`.
- Use cases and examples: `examples/usecases/` (in progress), `docs/API.md`.
- Absent, never to be fabricated: users, customers, testimonials, logos, pricing, a hosted service, a JevBench rank,
  download counts, stars.

## Product Principles

1. **Measured, not claimed.** Every number links to the script and result that produced it; losses are shown next to wins.
2. **Local and open.** The decision runs where the data lives; weights and code are inspectable.
3. **Drop-in.** Meet builders in the client code they already run; the wire format is the contract.
4. **Knowing when not to decide is a feature.** Confidence, abstain and escalate are first-class outputs.
5. **Honest limits.** Say where a fine-tuned or smaller competitor is the better tool.

## Accessibility & Inclusion

*(inferred)* WCAG 2.2 AA for every page: keyboard access to the playground, text alternatives for every chart,
reduced-motion support, and no information carried by colour alone (probabilities always also appear as numbers).
