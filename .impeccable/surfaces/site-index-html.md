---
version: 1
slug: "site-index-html"
primary_target: "site/index.html"
related_targets: ["site/usecases.html","site/playground.html","site/benchmarks.html","site/docs.html","site/research.html"]
---

## Scope

`site/index.html`, the landing page (Persuade). Same world for `site/usecases.html` (browse, Persuade),
`site/playground.html` (Operate), `site/benchmarks.html`, `site/docs.html` and `site/research.html` (Read).

## Audience, job, action

Builders first (confirmed): developers and ML engineers deciding whether Tez can replace a hosted-LLM call or a
per-task classifier. Action: install and run `tez serve`, or open the playground. Proof on hand:
`site/data/strips.json` (recorded runs), `site/data/benchmarks.json`, `examples/usecases/`, `docs/figures/`.

## Constraints

Static, GitHub Pages under `/Tez/`. Recorded replays labelled as replays. No invented users, logos, stars, prices or
JevBench rank. Memorable moment: the hero deciding a real message word by word.

## Unresolved

PyPI name (`tez-decisions` proposed); Hugging Face repository names.

## Direction contract

THESIS: The category standard for local-model tools at Ollama, LM Studio and Hugging Face craft: a real decision proves the product in viewport one; every number is sourced. Refuses glow, gradients, fake logos, hype.

OWN-WORLD: Light-first with automatic dark; ink #0B0D10 on white and #F6F7F9; hairline borders, 8px cards, model-card pill tags; one blue (#2F5BD8 for text and buttons, #3F6FF0 for bars) for actions and chosen options; Geist Sans and Geist Mono; probability bars recur.

STORY: A builder sees what Tez is, watches it decide, trusts sourced head-to-heads with losses shown, then installs it or opens the playground.

FIRST VIEWPORT: Left five columns: H1 "Typed decisions from one forward pass.", subhead, Get started and Playground buttons, copyable install. Right seven: a recorded run replayed word by word, live probability bars and milliseconds, JSON, Python and curl tabs. Signature: watch it decide; bars ease 120 ms; reduced motion shows the final state.

FORM: canon, the owner's standing exit; seed key bf88ebe9.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
