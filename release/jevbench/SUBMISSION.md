<!--
DRAFT. NOT POSTED. Issue text for https://github.com/fstandhartinger/jevbench (JevBench v1.4.0).
Before posting:
  [x] public-tier table filled from run_public.py (results/jevbench_tez_server_short_warm.json, _hard.json)
  [ ] check the install command and the `tez serve` flags against the released package (`tez serve --help`); they
      match the package in the working tree on 2026-09-23 (entry point `tez`, --backend default :8080, port 8787)
  [ ] make sure the repository is public and name the commit or tag the maintainer should run
  [ ] delete this comment
-->

**Title:** Evaluation request: Tez (frozen Gemma 4 12B Q8_0, one-pass letter readout, runs locally)

### System

- **Name:** Tez
- **Code:** https://github.com/Jibalmi/Tez (MIT)
- **Endpoint:** self-hosted, TypeSafe's wire format: `POST http://127.0.0.1:8787/v1/systemone`, plus `GET /v1/models`
  and `GET /healthz`. Any bearer token is accepted locally; `tez serve --api-key <key>` makes one required.
- **Model alias:** `tez-latest`

### What it is

Tez is an open, local "System One" decision engine. For every question it runs one forward pass of a frozen
**Gemma 4 12B** (instruction-tuned, **Q8_0** GGUF) served by **llama.cpp**, and reads the probability of each option's
letter at the answer position. No tokens are generated.

- **No fine-tuning.** No weights, adapters, heads or temperatures are trained or fitted for this submission.
  (Tez also has a probe readout trained from a user's own labels; it is not used here. With no probe trained, the
  default readout is letters for every question.)
- **Readout:** `/completion` for one token with the top-200 log-probabilities; a softmax over the option letters gives
  the distribution over exactly the question's labels. Instructions and options come first and the state last, so
  the constant part of each prompt is cached. Choice questions with more than 26 options run as a chunked
  tournament; the public tiers have at most 6 labels, so each item here is a single pass.
- **Answer shapes:** Jev's three: `noul` = P(yes); `choice` = label plus `probabilities` over exactly the label set;
  `score` = expected level plus `probabilities` over the levels.
- Nothing in Tez was trained on or derived from Jev outputs (Jev's terms forbid it).

### How to run it

1. **Weights.** Gemma 4 12B instruction-tuned, Q8_0. The file behind our numbers is Ollama's `gemma4:12b-it-q8_0`
   GGUF (blob `sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795`, 12,669,645,728 bytes),
   loaded directly by llama-server. The public equivalent is `gemma-4-12b-it-Q8_0.gguf` from
   [`unsloth/gemma-4-12b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF); that file was not separately
   measured.

2. **llama.cpp**, build b11100 (we used the `win-cuda-13.4` release; other builds are untested):

   ```
   llama-server -m gemma-4-12b-it-Q8_0.gguf -ngl 99 -c 8192 -b 512 -np 1 --swa-full --host 127.0.0.1 --port 8091 --no-webui
   ```

   `--swa-full` matters: without it llama-server cannot reuse Gemma's sliding-window cache and re-evaluates the whole
   prompt on every request. `-c 8192` holds the hard tier's longest states (about 3.9k tokens). One slot (`-np 1`).
   On a 16 GB GPU, `-c 8192` together with `--swa-full` does not fit; drop `--swa-full` there (answers are unchanged).

3. **Tez**, in front of it:

   ```
   git clone https://github.com/Jibalmi/Tez
   cd Tez
   pip install -e .
   tez serve --backend http://127.0.0.1:8091 --template gemma4     # serves http://127.0.0.1:8787
   ```

   (`--backend` is needed because Tez's default backend address is llama-server's own default, port 8080.)

4. **Check it answers:**

   ```
   curl -s http://127.0.0.1:8787/healthz
   curl -s http://127.0.0.1:8787/v1/systemone -H "Content-Type: application/json" -d '{"model": "tez-latest", "state": "Help! My payouts have been failing for 3 days.", "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'
   ```

5. **Our own public-tier run** (standard library only): `python release/jevbench/run_public.py` sends each of the
   231 public items once, one request at a time with no concurrency, times caller wall time including the network,
   and writes `results/jevbench_tez_server.json` with a manifest (endpoint, server model string, SHA-256 of each
   data file).

### Hardware behind our numbers

One laptop: NVIDIA RTX 5080 Laptop GPU (16 GB), Windows 11, llama.cpp b11100. The 12.7 GB Q8_0 file ran with
`-c 8192 -np 1` on that GPU. We have not measured any other GPU, a server card or CPU-only inference.

### Public-tier numbers through the wire format

From `release/jevbench/run_public.py` against `tez serve` on 2026-09-23 (formulas below). Standard and easy:
`results/jevbench_tez_server_short_warm.json`, llama-server `-c 4096 --swa-full`, second pass on a warm server (a first
pass right after a restart measured p50 0.192 s on the standard tier, with identical accuracy). Hard:
`results/jevbench_tez_server_hard.json`, llama-server `-c 8192` without `--swa-full`, because the full sliding-window
cache at 8k does not fit next to the 12.7 GB weights in 16 GB and spills into shared memory; answers are unchanged,
but every hard item's prompt is evaluated in full, so its times are pessimistic. A pass of all three tiers at 8k with
`--swa-full` gave the same accuracy on standard and easy before the spill made it too slow to finish.

| tier (n) | accuracy | chance | intelligence | ECE-15 | NLL | p50 s | p95 s | speed score p50 / p95 | invalid + errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard (original) (72) | 0.958 | 0.311 | 94.0 | 0.039 | 0.168 | 0.052 | 0.087 | 105.7 / 101.2 | 0 |
| easy (48) | 1.000 | 0.284 | 100.0 | 0.000 | 0.000 | 0.050 | 0.083 | 106.1 / 101.6 | 0 |
| hard (111) | 0.703 | 0.336 | 55.2 | 0.241 | 1.519 | 0.716 | 2.173 | 82.9 / 73.3 | 0 |

Public-tier intelligence with the leaderboard's tier weights (hard 0.30, easy 0.14, standard 0.28; the judge tier is
not public): **79.0** (indicative only; the leaderboard scores held-out items). Paraphrase-pair consistency on the
standard tier: 0.917.

Formulas, as in our `experiments/bench_jevbench.py`: intelligence = max(0, (accuracy - chance) / (1 - chance)) x 100
with chance = mean of 1 / |labels|; ECE-15 on the max probability; speed score = 100 - 20 log10(s / 0.1 s). A
distribution that does not cover exactly the label set, leaves [0, 1] or sums outside 1 +/- 2 % counts as wrong, as
your README states. Times are unadjusted: we have not applied your x2 (+0.15 s) self-hosted adjustment.

### For reference: the same model through our experiment harness

Measured earlier with `experiments/bench_jevbench.py` calling llama-server directly (same weights, flags and
hardware, no Tez server in between), from `results/jevbench_tez_gemma4-12b-q8_0.json`, scored with the formulas
above (summary in `BENCHMARKS.md` §5). The wire-format run adds an HTTP hop and Tez's own prompt handling, so its
numbers can differ.

| tier (n) | accuracy | chance | intelligence | ECE-15 | NLL | p50 s | p95 s | speed score p50 / p95 | invalid + errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard (original) (72) | 0.944 | 0.311 | 91.9 | 0.053 | 0.312 | 0.053 | 0.089 | 105.5 / 101.0 | 0 |
| easy (48) | 1.000 | 0.284 | 100.0 | 0.000 | 0.000 | 0.056 | 0.102 | 105.1 / 99.9 | 0 |
| hard (111) | 0.703 | 0.336 | 55.2 | 0.256 | 1.486 | 0.466 | 2.199 | 86.6 / 73.2 | 0 |

- Public-tier intelligence with the weights above: 78.2 (indicative only).
- Paraphrase-pair consistency on the standard tier: 0.944.
- Hard tier by family: routing_hard 1.000, trap 0.875, ambiguous 0.857, adversarial 0.833, long_policy 0.789,
  multi_hop 0.778, judge_hard 0.765, probability 0.700, tradeoff 0.667, temporal_numeric 0.133 (2 of 15: date and
  quantity arithmetic, which one forward pass does poorly).
- On the 10 hard items that carry `gold_probs`, the mean total variation distance between our distribution and the
  gold distribution is 0.388 (our own measure, not your calibration formula).
- `BENCHMARKS.md` §5 quotes the hard tier from a separate hard-only run (`results/jevbench_tez_gemma4-12b-q8_0_hard.json`):
  same accuracy and intelligence (0.703, 55.2), ECE 0.251, p50 0.456 s.

### What was and was not used

- **No training on anything.** The weights are Google's public release, used frozen.
- **Public tiers, evaluation only.** We have scored the public tiers before (the reference table above). Two
  experiments read them alongside other benchmarks: a Qwen3.5-9B backbone comparison on the standard and easy tiers,
  and a 64-token thinking budget on the hard tier, which lowered accuracy (0.703 to 0.658) and was not adopted.
  The held-out tiers were never available to us.
- **Gemma 4's own training data** is Google's; we cannot audit it for JevBench items.

### Cost

Tez runs locally and has no per-decision price. We have not measured energy or hardware cost, so we leave the
cost basis for a self-hosted run to your usual rule.

### Licences

- Gemma 4 weights: Apache-2.0
- Tez code: MIT
- The JevBench public items we ran are MIT-licensed (per their `provenance`).

Every number in this issue comes from a result file in the repository (`results/`); `run_public.py`'s output also
records the endpoint, the server's model string and the SHA-256 of each data file. Happy to adjust anything to your
harness's conventions.
