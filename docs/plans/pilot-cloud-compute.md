# Pilot compute options, 2026-09-27

## Measured reason to move GPU work

The local Gemma teacher's recovered segment completed 1,200 states in 1,587 seconds, including
962.3696 seconds cooling (60.6% of elapsed time). This is from `runs/pilot/logs/teacher.log`;
it is a laptop measurement, not a RunPod benchmark. Cloud throughput has not been measured.

The same machine successfully ran the GPU teacher alongside Codex labeling clients. On the
160-state / 800-decision sample, four Codex workers took 50.03 seconds, eight took 35.85 seconds,
and twenty took 35.00 seconds. All outputs validated. The clients shared one logical CPU to
limit interference with the GPU job, so the near-flat eight-to-twenty result does not establish
a provider concurrency limit. See [the scaling report](../../results/pilot/codex_teacher_scaling_20260927.md).

## Proposed RunPod trial

Use one on-demand GPU Pod for Gemma inference followed by student training, while Codex
continues using its existing hosted service. A RunPod GPU does not execute the hosted Astra model.

| GPU | VRAM | Listed GPU price/hour | Role in the trial |
|---|---:|---:|---|
| RTX 4090 | 24 GB | $0.74 | Suitable first option for the current 12B Q8 teacher |
| RTX 6000 Ada | 48 GB | $0.84 | Preferred if available: additional batch/training memory for a small price difference |
| A100 PCIe | 80 GB | $1.59 | Alternative if training needs more memory; speed advantage is unmeasured |

These are published prices checked on September 27, not an account-specific availability quote.
Source: [RunPod pricing](https://www.runpod.io/pricing), page updated September 13, 2026.

Proposed initial configuration: one RTX 6000 Ada, Linux/PyTorch CUDA environment, 50 GB persistent
workspace plus a 20 GB container disk, and a one-hour benchmark window with a scheduled stop.
One hour at the published rate is **$0.84 GPU cost**, plus storage; this is a budget calculation,
not a prediction of the full pilot's runtime. Volume storage is listed at $0.10/GB/month while
running and $0.20/GB/month while stopped. Download checkpoints before terminating a Pod, because
its volume disk is deleted with it. [Storage](https://docs.runpod.io/pods/storage/types),
[Pod lifecycle](https://docs.runpod.io/pods/manage-pods).

## Work required before launch

1. Make the teacher GGUF and llama.cpp library paths configurable. They currently point to
   this Windows user's Ollama cache and `C:/temp/llamacpp` in `common.py`.
2. Build the same llama.cpp b11100 backend for Linux and preserve the model SHA, template,
   option orders, context length, and calibrated temperatures. The Windows spill-memory probe
   must be recorded as unsupported on Linux rather than treated as a measured zero.
3. Provide a Linux stage runner; `seg_next.sh` currently selects Windows Python and Git Bash.
   Use the Pod's resource limits and GPU telemetry while retaining the laptop's existing guards.
4. Transfer the pool, question assignments, saved teacher shards, and evaluation inputs with
   a checksum manifest. The measured pool + assignment files total about 14.6 MB; saved teacher
   shards were about 2.6 MB at inspection. The exact Q8 model is 12,669,645,728 bytes, so model
   transfer or a matching remote download dominates setup size.
5. Test a small matched teacher sample, check output parity within documented floating-point
   tolerances, and measure complete wall throughput including loading. Preserve the existing
   temperature calibration and compare argmax and target distributions before resuming production.
6. Resume from a validated snapshot. If both GPUs label concurrently, assign disjoint state IDs
   and separate output directories; merge by ID with duplicate/conflict checks. Do not let two
   writers share `part-NNNN.jsonl` filenames. Stop renting compute when the bounded run ends.

Deployment status: the portable teacher package and bounded launcher are implemented and offline
tested. No Pod benchmark or cloud speedup claim exists yet. Provider account access, authenticated
capacity/price checks, and a spending cap are still launch inputs.

## Speed-focused trial prepared after the deployment request

Start with one H100 SXM (80 GB; listed at $3.49/hour) and benchmark one, two, and four isolated
Gemma processes. This is a speed-focused candidate, not an assertion that H100 is the fastest
GPU for this exact workload. H200 ($4.59/hour) and B200 ($6.79/hour) are higher-priced alternatives
from the same published pricing page; use actual capacity and measured throughput to choose.
The earlier RTX 6000 Ada option remains the lower-cost trial.

Implemented commands:

```powershell
.\.venv-pilot\Scripts\python.exe experiments\small_model\cloud_teacher.py prepare --output runs\pilot\runpod-prep-20260927 --canary-states 100
.\.venv-pilot\Scripts\python.exe experiments\small_model\runpod_trial.py --key-file runs\pilot\runpod.key catalog
```

`runpod.key` is an optional local raw-key file under the ignored `runs/` tree. An existing
`RUNPOD_API_KEY` environment variable also works. Do not send its contents in messages or include
it in an upload. The bundle contains project code, public evaluation data, the pool and question
assignments, and reference teacher outputs; it excludes credentials, virtual environments, and model binaries.

The model registry answered a HEAD request with HTTP 200 and the exact expected size, and
GitHub supplies prebuilt b11100 Linux CUDA 12.8 libraries with SHA-256 digests. The cloud bootstrap
downloads and verifies these directly. The verified container image is
`runpod/pytorch@sha256:0ad20c96682da30fd04c2f130c13edb68aa5ebe3d370f2d5096eb1fcbbbfddad`
(`1.3.3-cu1281-torch280-ubuntu2404`). Image and source checks do not replace a live GPU smoke test.

The prepared snapshot is `runs/pilot/runpod-prep-20260927-v2/tez-pilot-cloud.tar.gz`: 25,257,759
bytes, with 449 verified file hashes, 2,700 completed teacher states, 3,576 pending states, and
100 parity examples. Saved and pending IDs are disjoint and together cover all 6,276 assignments.
These are snapshot counts; the local teacher continues progressing. Preparation evidence is in
`results/pilot/runpod_preparation_20260927.json`.

After account access and budget are provided, resolve the exact GPU ID with `catalog`, generate a
dedicated SSH key, then use `runpod_trial.py launch` with explicit `--gpu-id`, `--max-hourly-usd`,
`--minutes`, `--public-key`, and `--receipt`. The launcher checks the current GPU quote and returned
Pod price and starts a local watcher to stop the created Pod at its deadline. The laptop and
network must remain available for that provider stop; the cloud runner's compute deadline alone
does not stop provider billing. Storage remains billable after a GPU stop.

Upload the bundle to the new Pod, extract it into `/workspace/tez-pilot`, and run:

```bash
cd /workspace/tez-pilot
bash experiments/small_model/cloud_bootstrap.sh --name h100-trial --workers 1 2 4 --minutes 45 --production
```

The script verifies input hashes before inference, selects worker counts with enough memory,
checks 100 saved states against the local target distributions, and only continues if the fixed
parity gates pass: maximum probability delta <= 0.02, mean total variation <= 0.005, and argmax
agreement >= 99%. It records timing including process startup and model loading, selects the
fastest tested worker count, and estimates remaining label time from the measured canary rate.
Those estimates exclude provisioning, downloads, setup, and later student training.

Cloud workers use disjoint state IDs and isolated shard/result directories. Cloud outputs remain
under `runs/cloud-output/<name>/`; they are not automatically merged with the still-running
laptop's shards. The initial job covers the remaining 12B teacher labeling, not all subsequent
student training, evaluation, and latency measurements.

## Combining Gemma and Codex targets

The baseline uses Gemma's measured logits from two option orders, temperature-adjusted and
averaged into a distribution. GPT-6 Astra does not expose `logprobs`, including through the API.
Source: [OpenAI's September 3 release notes](https://developers.openai.com/api/docs/changelog).

Codex can supply a selected label, a reported confidence estimate, or repeated-vote frequencies.
Those are different measurements. Repeated answers can share the same errors; a 90% vote share
is not evidence of 90% correctness. Neither model's raw answer-token distribution is automatically
a calibrated probability of correctness.

A useful additional training arm would combine Gemma soft-target loss with a separately weighted
Codex pseudo-label loss, retaining verified gold labels as their own source. Preserve a Gemma-only
control and compare on held-out gold tasks; do not overwrite gold or declare improvements based only
on agreement with either teacher. Requiring a Gemma distribution for every row still leaves Gemma
as the bottleneck; Codex-only supplemental rows would explicitly change that requirement.

The current benchmark labels are retained separately from training. Eight Codex workers are a
reasonable measured starting point for further tests on this machine. More workers share the
same account allowance; the test has not established long-running throughput or remaining quota.
[Codex usage guidance](https://learn.chatgpt.com/docs/pricing).
