"""Bounded non-benchmark CUDA graph compatibility check; no speed or quality claim."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-cuda-graph-probe-20260927"
REGISTRATION = Path(__file__).with_name("qwen_cuda_graph_registration.json")


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    path = DIRECTORY / "probe.json"
    if path.exists():
        raise ValueError("Preserve previous graph probe")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "source_files": {}, "rows": []}
    started = time.monotonic()
    write_json(path, receipt)
    try:
        for name, expected in spec["sources"].items():
            source = ROOT / name
            if digest(source) != expected:
                raise ValueError("Registered source changed")
            target = DIRECTORY / "executed_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            receipt["source_files"][name] = expected
        for name, expected in spec["prerequisites"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen prerequisite changed")
        import torch
        import torch.nn.functional as F
        from huggingface_hub import snapshot_download
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        import common
        from functools import partial
        common.thermal_guard = partial(common.thermal_guard, hot=80, cool=76, every=0.0)
        from common import THERMAL, thermal_guard
        from eval_arms import PredictC
        from restricted_qwen import RestrictedQwen
        from train_c import letter_logp, prompt_for
        if version("torch") != spec["torch_version"]:
            raise ValueError("Registered Torch environment changed")
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        weights = list(base.glob("*.safetensors"))
        if len(weights) != 1 or digest(weights[0]) != spec["base_sha256"]:
            raise ValueError("Frozen base weights changed")
        thermal_guard(force=True)
        reference = PredictC("c-qwen08-30k")
        candidate = RestrictedQwen(reference)
        prompts = [prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]) for c in spec["cases"]]
        counts = [len(c["options"]) for c in spec["cases"]]
        encoded = reference.tok(prompts, add_special_tokens=False, padding=True, truncation=False, return_tensors="pt", padding_side="left")
        if encoded["input_ids"].shape[1] > 4096:
            raise ValueError("No truncation is allowed")
        static_ids = encoded["input_ids"].to("cuda")
        static_padding = encoded["attention_mask"].to("cuda")
        batch, length = static_ids.shape
        positions = torch.arange(length, device="cuda").view(1, 1, -1).expand(4, batch, -1)
        allowed = torch.arange(max(counts), device="cuda")[None, :] < torch.tensor(counts, device="cuda")[:, None]

        def build_masks(ids, padding):
            # Native helpers may synchronize while inspecting the mask; run them outside capture.
            kwargs = {"config": candidate.lm.config, "inputs_embeds": candidate.lm.model.embed_tokens(ids),
                      "attention_mask": padding, "past_key_values": None, "position_ids": positions[0]}
            return {"full_attention": create_causal_mask(**kwargs), "linear_attention": create_recurrent_attention_mask(**kwargs)}

        def forward(masks):
            hidden = candidate.lm.model(input_ids=static_ids, attention_mask=masks, position_ids=positions,
                                        use_cache=False).last_hidden_state[:, -1:, :]
            logits = F.linear(hidden, candidate.weight, candidate.bias)[:, -1, :max(counts)].float()
            return F.log_softmax(torch.where(allowed, logits, torch.full_like(logits, -1e4)), -1)

        receipt.update(status="running", gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(),
                       packages={name: version(name) for name in ("torch", "transformers", "peft")},
                       attention_implementation=candidate.lm.config._attn_implementation,
                       weights=reference.weights, base_sha256=spec["base_sha256"], tensor_shape=[batch, length])
        write_json(path, receipt)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = build_masks(static_ids, static_padding)
            if any(value is None for value in masks.values()):
                raise ValueError("This probe requires explicit fixed-shape native masks")
            # Warm on a side stream and synchronize before capturing, per the pinned Torch guidance.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    forward(masks)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = forward(masks)
            torch.cuda.synchronize()
            for order in ([0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3]):
                thermal_guard(force=True)
                ordered_prompts = [prompts[i] for i in order]
                ordered_counts = [counts[i] for i in order]
                eager, _ = letter_logp(reference.model, reference.tok, reference.letter_ids,
                                        ordered_prompts, ordered_counts, "cuda", max_len=4096)
                fresh = reference.tok(ordered_prompts, add_special_tokens=False, padding=True, truncation=False,
                                      return_tensors="pt", padding_side="left").to("cuda")
                replacement = build_masks(fresh["input_ids"], fresh["attention_mask"])
                if fresh["input_ids"].shape != static_ids.shape:
                    raise ValueError("Graph input shape changed")
                static_ids.copy_(fresh["input_ids"])
                static_padding.copy_(fresh["attention_mask"])
                allowed.copy_(torch.arange(max(counts), device="cuda")[None, :] < torch.tensor(ordered_counts, device="cuda")[:, None])
                for key in masks:
                    if replacement[key] is None or replacement[key].shape != masks[key].shape:
                        raise ValueError("Graph native mask structure changed")
                    masks[key].copy_(replacement[key])
                uncaptured = forward(masks)
                graph.replay()
                torch.cuda.synchronize()
                full_p = eager.exp().double().cpu().tolist()
                uncaptured_p = uncaptured.exp().double().cpu().tolist()
                graph_p = static_output.exp().double().cpu().tolist()
                receipt["rows"].append({"order": order, "option_counts": ordered_counts,
                                        "full_probabilities": full_p, "uncaptured_probabilities": uncaptured_p,
                                        "graph_probabilities": graph_p})
                write_json(path, receipt)
        deltas, tvs, matched, replay_deltas = [], [], [], []
        for row in receipt["rows"]:
            for i, count in enumerate(row["option_counts"]):
                a, b, c = [row[key][i][:count] for key in ("full_probabilities", "graph_probabilities", "uncaptured_probabilities")]
                deltas.extend(abs(p - q) for p, q in zip(a, b, strict=True))
                tvs.append(sum(abs(p - q) for p, q in zip(a, b, strict=True)) / 2)
                matched.append(max(range(count), key=a.__getitem__) == max(range(count), key=b.__getitem__))
                replay_deltas.extend(abs(p - q) for p, q in zip(b, c, strict=True))
        metrics = {"max_probability_delta": max(deltas), "mean_total_variation": sum(tvs) / len(tvs),
                   "argmax_agreement": sum(matched) / len(matched), "max_graph_vs_uncaptured_delta": max(replay_deltas)}
        passed = (metrics["max_probability_delta"] <= .04 and metrics["mean_total_variation"] <= .005
                  and metrics["argmax_agreement"] >= .99 and metrics["max_graph_vs_uncaptured_delta"] <= 1e-5)
        receipt.update(status="complete", passed=passed, metrics=metrics, thermal_governor=THERMAL,
                       scope="Four non-benchmark cases in three batch orders only; no full-validation parity, benchmark accuracy or latency claim.")
        if not passed:
            raise ValueError("Graph compatibility probability limits failed")
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, receipt)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve previous execution")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    env = dict(os.environ)
    env.update(TEZ_PILOT_RUNS=str(ROOT / spec["model_directory"]), TEZ_PILOT_RESULTS=str(DIRECTORY),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": 300}
    write_json(path, receipt)
    with (DIRECTORY / "probe.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            proc.wait()
            code = 124
    receipt.update(status="complete" if code == 0 else "failed", returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if "--worker" in sys.argv:
        worker()
    else:
        raise SystemExit(driver())
