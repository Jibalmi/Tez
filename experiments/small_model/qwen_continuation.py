"""Bounded, checkpointed continuation; execution requires a completed trainability probe."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import fmean
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-continuation-20260927/training"
RUNS = ROOT / "runs/pilot-qwen-continuation-20260927/training"
REGISTRATION = Path(__file__).with_name("qwen_continuation_registration.json")


def data_for(tokenizer, spec):
    import dataset
    from train_c import prompt_for
    data = ROOT / spec["teacher_directory"]
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(data / name) != expected:
            raise ValueError("Original teacher data changed")
    dataset.RUNS = data
    train, _, validation = dataset.load_examples()
    if len(train) != 5998 or len(validation) != 278 or {e.id for e in train} & {e.id for e in validation}:
        raise ValueError("Training and validation membership changed")
    indexed = {(e.id, d.qid): (e, d) for e in train for d in e.decisions}
    schedule = json.loads((ROOT / spec["schedule"]).read_text(encoding="utf-8"))
    if (len(indexed) != 29990 or schedule["steps"] != 150 or len(schedule["batches"]) != 150
            or schedule["effective_batch"] != 16 or schedule["seed"] != 113
            or any(len(b) != 16 for b in schedule["batches"])
            or len({(r["state_id"], r["qid"]) for b in schedule["batches"] for r in b}) != 2400):
        raise ValueError("Training schedule shape changed")

    def render(e, d, order):
        if sorted(order) != list(range(len(d.options))):
            raise ValueError("Invalid option permutation")
        prompt = prompt_for(d.type, d.instructions, [d.options[i] for i in order], e.state)
        count = len(tokenizer.encode(prompt, add_special_tokens=False, truncation=False))
        if count > spec["context_bound"]:
            raise ValueError("Complete input exceeds the context bound")
        return {"state_id": e.id, "qid": d.qid, "qtype": d.type, "k": len(order), "prompt": prompt,
                "target": d.target[order].tolist(), "gold": order.index(d.gold) if d.gold is not None else None,
                "encoded_tokens": count, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}

    batches = []
    for batch in schedule["batches"]:
        rendered = []
        for row in batch:
            e, d = indexed[row["state_id"], row["qid"]]
            item = render(e, d, row["permutation"])
            if any(item[k] != row[k] for k in ("encoded_tokens", "prompt_sha256")):
                raise ValueError("Prepared training input differs")
            rendered.append(item)
        batches.append(rendered)
    heldout = [render(e, d, list(range(len(d.options)))) for e in validation for d in e.decisions]
    if len(heldout) != 1390 or len({(v["state_id"], v["qid"]) for v in heldout}) != 1390:
        raise ValueError("Incomplete or duplicate validation cases")
    return batches, heldout


def validate(model, tokenizer, letters, cases, directory, spec, guard, started):
    import torch
    import torch.nn.functional as F
    from parity_evidence import read_rows
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    order = sorted(range(len(cases)), key=lambda i: len(cases[i]["prompt"]))
    rows = []
    path = directory / "validation_rows.jsonl"
    with path.open("x", encoding="utf-8") as output, torch.inference_mode():
        for start in range(0, len(order), 16):
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Preserve partial validation; no incomplete score is eligible")
            batch = [cases[i] for i in order[start:start + 16]]
            guard(force=True)
            enc = tokenizer([x["prompt"] for x in batch], add_special_tokens=False, padding=True,
                            padding_side="left", truncation=False, return_tensors="pt").to("cuda")
            if enc["input_ids"].shape[1] > spec["context_bound"]:
                raise ValueError("Validation input exceeds the safety bound")
            k = max(x["k"] for x in batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc, logits_to_keep=1).logits[:, -1, letters[:k]].float()
                mask = torch.arange(k, device="cuda")[None, :] < torch.tensor([x["k"] for x in batch], device="cuda")
                logp = F.log_softmax(torch.where(mask, logits, torch.full_like(logits, -1e4)), dim=-1).cpu().tolist()
            for case, values in zip(batch, logp, strict=True):
                values = values[:case["k"]]
                p = [math.exp(v - max(values)) for v in values]
                p = [v / sum(p) for v in p]
                row = {key: case[key] for key in ("state_id", "qid", "qtype", "k", "target", "gold", "encoded_tokens", "prompt_sha256")}
                row.update(probabilities=p, log_probabilities=values,
                           kl=sum(t * (math.log(max(t, 1e-12)) - math.log(max(q, 1e-12))) for t, q in zip(case["target"], p, strict=True)),
                           teacher_argmax=max(range(len(p)), key=case["target"].__getitem__),
                           student_argmax=max(range(len(p)), key=p.__getitem__))
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows.append(row)
            output.flush()
    expected = [(x["state_id"], x["qid"], x["k"]) for x in cases]
    probabilities = read_rows(path, expected)
    metrics = {"n": len(rows), "mean_kl": fmean(r["kl"] for r in rows),
               "teacher_agreement": fmean(r["student_argmax"] == r["teacher_argmax"] for r in rows),
               "rows_sha256": digest(path)}
    return probabilities, metrics


def worker(args, directory, spec):
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "operation": args.operation,
               "target_step": args.target_step, "source_files": {}, "checkpoints": [], "updates": []}
    path = directory / "work.json"
    started = time.monotonic()
    write_json(path, receipt)
    try:
        for name, expected in spec["sources"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Registered source changed")
            destination = directory / "executed_source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            receipt["source_files"][name] = expected
        for name, expected in spec["prerequisites"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen prerequisite changed")
        probe = json.loads((ROOT / spec["trainability_probe"]).read_text(encoding="utf-8"))
        if probe["status"] != "complete" or not probe["passed"] or probe["weights_saved"]:
            raise ValueError("Require the successful probe with discarded weights")
        if any(version(name) != expected for name, expected in spec["packages"].items()):
            raise ValueError("Registered package versions changed")
        import torch
        import common
        from functools import partial
        common.thermal_guard = partial(common.thermal_guard, hot=80, cool=76, every=0.0)
        from common import THERMAL, thermal_guard
        from train_c import load_base
        from peft import PeftModel
        from huggingface_hub import snapshot_download
        from qwen_checkpoint import load_checkpoint, save_checkpoint
        from qwen_continuation_loss import decision_loss
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]:
            raise ValueError("Base weights changed")
        thermal_guard(force=True)
        torch.manual_seed(113)
        tokenizer, model, letters = load_base()
        model = PeftModel.from_pretrained(model, ROOT / spec["adapter_directory"], is_trainable=True).to("cuda")
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named or any("lora_" not in name for name, _ in named):
            raise ValueError("Only the frozen adapter layout may train")
        optimizer = torch.optim.AdamW([p for _, p in named], lr=5e-5, weight_decay=0.0)
        batches, heldout = data_for(tokenizer, spec)
        state = {"step": 0, "decisions": 0, "tokens": 0, "validations": {}}
        if args.checkpoint:
            checkpoint = (ROOT / args.checkpoint).resolve()
            if not checkpoint.is_relative_to(RUNS.resolve()):
                raise ValueError("Resume only this continuation's own checkpoint")
            state = load_checkpoint(checkpoint, args.checkpoint_sha256, named, optimizer, digest(REGISTRATION), cuda=True)
            receipt["input_checkpoint"] = {"path": checkpoint.relative_to(ROOT).as_posix(), "sha256": args.checkpoint_sha256}
        if state["decisions"] != state["step"] * 16 or not 0 <= state["step"] <= 150:
            raise ValueError("Checkpoint progress is inconsistent")
        receipt.update(status="running", initial_step=state["step"], gpu=torch.cuda.get_device_name(),
                       threads=torch.get_num_threads(), packages={n: version(n) for n in ("torch", "transformers", "peft")},
                       attention_implementation=model.config._attn_implementation,
                       trainable_parameters=sum(p.numel() for _, p in named))
        torch.cuda.reset_peak_memory_stats()
        write_json(path, receipt)

        def checkpoint_now():
            destination = RUNS / f"attempt-{args.attempt:02d}-step-{state['step']:03d}.pt"
            result = save_checkpoint(destination, named, optimizer, state, digest(REGISTRATION), cuda=True)
            result["path"] = destination.relative_to(ROOT).as_posix()
            receipt["checkpoints"].append(result)
            write_json(path, receipt)
            return result

        if args.operation in ("baseline", "validate"):
            if state["step"] != args.target_step or args.target_step not in (0, 50, 100, 150):
                raise ValueError("Validate only the declared checkpoint steps")
            probabilities, metrics = validate(model, tokenizer, letters, heldout, directory, spec, thermal_guard, started)
            metrics["directory"] = directory.relative_to(ROOT).as_posix()
            receipt["validation"] = metrics
            if args.operation == "baseline":
                from parity_evidence import compare_rows, read_rows
                expected = [(x["state_id"], x["qid"], x["k"]) for x in heldout]
                parity = compare_rows(read_rows(ROOT / spec["reference_rows"], expected), probabilities, spec)
                receipt["baseline_parity"] = parity
                if not parity["passed"]:
                    raise ValueError("Step zero does not preserve the frozen inference reference")
            state["validations"][str(state["step"])] = metrics
            checkpoint_now()
            receipt.update(status="complete", completed_step=state["step"])
        else:
            if args.target_step not in (50, 100, 150) or not args.target_step - 50 <= state["step"] < args.target_step:
                raise ValueError("Train only the next registered interval")
            if str(args.target_step - 50) not in state["validations"]:
                raise ValueError("Complete the preceding checkpoint validation before further training")
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
            model.train()
            segment_steps = 0
            while state["step"] < args.target_step:
                if (time.monotonic() - started > spec["soft_deadline_seconds"]
                        or segment_steps >= spec["max_updates_per_segment"]):
                    break
                batch = batches[state["step"]]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                update_started = time.monotonic()
                for item in batch:
                    thermal_guard(force=True)
                    enc = tokenizer(item["prompt"], add_special_tokens=False, truncation=False, return_tensors="pt").to("cuda")
                    if enc["input_ids"].shape[1] != item["encoded_tokens"]:
                        raise ValueError("Training tokenization changed")
                    logits = model(**enc, logits_to_keep=1).logits[0, -1, letters[:item["k"]]].float()
                    target = torch.tensor(item["target"], dtype=torch.float32, device="cuda")
                    loss = decision_loss(logits, target, item["gold"]) / 16
                    if not torch.isfinite(loss).item():
                        raise ValueError("Non-finite training loss")
                    loss.backward()
                    loss_sum += loss.detach().item()
                if any(p.grad is None for _, p in named):
                    raise ValueError("Missing LoRA gradient")
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0, error_if_nonfinite=True)
                thermal_guard(force=True)
                optimizer.step()
                torch.cuda.synchronize()
                state = {**state, "step": state["step"] + 1, "decisions": state["decisions"] + 16,
                         "tokens": state["tokens"] + sum(x["encoded_tokens"] for x in batch)}
                segment_steps += 1
                receipt["updates"].append({"step": state["step"], "mean_loss": loss_sum,
                                           "gradient_norm_before_clip": float(norm),
                                           "elapsed_seconds": time.monotonic() - update_started})
                if state["step"] % 10 == 0 or state["step"] == args.target_step:
                    checkpoint_now()
                write_json(path, receipt)
            if segment_steps == 0:
                raise TimeoutError("No optimizer update completed before the segment boundary")
            if not receipt["checkpoints"] or not receipt["checkpoints"][-1]["path"].endswith(f"step-{state['step']:03d}.pt"):
                checkpoint_now()
            receipt.update(status="complete" if state["step"] == args.target_step else "checkpointed",
                           completed_step=state["step"], interval_complete=state["step"] == args.target_step)
        receipt.update(peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                       scope="Original training-only continuation with complete inputs. Validation is teacher agreement, not broad gold accuracy. No public test fitting, quality promotion or speed claim.")
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, receipt)


def driver(args, directory, spec):
    import psutil
    if directory.exists() or list(DIRECTORY.glob(f"attempt-{args.attempt:02d}-*")):
        raise ValueError("Preserve prior attempts; use a new attempt number")
    directory.mkdir(parents=True)
    shutil.copy2(REGISTRATION, directory / "registration.json")
    request = vars(args).copy()
    request.pop("worker")
    write_json(directory / "request.json", request)
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               USE_TF="0", TOKENIZERS_PARALLELISM="false", TEZ_PILOT_RESULTS=str(directory))
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(directory / "execution.json", receipt)
    with (directory / "work.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=spec["worker_timeout_seconds"])
        except subprocess.TimeoutExpired:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            proc.wait()
            code = 124
    receipt.update(status="complete" if code == 0 else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("baseline", "train", "validate"), required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if args.attempt < 1 or (args.operation == "baseline") != (args.checkpoint is None):
        raise ValueError("Only baseline may start without an explicit checkpoint")
    if bool(args.checkpoint) != bool(args.checkpoint_sha256) or (args.operation == "baseline" and args.target_step != 0):
        raise ValueError("Pin the resume checkpoint and use step zero for baseline")
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if args.attempt > spec["max_attempts"]:
        raise ValueError("The registered attempt budget is exhausted")
    directory = DIRECTORY / f"attempt-{args.attempt:02d}-{args.operation}-{args.target_step:03d}"
    if args.worker:
        worker(args, directory, spec)
        return 0
    return driver(args, directory, spec)


if __name__ == "__main__":
    raise SystemExit(main())
