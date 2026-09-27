"""Matched frozen-base continuations: replay control versus filtered exact rule supervision."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from structure_probe import WEIGHTS, digest, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/pilot-reasoning-round2-20260927"
RUNS = ROOT / "runs/pilot-reasoning-round2-20260927"


def selection_key(old, fresh, baseline, weight):
    if old["kl"] > baseline["kl"] + .02 or old["agreement"] < baseline["agreement"] - .01:
        return None
    return (1 - weight) * old["kl"] + weight * fresh["kl"]


def oracle_examples(records):
    import numpy as np
    from dataset import Decision, Example
    out = []
    for record in records:
        decisions = []
        for q in record["questions"]:
            options = [tuple(item) for item in q["options"]]
            gold = [key for key, _ in options].index(q["gold_key"])
            target = np.eye(len(options), dtype=np.float32)[gold]
            decisions.append(Decision(q["operation"], q["qtype"], q["instructions"], options, target, target.copy(), gold))
        out.append(Example(record["id"], "authored_exact_rules", "en", record["state"], decisions))
    return out


def main(arm):
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    specification = ROOT / "experiments/small_model/reasoning_round2.json"
    config = json.loads(specification.read_text(encoding="utf-8"))
    if arm not in config["arms"]:
        raise ValueError("Unregistered training arm")
    os.environ.update(TEZ_PILOT_RUNS=str(ROOT / config["teacher_data"]), TEZ_PILOT_RESULTS=str(RESULTS),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoTokenizer
    from common import THERMAL, thermal_guard
    from dataset import load_examples, summary
    from reasoning_data import validate
    from student import Encoder, Student, lr_lambda
    from train_a import batches, evaluate_val, make_rows, step_loss
    from train_c import kernel_packages

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "models" / arm
    receipt_path = RESULTS / f"train_{arm}.json"
    if out.exists() or receipt_path.exists():
        raise ValueError("Preserve existing attempts; a partial checkpoint requires an explicit resume implementation")
    out.mkdir(parents=True)
    source_dir = RESULTS / "executed_source" / arm
    source_dir.mkdir(parents=True)
    source_hashes = {}
    for name in ("train_reasoning_round2.py", "reasoning_round2.json", "train_a.py", "student.py", "dataset.py",
                 "common.py", "reasoning_data.py", "structure_probe.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        source_hashes[name] = digest(source)
    receipt = {"status": "preparing", "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "specification_sha256": digest(specification), "sources": source_hashes, "packages": kernel_packages(),
               "step": 0, "validation": [], "selected_step": 0, "training_tokens": 0, "training_decisions": 0}
    write_json(receipt_path, receipt)
    started = time.monotonic()
    model = optimizer = scheduler = None
    best_path = out / "best.pt"
    checkpoint_path = out / "partial.pt"
    try:
        teacher = ROOT / config["teacher_data"]
        if digest(teacher / "manifest.json") != config["teacher_manifest_sha256"]:
            raise ValueError("Original teacher manifest changed")
        manifest = json.loads((teacher / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest(teacher / name) != expected:
                raise ValueError(f"Original teacher file changed: {name}")
        data_root = ROOT / config["new_data"]
        filtered = json.loads((data_root / "filter_receipt.json").read_text(encoding="utf-8"))
        data_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
        if (filtered["status"] != "complete" or filtered["filtered_train_sha256"] != config["filtered_train_sha256"]
                or digest(data_root / "train.filtered.json") != config["filtered_train_sha256"]):
            raise ValueError("New data lacks the required completed leakage checks")
        if digest(data_root / "validation.json") != data_manifest["splits"]["validation"]["sha256"]:
            raise ValueError("New validation split changed")
        raw_train = json.loads((data_root / "train.filtered.json").read_text(encoding="utf-8"))
        raw_val = json.loads((data_root / "validation.json").read_text(encoding="utf-8"))
        # Confirmation labels are deliberately not opened by the training process.
        validate({"train": raw_train, "validation": raw_val})
        new_train, new_val = oracle_examples(raw_train), oracle_examples(raw_val)
        old_train, _, old_val = load_examples()
        if (len(old_train), sum(len(e.decisions) for e in old_train), len(new_train), sum(len(e.decisions) for e in new_train)) != (5998, 29990, 1800, 5400):
            raise ValueError("Training counts differ from registration")
        rng = random.Random(config["seed"])
        if arm == "a-base-replay150":
            extra = []
            for i, original in enumerate(rng.sample(old_train, len(new_train))):
                example = copy.copy(original)
                example.id = f"control-replay/{i}/{original.id}"
                example.decisions = rng.sample(original.decisions, 3)
                extra.append(example)
        else:
            extra = new_train
        train = old_train + extra
        weight = 5400 / (29990 + 5400)
        receipt.update(training=summary(train), original_validation=summary(old_val), new_validation=summary(new_val),
                       new_validation_weight=weight, data_manifest_sha256=digest(data_root / "manifest.json"),
                       filter_receipt_sha256=digest(data_root / "filter_receipt.json"),
                       original_teacher_files_verified=len(manifest["files"]))
        base = ROOT / config["base_directory"]
        base_weights = {"encoder": digest(base / "encoder/model.safetensors"), "head": digest(base / "head.pt")}
        if base_weights != WEIGHTS[config["base_arm"]]:
            raise ValueError("Base checkpoint changed")
        receipt["base_weights"] = base_weights
        torch.manual_seed(config["seed"])
        device = torch.device("cuda")
        thermal_guard(force=True)
        tokenizer = AutoTokenizer.from_pretrained(base / "encoder")
        encoder = Encoder(tokenizer, config["max_sequence_length"])
        model = Student(str(base / "encoder")).to(device)
        model.head.load_state_dict(torch.load(base / "head.pt", map_location=device, weights_only=True))
        optimizer = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": config["encoder_lr"]},
                                       {"params": model.head.parameters(), "lr": config["head_lr"]}], weight_decay=.01, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(config["warmup_steps"], config["steps"]))
        pack_rng = random.Random(config["seed"] * 1000)
        training_batches = batches(make_rows(train, encoder, pack_rng), config["padded_token_budget"], pack_rng)
        if len(training_batches) < config["steps"]:
            raise ValueError("Not enough batches for the registered fixed-step schedule")
        receipt["available_batches"] = len(training_batches)

        def guard():
            if time.monotonic() - started > 12 * 60:
                raise TimeoutError("Training worker reached its registered 12-minute bound")
            thermal_guard(force=True)

        def validation():
            old_agree, old_kl, _ = evaluate_val(model, old_val, encoder, device, config["padded_token_budget"], guard=guard)
            new_agree, new_kl, _ = evaluate_val(model, new_val, encoder, device, config["padded_token_budget"], guard=guard)
            return {"agreement": old_agree, "kl": old_kl}, {"agreement": new_agree, "kl": new_kl}

        old, fresh = validation()
        baseline = old.copy()
        best_key = selection_key(old, fresh, baseline, weight)
        receipt["validation"].append({"step": 0, "original": old, "new": fresh, "eligible": True, "selection_key": best_key})
        torch.save(model.state_dict(), best_path)
        receipt["status"] = "training"
        write_json(receipt_path, receipt)
        for step in range(1, config["steps"] + 1):
            guard()
            batch = training_batches[step - 1]
            optimizer.zero_grad(set_to_none=True)
            loss, *_ = step_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss before optimizer update")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            receipt["step"] = step
            receipt["training_tokens"] += sum(len(row.ids) for row in batch)
            receipt["training_decisions"] += sum(len(row.meta["types"]) for row in batch)
            if step in config["validation_steps"]:
                old, fresh = validation()
                key = selection_key(old, fresh, baseline, weight)
                receipt["validation"].append({"step": step, "original": old, "new": fresh, "eligible": key is not None, "selection_key": key})
                if key is not None and key < best_key:
                    best_key = key
                    receipt["selected_step"] = step
                    temporary = out / "best.pt.tmp"
                    torch.save(model.state_dict(), temporary)
                    temporary.replace(best_path)
                print(json.dumps({"arm": arm, **receipt["validation"][-1], "selected_step": receipt["selected_step"]}), flush=True)
            if step % 10 == 0:
                receipt["last_loss"] = float(loss.detach())
                write_json(receipt_path, receipt)
                print(json.dumps({"arm": arm, "step": step, "loss": receipt["last_loss"]}), flush=True)
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.enc.save_pretrained(out / "encoder")
        tokenizer.save_pretrained(out / "encoder")
        torch.save(model.head.state_dict(), out / "head.pt")
        info = {"arm": arm, "backbone": config["base_directory"], "steps": receipt["step"],
                "selected_step": receipt["selected_step"], "temperatures": {}, "seed": config["seed"],
                "selection": config["selection"], "supervision": config["treatment"] if arm.endswith("rules150") else config["control"],
                "base_weights": base_weights, "training_specification_sha256": digest(specification)}
        write_json(out / "student.json", info)
        receipt.update(status="complete", weights={"encoder": digest(out / "encoder/model.safetensors"), "head": digest(out / "head.pt")},
                       peak_allocated_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
    except Exception as exc:
        receipt.update(status="stopped", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if model is not None and optimizer is not None and scheduler is not None:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "receipt": receipt, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint_path)
            receipt["partial_checkpoint"] = str(checkpoint_path.relative_to(ROOT))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                       thermal_governor=THERMAL)
        write_json(receipt_path, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    main(args.arm)
