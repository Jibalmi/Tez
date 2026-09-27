"""Check finite LoRA backward/update on the longest planned training input; discard the probe weights."""
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
DIRECTORY = ROOT / "results/pilot-qwen-continuation-20260927/trainability"
REGISTRATION = Path(__file__).with_name("qwen_trainability_registration.json")


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    output = DIRECTORY / "probe.json"
    if output.exists():
        raise ValueError("Preserve prior probe")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "source_files": {}}
    started = time.monotonic()
    write_json(output, receipt)
    try:
        for name, expected in spec["sources"].items():
            source = ROOT / name
            if digest(source) != expected:
                raise ValueError("Source changed")
            destination = DIRECTORY / "executed_source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            receipt["source_files"][name] = expected
        for name, expected in spec["prerequisites"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen prerequisite changed")
        import torch
        import torch.nn.functional as F
        import common
        from functools import partial
        common.thermal_guard = partial(common.thermal_guard, hot=80, cool=76, every=0.0)
        from common import THERMAL, thermal_guard
        import dataset
        from train_c import load_base, prompt_for
        from peft import PeftModel, get_peft_model_state_dict
        from huggingface_hub import snapshot_download
        teacher = ROOT / spec["teacher_directory"]
        manifest = json.loads((teacher / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest(teacher / name) != expected:
                raise ValueError("Teacher data changed")
        dataset.RUNS = teacher
        train, _, validation = dataset.load_examples()
        schedule = json.loads((ROOT / spec["schedule"]).read_text(encoding="utf-8"))
        row = max((r for b in schedule["batches"] for r in b), key=lambda r: r["encoded_tokens"])
        indexed = {(e.id, d.qid): (e, d) for e in train for d in e.decisions}
        if row["state_id"] in {e.id for e in validation} or (row["state_id"], row["qid"]) not in indexed:
            raise ValueError("Probe must use an original training example")
        example, decision = indexed[row["state_id"], row["qid"]]
        order = row["permutation"]
        if sorted(order) != list(range(len(decision.options))):
            raise ValueError("Invalid option permutation")
        prompt = prompt_for(decision.type, decision.instructions, [decision.options[i] for i in order], example.state)
        import hashlib
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Prepared prompt changed")
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]:
            raise ValueError("Base weights changed")
        thermal_guard(force=True)
        torch.manual_seed(113)
        tok, model, letters = load_base()
        model = PeftModel.from_pretrained(model, ROOT / spec["adapter_directory"], is_trainable=True).to("cuda")
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise ValueError("Only the original LoRA adapter may train")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
        encoded = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).to("cuda")
        if encoded["input_ids"].shape[1] != row["encoded_tokens"] or row["encoded_tokens"] > 2048:
            raise ValueError("Input length changed or exceeds the bound")
        target = torch.tensor(decision.target[order], device="cuda", dtype=torch.float32)
        before = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=5e-5, weight_decay=0.0)
        torch.cuda.reset_peak_memory_stats()
        thermal_guard(force=True)
        receipt.update(status="running", gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(),
                       packages={name: version(name) for name in ("torch", "transformers", "peft")},
                       encoded_tokens=row["encoded_tokens"], prompt_sha256=row["prompt_sha256"],
                       trainable_parameters=sum(p.numel() for _, p in trainable))
        write_json(output, receipt)
        logits = model(**encoded, logits_to_keep=1).logits[0, -1, letters[:len(order)]].float()
        logp = F.log_softmax(logits, -1)
        loss = (target * (target.clamp_min(1e-12).log() - logp)).sum()
        if decision.gold is not None:
            loss = loss - .5 * logp[order.index(decision.gold)]
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        grads = [p.grad for _, p in trainable if p.grad is not None]
        if len(grads) != len(trainable) or not all(torch.isfinite(g).all().item() for g in grads):
            raise ValueError("Missing or non-finite LoRA gradients")
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
        if not norm.item() > 0:
            raise ValueError("Zero gradient does not establish trainability")
        thermal_guard(force=True)
        optimizer.step()
        torch.cuda.synchronize()
        after = get_peft_model_state_dict(model)
        changed = sum(not torch.equal(before[name], value.detach().cpu()) for name, value in after.items())
        if changed == 0 or not all(torch.isfinite(value).all().item() for value in after.values()):
            raise ValueError("No finite adapter update")
        receipt.update(status="complete", passed=True, loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                       adapter_tensors_changed=changed, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                       thermal_governor=THERMAL, weights_saved=False,
                       scope="One optimizer step on the longest planned training input only. Probe weights are discarded; this is not a trained candidate, quality score, throughput benchmark or release result.")
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(output, receipt)


def driver():
    import psutil
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve prior execution")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
               TEZ_PILOT_RESULTS=str(DIRECTORY), TOKENIZERS_PARALLELISM="false")
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
