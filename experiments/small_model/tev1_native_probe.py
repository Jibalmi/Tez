"""Verify pinned Tev1 native rendering and full option probabilities on four non-benchmark cases."""
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-tev1-probe-20260927"
REGISTRATION = Path(__file__).with_name("tev1_probe_registration.json")


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    path = DIRECTORY / "probe.json"
    if path.exists():
        raise ValueError("Preserve previous probe results")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "rows": [], "source_files": {},
               "scope": "Non-benchmark interface check only; no accuracy, calibration or latency score"}
    start = time.monotonic()
    write_json(path, receipt)
    try:
        for relative, expected in spec["source_files"].items():
            source = ROOT / relative
            if digest(source) != expected:
                raise ValueError("Registered source changed")
            destination = DIRECTORY / "executed_source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            receipt["source_files"][relative] = expected
        prep_path = ROOT / spec["preparation"]
        if digest(prep_path) != spec["preparation_sha256"]:
            raise ValueError("Pinned preparation changed")
        prep = json.loads(prep_path.read_text(encoding="utf-8"))
        if prep["status"] != "staged" or not prep["baseline_unchanged"]:
            raise ValueError("Complete unchanged model preparation first")
        model_path = ROOT / prep["directory"]
        for filename, expected in prep["files"].items():
            if digest(model_path / filename) != expected["sha256"]:
                raise ValueError("Staged model file changed")
        module_spec = importlib.util.spec_from_file_location("tev1_native_request", ROOT / spec["native_helper"])
        native = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(native)
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from common import THERMAL, thermal_guard
        from importlib.metadata import version
        receipt.update(packages={name: version(name) for name in ("torch", "transformers", "tokenizers")},
                       gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(), model=prep["repository"],
                       revision=prep["revision"], preparation_sha256=digest(prep_path),
                       weights={name: value["sha256"] for name, value in prep["files"].items() if name.endswith(".safetensors")})
        thermal_guard(force=True)
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
        receipt.update(status="running", parameter_count=sum(p.numel() for p in model.parameters()),
                       dtype=str(next(model.parameters()).dtype), cases=spec["cases"])
        write_json(path, receipt)
        for index, case in enumerate(spec["cases"]):
            record = {"state": case["state"], "question": case["instructions"],
                      "options": [{"label": chr(65 + i), "key": key, "description": description}
                                  for i, (key, description) in enumerate(case["options"])]}
            body = native.payload(record, prep["repository"])
            if json.loads(body["messages"][1]["content"]) != record:
                raise ValueError("Native helper changed the supplied decision")
            prompt = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = [tok.encode(option["label"], add_special_tokens=False) for option in record["options"]]
            if any(len(value) != 1 for value in ids) or len(set(value[0] for value in ids)) != len(ids):
                raise ValueError("Option letters must be distinct single tokens")
            token_ids = [value[0] for value in ids]
            encoded = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).to("cuda")
            if encoded["input_ids"].shape[1] > 4096:
                raise ValueError("Non-benchmark probe exceeds its fixed context bound")
            thermal_guard(force=True)
            with torch.inference_mode():
                full = model(**encoded, logits_to_keep=1).logits[0, -1].float()
                logits = full[token_ids].double()
                probabilities = torch.softmax(logits, -1).cpu().tolist()
                vocabulary_mass = float(torch.softmax(full.double(), -1)[token_ids].sum().item())
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=1,
                             prefix_allowed_tokens_fn=lambda batch_id, input_ids, allowed=token_ids: allowed, pad_token_id=tok.pad_token_id)
            chosen = int(generated[0, -1].item())
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            if (chosen != token_ids[predicted] or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                    or abs(sum(probabilities) - 1) > 1e-12):
                raise ValueError("Native constrained generation and option readout disagree")
            receipt["rows"].append({"case": index, "record": record, "messages": body["messages"], "prompt": prompt,
                     "encoded_tokens": int(encoded["input_ids"].shape[1]), "letter_token_ids": token_ids,
                     "logits": logits.cpu().tolist(), "probabilities": probabilities, "unrestricted_option_mass": vocabulary_mass,
                     "generated_token": chosen, "generated_letter": tok.decode([chosen]), "pred": predicted})
            write_json(path, receipt)
        receipt.update(status="complete", passed=True, cases_verified=len(spec["cases"]), thermal_governor=THERMAL)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start)
        write_json(path, receipt)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    prep = ROOT / spec["preparation"]
    if digest(prep) != spec["preparation_sha256"] or json.loads(prep.read_text())["status"] != "staged":
        raise ValueError("Complete the registered preparation first")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve previous execution")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": 300}
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
