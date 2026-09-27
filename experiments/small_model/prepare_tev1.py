"""Download only the pinned Tev1 0.8B checkpoint for a later native-interface comparison."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/pilot-tev1-discovery-20260927"
DEST = ROOT / "runs/pilot-tev1-preparation-20260927"
REPO = "togethercomputer/Tev1-0.8B-experimental"


def main():
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    path = RESULTS / "preparation.json"
    if DEST.exists() or path.exists():
        raise ValueError("Preserve existing preparation attempts")
    inventory_path = RESULTS / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    entry = inventory["models"][REPO]
    metadata_path = ROOT / "runs/research/tev1-discovery-20260927/Tev1-0.8B-experimental.metadata.json"
    if digest(metadata_path) != entry["metadata_sha256"]:
        raise ValueError("Pinned model metadata changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["sha"] != entry["revision"]:
        raise ValueError("Metadata revision differs")
    files = [item for item in metadata["siblings"] if item["rfilename"] != ".gitattributes"]
    required_bytes = sum(item["size"] for item in files)
    if shutil.disk_usage(ROOT).free < required_bytes + 8 * 2**30:
        raise ValueError("Leave at least 8 GiB free after staging")
    DEST.mkdir()
    model = DEST / "model"
    model.mkdir()
    shutil.copy2(__file__, DEST / "prepare_tev1.py")
    started = time.monotonic()
    baseline = {name: version(name) for name in ("torch", "transformers", "peft", "tokenizers", "huggingface-hub")}
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(), "repository": REPO,
               "revision": entry["revision"], "inventory_sha256": digest(inventory_path), "source_sha256": digest(__file__),
               "directory": model.relative_to(ROOT).as_posix(), "files": {}, "baseline_environment": baseline,
               "scope": "Public checkpoint staging only. No inference, package installation, redistribution or paid API call."}
    write_json(path, receipt)
    try:
        for item in files:
            name = item["rfilename"]
            if Path(name).parts != (name,):
                raise ValueError("Unexpected model file path")
            url = f"https://huggingface.co/{REPO}/resolve/{entry['revision']}/{name}"
            target = model / name
            h = hashlib.sha256()
            git_blob = hashlib.sha1(f"blob {item['size']}\0".encode())
            size = 0
            with urllib.request.urlopen(url, timeout=60) as response, target.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Ten-minute preparation bound reached")
                    size += len(chunk)
                    if size > item["size"]:
                        raise ValueError("Remote file exceeded pinned length")
                    output.write(chunk)
                    h.update(chunk)
                    git_blob.update(chunk)
            if size != item["size"] or (h.hexdigest() != item["lfs"]["sha256"] if "lfs" in item else git_blob.hexdigest() != item["blobId"]):
                raise ValueError("Downloaded model file differs from pinned remote object")
            if name in entry["files"] and h.hexdigest() != entry["files"][name]["sha256"]:
                raise ValueError("Downloaded model configuration differs from discovery")
            receipt["files"][name] = {"sha256": h.hexdigest(), "bytes": size}
            write_json(path, receipt)
        after = {name: version(name) for name in baseline}
        if baseline != after:
            raise ValueError("Baseline package environment changed")
        receipt.update(status="staged", baseline_unchanged=True, baseline_environment_after=after)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, receipt)
    print(json.dumps({"status": receipt["status"], "revision": entry["revision"], "files_verified": len(receipt["files"])}))


if __name__ == "__main__":
    main()
