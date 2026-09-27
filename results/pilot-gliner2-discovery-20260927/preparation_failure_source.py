"""Stage two pinned GLiNER2 competitors and a separate compatible import overlay; no inference."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/pilot-gliner2-discovery-20260927"
DEST = ROOT / "runs/pilot-gliner2-20260927"
PACKAGES = {"gliner2": "2.0.0", "transformers": "4.57.6", "tokenizers": "0.22.2", "huggingface-hub": "0.36.2"}
MODELS = ("fastino/GLiNER2.5-Decide", "fastino/GLiNER2.5-multi-Decide")


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def download(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, target.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = digest(target)
    if expected is not None and actual != expected:
        raise ValueError("Downloaded artifact checksum mismatch")
    return {"url": url, "sha256": actual, "bytes": target.stat().st_size}


def main():
    if DEST.exists() or (RESULTS / "preparation.json").exists():
        raise ValueError("Preserve existing preparation attempts")
    inventory = json.loads((RESULTS / "inventory.json").read_text(encoding="utf-8"))
    DEST.mkdir()
    baseline = {name: installed(name) for name in ("torch", "peft", *PACKAGES)}
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": digest(__file__), "inventory_sha256": digest(RESULTS / "inventory.json"),
               "baseline_environment": baseline, "packages": {}, "models": {},
               "scope": "Download/install into an isolated overlay only; no model imports, inference or benchmark scores"}
    path = RESULTS / "preparation.json"
    write_json(path, receipt)
    try:
        supported = list(sys_tags())
        ranks = {tag: i for i, tag in enumerate(supported)}
        wheels = []
        for name, selected in PACKAGES.items():
            metadata = fetch_json(f"https://pypi.org/pypi/{name}/{selected}/json")
            choices = []
            for wheel in metadata["urls"]:
                if not wheel["filename"].endswith(".whl"):
                    continue
                tags = parse_wheel_filename(wheel["filename"])[3]
                if tags.intersection(ranks):
                    choices.append((min(ranks[tag] for tag in tags if tag in ranks), wheel))
            if not choices:
                raise ValueError("No compatible wheel")
            wheel = min(choices, key=lambda pair: pair[0])[1]
            target = DEST / "wheels" / wheel["filename"]
            saved = download(wheel["url"], target, wheel["digests"]["sha256"])
            receipt["packages"][name] = {**saved, "version": selected, "requires_dist": metadata["info"].get("requires_dist")}
            wheels.append(str(target))
            write_json(path, receipt)
        checks = {}
        for name, metadata in receipt["packages"].items():
            for text in metadata["requires_dist"] or []:
                req = Requirement(text)
                if req.marker and not req.marker.evaluate({"extra": "local" if name == "gliner2" else ""}):
                    continue
                candidate = PACKAGES.get(canonicalize_name(req.name), installed(req.name))
                if candidate is None or candidate not in req.specifier:
                    raise ValueError(f"Missing/incompatible dependency: {req.name}")
                checks[req.name] = candidate
        receipt["dependency_versions_checked"] = checks
        overlay = DEST / "site-packages"
        with (DEST / "install.log").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                            "--target", str(overlay), *wheels], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        receipt.update(overlay=overlay.relative_to(ROOT).as_posix(),
                       installed_file_hashes={p.relative_to(overlay).as_posix(): digest(p) for p in overlay.rglob("*") if p.is_file()})
        write_json(path, receipt)
        for name in MODELS:
            revision = inventory["models"][name]["revision"]
            metadata = fetch_json(f"https://huggingface.co/api/models/{name}/revision/{revision}?blobs=true")
            if metadata["sha"] != revision:
                raise ValueError("Model revision changed")
            chosen = [f for f in metadata["siblings"] if f["rfilename"].endswith(".json") or f["rfilename"] == "model.safetensors"]
            directory = DEST / "models" / name.split("/")[-1]
            receipt["models"][name] = {"revision": revision, "directory": directory.relative_to(ROOT).as_posix(), "files": {}}
            for file in chosen:
                filename = file["rfilename"]
                target = (directory / filename).resolve()
                if not target.is_relative_to(directory.resolve()):
                    raise ValueError("Unsafe model filename")
                expected = (file.get("lfs") or {}).get("sha256")
                if filename.endswith(".safetensors") and not expected:
                    raise ValueError("Weights lack an upstream checksum")
                saved = download(f"https://huggingface.co/{name}/resolve/{revision}/{filename}", target, expected)
                receipt["models"][name]["files"][filename] = saved
                write_json(path, receipt)
            print(json.dumps({"model_staged": name, "files": len(chosen)}), flush=True)
        after = {name: installed(name) for name in baseline}
        if baseline != after:
            raise ValueError("Baseline environment changed")
        receipt.update(status="staged", baseline_unchanged=True, baseline_environment_after=after)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, receipt)
    print(json.dumps({"status": receipt["status"], "packages": PACKAGES, "models": list(receipt["models"])}))


if __name__ == "__main__":
    main()
