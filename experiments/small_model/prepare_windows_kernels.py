"""Stage pinned Windows attention dependencies in an isolated overlay without importing them."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "runs/pilot-windows-kernels-20260927"
RESULTS = ROOT / "results/pilot-windows-kernels-20260927"
PACKAGES = {"triton-windows": "3.6.0.post26", "fla-core": "0.5.2", "einops": "0.8.2"}


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    if sys.version_info[:2] != (3, 14) or sys.platform != "win32" or installed("torch") != "2.10.0+cu130":
        raise ValueError("This wheel selection is for the recorded Python/PyTorch/Windows environment")
    if DEST.exists():
        raise ValueError("Preserve existing overlay preparation")
    DEST.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    wheel_dir = DEST / "wheels"
    wheel_dir.mkdir()
    receipt_path = RESULTS / "preparation.json"
    before = {name: installed(name) for name in ("torch", "transformers", "peft", *PACKAGES)}
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": digest(__file__), "python": sys.version, "baseline_environment": before,
               "packages": {}, "scope": "Package staging only; no GPU import, compilation, inference or speed measurement"}
    write_json(receipt_path, receipt)
    try:
        wheels = []
        for name, selected in PACKAGES.items():
            url = f"https://pypi.org/pypi/{name}/{selected}/json"
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.load(response)
            candidates = [f for f in metadata["urls"] if f["filename"].endswith(
                "cp314-cp314-win_amd64.whl" if name == "triton-windows" else "py3-none-any.whl")]
            if len(candidates) != 1:
                raise ValueError("No unique matching wheel")
            wheel = candidates[0]
            if not wheel["url"].startswith("https://files.pythonhosted.org/"):
                raise ValueError("Unexpected wheel host")
            target = wheel_dir / Path(wheel["filename"]).name
            with urllib.request.urlopen(wheel["url"], timeout=30) as response, target.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if digest(target) != wheel["digests"]["sha256"] or target.stat().st_size != wheel["size"]:
                raise ValueError("Wheel checksum mismatch")
            wheels.append(str(target))
            receipt["packages"][name] = {"version": selected, "metadata_url": url, "wheel_url": wheel["url"],
                "filename": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
                "requires_dist": metadata["info"].get("requires_dist")}
            write_json(receipt_path, receipt)
        target = DEST / "site-packages"
        command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile", "--target", str(target), *wheels]
        with (DEST / "install.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        after = {name: installed(name) for name in before}
        if before != after:
            raise ValueError("Baseline environment changed")
        receipt.update(status="staged", overlay=str(target.relative_to(ROOT)), baseline_environment_after=after,
                       install_log_sha256=digest(DEST / "install.log"), baseline_unchanged=True,
                       installed_file_hashes={p.relative_to(target).as_posix(): digest(p) for p in target.rglob("*") if p.is_file()})
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "overlay": receipt["overlay"], "packages": PACKAGES,
                      "baseline_unchanged": receipt["baseline_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
