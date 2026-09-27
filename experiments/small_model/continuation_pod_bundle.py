"""Package the registered Qwen continuation for a RunPod GPU, with a checksum manifest (no credentials, no base model).

The bundle is the committed tree (working-copy bytes of `git ls-files`) plus the ignored inputs the worker needs: the frozen teacher
directory, the original adapter, the training schedule, the verified resume checkpoint, and the three installed
library files whose hashes the registration pins (kept at their registered `.venv-pilot/Lib/site-packages/...` paths
so the pod can compare them with its own install). The base model is downloaded on the pod at the pinned revision.

  .venv-pilot/Scripts/python.exe experiments/small_model/continuation_pod_bundle.py --checkpoint runs/.../attempt-09-step-100.pt
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "experiments/small_model/qwen_continuation_registration.json"
OUT = ROOT / "runs/pilot/continuation-pod-20260927"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                      text=True).stdout.strip():
        raise SystemExit("Commit tracked changes first: the bundle is built from HEAD")
    extra = [ROOT / spec["teacher_directory"], ROOT / spec["adapter_directory"],
             (ROOT / spec["adapter_directory"]).parent / "student.json", ROOT / spec["schedule"], ROOT / a.checkpoint]
    extra += [ROOT / name for name in spec["sources"] if name.startswith(".venv-pilot/")]
    files = {}
    for item in extra:
        for path in ([item] if item.is_file() else sorted(p for p in item.rglob("*") if p.is_file())):
            if any(s in path.name.lower() for s in (".key", "credential", ".env")):
                raise SystemExit(f"Refusing a credential-like file: {path}")
            files[path.relative_to(ROOT).as_posix()] = path
    for name, expected in spec["sources"].items():
        if name in files and sha(files[name]) != expected:
            raise SystemExit(f"Registered source differs locally: {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = OUT / "bundle.tar"
    if bundle.exists():
        raise SystemExit("Bundle exists; remove it deliberately to rebuild")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # Working-copy bytes, not `git archive` output: the registration hashes the laptop's files as they are on disk,
    # and on Windows those carry CRLF line endings that the repository stores as LF (attempt 11 failed on this).
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    for name in filter(None, tracked):
        if (ROOT / name).is_file():
            files.setdefault(name, ROOT / name)
    manifest = {}
    with tarfile.open(bundle, "w") as out:
        for name, path in sorted(files.items()):
            manifest[name] = sha(path)
            out.add(path, arcname=name, recursive=False)
    receipt = {"git_head": head, "files": len(manifest), "bundle_bytes": bundle.stat().st_size,
               "bundle_sha256": sha(bundle), "checkpoint": a.checkpoint, "checkpoint_sha256": manifest[Path(a.checkpoint).as_posix()],
               "registration_sha256": sha(REGISTRATION)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "bundle_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
