"""Package the v2 data-scaling round for a RunPod GPU: the committed tree (working-copy bytes) plus the ignored inputs.

Inputs: the new pool and its clusters/embeddings (runs/pilot-v2), the original exported teacher dataset
(runs/pilot-cloud-h200-20260927), the gold question wording written for the original pool, and the frozen evaluation
suites of the 30k run. No credentials, model weights or virtual environments.

  .venv-pilot/Scripts/python.exe experiments/small_model/v2_pod_bundle.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/pilot/v2-pod-20260927"
EXTRA = ["runs/pilot-v2/pool.jsonl", "runs/pilot-v2/pool_emb.npy", "runs/pilot-v2/clusters.jsonl",
         "runs/pilot-v2/gold_questions.jsonl", "runs/pilot-v2/eval_minhash.npz", "runs/pilot-v2/eval_emb.npz",
         "runs/pilot-cloud-h200-20260927", "runs/pilot/student-prep-20260927/eval-suites.json"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import argparse
    global OUT, EXTRA
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="bundle directory (default: the v2 round's)")
    ap.add_argument("--extra", nargs="*", help="ignored inputs to include instead of the v2 round's")
    a = ap.parse_args()
    OUT = ROOT / a.out if a.out else OUT
    EXTRA = a.extra if a.extra is not None else EXTRA
    files = {}
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    for name in filter(None, tracked):
        if (ROOT / name).is_file():
            files[name] = ROOT / name
    for item in map(ROOT.joinpath, EXTRA):
        for path in ([item] if item.is_file() else sorted(p for p in item.rglob("*") if p.is_file())):
            if any(s in path.name.lower() for s in (".key", "credential", ".env")):
                raise SystemExit(f"Refusing a credential-like file: {path}")
            files[path.relative_to(ROOT).as_posix()] = path
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = OUT / "bundle.tar"
    if bundle.exists():
        raise SystemExit("Bundle exists; remove it deliberately to rebuild")
    manifest = {}
    with tarfile.open(bundle, "w") as out:
        for name, path in sorted(files.items()):
            manifest[name] = sha(path)
            info = out.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as f:
                out.addfile(info, f)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "manifest.json").write_bytes(json.dumps(manifest, indent=1).encode())
    receipt = {"git_head": head, "files": len(manifest), "bundle_bytes": bundle.stat().st_size, "bundle_sha256": sha(bundle)}
    (OUT / "bundle_receipt.json").write_bytes(json.dumps(receipt, indent=2).encode())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
