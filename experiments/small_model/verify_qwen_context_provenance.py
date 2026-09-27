"""Link the token-limit reconstruction to the archived H200 training implementation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-context-audit-20260927"


def main():
    original = ROOT / "runs/student-artifacts-h200-20260927/experiments/small_model/train_c.py"
    current = ROOT / "experiments/small_model/train_c.py"
    driver = original.with_name("cloud_student.py")
    output = DIRECTORY / "historical_source_check.json"
    if output.exists():
        raise ValueError("Preserve original provenance verification")
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (original, current)]
    functions = {}
    for name in ("letter_logp", "prompt_for", "items", "epoch_batches"):
        bodies = [[ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == name] for tree in trees]
        functions[name] = len(bodies[0]) == len(bodies[1]) == 1 and bodies[0] == bodies[1]
    if not all(functions.values()):
        raise ValueError("Historical tokenization, prompting or schedule function differs")
    original_text = original.read_text(encoding="utf-8")
    driver_text = driver.read_text(encoding="utf-8")
    if ('ap.add_argument("--seed", type=int, default=13)' not in original_text
            or '("train_c-qwen08-30k", "train_c.py", ["--arm", "c-qwen08-30k", "--max-segments", "99"])' not in driver_text):
        raise ValueError("Historical training seed or launch arguments differ")
    receipt = {"status": "complete", "verified_utc": datetime.now(timezone.utc).isoformat(),
               "audit_sha256": digest(DIRECTORY / "audit.json"), "historical_source_sha256": digest(original),
               "current_source_sha256": digest(current), "driver_sha256": digest(driver),
               "identical_function_asts": functions, "historical_default_seed": 13, "driver_overrides_seed": False,
               "verification_source_sha256": digest(__file__),
               "scope": "Archived training code and driver agree with the reconstruction's relevant functions and seed. This is a source/data reconstruction, not retained historical token tensors or a causal accuracy experiment."}
    write_json(output, receipt)
    files = {"historical/train_c.py": original, "historical/cloud_student.py": driver,
             "current/train_c.py": current, "verify_qwen_context_provenance.py": Path(__file__),
             output.name: output}
    expected = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in files.items()}
    archive = DIRECTORY / "provenance-artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as bundle:
        actual = {m.name: {"sha256": hashlib.sha256(bundle.extractfile(m).read()).hexdigest(), "bytes": m.size}
                  for m in bundle.getmembers()}
    if actual != expected:
        raise ValueError("Provenance archive differs")
    write_json(DIRECTORY / "provenance_manifest.json", expected)
    write_json(DIRECTORY / "provenance_receipt.json", {"sha256": digest(archive), "bytes": archive.stat().st_size,
                                                     "files_verified": len(actual)})
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
