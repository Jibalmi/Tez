"""Record primary-source kernel/package metadata without installing packages or using the GPU."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from packaging.tags import sys_tags

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-kernel-discovery-20260927"
CACHE = ROOT / "runs/research/kernel-discovery-20260927"


def main():
    if DIRECTORY.exists() or CACHE.exists():
        raise ValueError("Preserve previous metadata discovery")
    DIRECTORY.mkdir(parents=True)
    CACHE.mkdir(parents=True)
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": digest(__file__), "python": sys.version, "requests": {}, "packages": {}}
    write_json(DIRECTORY / "discovery.json", receipt)
    try:
        def fetch(url, filename):
            with urlopen(Request(url, headers={"User-Agent": "Tez-research-metadata"}), timeout=30) as response:
                data = response.read()
            path = CACHE / filename
            path.write_bytes(data)
            receipt["requests"][filename] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            return data

        commit = json.loads(fetch("https://api.github.com/repos/QwenLM/FlashQLA/commits/main", "flashqla_commit.json"))["sha"]
        receipt["flashqla_commit"] = commit
        for filename in ("README.md", "setup.py", "LICENSE"):
            fetch(f"https://raw.githubusercontent.com/QwenLM/FlashQLA/{commit}/{filename}", "flashqla_" + filename)
        compatible = {str(tag) for tag in sys_tags()}
        from packaging.utils import parse_wheel_filename
        for name, revision in (("flash-qla", None), ("tilelang", "0.1.12")):
            url = f"https://pypi.org/pypi/{name}/" + (revision + "/" if revision else "") + "json"
            metadata = json.loads(fetch(url, name + ".json"))
            wheels = []
            for wheel in metadata["urls"]:
                if wheel["packagetype"] != "bdist_wheel":
                    continue
                _, _, _, tags = parse_wheel_filename(wheel["filename"])
                wheels.append({"filename": wheel["filename"], "bytes": wheel["size"], "sha256": wheel["digests"]["sha256"],
                               "compatible_tags": bool(compatible & {str(tag) for tag in tags}), "yanked": wheel["yanked"]})
            receipt["packages"][name] = {"version": metadata["info"]["version"], "requires_python": metadata["info"]["requires_python"],
                                         "requires_dist": metadata["info"]["requires_dist"], "wheels": wheels}
        receipt.update(status="complete", scope="Metadata and local wheel-tag comparison only. No install, model loading, GPU execution, compatibility success, probability parity or speed measurement.")
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(DIRECTORY / "discovery.json", receipt)
    print(json.dumps({"status": receipt["status"], "flashqla_commit": commit,
                      "packages": {name: {"version": item["version"], "requires_dist": item["requires_dist"],
                                          "compatible_wheels": [w for w in item["wheels"] if w["compatible_tags"] and not w["yanked"]]}
                                   for name, item in receipt["packages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
