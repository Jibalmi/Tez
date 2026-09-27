"""Complete only three missing text metadata files in the pinned local Qwen snapshot."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-cache-metadata-20260927"
REPOSITORY = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
NAMES = (".gitattributes", "LICENSE", "README.md")


def main():
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    if DIRECTORY.exists():
        raise ValueError("Preserve prior metadata repair")
    cache = Path(HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / REVISION
    before = {p.relative_to(cache).as_posix(): digest(p) for p in cache.rglob("*") if p.is_file()}
    if (not before or set(before) & set(NAMES)
            or [v for k, v in before.items() if k.endswith(".safetensors")]
            != ["04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"]):
        raise ValueError("Expected intact pinned weights and only missing text metadata")
    DIRECTORY.mkdir(parents=True)
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "repository": REPOSITORY, "revision": REVISION, "source_sha256": digest(__file__),
               "before_sha256": before, "downloaded": {}}
    write_json(DIRECTORY / "repair.json", receipt)
    try:
        for name in NAMES:
            metadata = get_hf_file_metadata(hf_hub_url(REPOSITORY, name, revision=REVISION))
            if metadata.commit_hash != REVISION or metadata.size is None or metadata.size > 1024 * 1024:
                raise ValueError("Expected a small metadata file from the pinned revision")
            path = Path(hf_hub_download(REPOSITORY, name, revision=REVISION))
            raw = path.read_bytes()
            git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
            if len(raw) != metadata.size or git_hash != metadata.etag:
                raise ValueError("Downloaded metadata does not match its remote Git object")
            receipt["downloaded"][name] = {"sha256": digest(path), "bytes": len(raw), "git_blob_sha1": git_hash}
            write_json(DIRECTORY / "repair.json", receipt)
        after = {p.relative_to(cache).as_posix(): digest(p) for p in cache.rglob("*") if p.is_file()}
        if set(after) != set(before) | set(NAMES) or any(after[k] != value for k, value in before.items()):
            raise ValueError("Unexpected change to existing model/tokenizer files")
        resolved = Path(snapshot_download(REPOSITORY, revision=REVISION, local_files_only=True))
        if resolved.resolve() != cache.resolve():
            raise ValueError("Offline lookup resolved another snapshot")
        receipt.update(status="complete", after_sha256=after, offline_full_snapshot_lookup_passed=True,
                       existing_model_and_tokenizer_unchanged=True,
                       scope="Only three text metadata files were downloaded, with Git-object verification. No model inference, package update, weight or tokenizer change, training, GPU work or cloud spend.")
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(DIRECTORY / "repair.json", receipt)
    print(json.dumps({"status": receipt["status"], "downloaded": receipt["downloaded"],
                      "unchanged_existing_files": len(before)}, indent=2))


if __name__ == "__main__":
    main()
