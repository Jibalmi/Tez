#!/usr/bin/env python3
"""Upload the 24-block Qwen3.5-4B GGUF and its model card to a Hugging Face model repository.

  hf auth login                                        # once (or set HF_TOKEN for a single run)
  python release/hf/upload.py                          # -> Jibalmi/tez-qwen3.5-4b-L24-gguf
  python release/hf/upload.py someone/other-name --private
  python release/hf/upload.py --dry-run                # every check, no upload

Steps, stopping at the first problem (nothing is uploaded unless every check passes):
  1. find your token (HF_TOKEN, or the one `hf auth login` stored) and check it can write to the repo's namespace;
     without a token it prints the exact commands to run and exits
  2. check the GGUF is the documented file: its size and SHA-256 must match the model card
  3. fetch LICENSE (and NOTICE, if there is one) from Qwen/Qwen3.5-4B: Apache-2.0 asks redistributors to pass on a
     copy of the licence
  4. create the repository if it does not exist and upload README.md, LICENSE and the GGUF in one commit

Needs huggingface_hub (pip install -U huggingface_hub). The upload is about 3.5 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "Jibalmi/tez-qwen3.5-4b-L24-gguf"
DEFAULT_GGUF = ROOT / "tools" / "models" / "Qwen3.5-4B-Q8_0-L24.gguf"
DEFAULT_CARD = Path(__file__).resolve().parent / "README.md"
BASE_REPO = "Qwen/Qwen3.5-4B"
EXPECTED_SIZE = 3_533_398_528
EXPECTED_SHA256 = "f6ab76cf738d0293392488e18da7088d9952b59f0c11f3ff4ab2360a61469fd6"


def this_command(repo: str) -> str:
    try:
        script = Path(os.path.relpath(__file__)).as_posix()   # forward slashes work in PowerShell and bash
    except ValueError:                             # another drive on Windows
        script = Path(__file__).as_posix()
    return f"python {script} {repo}"


def no_token(repo: str) -> int:
    cmd = this_command(repo)
    print("No Hugging Face token found, so nothing was uploaded.\n"
          "\n"
          "Log in once (the token needs write access), then run this script again:\n"
          "\n"
          "    hf auth login\n"
          f"    {cmd}\n"
          "\n"
          "or pass a token for this run only:\n"
          "\n"
          f"    PowerShell:  $env:HF_TOKEN = \"hf_...\"; {cmd}\n"
          f"    bash:        HF_TOKEN=hf_... {cmd}\n"
          "\n"
          "(Older huggingface_hub releases name the login command `huggingface-cli login`.)")
    return 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload the 24-block Qwen3.5-4B GGUF and its card to Hugging Face.")
    ap.add_argument("repo", nargs="?", default=DEFAULT_REPO, help=f"model repository id (default {DEFAULT_REPO})")
    ap.add_argument("--gguf", default=str(DEFAULT_GGUF))
    ap.add_argument("--card", default=str(DEFAULT_CARD))
    ap.add_argument("--private", action="store_true", help="create the repository as private")
    ap.add_argument("--dry-run", action="store_true", help="run every check, upload nothing")
    args = ap.parse_args()
    if args.repo.count("/") != 1:
        print(f"repository id must look like owner/name, got {args.repo!r}")
        return 2

    try:
        from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed: pip install -U huggingface_hub")
        return 2

    # 1. token ------------------------------------------------------------------------------------------
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        return no_token(args.repo)
    api = HfApi(token=token)
    try:
        me = api.whoami()
    except Exception as exc:  # noqa: BLE001
        print(f"The Hugging Face token was rejected ({type(exc).__name__}: {exc}).")
        return no_token(args.repo)
    owners = {me.get("name")} | {o.get("name") for o in me.get("orgs", []) if isinstance(o, dict)}
    namespace = args.repo.split("/")[0]
    if namespace not in owners:
        print(f"This token belongs to {me.get('name')!r} (organisations: {sorted(o for o in owners if o and o != me.get('name'))}); "
              f"it cannot create repositories under {namespace!r}. Log in as {namespace!r} or pick another repository id.")
        return 2
    print(f"token ok: {me.get('name')}")

    # 2. the file and the card agree ---------------------------------------------------------------------
    gguf, card = Path(args.gguf), Path(args.card)
    for p in (gguf, card):
        if not p.is_file():
            print(f"missing file: {p}")
            return 2
    size = gguf.stat().st_size
    if size != EXPECTED_SIZE:
        print(f"{gguf} is {size:,} bytes, expected {EXPECTED_SIZE:,}: not the documented file")
        return 2
    print(f"hashing {gguf.name} ({size / 1e9:.2f} GB) ...", flush=True)
    digest = sha256(gguf)
    if digest != EXPECTED_SHA256:
        print(f"SHA-256 {digest} does not match the documented {EXPECTED_SHA256}")
        return 2
    card_text = card.read_text(encoding="utf-8")
    if EXPECTED_SHA256 not in card_text or not card_text.startswith("---"):
        print(f"{card} does not start with YAML front matter or does not state the file's SHA-256")
        return 2
    print(f"file ok: SHA-256 {digest}")

    # 3. the base model's licence ------------------------------------------------------------------------
    try:
        licence = hf_hub_download(BASE_REPO, "LICENSE", token=token)
        notice = hf_hub_download(BASE_REPO, "NOTICE", token=token) if api.file_exists(BASE_REPO, "NOTICE") else None
    except Exception as exc:  # noqa: BLE001
        print(f"could not fetch the licence from {BASE_REPO} ({type(exc).__name__}: {exc}); nothing was uploaded")
        return 2
    print(f"licence ok: LICENSE{' and NOTICE' if notice else ''} from {BASE_REPO}")

    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)),
                  CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=licence)]
    if notice:
        operations.append(CommitOperationAdd(path_in_repo="NOTICE", path_or_fileobj=notice))
    operations.append(CommitOperationAdd(path_in_repo=gguf.name, path_or_fileobj=str(gguf)))
    plan = ", ".join(op.path_in_repo for op in operations)
    if args.dry_run:
        print(f"dry run: would upload {plan} to https://huggingface.co/{args.repo}"
              f" ({'private' if args.private else 'public'}); nothing was uploaded")
        return 0

    # 4. upload ------------------------------------------------------------------------------------------
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    info = api.create_commit(args.repo, operations=operations, repo_type="model",
                             commit_message="Qwen3.5-4B Q8_0 cut to its first 24 blocks (GGUF), model card and licence")
    print(f"uploaded {plan}\nhttps://huggingface.co/{args.repo}  (commit {getattr(info, 'oid', '?')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
