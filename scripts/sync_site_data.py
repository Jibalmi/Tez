"""Copy the use-case gallery into the static site so the browser never parses YAML.

    python scripts/sync_site_data.py

Reads examples/usecases/usecases.json and, for every use case, its schema.yaml and samples.jsonl. Writes

    site/data/usecases.json          the manifest, copied byte for byte
    site/data/usecases/<id>.json     {"schema": {name, description, state, questions, gate},
                                      "samples": [...], "yaml": "<schema.yaml as written>"}

`questions` is the /v1/systemone wire format (docs/API.md): noul criteria keyed "true"/"false", choice criteria
keyed by label, score criteria a list. YAML is read with YAML 1.2 booleans (only true/false), as `tez` reads it,
so options called yes/no/on/off stay strings. Files are rewritten only when their content changes, so running it
twice writes nothing the second time. Standard library plus PyYAML.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "examples" / "usecases"
OUT = ROOT / "site" / "data"

CHOICE_MIN, CHOICE_MAX = 2, 255
SCORE_MIN, SCORE_MAX = 2, 10


class _Loader(yaml.SafeLoader):
    """YAML 1.2 booleans, matching tez/schema.py: only true/false resolve to booleans."""


_Loader.yaml_implicit_resolvers = {
    ch: [r for r in rs if r[0] != "tag:yaml.org,2002:bool"] for ch, rs in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def _key(k) -> str:
    if isinstance(k, bool):
        return "true" if k else "false"
    return k if isinstance(k, str) else str(k)


def wire_question(qid: str, raw: dict, where: str) -> dict:
    """One schema question in the wire format, checked against the rules in docs/API.md."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.{qid} must be a mapping")
    qtype = raw.get("type")
    if qtype not in ("noul", "choice", "score"):
        raise ValueError(f"{where}.{qid}.type must be noul, choice or score, got {qtype!r}")
    instructions = raw.get("instructions")
    if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
        raise ValueError(f"{where}.{qid}.instructions is required")
    out = {"type": qtype, "instructions": instructions}
    crit = raw.get("criteria")
    if qtype == "noul":
        if crit is not None:
            crit = {_key(k): v for k, v in crit.items()}
            extra = sorted(set(crit) - {"true", "false"})
            if extra:
                raise ValueError(f"{where}.{qid}.criteria keys must be true/false, got {extra}")
            out["criteria"] = {k: crit[k] for k in ("true", "false") if k in crit}
    elif qtype == "choice":
        if not isinstance(crit, dict):
            raise ValueError(f"{where}.{qid}.criteria must map each label to a description")
        norm = {_key(k): v for k, v in crit.items()}
        if not CHOICE_MIN <= len(norm) <= CHOICE_MAX:
            raise ValueError(f"{where}.{qid}.criteria must have {CHOICE_MIN} to {CHOICE_MAX} options, got {len(norm)}")
        out["criteria"] = norm
    else:
        if not isinstance(crit, list) or not SCORE_MIN <= len(crit) <= SCORE_MAX:
            raise ValueError(f"{where}.{qid}.criteria must be a list of {SCORE_MIN} to {SCORE_MAX} levels")
        out["criteria"] = [str(c) for c in crit]
    return out


def convert(uid: str, schema_path: Path, samples_path: Path) -> dict:
    text = schema_path.read_text(encoding="utf-8")
    raw = yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    if not isinstance(raw, dict):
        raise ValueError(f"{schema_path}: a schema must be a mapping")
    where = f"{uid}/schema.yaml: questions"
    questions = {_key(qid): wire_question(_key(qid), q, where) for qid, q in (raw.get("questions") or {}).items()}
    if not questions:
        raise ValueError(f"{schema_path}: no questions")
    gate = raw.get("gate")
    samples = []
    for n, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "state" not in row:
            raise ValueError(f"{samples_path}:{n}: each line must be an object with a state")
        samples.append(row)
    return {
        "schema": {
            "name": _key(raw.get("name") or uid),
            "description": raw.get("description") or "",
            "state": raw.get("state") or "",
            "questions": questions,
            "gate": gate if isinstance(gate, dict) else None,
        },
        "samples": samples,
        "yaml": text,
    }


def write_if_changed(path: Path, content: bytes, wrote: list, kept: list) -> None:
    if path.exists() and path.read_bytes() == content:
        kept.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    wrote.append(path)


def main() -> int:
    manifest_path = SRC / "usecases.json"
    if not manifest_path.exists():
        print(f"nothing to do: {manifest_path.relative_to(ROOT)} does not exist yet", file=sys.stderr)
        return 1
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    items = manifest if isinstance(manifest, list) else manifest.get("usecases", [])
    wrote: list[Path] = []
    kept: list[Path] = []
    write_if_changed(OUT / "usecases.json", manifest_bytes, wrote, kept)
    ids = []
    for uc in items:
        uid = uc["id"]
        ids.append(uid)
        schema_path = SRC / uc.get("schema", f"{uid}/schema.yaml")
        samples_path = SRC / uc.get("samples", f"{uid}/samples.jsonl")
        data = convert(uid, schema_path, samples_path)
        body = (json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        write_if_changed(OUT / "usecases" / f"{uid}.json", body, wrote, kept)
        print(f"  {uid:24s} {len(data['schema']['questions'])} question(s), {len(data['samples'])} sample(s)")
    for p in wrote:
        print(f"wrote     {p.relative_to(ROOT).as_posix()}")
    for p in kept:
        print(f"unchanged {p.relative_to(ROOT).as_posix()}")
    stale = sorted(p for p in (OUT / "usecases").glob("*.json") if p.stem not in ids)
    for p in stale:
        print(f"note: {p.relative_to(ROOT).as_posix()} is not in the manifest (left in place; delete it by hand if unwanted)")
    print(f"{len(ids)} use case(s): {len(wrote)} file(s) written, {len(kept)} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
