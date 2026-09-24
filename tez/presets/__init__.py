"""Presets: the ten worked use cases of examples/usecases/ shipped with the package as ready-made schemas.

    import tez.presets
    tez.presets.names()                      # ['agent-trace-review', 'intent-router', ...]
    schema = tez.presets.load("support-triage")
    Tez(backend=..., schemas=[schema]).decide("My invoice is wrong", schema="support-triage")

A preset is zero-shot and marked builtin: it never loads fitted artefacts and `tez fit` refuses it. To fit one, copy it
into a schema directory first (`tez presets --show support-triage > schemas/support-triage.yaml`). On the command
line: `tez serve --presets` loads them all (listed with "builtin": true in GET /v1/schemas), and
`tez decide "text" --preset support-triage` decides with one. The evidence and limits of each use case are in
examples/usecases/usecases.json.
"""
from __future__ import annotations

from importlib import resources

from ..errors import InvalidRequest
from ..schema import Schema, load_yaml, parse_schema

_PACKAGE = "tez.presets"


def _files() -> dict[str, object]:
    return {p.name[: -len(".yaml")]: p for p in resources.files(_PACKAGE).iterdir()
            if p.name.endswith(".yaml") and p.is_file()}


def names() -> list[str]:
    """The preset names, sorted."""
    return sorted(_files())


def text(name: str) -> str:
    """A preset's YAML, as shipped (comments included)."""
    files = _files()
    if name not in files:
        raise InvalidRequest(f"unknown preset '{name}' (presets: {', '.join(sorted(files))})")
    return files[name].read_text(encoding="utf-8")


def load(name: str) -> Schema:
    """A preset as a Schema (builtin=True)."""
    schema = parse_schema(load_yaml(text(name)), source=f"preset {name}", default_name=name)
    schema.builtin = True
    return schema


def load_all() -> list[Schema]:
    """Every preset, in name order."""
    return [load(n) for n in names()]


__all__ = ["names", "text", "load", "load_all"]
