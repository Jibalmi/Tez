"""Static check: no unsafe deserialisation and no dynamic code execution in the runtime (tez/). Used by
.github/workflows/security.yml; run it locally with `python scripts/check_banned_calls.py`.

Banned:
  pickle and relatives  importing pickle, cPickle, _pickle, dill, cloudpickle, joblib, shelve or marshal
  torch.load            it unpickles
  unsafe YAML           yaml.unsafe_load / full_load (and *_all); yaml.load / load_all without a safe Loader
  eval / exec           calls to the builtins
  numpy.load            unless allow_pickle=False is passed as a keyword

Allowed: yaml.load(..., Loader=L) where L is SafeLoader / CSafeLoader or a class in the same module that subclasses one
(tez/schema.py's _Loader), and np.load(..., allow_pickle=False) (tez/artifacts.py). The check reads the syntax tree,
so comments and docstrings that mention these names do not count, and calls split over several lines are still seen.

    python scripts/check_banned_calls.py [path ...]       # default: tez/ next to this script; exit 1 on any finding
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED_MODULES = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "joblib", "shelve", "marshal"}
SAFE_LOADERS = {"SafeLoader", "CSafeLoader"}
UNSAFE_YAML = {"unsafe_load", "full_load", "unsafe_load_all", "full_load_all"}
ROOT = Path(__file__).resolve().parents[1]


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _resolve(name: str, aliases: dict[str, str]) -> str:
    head, _, rest = name.partition(".")
    target = aliases.get(head, head)
    return f"{target}.{rest}" if rest else target


def _safe_loader_classes(tree: ast.AST, aliases: dict[str, str]) -> set[str]:
    """Classes of this module that subclass a safe YAML loader, directly or through each other."""
    safe = {f"yaml.{n}" for n in SAFE_LOADERS}
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    found: set[str] = set()
    changed = True
    while changed:
        changed = False
        for c in classes:
            if c.name not in found and any(
                    (d := dotted(b)) is not None and (_resolve(d, aliases) in safe or d in found) for b in c.bases):
                found.add(c.name)
                changed = True
    return found


def check_source(source: str, filename: str = "<source>") -> list[tuple[int, str]]:
    """(line, reason) for every banned construct in one module."""
    tree = ast.parse(source, filename)
    problems: list[tuple[int, str]] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_MODULES:
                    problems.append((node.lineno, f"import of {a.name} (unpickling)"))
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in BANNED_MODULES:
                problems.append((node.lineno, f"import from {node.module} (unpickling)"))
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    loaders = _safe_loader_classes(tree, aliases)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if name is None:
            continue
        full = _resolve(name, aliases)
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if full in ("eval", "exec", "builtins.eval", "builtins.exec"):
            problems.append((node.lineno, f"call to {full.split('.')[-1]}()"))
        elif full == "torch.load":
            problems.append((node.lineno, "torch.load (unpickles)"))
        elif full in {f"yaml.{f}" for f in UNSAFE_YAML}:
            problems.append((node.lineno, f"{full} (unsafe YAML loader)"))
        elif full in ("yaml.load", "yaml.load_all"):
            loader = kwargs.get("Loader", node.args[1] if len(node.args) > 1 else None)
            ref = dotted(loader) if loader is not None else None
            ok = ref is not None and (ref in loaders or _resolve(ref, aliases) in {f"yaml.{n}" for n in SAFE_LOADERS})
            if not ok:
                problems.append((node.lineno, f"{full} without a safe Loader (use yaml.safe_load or a SafeLoader)"))
        elif full == "numpy.load":
            flag = kwargs.get("allow_pickle")
            if not (isinstance(flag, ast.Constant) and flag.value is False):
                problems.append((node.lineno, "numpy.load without allow_pickle=False"))
    return sorted(problems)


def check_paths(paths: list[Path]) -> list[str]:
    out = []
    for base in paths:
        files = sorted(base.rglob("*.py")) if base.is_dir() else [base]
        for f in files:
            for line, reason in check_source(f.read_text(encoding="utf-8"), str(f)):
                out.append(f"{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}:{line}: {reason}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    paths = [Path(a).resolve() for a in args] or [ROOT / "tez"]
    findings = check_paths(paths)
    for f in findings:
        print(f)
    print(f"{len(findings)} banned construct(s) in {', '.join(str(p) for p in paths)}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
