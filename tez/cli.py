"""Command line: tez serve | decide | fit | suggest | eval | truncate."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .backends import DEFAULT_BACKEND
from .engine import READOUTS
from .errors import TezError
from .prompt import TEMPLATES


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _backend_args(p: argparse.ArgumentParser, embed: bool = True) -> None:
    p.add_argument("--backend", default=os.environ.get("TEZ_BACKEND", DEFAULT_BACKEND),
                   help=f"llama-server URL, or 'fake' for the offline demo backend (env TEZ_BACKEND; default {DEFAULT_BACKEND})")
    p.add_argument("--template", default=os.environ.get("TEZ_TEMPLATE", "gemma4"), choices=sorted(TEMPLATES),
                   help="prompt template of the model behind --backend (env TEZ_TEMPLATE; default gemma4)")
    if embed:
        p.add_argument("--embed-backend", default=os.environ.get("TEZ_EMBED_BACKEND"),
                       help="separate llama-server for probe features, started with --embeddings --pooling last "
                            "(default: --backend)")
        p.add_argument("--embed-template", default=None, choices=sorted(TEMPLATES),
                       help="prompt template of the embedding model (default: --template)")
    p.add_argument("--no-cache-prompt", action="store_true",
                   help="switch llama.cpp prompt caching off (hybrid Qwen3.5 GGUFs crash b11100 on partial prefix reuse)")
    p.add_argument("--model-name", default=None, help="model name used in responses and calibration (default: read from the server)")


def _make_tez(args: argparse.Namespace, schemas: Any = None):
    from .engine import Tez
    return Tez(backend=args.backend, template=args.template, schemas=schemas,
               embed_backend=getattr(args, "embed_backend", None), embed_template=getattr(args, "embed_template", None),
               cache_prompt=not args.no_cache_prompt, data_dir=getattr(args, "data_dir", None), model_name=args.model_name)


def _read_state(path: str) -> Any:
    if path == "-":
        text = sys.stdin.read()
    else:
        p = Path(path)
        text = p.read_text(encoding="utf-8-sig")
        if p.suffix.lower() == ".json":
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise TezError(f"{path}: invalid JSON ({exc.msg})") from exc
    return text.rstrip("\r\n")


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


# ---------------------------------------------------------------------------------------------- commands
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app
    tez = _make_tez(args, schemas=args.schemas)
    app = create_app(tez, api_key=args.api_key, cors=not args.no_cors)
    probes = sum(len(v) for v in tez.probe_index().values())
    print(f"tez {__version__} on http://{args.host}:{args.port}  backend {tez.backend.url} ({tez.template})"
          + (f", embeddings {tez.embedder.url} ({tez.embedder.template})" if tez.embedder is not tez.backend else "")
          + f", {len(tez.schemas)} schema(s), {probes} probe(s)" + (", API key required" if args.api_key else ""),
          file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    from .schema import load_schema
    if not args.schema and not args.questions:
        raise TezError("give --schema FILE and/or --questions FILE")
    schema = load_schema(args.schema) if args.schema else None
    questions = None
    if args.questions:
        try:
            raw = json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TezError(f"cannot read questions from {args.questions}: {exc}") from exc
        questions = raw["questions"] if isinstance(raw, dict) and isinstance(raw.get("questions"), dict) else raw
    state = args.state if args.state is not None else _read_state(args.state_file)
    tez = _make_tez(args, schemas=[schema] if schema else None)
    res = tez.decide(state, questions=questions, schema=schema.name if schema else None, readout=args.readout,
                     abstain=args.abstain, alpha=args.alpha)
    print(json.dumps(res, indent=None if args.compact else 2, ensure_ascii=False))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from .fit import fit
    from .schema import load_schema
    schema = load_schema(args.schema)
    tez = _make_tez(args, schemas=[schema])
    cal = fit(tez, schema, label_files=args.labels, feedback=not args.no_feedback, holdout=args.holdout,
              min_labels=args.min_labels, seed=args.seed, letters=not args.no_letters, use_blend=not args.no_blend)
    rows = [["question", "type", "labels", "probe", "letters acc", "probe acc", "T letters", "T probe", "act@0.05"]]
    for qid, e in cal["questions"].items():
        lt, pr = e.get("letters") or {}, e.get("probe") or {}
        best = pr or lt
        cut = (best.get("thresholds") or {}).get("0.05")
        probe = (f"trained on {pr['n_train']}" + (f", blend w={pr['weight']:.2f}" if pr.get("blend") else "")) if pr else \
            ("skipped" if e.get("n_labels") else "no labels")
        rows.append([qid, e["type"], str(e["n_labels"]), probe, _fmt(lt.get("accuracy")), _fmt(pr.get("accuracy")),
                     _fmt(lt.get("temperature"), 2), _fmt(pr.get("temperature"), 2), "never" if best and cut is None else _fmt(cut)])
    print(_table(rows))
    for qid, e in cal["questions"].items():
        if e.get("note"):
            print(f"{qid}: {e['note']}")
    print(f"calibration {cal['calibration_id']} written to {schema.artifact_dir}")
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    from .fit import read_states, suggest
    from .schema import load_schema
    schema = load_schema(args.schema)
    tez = _make_tez(args, schemas=[schema])
    picks = suggest(tez, schema, read_states(args.unlabelled), n=args.n, question=args.question, seed=args.seed)
    text = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picks)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {len(picks)} rows to {args.out}; fill in labels and pass it to tez fit --labels", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .fit import evaluate
    from .schema import load_schema
    schema = load_schema(args.schema)
    tez = _make_tez(args, schemas=[schema])
    res = evaluate(tez, schema, args.labels, readout=args.readout, alpha=args.alpha, include_seen=args.include_seen)
    head = ["question", "n", "readout", "accuracy", "ECE", "mean conf"] + (["acted", "error|acted"] if args.alpha else [])
    rows = [head]
    for qid, r in res["questions"].items():
        if not r.get("n"):
            rows.append([qid, "0", "-", "-", "-", "-"] + (["-", "-"] if args.alpha else []))
            continue
        ro = "+".join(r["readouts"])
        row = [qid, str(r["n"]), ro, _fmt(r["accuracy"]), _fmt(r["ece"]), _fmt(r["mean_confidence"])]
        if args.alpha:
            row += [_fmt(r.get("acted")), _fmt(r.get("error_among_acted"))]
        rows.append(row)
    if res.get("overall"):
        o = res["overall"]
        rows.append(["overall", str(o["n"]), "", _fmt(o["accuracy"]), _fmt(o["ece"]), ""] + (["", ""] if args.alpha else []))
    print(_table(rows))
    for qid, r in res["questions"].items():
        if r.get("skipped_seen_in_training"):
            print(f"{qid}: {r['skipped_seen_in_training']} row(s) the probe was trained on were skipped (--include-seen to keep them)")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def cmd_truncate(args: argparse.Namespace) -> int:
    from .truncate import truncate_gguf
    info = truncate_gguf(args.src, args.n, args.dst)
    print(f"{info['src']}: {info['blocks'][0]} -> {info['blocks'][1]} blocks, "
          f"{info['tensors'][0]} -> {info['tensors'][1]} tensors -> {info['dst']}")
    return 0


# ---------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tez", description="Tez: an open local System One decision engine.")
    parser.add_argument("--version", action="version", version=f"tez {__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p = sub.add_parser("serve", help="run the HTTP server (Jev-compatible /v1/systemone)")
    _backend_args(p)
    p.add_argument("--schemas", help="directory of *.yaml schemas (trained artefacts in <dir>/.tez/)")
    p.add_argument("--data-dir", help="where POST /v1/feedback writes (default: <schemas>/.tez)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--api-key", default=os.environ.get("TEZ_API_KEY"), help="require Authorization: Bearer <key> (env TEZ_API_KEY)")
    p.add_argument("--no-cors", action="store_true", help="do not send CORS headers (open to every origin by default)")
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("decide", help="decide one state and print the wire-format JSON")
    _backend_args(p)
    p.add_argument("--schema", help="schema YAML file (its fitted probes and calibration are used)")
    p.add_argument("--questions", help="JSON file with Jev questions ({id: {type, instructions, criteria}})")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--state", help="the state as text")
    g.add_argument("--state-file", help="file with the state ('-' = stdin; *.json is parsed as JSON)")
    p.add_argument("--readout", choices=READOUTS, default="auto")
    p.add_argument("--abstain", action="store_true", help="add the implicit __none__ option to choice questions")
    p.add_argument("--alpha", type=float, default=None, help="gate: target error rate among acted decisions")
    p.add_argument("--compact", action="store_true", help="one-line JSON")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("fit", help="train probes, temperatures and gate thresholds for a schema")
    _backend_args(p)
    p.add_argument("--schema", required=True, help="schema YAML file; artefacts go to <dir>/.tez/<name>/")
    p.add_argument("--labels", action="append", default=[], metavar="FILE",
                   help="labelled JSONL rows {\"state\": ..., \"labels\": {question: label}} (repeatable)")
    p.add_argument("--data-dir", help="where recorded feedback lives (default: <schema dir>/.tez)")
    p.add_argument("--no-feedback", action="store_true", help="ignore rows recorded through POST /v1/feedback")
    p.add_argument("--holdout", type=float, default=0.3, help="share of each question's labels held out for calibration")
    p.add_argument("--min-labels", type=int, default=20, help="fewest labels for which a probe is trained")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-letters", action="store_true", help="skip the letters readout (no letters calibration, no blend)")
    p.add_argument("--no-blend", action="store_true", help="serve the probe alone, without the zero-shot letters prior")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("suggest", help="pick the most typical unlabelled states to label first")
    _backend_args(p)
    p.add_argument("--schema", required=True)
    p.add_argument("--unlabelled", required=True, help="JSONL rows with a state (or one state per line)")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--question", help="cluster on one question's prompt (default: all questions averaged)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the picks here (JSONL with empty labels) instead of stdout")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("eval", help="accuracy and calibration (ECE) per question on labelled rows")
    _backend_args(p)
    p.add_argument("--schema", required=True)
    p.add_argument("--labels", action="append", required=True, metavar="FILE")
    p.add_argument("--readout", choices=READOUTS, default="auto")
    p.add_argument("--alpha", type=float, default=None, help="also report the gate: share acted and error among acted")
    p.add_argument("--include-seen", action="store_true", help="keep rows a probe was trained on")
    p.add_argument("--json", help="also write the results as JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("truncate", help="keep the first N transformer blocks of a GGUF (needs the gguf package)")
    p.add_argument("src", help="input .gguf")
    p.add_argument("n", type=int, help="blocks to keep")
    p.add_argument("dst", help="output .gguf")
    p.set_defaults(func=cmd_truncate)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    logging.basicConfig(level=logging.WARNING, format="tez: %(message)s")
    try:
        return int(args.func(args) or 0)
    except TezError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
