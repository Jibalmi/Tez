"""Command line: tez serve | decide | fit | suggest | eval | presets | truncate.

Settings can come from the environment (containers, services): TEZ_BACKEND, TEZ_TEMPLATE, TEZ_EMBED_BACKEND for every
command; for tez serve also TEZ_HOST, TEZ_PORT, TEZ_SCHEMAS, TEZ_DATA_DIR, TEZ_API_KEY, TEZ_LOG_LEVEL, TEZ_CORS_ORIGINS,
TEZ_PRESETS and TEZ_LAYOUT. Each can be read from a file instead, for Docker and Compose secrets: TEZ_API_KEY_FILE=
/run/secrets/tez_api_key (surrounding whitespace is stripped; setting both VAR and VAR_FILE is an error). A flag on the
command line wins over the environment.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from ._version import __version__
from .backends import DEFAULT_BACKEND, DEFAULT_N_PROBS
from .config import DEFAULT_CORS_ORIGINS, Env, Limits
from .engine import READOUTS
from .errors import TezError
from .prompt import TEMPLATES
from .schema import LAYOUTS

LOG_LEVELS = ("critical", "error", "warning", "info", "debug")
ENV_VARS = ("TEZ_BACKEND", "TEZ_TEMPLATE", "TEZ_EMBED_BACKEND", "TEZ_HOST", "TEZ_PORT", "TEZ_SCHEMAS", "TEZ_DATA_DIR",
            "TEZ_API_KEY", "TEZ_LOG_LEVEL", "TEZ_CORS_ORIGINS", "TEZ_PRESETS", "TEZ_LAYOUT")


def _port(text: Any) -> int:
    try:
        n = int(str(text).strip())
    except ValueError:
        n = 0
    if not 0 < n < 65536:
        raise argparse.ArgumentTypeError(f"a port number from 1 to 65535 (--port or TEZ_PORT), got {text!r}")
    return n


def _choice(name: str, options: tuple) -> Any:
    """An argparse type that checks a value (and an environment default, which argparse does not check) is one of
    the options."""
    def parse(text: Any) -> str:
        value = str(text).strip().lower()
        if value not in options:
            raise argparse.ArgumentTypeError(f"{name} must be one of {', '.join(options)}, got {text!r}")
        return value
    parse.__name__ = name
    return parse


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _backend_args(p: argparse.ArgumentParser, env: Env, embed: bool = True) -> None:
    p.add_argument("--backend", default=env.get("TEZ_BACKEND", DEFAULT_BACKEND),
                   help=f"llama-server URL, or 'fake' for the offline demo backend (env TEZ_BACKEND; default {DEFAULT_BACKEND})")
    p.add_argument("--template", default=env.get("TEZ_TEMPLATE", "gemma4"), type=_choice("template", tuple(sorted(TEMPLATES))),
                   help="prompt template of the model behind --backend: gemma4 or qwen3 (env TEZ_TEMPLATE; default gemma4)")
    if embed:
        p.add_argument("--embed-backend", default=env.get("TEZ_EMBED_BACKEND"),
                       help="separate llama-server for probe features, started with --embeddings --pooling last "
                            "(env TEZ_EMBED_BACKEND; default: --backend)")
        p.add_argument("--embed-template", default=None, choices=sorted(TEMPLATES),
                       help="prompt template of the embedding model (default: --template)")
    p.add_argument("--no-cache-prompt", action="store_true",
                   help="switch llama.cpp prompt caching off (hybrid Qwen3.5 GGUFs crash b11100 on partial prefix reuse)")
    p.add_argument("--model-name", default=None, help="model name used in responses and calibration (default: read from the server)")
    p.add_argument("--n-probs", type=int, default=DEFAULT_N_PROBS, metavar="N",
                   help=f"next-token log-probabilities a letter readout asks for; letters outside them get the floor "
                        f"(default {DEFAULT_N_PROBS})")


def _non_negative(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = -1
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a whole number (0 = no limit), got {text!r}")
    return n


def _layout_arg(p: argparse.ArgumentParser, what: str) -> None:
    p.add_argument("--layout", choices=LAYOUTS, default=None, help=what)


def _hook_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hook", action="append", default=[], metavar="MODULE:OBJECT",
                   help="load a hook (docs/HOOKS.md): a class is instantiated, a hook instance used as is, a factory "
                        "called (repeatable; they run in the order given)")
    p.add_argument("--decision-log", metavar="PATH",
                   help="append every decision to this JSONL file (rows tez fit reads once labels are filled in)")
    p.add_argument("--hook-errors", choices=["raise", "log"], default="raise",
                   help="a hook that raises fails the decision (raise, the default) or is logged and skipped (log)")


def _hooks(args: argparse.Namespace) -> list:
    from .hooks import DecisionLog, load_hook
    try:
        hooks = [load_hook(spec) for spec in getattr(args, "hook", None) or []]
    except (ValueError, TypeError) as exc:
        raise TezError(str(exc)) from exc
    if getattr(args, "decision_log", None):
        hooks.append(DecisionLog(args.decision_log))
    return hooks


def _make_tez(args: argparse.Namespace, schemas: Any = None, layout: str = "auto"):
    from .engine import Tez
    try:
        return Tez(backend=args.backend, template=args.template, schemas=schemas,
                   embed_backend=getattr(args, "embed_backend", None), embed_template=getattr(args, "embed_template", None),
                   cache_prompt=not args.no_cache_prompt, data_dir=getattr(args, "data_dir", None),
                   model_name=args.model_name, n_probs=args.n_probs, layout=layout, hooks=_hooks(args),
                   hooks_raise=getattr(args, "hook_errors", "raise") == "raise")
    except ValueError as exc:
        raise TezError(str(exc)) from exc


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

    from .server import OriginPolicy, create_app
    logging.getLogger("tez").setLevel(args.log_level.upper())
    try:
        origins = OriginPolicy(cors_origins(args.cors_origins))
    except ValueError as exc:
        raise TezError(str(exc)) from exc
    tez = _make_tez(args, schemas=args.schemas, layout=args.layout or "auto")
    if args.presets:
        tez.add_presets()
    limits = Limits(*(v or None for v in (args.max_body_bytes, args.max_questions, args.max_state_chars, args.max_batch)))
    app = create_app(tez, api_key=args.api_key, cors=not args.no_cors, limits=limits, cors_origins=origins.patterns)
    probes = sum(len(v) for v in tez.probe_index().values())
    print(f"tez {__version__} on http://{args.host}:{args.port}  backend {tez.backend.url} ({tez.template})"
          + (f", embeddings {tez.embedder.url} ({tez.embedder.template})" if tez.embedder is not tez.backend else "")
          + f", {len(tez.schemas)} schema(s), {probes} probe(s), layout {tez.layout}"
          + (f", hooks: {', '.join(type(h).__name__ for h in tez.hooks)}" if tez.hooks else "")
          + (", API key required" if args.api_key else "")
          + (", no CORS" if args.no_cors else f", browsers: {', '.join(origins.patterns) or 'none'}"),
          file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cors_origins(value: str | None) -> list[str]:
    """The --cors-origins / TEZ_CORS_ORIGINS list: comma- or space-separated origins."""
    return [p for p in str(value or "").replace(",", " ").split() if p]


def cmd_decide(args: argparse.Namespace) -> int:
    from . import presets
    from .schema import load_schema
    if not args.schema and not args.preset and not args.questions:
        raise TezError("give --schema FILE, --preset NAME and/or --questions FILE")
    if args.schema and args.preset:
        raise TezError("give --schema or --preset, not both")
    given = [x for x in (args.text, args.state, args.state_file, args.states_file) if x is not None]
    if len(given) != 1:
        raise TezError("give the state once: as TEXT, --state, --state-file or --states-file")
    schema = load_schema(args.schema) if args.schema else (presets.load(args.preset) if args.preset else None)
    questions = None
    if args.questions:
        try:
            raw = json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TezError(f"cannot read questions from {args.questions}: {exc}") from exc
        questions = raw["questions"] if isinstance(raw, dict) and isinstance(raw.get("questions"), dict) else raw
    tez = _make_tez(args, schemas=[schema] if schema else None)
    opts = dict(questions=questions, schema=schema.name if schema else None, readout=args.readout,
                abstain=args.abstain, alpha=args.alpha, layout=args.layout)
    if args.states_file is not None:
        return _decide_file(tez, args, opts)
    if args.out:
        raise TezError("--out goes with --states-file")
    state = args.text if args.text is not None else args.state if args.state is not None else _read_state(args.state_file)
    res = tez.decide(state, **opts)
    print(json.dumps(res, indent=None if args.compact else 2, ensure_ascii=False))
    return 0


def _decide_file(tez: Any, args: argparse.Namespace, opts: dict) -> int:
    """Every state of a JSONL file as one batch: one result line per input row, in order."""
    from .fit import read_states
    try:
        states = read_states(args.states_file)
    except OSError as exc:
        raise TezError(f"cannot read {args.states_file}: {exc}") from exc
    if not states:
        raise TezError(f"{args.states_file} holds no states")
    results = tez.decide_many(states, **opts)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    errors = sum(1 for r in results if "error" in r)
    print(f"decided {len(results) - errors} of {len(results)} states" + (f", {errors} failed" if errors else "")
          + (f" -> {args.out}" if args.out else ""), file=sys.stderr)
    return 1 if errors == len(results) else 0


def cmd_fit(args: argparse.Namespace) -> int:
    from .fit import fit
    from .schema import load_schema
    schema = load_schema(args.schema)
    tez = _make_tez(args, schemas=[schema])
    cal = fit(tez, schema, label_files=args.labels, feedback=not args.no_feedback, holdout=args.holdout,
              min_labels=args.min_labels, seed=args.seed, letters=not args.no_letters, use_blend=not args.no_blend,
              layout=args.layout)
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
    res = evaluate(tez, schema, args.labels, readout=args.readout, alpha=args.alpha, include_seen=args.include_seen,
                   layout=args.layout)
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


def cmd_presets(args: argparse.Namespace) -> int:
    from . import presets
    if args.show:
        sys.stdout.write(presets.text(args.show))
        return 0
    rows = [["preset", "questions", "description"]]
    for s in presets.load_all():
        kinds = ", ".join(f"{qid} ({q.type})" for qid, q in s.questions.items())
        rows.append([s.name, kinds, s.description])
    print(_table(rows))
    return 0


def cmd_truncate(args: argparse.Namespace) -> int:
    from .truncate import truncate_gguf
    info = truncate_gguf(args.src, args.n, args.dst)
    print(f"{info['src']}: {info['blocks'][0]} -> {info['blocks'][1]} blocks, "
          f"{info['tensors'][0]} -> {info['tensors'][1]} tensors -> {info['dst']}")
    return 0


# ---------------------------------------------------------------------------------------------- parser
def build_parser(environ: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """The command line. `environ` (default os.environ) supplies the TEZ_* defaults; raises TezError for a bad
    VAR_FILE."""
    env = Env(environ)
    parser = argparse.ArgumentParser(prog="tez", description="Tez: an open local System One decision engine.")
    parser.add_argument("--version", action="version", version=f"tez {__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p = sub.add_parser("serve", help="run the HTTP server (Jev-compatible /v1/systemone)")
    _backend_args(p, env)
    p.add_argument("--schemas", default=env.get("TEZ_SCHEMAS"),
                   help="directory of *.yaml schemas (trained artefacts in <dir>/.tez/) (env TEZ_SCHEMAS)")
    p.add_argument("--presets", action="store_true", default=env.flag("TEZ_PRESETS"),
                   help="also load the built-in presets (tez presets); a schema of the same name in --schemas wins "
                        "(env TEZ_PRESETS=1)")
    p.add_argument("--data-dir", default=env.get("TEZ_DATA_DIR"),
                   help="where POST /v1/feedback writes (env TEZ_DATA_DIR; default: <schemas>/.tez)")
    p.add_argument("--host", default=env.get("TEZ_HOST", "127.0.0.1"), help="address to bind (env TEZ_HOST; default 127.0.0.1)")
    p.add_argument("--port", type=_port, default=env.get("TEZ_PORT", "8787"), help="port (env TEZ_PORT; default 8787)")
    p.add_argument("--api-key", default=env.get("TEZ_API_KEY"),
                   help="require Authorization: Bearer <key> (env TEZ_API_KEY or TEZ_API_KEY_FILE)")
    p.add_argument("--cors-origins", default=env.get("TEZ_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)), metavar="LIST",
                   help="browser origins allowed by CORS, comma-separated: an origin, host:* for any port, "
                        "https://*.domain, null, or * for any origin (env TEZ_CORS_ORIGINS; default "
                        f"{','.join(DEFAULT_CORS_ORIGINS)}); POST /v1/feedback refuses other origins")
    p.add_argument("--no-cors", action="store_true", help="send no CORS headers (browsers on other origins cannot call it)")
    p.add_argument("--log-level", default=env.get("TEZ_LOG_LEVEL", "info"), type=_choice("log level", LOG_LEVELS),
                   help=f"{', '.join(LOG_LEVELS)} (env TEZ_LOG_LEVEL; default info)")
    p.add_argument("--layout", default=env.get("TEZ_LAYOUT"), type=_choice("layout", LAYOUTS),
                   help="default prompt layout for requests and schemas that name none: auto (state_first for 2+ "
                        "questions, question_first for one), question_first or state_first (env TEZ_LAYOUT; default auto)")
    _hook_args(p)
    lim = Limits()
    for flag, default, what in (("--max-body-bytes", lim.max_body_bytes, "request body size in bytes"),
                                ("--max-questions", lim.max_questions, "questions per request"),
                                ("--max-state-chars", lim.max_state_chars, "characters per state (objects: as JSON)"),
                                ("--max-batch", lim.max_batch, "states per batch request")):
        p.add_argument(flag, type=_non_negative, default=default, metavar="N",
                       help=f"limit on {what}, 413 past it (default {default:,}; 0 = no limit)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("decide", help="decide one state (or a file of states) and print the wire-format JSON")
    _backend_args(p, env)
    p.add_argument("text", nargs="?", help="the state as text (or use --state, --state-file or --states-file)")
    p.add_argument("--schema", help="schema YAML file (its fitted probes and calibration are used)")
    p.add_argument("--preset", metavar="NAME", help="a built-in preset schema (tez presets lists them)")
    p.add_argument("--questions", help="JSON file with Jev questions ({id: {type, instructions, criteria}})")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--state", help="the state as text")
    g.add_argument("--state-file", help="file with the state ('-' = stdin; *.json is parsed as JSON)")
    g.add_argument("--states-file", help="JSONL file of states (rows with a state, JSON values or text lines), decided "
                                         "as one batch: one result line per row, in order")
    p.add_argument("--out", help="with --states-file: write the result lines here instead of stdout")
    p.add_argument("--readout", choices=READOUTS, default="auto")
    p.add_argument("--abstain", action="store_true", help="add the implicit __none__ option to choice questions")
    p.add_argument("--alpha", type=float, default=None, help="gate: target error rate among acted decisions")
    p.add_argument("--compact", action="store_true", help="one-line JSON")
    _layout_arg(p, "prompt layout (default: the schema's, else auto)")
    _hook_args(p)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("fit", help="train probes, temperatures and gate thresholds for a schema")
    _backend_args(p, env)
    p.add_argument("--schema", required=True, help="schema YAML file; artefacts go to <dir>/.tez/<name>/")
    p.add_argument("--labels", action="append", default=[], metavar="FILE",
                   help="labelled JSONL rows {\"state\": ..., \"labels\": {question: label}} (repeatable)")
    p.add_argument("--data-dir", default=env.get("TEZ_DATA_DIR"),
                   help="where recorded feedback lives (env TEZ_DATA_DIR; default: <schema dir>/.tez)")
    p.add_argument("--no-feedback", action="store_true", help="ignore rows recorded through POST /v1/feedback")
    p.add_argument("--holdout", type=float, default=0.3, help="share of each question's labels held out for calibration")
    p.add_argument("--min-labels", type=int, default=20, help="fewest labels for which a probe is trained")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-letters", action="store_true", help="skip the letters readout (no letters calibration, no blend)")
    p.add_argument("--no-blend", action="store_true", help="serve the probe alone, without the zero-shot letters prior")
    _layout_arg(p, "prompt layout to fit under (default: the schema's layout when it names one, else question_first); "
                   "under auto a request reads each fitted question in the layout it was fitted under")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("suggest", help="pick the most typical unlabelled states to label first")
    _backend_args(p, env)
    p.add_argument("--schema", required=True)
    p.add_argument("--unlabelled", required=True, help="JSONL rows with a state (or one state per line)")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--question", help="cluster on one question's prompt (default: all questions averaged)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the picks here (JSONL with empty labels) instead of stdout")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("eval", help="accuracy and calibration (ECE) per question on labelled rows")
    _backend_args(p, env)
    p.add_argument("--schema", required=True)
    p.add_argument("--labels", action="append", required=True, metavar="FILE")
    p.add_argument("--readout", choices=READOUTS, default="auto")
    p.add_argument("--alpha", type=float, default=None, help="also report the gate: share acted and error among acted")
    p.add_argument("--include-seen", action="store_true", help="keep rows a probe was trained on")
    p.add_argument("--json", help="also write the results as JSON")
    _layout_arg(p, "prompt layout (default: the schema's, else auto for a request with all of its questions)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("presets", help="list the built-in preset schemas, or print one (--show NAME)")
    p.add_argument("--show", metavar="NAME", help="print this preset's YAML (redirect it into a schema file to fit it)")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("truncate", help="keep the first N transformer blocks of a GGUF (needs the gguf package)")
    p.add_argument("src", help="input .gguf")
    p.add_argument("n", type=int, help="blocks to keep")
    p.add_argument("dst", help="output .gguf")
    p.set_defaults(func=cmd_truncate)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    try:
        parser = build_parser()
    except TezError as exc:            # a TEZ_*_FILE that cannot be read
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
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
