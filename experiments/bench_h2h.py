"""Head-to-head on Laya's own public benchmarks, same rows, same machine.

Models
  tez        frozen Gemma 4 12B via llama-server, one-pass letter readout (A..Z), zero-shot.
             Constant material (instruction, options) first, the state LAST, so the prefix is
             cached and only the state is evaluated. >26 options => chunked tournament.
  laya-en    convaiinnovations/laya            (ModernBERT-large 421M, English)
  laya-ml    convaiinnovations/laya/multilingual (mmBERT-base 322M)
  laya-td    convaiinnovations/laya/typed-decisions (fine-tuned on that benchmark's train split)

Tasks (Laya's protocol where one is published: MASSIVE = 20 sampled options, seed 13)
  ag_news, emotion, banking77, sst5 (score), boolq (noul), prompt_injections (noul),
  massive:<lang>, xnli:<lang>, typed_decisions (400 cases / 2,000 decisions, 4 workflows)

Metrics: accuracy, macro-F1, ECE-15, Brier, NLL, mean confidence, acc@50% coverage (Laya's
set), plus 2-fold out-of-fold temperature refit (ECE after), option-order flip rate on a
subset, soft accuracy / score MAE on typed-decisions, and wall-clock ms per decision.

Usage
  py experiments/bench_h2h.py --model tez     --tasks all --out results/h2h
  py experiments/bench_h2h.py --model laya-en --tasks all --out results/h2h
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import string
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
LETTERS = string.ascii_uppercase
SEED = 13
TAIL = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
MASSIVE_LANGS = ["en", "de", "fr", "es", "ja", "zh-CN", "ar", "hi", "th", "ko", "km"]
XNLI_LANGS = ["en", "de", "fr", "es", "ar", "hi", "th", "zh", "ru", "tr"]

# ----------------------------------------------------------------------------- tasks
def humanise(label: str) -> str:
    return label.replace("_", " ").replace(".", ": ").replace("?", "").strip()


def build_task(name: str, n: int):
    """Return list of cases: dict(id, task, lang, state, qtype, instructions, options[(key, desc)], gold, extra)."""
    from datasets import load_dataset
    rng = random.Random(SEED)
    cases = []
    if name == "ag_news":
        d = load_dataset("ag_news", split="test")
        names = d.features["label"].names
        desc = {"World": "world news, politics, international affairs", "Sports": "sports", "Business": "business, economy, finance", "Sci/Tech": "science and technology"}
        idx = rng.sample(range(len(d)), n)
        for i in idx:
            r = d[i]
            cases.append(dict(id=f"ag_news-{i}", task=name, lang="en", state={"text": r["text"]}, qtype="choice",
                              instructions="What is the topic of `text`?", options=[(k, desc[k]) for k in names], gold=r["label"]))
    elif name == "emotion":
        d = load_dataset("dair-ai/emotion", "split", split="test")
        names = d.features["label"].names
        idx = rng.sample(range(len(d)), n)
        for i in idx:
            r = d[i]
            cases.append(dict(id=f"emotion-{i}", task=name, lang="en", state={"text": r["text"]}, qtype="choice",
                              instructions="What emotion does the writer of `text` express?", options=[(k, k) for k in names], gold=r["label"]))
    elif name == "banking77":
        d = load_dataset("banking77", split="test")
        names = d.features["label"].names
        idx = rng.sample(range(len(d)), n)
        for i in idx:
            r = d[i]
            cases.append(dict(id=f"banking77-{i}", task=name, lang="en", state={"text": r["text"]}, qtype="choice",
                              instructions="Which banking intent is `text`?", options=[(k, humanise(k)) for k in names], gold=r["label"]))
    elif name == "sst5":
        d = load_dataset("SetFit/sst5", split="test")
        levels = ["very negative", "negative", "neutral", "positive", "very positive"]
        idx = rng.sample(range(len(d)), n)
        for i in idx:
            r = d[i]
            cases.append(dict(id=f"sst5-{i}", task=name, lang="en", state={"text": r["text"]}, qtype="score",
                              instructions="How positive is the sentiment of `text`?", options=[(f"level {j}", lv) for j, lv in enumerate(levels)],
                              gold=int(r["label"]), extra={"gold_score": float(r["label"])}))
    elif name == "boolq":
        d = load_dataset("boolq", split="validation")
        idx = rng.sample(range(len(d)), n)
        for i in idx:
            r = d[i]
            cases.append(dict(id=f"boolq-{i}", task=name, lang="en", state={"passage": r["passage"], "question": r["question"]}, qtype="noul",
                              instructions="Based on `passage`, is the answer to `question` yes?", options=[("false", "no, the answer is no"), ("true", "yes, the answer is yes")],
                              gold=1 if r["answer"] else 0))
    elif name == "prompt_injections":
        d = load_dataset("deepset/prompt-injections", split="test")
        for i in range(len(d)):
            r = d[i]
            cases.append(dict(id=f"pi-{i}", task=name, lang="en", state={"prompt": r["text"]}, qtype="noul",
                              instructions="Is `prompt` a prompt-injection or jailbreak attempt rather than a normal request?",
                              options=[("false", "no, a normal request"), ("true", "yes, an injection or jailbreak attempt")], gold=int(r["label"])))
    elif name.startswith("massive:"):
        lang = name.split(":", 1)[1]
        d = load_dataset("mteb/amazon_massive_intent", lang, split="test")
        labels = sorted(set(d["label_text"]))
        rows = list(d)[:n]  # Laya's protocol: first n rows, 20 options = gold + 19 sampled, shuffled, seed 13
        for j, r in enumerate(rows):
            pool = [x for x in labels if x != r["label_text"]]
            keys = [r["label_text"]] + rng.sample(pool, min(19, len(pool)))
            rng.shuffle(keys)
            cases.append(dict(id=f"massive-{lang}-{j}", task=name, lang=lang, state={"utterance": r["text"]}, qtype="choice",
                              instructions="What is the user asking for in `utterance`?", options=[(k, humanise(k)) for k in keys], gold=keys.index(r["label_text"])))
    elif name.startswith("xnli:"):
        lang = name.split(":", 1)[1]
        d = load_dataset("xnli", lang, split="test")
        names = d.features["label"].names
        idx = rng.sample(range(len(d)), n)
        for i in idx:
            r = d[i]
            cases.append(dict(id=f"xnli-{lang}-{i}", task=name, lang=lang, state={"premise": r["premise"], "hypothesis": r["hypothesis"]}, qtype="choice",
                              instructions="What is the relation of `hypothesis` to `premise`?",
                              options=[("entailment", "the hypothesis follows from the premise"), ("neutral", "the hypothesis may or may not be true given the premise"), ("contradiction", "the hypothesis contradicts the premise")],
                              gold=r["label"]))
    elif name == "typed_decisions":
        d = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
        for r in d:
            qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
            try:
                st = json.loads(r["state"])
            except Exception:  # noqa: BLE001
                st = r["state"]
            for qid, qd in qs.items():
                g = gold[qid]
                if qd["type"] == "choice":
                    keys = list(qd["criteria"].keys())
                    opts = [(k, qd["criteria"][k] if isinstance(qd["criteria"][k], str) else json.dumps(qd["criteria"][k])) for k in keys]
                    gi = keys.index(str(g["label"])); soft = [float(g.get("probabilities", {}).get(k, 0.0)) for k in keys]; extra = {"soft": soft}
                elif qd["type"] == "noul":
                    opts = [("false", "no, the statement does not hold"), ("true", "yes, the statement holds")]
                    pt = float(g.get("probabilities", {}).get("true", g.get("noul", 0.5)))
                    gi = 1 if str(g["label"]).lower() == "true" else 0; extra = {"soft": [1 - pt, pt]}
                else:
                    crit = qd["criteria"]
                    opts = [(f"level {i}", c if isinstance(c, str) else json.dumps(c)) for i, c in enumerate(crit)]
                    gi = int(g["label"]); extra = {"soft": [float(g.get("probabilities", {}).get(str(i), 0.0)) for i in range(len(crit))], "gold_score": float(g.get("score", g["label"]))}
                cases.append(dict(id=f"td-{r['id']}-{qid}", task=name, lang="en", state=st, qtype=qd["type"], instructions=qd["instructions"],
                                  options=opts, gold=gi, extra=extra, workflow=r["workflow"], laya_q=qd))
    else:
        raise ValueError(name)
    return cases


# ----------------------------------------------------------------------------- tez scorer
SESSION = requests.Session()


def tez_prompt(case, options):
    head = ("You are a decision engine. Read the question and the options, then look at the input and answer "
            "with the single letter of the best option. Answer with the letter only.")
    q = f"Question ({case['qtype']}): {case['instructions']}"
    opts = "Options:\n" + "\n".join(f"{LETTERS[i]}. {k}: {d}" if d and d != k else f"{LETTERS[i]}. {k}" for i, (k, d) in enumerate(options))
    st = json.dumps(case["state"], ensure_ascii=False) if not isinstance(case["state"], str) else case["state"]
    user = f"{head}\n\n{q}\n\n{opts}\n\nInput:\n{st}"
    tmpl = os.environ.get("TEZ_TEMPLATE", "gemma4")
    if tmpl == "qwen3":   # Qwen3 / Qwen3.5 chat format, thinking disabled (empty think block), letter read at the next position
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<|turn>user\n{user}{TAIL}"


def tez_score_letters(server, prompt, k, n_probs=200):
    if os.environ.get("TEZ_BACKEND", "llamacpp") != "llamacpp":   # e.g. TEZ_BACKEND=ollama when llama-server is unavailable
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("backend", ROOT / "experiments" / "backend.py"); _b = _iu.module_from_spec(_s); _s.loader.exec_module(_b)  # type: ignore[union-attr]
        return _b.score_letters(prompt, k, n_probs)
    body = {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0, "samplers": [],
            "cache_prompt": os.environ.get("TEZ_CACHE_PROMPT", "1") != "0"}   # qwen35 GGUFs crash llama.cpp b11100 on partial prefix reuse
    data = SESSION.post(f"{server}/completion", json=body, timeout=300).json()
    if "completion_probabilities" not in data:   # the first greedy token was end-of-sequence (depth-pruned GGUFs): suppress it and re-read
        data = SESSION.post(f"{server}/completion", json=dict(body, ignore_eos=True), timeout=300).json()
    tops = data["completion_probabilities"][0]["top_logprobs"]
    lp = {}
    for t in tops:
        lp.setdefault(t["token"], t["logprob"])
    floor = min(t["logprob"] for t in tops) - 2.0
    z = np.array([lp.get(LETTERS[i], floor) for i in range(k)], dtype=float)
    p = np.exp(z - z.max()); p /= p.sum()
    return p, z, data.get("timings", {}).get("prompt_n")


def tez_decide(server, case, options=None):
    """Return probabilities over `options` (default case['options']); chunked tournament if > 26."""
    options = options if options is not None else case["options"]
    if len(options) <= 26:
        p, z, pn = tez_score_letters(server, tez_prompt(case, options), len(options))
        return p, z, pn, 1
    # tournament: chunks of <=20 with 'none of these' appended; winners meet in a final round
    n_calls, winners, wprob = 0, [], []
    for s in range(0, len(options), 20):
        chunk = options[s:s + 20]
        p, _, _ = tez_score_letters(server, tez_prompt(case, chunk + [("none", "none of the options above fits")]), len(chunk) + 1)
        n_calls += 1
        j = int(np.argmax(p[:-1]))
        winners.append(s + j); wprob.append(float(p[j]))
    order = np.argsort(wprob)[::-1][:20]
    finalists = [winners[i] for i in order]
    fopts = [options[i] for i in finalists]
    p, z, pn = tez_score_letters(server, tez_prompt(case, fopts), len(fopts)); n_calls += 1
    full = np.zeros(len(options)); full[finalists] = p
    zf = np.full(len(options), z.min() - 5.0); zf[finalists] = z
    return full, zf, pn, n_calls


# ----------------------------------------------------------------------------- laya scorer
def laya_question(case):
    if "laya_q" in case:
        return case["laya_q"]
    if case["qtype"] == "choice":
        return {"type": "choice", "instructions": case["instructions"], "criteria": {k: d for k, d in case["options"]}}
    if case["qtype"] == "score":
        return {"type": "score", "instructions": case["instructions"], "criteria": [d for _, d in case["options"]]}
    return {"type": "noul", "instructions": case["instructions"]}


def laya_decide(agent, case):
    q = laya_question(case)
    res = agent.predict(case["state"], {"q": q})["answers"]["q"]
    k = len(case["options"])
    if case["qtype"] == "choice":
        keys = list(q["criteria"].keys())
        p = np.array([res["probabilities"][kk] for kk in keys], dtype=float)
    elif case["qtype"] == "score":
        p = np.array([res["probabilities"][str(i)] for i in range(k)], dtype=float)
    else:
        p = np.array([1 - res["noul"], res["noul"]], dtype=float)
    p = np.clip(p, 1e-9, None); p /= p.sum()
    return p, np.log(p), None, 1


# ----------------------------------------------------------------------------- metrics
def ece15(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def macro_f1(g, p):
    g, p = np.asarray(g), np.asarray(p)
    f = []
    for c in sorted(set(g.tolist()) | set(p.tolist())):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum()); fn = int(((p != c) & (g == c)).sum())
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def softmax_t(z, t):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max()); return e / e.sum()


def fit_temperature(rows):
    """rows: list of (gold, logits). Grid-search T minimising NLL."""
    best, bt = 1e9, 1.0
    for t in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        nll = -np.mean([math.log(max(softmax_t(z, t)[g], 1e-12)) for g, z in rows])
        if nll < best:
            best, bt = nll, float(t)
    return bt


def summarise(recs):
    g = np.array([r["gold"] for r in recs]); P = [np.asarray(r["probabilities"]) for r in recs]
    pred = np.array([int(np.argmax(p)) for p in P]); conf = np.array([float(np.max(p)) for p in P]); corr = (pred == g).astype(float)
    out = dict(n=len(recs), accuracy=float(corr.mean()), macro_f1=macro_f1(g, pred), ece=ece15(conf, corr),
               brier=float(np.mean([((p - np.eye(len(p))[gg]) ** 2).sum() for p, gg in zip(P, g)])),
               nll=float(np.mean([-math.log(max(float(p[gg]), 1e-12)) for p, gg in zip(P, g)])),
               mean_confidence=float(conf.mean()),
               acc_at_50_coverage=float(corr[np.argsort(-conf)[: max(1, len(conf) // 2)]].mean()),
               ms_p50=float(np.median([r["ms"] for r in recs])), ms_p95=float(np.percentile([r["ms"] for r in recs], 95)))
    # 2-fold OOF temperature refit on logits
    rows = [(r["gold"], np.asarray(r["logits"])) for r in recs]
    if len(rows) >= 40:
        conf2, corr2, nll2 = [], [], []
        for fold in (0, 1):
            fit = [rw for i, rw in enumerate(rows) if i % 2 == fold]; ev = [rw for i, rw in enumerate(rows) if i % 2 != fold]
            T = fit_temperature(fit)
            for gg, z in ev:
                p = softmax_t(z, T); conf2.append(float(p.max())); corr2.append(float(np.argmax(p) == gg)); nll2.append(-math.log(max(p[gg], 1e-12)))
        out.update(ece_refit=ece15(conf2, corr2), nll_refit=float(np.mean(nll2)))
    soft = [r for r in recs if r.get("extra", {}).get("soft")]
    if soft:
        sa, br = [], []
        for r in soft:
            gp = np.asarray(r["extra"]["soft"], float)
            if gp.sum() > 0:
                gp = gp / gp.sum(); pp = np.asarray(r["probabilities"]); sa.append(float((pp * gp).sum())); br.append(float(((pp - gp) ** 2).sum()))
        out.update(soft_accuracy=float(np.mean(sa)), brier_vs_soft=float(np.mean(br)))
    sc = [r for r in recs if "gold_score" in r.get("extra", {})]
    if sc:
        mae = [abs(float((np.arange(len(r["probabilities"])) * np.asarray(r["probabilities"])).sum()) - r["extra"]["gold_score"]) for r in sc]
        out.update(score_mae=float(np.mean(mae)), within_1_level=float(np.mean([m <= 1 for m in mae])))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tez", "laya-en", "laya-ml", "laya-td"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-lang", type=int, default=100)
    ap.add_argument("--flip-n", type=int, default=100, help="rows re-scored with reversed options for the flip rate")
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--out", default="results/h2h")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks != "all" else (
        ["ag_news", "emotion", "banking77", "sst5", "boolq", "prompt_injections", "typed_decisions"]
        + [f"massive:{l}" for l in MASSIVE_LANGS] + [f"xnli:{l}" for l in XNLI_LANGS])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    if args.model == "tez":
        props = SESSION.get(f"{args.server}/props", timeout=30).json()
        model_desc = props.get("model_path")
        decide = lambda case, options=None: tez_decide(args.server, case, options)  # noqa: E731
    else:
        import laya
        sub = {"laya-en": None, "laya-ml": "multilingual", "laya-td": "typed-decisions"}[args.model]
        agent = laya.load("convaiinnovations/laya", subfolder=sub, device=args.device)
        model_desc = f"convaiinnovations/laya/{sub or 'root'} on {agent.device} {agent.dtype}"
        # warm
        agent.predict({"text": "hello"}, {"q": {"type": "choice", "instructions": "topic?", "criteria": {"a": "x", "b": "y"}}})
        decide = lambda case, options=None: laya_decide(agent, case if options is None else dict(case, options=options))  # noqa: E731

    for task in tasks:
        n = args.n_lang if (":" in task) else args.n
        try:
            cases = build_task(task, n)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {task}: {exc}"); continue
        recs, t0 = [], time.perf_counter()
        for c in cases:
            t = time.perf_counter()
            try:
                p, z, pn, calls = decide(c)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERR {c['id']}: {str(exc)[:120]}"); continue
            ms = (time.perf_counter() - t) * 1000
            recs.append(dict(id=c["id"], task=task, lang=c["lang"], qtype=c["qtype"], gold=c["gold"], k=len(c["options"]),
                             probabilities=[float(x) for x in p], logits=[float(x) for x in z], pred=int(np.argmax(p)), ms=ms,
                             prompt_n=pn, calls=calls, extra=c.get("extra", {}), workflow=c.get("workflow")))
        wall = time.perf_counter() - t0
        # option-order flip rate on a subset (reversed options), only when the letters fit
        flips, fl_n = 0, 0
        for c, r in list(zip(cases, recs))[: args.flip_n]:
            if len(c["options"]) > 26 or c["qtype"] != "choice":   # noul/score orders are semantic, not a bias test
                break
            try:
                p2, _, _, _ = decide(c, c["options"][::-1])
            except Exception:  # noqa: BLE001
                continue
            fl_n += 1
            flips += int(np.argmax(p2[::-1])) != r["pred"]
        s = summarise(recs) if recs else {"n": 0}
        s.update(wall_s=wall, flip_rate=(flips / fl_n) if fl_n else None, flip_n=fl_n, model=model_desc)
        if task == "typed_decisions" and recs:
            s["by_workflow"] = {w: summarise([r for r in recs if r["workflow"] == w]) for w in sorted({r["workflow"] for r in recs})}
            s["by_qtype"] = {q: summarise([r for r in recs if r["qtype"] == q]) for q in sorted({r["qtype"] for r in recs})}
        summary.setdefault(task, {})[args.model] = s
        (out / f"rows_{task.replace(':', '_')}_{args.model}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{task:22s} {args.model:8s} n={s.get('n')} acc={s.get('accuracy', 0):.3f} f1={s.get('macro_f1', 0):.3f} ece={s.get('ece', 0):.3f}"
              f" refit={s.get('ece_refit', float('nan')):.3f} brier={s.get('brier', 0):.3f} conf={s.get('mean_confidence', 0):.3f}"
              f" flip={s.get('flip_rate')} ms={s.get('ms_p50', 0):.0f} ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
