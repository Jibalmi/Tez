"""Pilot day 4: every arm on the held-out suites, scored beside the 12B and Laya on the same rows.

    python experiments/small_model/eval_arms.py --arm a-base-30k
    python experiments/small_model/eval_arms.py --arm c-qwen08-30k

Suites (plan §6 and G1): typed-decisions test (400 states, 2,000 decisions; the student reads each state's 5
questions packed in one row, and again one question per row for the packing check), Laya's seven application
workflows (the three held-out ones gate G1: model routing, jailbreak, toxicity), MASSIVE intent in 11 languages (20
options; km unseen in training), SemIf authored144 and perturbations108 (mean family balanced accuracy, SemIf's own
evaluator). MASSIVE is also read with the options reversed (order flip at 20 options).

Probabilities are reported as read (T = 1; the rows files hold these) and with the student's per-type temperatures
fitted on the teacher-labelled validation split ("cal"; no evaluation labels involved). Rows: results/h2h/rows_<task>_tez-s-<arm>.jsonl (the plan's
naming), metrics: results/pilot/eval_<arm>.json, with the weights' SHA-256 in the manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from collections import defaultdict

import numpy as np
import torch

from common import MASSIVE11, RESULTS, ROOT, RUNS, THERMAL, dump, file_sha256, read_jsonl, thermal_guard, write_jsonl

APPS = ["model_routing_domain", "guardrails_jailbreak", "moderation_toxicity", "email_spam", "phishing",
        "rag_relevance", "support_triage"]
HELD_OUT_APPS = APPS[:3]


# ------------------------------------------------------------------------------------------------ suites
def suites() -> dict[str, list[dict]]:
    """task -> cases: {id, state, qtype, instructions, options [(key, desc)], gold, extra, workflow, group}"""
    snapshot = os.environ.get("TEZ_PILOT_EVAL_SUITES")
    if snapshot:
        from pathlib import Path
        return json.loads(Path(snapshot).read_text(encoding="utf-8"))
    import bench_h2h
    from tez.schema import parse_question, state_key
    out = {}
    td = bench_h2h.build_task("typed_decisions", 400)
    for c in td:
        c["group"] = state_key(c["state"])
    out["typed_decisions"] = td
    for task in APPS:
        cs = []
        for r in read_jsonl(ROOT / "results" / "vs_laya" / "apps_cases.jsonl"):
            if r["task"] != task:
                continue
            q = parse_question(r["qid"], r["question"])
            cs.append({"id": r["id"], "state": r["state"], "qtype": q.type, "instructions": q.instructions_text,
                       "options": q.options(), "gold": r["gold"], "extra": {}, "lang": "en"})
        out[f"app_{task}"] = cs
    for lang in MASSIVE11:
        out[f"massive_{lang}"] = bench_h2h.build_task(f"massive:{lang}", 100)
    for name in ("authored144", "perturbations108"):
        cs = []
        for r in read_jsonl(ROOT / "data" / "semif" / f"{name}.jsonl"):
            cs.append({"id": r["id"], "state": r["state"], "qtype": "choice", "instructions": r["question"],
                       "options": [(o["id"], o["description"]) for o in r["options"]], "gold": r["label"],
                       "extra": {}, "lang": "en", "family": r["family"]})
        out[f"semif_{name}"] = cs
    return out


# ------------------------------------------------------------------------------------------------ predictors
class PredictA:
    def __init__(self, arm: str, device: str = "cuda"):
        from transformers import AutoTokenizer
        from student import Encoder, Student
        d = RUNS / "models" / arm
        self.info = json.loads((d / "student.json").read_text(encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(d / "encoder")
        self.enc = Encoder(self.tok, 8192)
        self.model = Student(str(d / "encoder")).to(device).eval()
        self.model.head.load_state_dict(torch.load(d / "head.pt", map_location=device))
        self.device = device
        self.weights = {"encoder": file_sha256(d / "encoder" / "model.safetensors"), "head": file_sha256(d / "head.pt")}

    @torch.no_grad()
    def groups(self, groups: list[list[dict]], budget: int = 32768) -> list[list[np.ndarray]]:
        """Each group = cases about one state, read together in one row. Returns log-probs per case."""
        from student import Q, collate
        rows = [self.enc.encode(g[0]["state"], [Q(c["qtype"], c["instructions"], list(c["options"])) for c in g])
                for g in groups]
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids))
        out: list = [None] * len(rows)
        i = 0
        while i < len(order):
            j, mx = i, 0
            while j < len(order) and max(mx, len(rows[order[j]].ids)) * (j - i + 1) <= budget:
                mx = max(mx, len(rows[order[j]].ids))
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            if self.device == "cuda":
                thermal_guard()
            batch = collate([rows[k] for k in idx], self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = self.model(batch).float().cpu().numpy()
            q = 0
            for k in idx:
                res = []
                for c in groups[k]:
                    res.append(lp[q, : len(c["options"])])
                    q += 1
                out[k] = res
            i = j
        return out


class PredictC:
    def __init__(self, arm: str, device: str = "cuda", merge: bool = False, disable_cache: bool = False):
        from peft import PeftModel
        from train_c import load_base
        d = RUNS / "models" / arm
        self.info = json.loads((d / "student.json").read_text(encoding="utf-8"))
        tok, base, self.letter_ids = load_base(device=device)
        self.tok = tok
        self.tok.padding_side = "left"
        self.model = PeftModel.from_pretrained(base, d / "adapter").to(device).eval()
        if merge:
            self.model = self.model.merge_and_unload(safe_merge=True).eval()
        if merge or disable_cache:
            self.model.config.use_cache = False
        self.info = {**self.info, "lora_merged": merge, "cache_disabled": merge or disable_cache}
        self.batch_size = 16
        self.device = device
        self.weights = {"adapter": file_sha256(d / "adapter" / "adapter_model.safetensors")}

    @torch.no_grad()
    def groups(self, groups: list[list[dict]], batch: int | None = None) -> list[list[np.ndarray]]:
        from train_c import letter_logp, prompt_for
        batch = batch if batch is not None else getattr(self, "batch_size", 16)
        flat = [(gi, c) for gi, g in enumerate(groups) for c in g]
        prompts = [prompt_for(c["qtype"], c["instructions"], list(c["options"]), c["state"]) for _, c in flat]
        order = sorted(range(len(flat)), key=lambda i: len(prompts[i]))
        res: list = [None] * len(flat)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            thermal_guard()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, _ = letter_logp(self.model, self.tok, self.letter_ids, [prompts[i] for i in idx],
                                    [len(flat[i][1]["options"]) for i in idx], self.device, max_len=4096)
            lp = lp.float().cpu().numpy()
            for r, i in zip(lp, idx):
                res[i] = r[: len(flat[i][1]["options"])]
        out = [[] for _ in groups]
        for (gi, _), r in zip(flat, res):
            out[gi].append(r)
        return out


class PredictGLiClass:
    """Baseline: knowledgator/gliclass-x-base (mdeberta-v3-base, Apache-2.0), zero-shot, off the shelf. One question
    per call: the rendered state is the text, the question's instructions the task prompt, each option's text
    ("label: description", as the student sees it) a label; single-label softmax over the labels."""

    def __init__(self, arm: str = "gliclass-x-base", device: str = "cuda"):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from competitor_models import model_revision
        path = snapshot_download("knowledgator/" + arm, revision=model_revision("knowledgator/" + arm))
        model = GLiClassModel.from_pretrained(path)
        tok = AutoTokenizer.from_pretrained(path, **({"add_prefix_space": True} if arm == "gliclass-x-base" else {}))
        self.pipe = ZeroShotClassificationPipeline(model, tok, classification_type="single-label", device=f"{device}:0",
                                                   progress_bar=False).pipe
        self.info = {"temperatures": {}, "model": "knowledgator/" + arm, "revision": Path(path).name}
        self.weights = {"model": file_sha256(f"{path}/model.safetensors")}

    @torch.no_grad()
    def groups(self, groups: list[list[dict]], batch: int = 16) -> list[list[np.ndarray]]:
        from student import option_text, render_state
        flat = [(gi, c) for gi, g in enumerate(groups) for c in g]
        res: list = [None] * len(flat)
        for s in range(0, len(flat), batch):
            chunk = flat[s:s + batch]
            thermal_guard()
            texts = [render_state(c["state"]) for _, c in chunk]
            labels = [[option_text(k, d) for k, d in c["options"]] for _, c in chunk]
            # duplicate option texts would collapse in the pipeline's label handling: make them unique
            labels = [[f"{t} ({i})" if lab.count(t) > 1 else t for i, t in enumerate(lab)] for lab in labels]
            prompts = [c["instructions"] for _, c in chunk]
            inputs = self.pipe.prepare_inputs(texts, labels, same_labels=False, prompt=prompts)
            out = self.pipe.model(**inputs, max_num_classes=self.pipe._resolve_max_num_classes(labels, False))
            logits = out.logits.float()
            for j, (_, c) in enumerate(chunk):
                z = logits[j, : len(c["options"])]
                res[s + j] = torch.log_softmax(z, -1).cpu().numpy()
        out_g = [[] for _ in groups]
        for (gi, _), r in zip(flat, res):
            out_g[gi].append(r)
        return out_g


# ------------------------------------------------------------------------------------------------ metrics
def temp_of(temps: dict, qtype: str, k: int) -> float:
    if qtype == "choice":
        return temps.get("choice_many" if k > 10 else "choice", 1.0)
    return temps.get(qtype, 1.0)


def softmax(z, t=1.0):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar on paired correctness vectors."""
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p": 1.0}
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b01": b01, "b10": b10, "p": min(1.0, 2 * p)}


def load_ref_rows(task: str) -> dict[str, dict[str, dict]]:
    """System -> id -> row, for the 12B and Laya on the same cases (existing rows only)."""
    refs: dict[str, dict[str, dict]] = {}
    h2h = ROOT / "results" / "h2h"
    if task == "typed_decisions" or task.startswith("massive_"):
        for sysname, suffix in (("tez-12b", "tez"), ("laya-en", "laya-en"), ("laya-ml", "laya-ml"), ("laya-td", "laya-td")):
            p = h2h / f"rows_{task}_{suffix}.jsonl"
            if p.exists():
                refs[sysname] = {r["id"]: r for r in read_jsonl(p)}
    elif task.startswith("app_"):
        t = task[4:]
        for sysname, suffix in (("tez-12b", "tez"), ("laya-en", "laya"), ("laya-ml", "laya-multilingual"),
                                ("laya-td", "laya-typed-decisions")):
            p = ROOT / "results" / "vs_laya" / "apps_rows" / f"{t}_{suffix}.jsonl"
            if p.exists():
                refs[sysname] = {r["id"]: r for r in read_jsonl(p)}
    elif task.startswith("semif_"):
        name = {"semif_authored144": "authored144", "semif_perturbations108": "perturb108"}[task]
        p = ROOT / "results" / f"{name}_gemma4-12b-q8_0.jsonl"
        if p.exists():
            refs["tez-12b"] = {r["id"]: r for r in read_jsonl(p)}
    return refs


def summarise(recs: list[dict]) -> dict:
    import bench_h2h
    s = bench_h2h.summarise(recs)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("ms_p50", "ms_p95")}


def semif_score(task: str, recs: list[dict]) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("semif_eval", ROOT / "data" / "semif" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)  # type: ignore[union-attr]
    name = task.split("_", 1)[1]
    gold = ev.read_jsonl(ROOT / "data" / "semif" / f"{name}.jsonl")
    preds = [{"id": r["id"], "probabilities": [float(x) for x in np.asarray(r["probabilities"]) / np.sum(r["probabilities"])]}
             for r in recs]
    m = ev.evaluate(gold, preds)
    rows = ev.align(gold, preds)
    return {"mean_family_balanced_accuracy": round(m["mean_family_balanced_accuracy"], 4),
            "accuracy": round(ev.basic(rows)["accuracy"], 4)}


def score_task(task: str, cases: list[dict], lps: list[np.ndarray], temps: dict, arm: str) -> tuple[dict, list[dict]]:
    recs, recs_cal = [], []
    for c, lp in zip(cases, lps):
        p1 = softmax(lp)
        pc = softmax(lp, temp_of(temps, c["qtype"], len(c["options"])))
        base = {"id": c["id"], "task": task, "lang": c.get("lang"), "qtype": c["qtype"], "gold": c["gold"],
                "k": len(c["options"]), "extra": c.get("extra", {}), "workflow": c.get("workflow"), "ms": 0.0,
                "calls": 1, "prompt_n": None}
        recs.append({**base, "probabilities": p1.tolist(), "logits": [float(x) for x in lp], "pred": int(np.argmax(p1))})
        recs_cal.append({**base, "probabilities": pc.tolist(), "logits": [float(x) for x in lp], "pred": int(np.argmax(pc))})
    res = {"n": len(recs), "t1": summarise(recs), "cal": summarise(recs_cal)}
    if task.startswith("semif_"):
        res["semif"] = semif_score(task, recs)
    refs = load_ref_rows(task)
    ids = [r["id"] for r in recs]
    mine = np.array([int(r["pred"] == r["gold"]) for r in recs])
    comp = {}
    for sysname, rows in refs.items():
        if not all(i in rows for i in ids):
            comp[sysname] = {"note": f"only {sum(i in rows for i in ids)} of {len(ids)} ids present"}
            continue
        other = np.array([int(rows[i].get("pred", int(np.argmax(rows[i]["probabilities"]))) == rows[i]["gold"])
                          for i in ids])
        agree = np.mean([int(rows[i].get("pred", int(np.argmax(rows[i]["probabilities"])))) == r["pred"]
                         for i, r in zip(ids, recs)])
        comp[sysname] = {"accuracy": round(float(other.mean()), 4), "agreement_with_student": round(float(agree), 4),
                         "mcnemar_vs_student": mcnemar_exact(other, mine)}
        if sysname == "tez-12b":
            res["retention_vs_12b"] = round(float(mine.mean() / max(other.mean(), 1e-9)), 4)
            res["acc_where_12b_wrong"] = round(float(mine[other == 0].mean()), 4) if (other == 0).any() else None
            res["acc_where_12b_right"] = round(float(mine[other == 1].mean()), 4) if (other == 1).any() else None
            if task.startswith("semif_"):
                res["semif_12b"] = semif_score(task, [{"id": i, "probabilities": rows[i]["probabilities"]} for i in ids])
    res["same_rows"] = comp
    return res, recs          # rows as read (T = 1), like the 12B's rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tasks", default="all")
    args = ap.parse_args()
    t0 = time.time()
    from competitor_models import predictor
    from eval_arms import PredictA as EncoderPredictor
    pred = predictor(args.arm)
    temps = pred.info.get("temperatures", {})
    S = suites()
    if args.tasks != "all":
        S = {k: v for k, v in S.items() if k in args.tasks.split(",")}
    out = {"arm": args.arm, "weights_sha256": pred.weights, "temperatures": temps, "model_info": pred.info, "tasks": {}}
    progress_path = RESULTS / f"progress_eval_{args.arm}.json"
    dump(progress_path, {**out, "status": "running", "expected_tasks": list(S)})
    for task, cases in S.items():
        t1 = time.time()
        if task == "typed_decisions":
            groups = defaultdict(list)
            for c in cases:
                groups[c["group"]].append(c)
            glist = list(groups.values())
            lp_packed = pred.groups(glist)
            flat_cases = [c for g in glist for c in g]
            flat_lp = [x for g in lp_packed for x in g]
            res, recs = score_task(task, flat_cases, flat_lp, temps, args.arm)
            if isinstance(pred, EncoderPredictor):
                single = pred.groups([[c] for c in flat_cases])
                res1, _ = score_task(task, flat_cases, [g[0] for g in single], temps, args.arm)
                res["one_question_per_row"] = {"t1": res1["t1"], "retention_vs_12b": res1.get("retention_vs_12b")}
                res["packing_delta_points"] = round(100 * (res["t1"]["accuracy"] - res1["t1"]["accuracy"]), 2)
            by_wf = defaultdict(list)
            for r in recs:
                by_wf[r["workflow"]].append(r)
            res["by_workflow"] = {w: summarise(v) for w, v in sorted(by_wf.items())}
            by_q = defaultdict(list)
            for r in recs:
                by_q[r["qtype"]].append(r)
            res["by_qtype"] = {w: {"n": len(v), "accuracy": summarise(v)["accuracy"]} for w, v in sorted(by_q.items())}
        else:
            lps = [g[0] for g in pred.groups([[c] for c in cases])]
            res, recs = score_task(task, cases, lps, temps, args.arm)
            cases_ordered = cases
            if task.startswith("massive_"):
                rev = [dict(c, options=list(c["options"])[::-1]) for c in cases]
                lps_r = [g[0][::-1] for g in pred.groups([[c] for c in rev])]
                flips = np.mean([int(np.argmax(a) != np.argmax(b)) for a, b in zip(lps, lps_r)])
                res["order_flip_20"] = round(float(flips), 4)
            del cases_ordered
        res["seconds"] = round(time.time() - t1, 1)
        out["tasks"][task] = res
        rows_dir = RESULTS / "rows" if os.environ.get("TEZ_PILOT_RESULTS") else ROOT / "results" / "h2h"
        write_jsonl(rows_dir / f"rows_{task}_tez-s-{args.arm}.jsonl", recs)
        dump(progress_path, {**out, "status": "running", "expected_tasks": list(S)})
        print(f"{task:28s} acc {res['t1']['accuracy']:.4f} ece(t1) {res['t1']['ece']:.3f} ece(cal) {res['cal']['ece']:.3f}"
              f" retention {res.get('retention_vs_12b')} ({res['seconds']} s)", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["thermal"] = dict(THERMAL)
    dump(RESULTS / f"eval_{args.arm}.json", out)
    dump(progress_path, {**out, "status": "complete", "expected_tasks": list(S)})


if __name__ == "__main__":
    main()
