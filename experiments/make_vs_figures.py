"""Laya's published comparison charts, redrawn with Tez in the lead role (docs/figures/vs/).

    python experiments/make_vs_figures.py

Laya (github.com/NandhaKishorM/laya) publishes its comparison figures from research/scripts/make_plots.py.
This script draws the same charts with the same surface, ink, grid, typography, value labels, delta annotations
and panel structure, but Tez takes Laya's blue position, Jev stays the grey published reference and Laya's three
checkpoints become the comparators.

Reads only
  site/data/benchmarks.json      every number in BENCHMARKS.md, keyed by section
  BENCHMARKS.md                  the section 4b task-probe table (parsed below)
  results/h2h/summary.json       experiments/bench_h2h.py per-system, per-task summaries (same rows, same GPU)
  results/h2h/rows_*.jsonl       its per-decision rows (only when results/vs_laya/selective.json is missing)
  results/vs_laya/apps.json, massive51.json, speed_per_call.json, selective.json   (optional)
  README.md, tez/schema.py, docs/REPORT.md section 1   licence, option cap and Jev's documented terms

A panel whose data file is missing is not drawn (no placeholder numbers); every skip is printed.
Writes docs/figures/vs/*.png and docs/figures/vs/panels/*.png at 200 dpi.
"""
from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "vs"
PANEL_DIR = OUT / "panels"
DPI = 200

# ------------------------------------------------------------------ look: Laya's make_plots.py surface, ink and grid
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
REF = "#e4e3df"                      # Laya's light reference bars (teacher ceiling, majority class)
BELOW = "#b9b8b2"                    # a Tez bar below 3x random
# roles. Tez takes Laya's blue position (DESIGN.md signal-blue-fill for marks, signal-blue for text).
TEZ, TEZ_INK = "#3F6FF0", "#2F5BD8"
# Laya's checkpoints and Jev, validated as one set with the dataviz palette checker (all pairs, light surface):
# worst colour-blind separation between the four series colours dE 16.2, Jev's grey at least 9.2 from each,
# normal-vision floor >= 15. Amber and magenta sit below 3:1 on the surface, so every mark carries a value label.
LAYA_EN, LAYA_ML, LAYA_TD = "#e87ba4", "#eda100", "#008300"
LAYA_RT = "#d95926"                  # laya-routed (Laya's router over laya / laya-multilingual)
JEV = "#7a7975"
HATCH = "////"                       # hatched Tez = the probe readout (trained on labelled rows)
SHIPPED_ALPHA = 0.38                 # tinted = as shipped / before the fix; solid = after

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "hatch.linewidth": 1.5,
})

KEYS = ("tez", "laya", "laya-multilingual", "laya-typed-decisions")
LAYA3 = ("laya", "laya-multilingual", "laya-typed-decisions")
NAME = {"tez": "Tez", "tez-probe": "Tez probe", "laya": "laya", "laya-multilingual": "laya-multilingual",
        "laya-typed-decisions": "laya-typed-decisions", "laya-routed": "laya-routed", "jev": "Jev (published)"}
COLOR = {"tez": TEZ, "tez-probe": TEZ, "laya": LAYA_EN, "laya-multilingual": LAYA_ML,
         "laya-typed-decisions": LAYA_TD, "laya-routed": LAYA_RT, "jev": JEV}
H2H_KEY = {"tez": "tez", "laya": "laya-en", "laya-multilingual": "laya-ml", "laya-typed-decisions": "laya-td"}

CONDITIONS = ("Tez and every Laya checkpoint answered byte-identical rows on one RTX 5080 laptop GPU. "
              "Jev figures are third-party published; Jev was never run here.")
TEZ_LETTERS = "Tez = frozen Gemma 4 12B Q8_0 on llama.cpp b11100, one forward pass, zero-shot letter readout."
TEZ_PROBE = "Tez probe = logistic probe on a frozen Qwen3.5-4B's hidden state, trained on labelled rows."
APPS_CPU = ("Laya's application-workflow scores come from Laya's own harness on the same laptop's CPU (fp32), "
            "as in its published run.")

# Laya's own published results (github.com/NandhaKishorM/laya README.md and BENCHMARKS.md), never our rerun.
# Every chart that shows one says so on the chart.
LAYA_PUB = {"routed_usable_51": 45, "teacher_ceiling": 0.735, "majority_class": 0.461}
# Jev's documented terms, docs/REPORT.md section 1 ("What Jev is", from docs.typesafe.ai).
JEV_FACTS = {"weights": "closed; no self-host, no open weights", "cost": "$0.042 per 1M input tokens",
             "deployment": "hosted only, one US West Coast region", "options": "up to 255"}


# ================================================================== small helpers
def _json(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def canon(name):
    n = str(name).strip().lower()
    if n.startswith("tez"):
        return "tez-probe" if "probe" in n else "tez"
    for k, alias in {"laya": ("laya", "laya-en", "english", "laya (english)", "laya-english"),
                     "laya-multilingual": ("laya-multilingual", "laya-ml", "multilingual"),
                     "laya-typed-decisions": ("laya-typed-decisions", "laya-td", "typed-decisions"),
                     "laya-routed": ("laya-routed", "routed", "laya (routed)", "router", "laya-router"),
                     "jev": ("jev", "jev (published)")}.items():
        if n in alias:
            return k
    return n


def display(name):
    k = canon(name)
    if k in ("tez", "tez-probe"):
        s = str(name).strip()
        return "Tez" if s.lower() == "tez" else s[:1].upper() + s[1:]
    return NAME.get(k, str(name))


def f3(v):
    """A value as BENCHMARKS.md / benchmarks.json print it (three decimals)."""
    return f"{v:.3f}"


def fnum(v):
    """A value from results/vs_laya as the file gives it (up to 4 decimals, at least 3); never re-rounded to 3."""
    s = f"{float(v):.4f}"
    return s[:-1] if s.endswith("0") else s


def fms(v):
    """Milliseconds as the file gives them: integers as integers, otherwise one decimal."""
    v = float(v)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def fdelta(d):
    d = round(d, 6)
    return ("+" if d > 0 else "−" if d < 0 else "±") + f"{abs(d):.3f}"


def rng(s):
    a, b = (int(x) for x in re.split(r"[-–]", str(s)))
    return a, b


def join_and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ================================================================== data
class Data:
    def __init__(self):
        B = _json("site/data/benchmarks.json")
        S = _json("results/h2h/summary.json")
        if B is None or S is None:
            raise SystemExit("site/data/benchmarks.json and results/h2h/summary.json are required")
        self.B, self.S = B, S
        cols = ("tez", "laya", "laya-multilingual", "laya-typed-decisions", "jev")
        self.h2h = {}
        for r in B["h2h"]["rows"]:
            d = dict(zip(cols, r["values"]))
            d.update(n=r["n"], note=r.get("note"))
            self.h2h[r["task"]] = d
        # typed-decisions ladder (BENCHMARKS.md sections 1, 4b, 4c)
        td = self.h2h["typed-decisions"]
        probe_row = next(r for r in B["readouts_typed_decisions"]["rows"] if r["readout"].startswith("4B probe"))
        typical = next(s for s in B["few_labels"]["series"] if s["name"].startswith("typical rows + 12B"))
        self.td = dict(zero=td["tez"], four=B["typed_decisions_detail"]["tez_4shot"]["accuracy"],
                       labels50=typical["values"][B["few_labels"]["n"].index(50)], probe=probe_row["accuracy"],
                       probe_ece=probe_row["ece"], laya_td=td["laya-typed-decisions"], jev=td["jev"],
                       laya=td["laya"], laya_ml=td["laya-multilingual"])
        # calibration, order flip, latency (BENCHMARKS.md section 1 "Calibration, order robustness, latency")
        rows = {r["metric"]: r["values"] for r in B["calibration_order_latency"]["rows"]}
        sysc = ("tez", "laya", "laya-multilingual", "laya-typed-decisions", "jev")
        pick = lambda prefix: dict(zip(sysc, next(v for m, v in rows.items() if m.startswith(prefix))))  # noqa: E731
        self.ece_shipped, self.ece_temp = pick("mean ECE-15 as shipped"), pick("mean ECE-15 after")
        self.flip20, self.ms = pick("option-order flip"), {k: rng(v) for k, v in pick("ms per decision").items()}
        self.probe_ms = next(r["ms_probe"] for r in B["pruned"]["rows"] if r["blocks"].startswith("24"))
        vw = {r["variant"]: r for r in B["voice_per_word"]["rows"]}
        self.voice_after = next(v for k, v in vw.items() if "--swa-full (used)" in k)
        self.voice_before = next(v for k, v in vw.items() if "before --swa-full" in k)
        # languages (BENCHMARKS.md section 1 Languages, XNLI)
        L = B["languages"]
        by = {("probe-en" if s["system"].startswith("4B") else canon(s["system"].split(" (")[0])): s for s in L["series"]}
        self.langs11 = L["langs"]
        self.lang_acc = {k: dict(zip(L["langs"], by[k]["values"])) for k in KEYS}
        self.lang_macro = {k: by[k]["macro"] for k in KEYS}
        self.usable11 = {k: sum(v > 3 * 0.05 for v in self.lang_acc[k].values()) for k in KEYS}
        self.xnli = {canon(s["system"]): s for s in B["xnli_loss"]["series"]}
        self.xnli_langs = B["xnli_loss"]["langs"]
        self.probes = self._task_probes()
        self.probes["typed-decisions"] = (self.td["probe"], "layer 26")
        m = re.search(r"CHOICE_MIN,\s*CHOICE_MAX\s*=\s*(\d+),\s*(\d+)",
                      (ROOT / "tez" / "schema.py").read_text(encoding="utf-8"))
        self.option_cap = int(m.group(2))
        self.gemma_apache = "Gemma 4 is Apache-2.0" in (ROOT / "README.md").read_text(encoding="utf-8")
        # optional measurements from results/vs_laya
        self.apps = self._opt("results/vs_laya/apps.json", self._apps)
        self.m51 = self._opt("results/vs_laya/massive51.json", self._m51)
        self.spc = self._opt("results/vs_laya/speed_per_call.json", self._spc)
        self.sel = self._opt("results/vs_laya/selective.json", self._sel)
        self.sel_source = "results/vs_laya/selective.json" if self.sel else None
        if self.sel is None:
            self.sel = self._selective_from_rows()
            if self.sel:
                self.sel_source = "results/h2h/rows_typed_decisions_*.jsonl (selective.json missing)"
                print("  computed selective / reliability / risk-coverage from results/h2h typed-decisions rows")

    # -------------------------------------------------------------- BENCHMARKS.md section 4b task probes
    def _task_probes(self):
        text = (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")
        sec = text.split("### Task probes on Laya's public benchmarks", 1)[1].split("\n\n", 2)[1]
        lines = [ln for ln in sec.splitlines() if ln.startswith("|")]
        head = [c.strip() for c in lines[0].strip("|").split("|")]
        probe_cols = [i for i, c in enumerate(head) if c.startswith("4B probe") or c in ("L22", "L26", "final")]
        out = {}
        for ln in lines[2:]:
            cells = [c.strip().replace("*", "") for c in ln.strip("|").split("|")]
            vals = [(float(cells[i]), head[i]) for i in probe_cols if re.fullmatch(r"\d\.\d+", cells[i])]
            task = {"Banking77 (77)": "Banking77", "Emotion (6)": "DAIR Emotion"}.get(cells[0])
            if vals and task:
                best = max(vals)
                out[task] = (best[0], best[1].replace("4B probe ", "").replace("L", "layer "))
        return out

    # -------------------------------------------------------------- optional files
    def _opt(self, rel, check):
        d = _json(rel)
        if d is None:
            print(f"  missing  {rel}")
            return None
        try:
            d = check(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  unusable {rel}: {exc!r}")
            return None
        print(f"  using    {rel}")
        return d

    @staticmethod
    def _apps(d):
        tasks = []
        for t in d["tasks"]:
            sy = {}
            for k, v in (t.get("systems") or {}).items():
                if isinstance(v, dict) and isinstance(v.get("accuracy"), (int, float)):
                    sy[canon(k)] = v
            if "tez" not in sy:
                continue
            pub = {}
            for k, v in (t.get("laya_published") or {}).items():
                acc = v.get("accuracy") if isinstance(v, dict) else v
                if isinstance(acc, (int, float)):
                    pub[canon(k)] = float(acc)
            tasks.append(dict(id=t.get("id"), label=t.get("label") or t.get("id"), n=t.get("n"),
                              held_out=t.get("held_out"), systems=sy, pub=pub))
        if not tasks:
            raise ValueError("no task with a tez result")
        return dict(meta=d.get("meta", {}), tasks=tasks)

    @staticmethod
    def _m51(d):
        langs = list(d["langs"])
        sy = {}
        for k, v in d["systems"].items():            # {lang: accuracy} or {lang: {"accuracy": ...}}
            acc = {lg: (x.get("accuracy") if isinstance(x, dict) else x) for lg, x in v.items()}
            sy[canon(k)] = {lg: float(x) for lg, x in acc.items() if isinstance(x, (int, float))}
        if "tez" not in sy:
            raise ValueError("no tez series")
        rnd = float(d.get("random", 0.05))
        usable = {}
        for k, v in (d.get("usable_3x_random") or {}).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                usable[canon(k)] = int(v)
            elif isinstance(v, (list, tuple)):
                usable[canon(k)] = len(v)
            elif isinstance(v, dict):
                n = v.get("n", v.get("count", v.get("usable")))
                if isinstance(n, (list, tuple)):
                    n = len(n)
                if isinstance(n, (int, float)):
                    usable[canon(k)] = int(n)
        for k, accs in sy.items():
            usable.setdefault(k, sum(float(accs[lg]) > 3 * rnd for lg in langs if lg in accs))
        return dict(meta=d.get("meta", {}), langs=langs, random=rnd, systems=sy, usable=usable)

    @staticmethod
    def _spc(d):
        q = [int(x) for x in d["questions_per_call"]]
        num = lambda xs: [float(x) if isinstance(x, (int, float)) else float("nan") for x in xs]  # noqa: E731
        sy = {}
        for k, v in d["systems"].items():
            sy[canon(k)] = dict(name=k, call=num(v["p50_ms_per_call"]),
                                per_q=num(v.get("p50_ms_per_question") or []) or None)
        if "tez" not in sy:
            raise ValueError("no tez series")
        return dict(meta=d.get("meta", {}), q=q, systems=sy)

    @staticmethod
    def _sel(d):
        cov = [float(c) for c in d["coverage"]]
        sy = {k: [float(a) for a in v] for k, v in d["systems"].items()}
        raw = {}
        for k, v in (d.get("reliability") or {}).items():
            if k == "extras" and isinstance(v, dict):
                raw.update(v)                              # the Laya checkpoints' shipped reliability, for context
            else:
                raw[k] = v
        bins_n = max((len(p) for v in raw.values() for p in v.values()), default=0)
        rel = {k: {var: [b for b in pts if b.get("n") and b.get("conf") is not None and b.get("acc") is not None]
                   for var, pts in v.items()} for k, v in raw.items()}
        ece = {}
        for k, v in (d.get("summary") or {}).items():
            e = {}
            if isinstance(v.get("ece"), (int, float)):
                e["shipped"] = float(v["ece"])
            if isinstance(v.get("ece_after_temperature"), (int, float)):
                e["temperature"] = float(v["ece_after_temperature"])
            if e:
                ece[k] = e
        risk = {k: v for k, v in (d.get("risk_coverage") or {}).items() if "(temperature)" not in k}
        return dict(meta=d.get("meta", {}), coverage=cov, systems=sy, reliability=rel, risk=risk, ece=ece or None,
                    bins_n=bins_n)

    # -------------------------------------------------------------- fallback: typed-decisions rows of bench_h2h.py
    def _selective_from_rows(self):
        cov = [round(0.1 * i, 1) for i in range(3, 11)]
        out = dict(meta={}, coverage=cov, systems={}, reliability={}, risk={}, ece={}, bins_n=15)
        for k in KEYS:
            p = ROOT / "results" / "h2h" / f"rows_typed_decisions_{H2H_KEY[k]}.jsonl"
            if not p.exists():
                return None
            recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            g = np.array([r["gold"] for r in recs])
            P = [np.asarray(r["probabilities"], float) for r in recs]
            conf = np.array([float(x.max()) for x in P])
            corr = (np.array([int(np.argmax(x)) for x in P]) == g).astype(float)
            order = np.argsort(-conf)                      # the call bench_h2h.summarise uses for acc@50% coverage
            n = len(recs)
            cum = np.cumsum(corr[order]) / np.arange(1, n + 1)
            name = "tez letters" if k == "tez" else k
            out["systems"][name] = [float(cum[max(1, int(round(c * n))) - 1]) for c in cov]
            out["risk"][name] = dict(coverage=(np.arange(1, n + 1) / n).tolist(), error=(1 - cum).tolist())
            summ = self.S["typed_decisions"][H2H_KEY[k]]
            out["ece"][name] = dict(shipped=summ["ece"])
            if k == "tez":
                conf2, corr2 = _oof_temperature(recs)
                if abs(_ece15(conf2, corr2) - summ["ece_refit"]) > 1e-6:
                    print("  WARNING tez: recomputed ECE after temperature differs from results/h2h/summary.json")
                out["reliability"][name] = dict(shipped=_bins(conf, corr), temperature=_bins(np.array(conf2), np.array(corr2)))
                out["ece"][name]["temperature"] = summ["ece_refit"]
            else:
                out["reliability"][name] = dict(shipped=_bins(conf, corr))
        return out


def _ece15(conf, corr, bins=15):                 # bench_h2h.ece15
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def _softmax_t(z, t):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def _fit_temperature(rows):                      # bench_h2h.fit_temperature
    best, bt = 1e9, 1.0
    for t in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        nll = -np.mean([math.log(max(_softmax_t(z, t)[g], 1e-12)) for g, z in rows])
        if nll < best:
            best, bt = nll, float(t)
    return bt


def _oof_temperature(recs):                      # bench_h2h.summarise: 2-fold out-of-fold temperature
    rows = [(r["gold"], np.asarray(r["logits"], float)) for r in recs]
    conf2, corr2 = [], []
    for fold in (0, 1):
        fit = [rw for i, rw in enumerate(rows) if i % 2 == fold]
        ev = [rw for i, rw in enumerate(rows) if i % 2 != fold]
        T = _fit_temperature(fit)
        for g, z in ev:
            p = _softmax_t(z, T)
            conf2.append(float(p.max()))
            corr2.append(float(np.argmax(p) == g))
    return conf2, corr2


def _bins(conf, corr, nb=15):
    out, edges = [], np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            out.append(dict(conf=float(conf[s].mean()), acc=float(corr[s].mean()), n=int(s.sum())))
    return out


# ================================================================== drawing helpers
def strip(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def rel_lum(hexc):
    c = [int(hexc.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def text_on(hexc):
    """White on a dark fill (as Laya's in-bar labels), near-black ink on a light one."""
    return "white" if 1.05 / (rel_lum(hexc) + 0.05) >= 3.0 else INK


def vlab(ax, x, v, s, fs=9, color=INK2, weight="normal", dy=3, rot=0, ha="center"):
    ax.annotate(s, (x, v), xytext=(0, dy), textcoords="offset points", ha=ha, va="bottom",
                fontsize=fs, color=color, fontweight=weight, rotation=rot, zorder=5)


def hlab(ax, v, y, s, fs=8.5, color=INK2, weight="normal", dx=4):
    ax.annotate(s, (v, y), xytext=(dx, 0), textcoords="offset points", ha="left", va="center",
                fontsize=fs, color=color, fontweight=weight, zorder=5)


def bar(ax, x, v, w, key=None, color=None, hatch=None, alpha=1.0, horizontal=False, **kw):
    c = color or COLOR[key]
    args = dict(color=c, zorder=3, alpha=alpha, linewidth=0)
    if hatch:
        args.update(hatch=hatch, edgecolor=SURFACE)
    args.update(kw)
    return (ax.barh if horizontal else ax.bar)(x, v, w, **args)


def patch(key=None, label=None, color=None, hatch=None, alpha=1.0):
    c = color or COLOR[key]
    if hatch:
        return Patch(facecolor=c, edgecolor=SURFACE, hatch=hatch, linewidth=0, label=label, alpha=alpha)
    return Patch(facecolor=c, linewidth=0, label=label, alpha=alpha)


def ax_width_in(ax):
    return ax.get_position().width * ax.figure.get_figwidth()


def head(ax, title, handles=None, ncol=None, note=None, show_title=True, fs_leg=9.5, fs_title=13.5, align=True):
    """Laya's panel head: bold left title, the legend under it, a small grey note, then the plot."""
    y = 4.0
    if note:
        note = wrap(note, ax_width_in(ax) + 0.4, 8.5)
        ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=INK3, linespacing=1.3)
        y += 8.5 * 1.36 * (note.count("\n") + 1) + 3
    if handles:
        ncol = ncol or len(handles)
        nrow = math.ceil(len(handles) / ncol)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1),
                  bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, x=-5, y=y, units="points"),
                  frameon=False, ncol=ncol, fontsize=fs_leg, labelcolor=INK2, handlelength=1.4, handleheight=0.9,
                  columnspacing=1.4, borderaxespad=0, borderpad=0, labelspacing=0.45)
        y += nrow * fs_leg * 1.62 + 2
    ax._tez_head = y                                    # points above the axes used by the legend and the note
    if show_title and title:
        ax.set_title(title, loc="left", fontsize=fs_title, fontweight="bold", color=INK, pad=y + 5)
        if align:
            ax._tez_title = (title, fs_title, y)        # lets compose() align the titles of one row
        ax._tez_head = y + 5 + fs_title * 1.3
    return y


def align_titles(fig):
    """Raise every panel title in a row to the highest one, as in Laya's composites."""
    rows = {}
    for ax in fig.axes:
        if hasattr(ax, "_tez_title"):
            rows.setdefault(round(ax.get_position().y1, 3), []).append(ax)
    for axes in rows.values():
        top = max(ax._tez_title[2] for ax in axes)
        for ax in axes:
            title, fs, _ = ax._tez_title
            ax.set_title(title, loc="left", fontsize=fs, fontweight="bold", color=INK, pad=top + 5)
            ax._tez_head = top + 5 + fs * 1.3


def top_head_in(fig):
    """Inches the top row of panels needs above its axes (title, legend, note)."""
    axes = [ax for ax in fig.axes if hasattr(ax, "_tez_head")]
    if not axes:
        return 0.0
    y1 = max(ax.get_position().y1 for ax in axes)
    return max(ax._tez_head for ax in axes if abs(ax.get_position().y1 - y1) < 1e-3) / 72


def foot(ax, text, dy=-34, fs=8.2, extra_in=0.3):
    t = wrap(text, ax_width_in(ax) + extra_in, fs)
    ax.annotate(t, xy=(0, 0), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                ha="left", va="top", fontsize=fs, color=INK3, linespacing=1.35)


def wrap(text, width_in, fs, bold=False):
    cpl = max(24, int(width_in * 72 / (fs * (0.64 if bold else 0.56))))
    return "\n".join("\n".join(textwrap.wrap(par, cpl)) if par else "" for par in text.split("\n"))


class Ctx:
    def __init__(self, D, show_title=True):
        self.D, self.show_title = D, show_title


# ================================================================== panels
# ---------------------------------------------------------------- accuracy vs Jev
ACC_GROUPS = [("typed-decisions", "typed-decisions", "2,000 decisions"),
              ("AG News", "AG News", "4 labels · 400 rows"),
              ("DAIR Emotion", "DAIR Emotion", "6 labels · 400 rows"),
              ("Banking77", "Banking77", "77 labels · 400 rows")]


def draw_accuracy(fig, spec, ctx, context=True):
    D = ctx.D
    ax = fig.add_subplot(spec)
    wm, gm = 0.40, 0.07                      # Jev, Tez, Tez probe
    wc, gc, sep = 0.15, 0.025, 0.22          # laya, laya-multilingual, laya-typed-decisions (context)
    main_w = 3 * wm + 2 * gm
    total = main_w + ((sep + 3 * wc + 2 * gc) if context else 0.0)
    pitch = total + 0.55
    centers = []
    for gi, (task, name, sub) in enumerate(ACC_GROUPS):
        c = gi * pitch
        centers.append(c)
        x0 = c - total / 2
        xj, xt, xp = (x0 + wm / 2 + i * (wm + gm) for i in range(3))
        r = D.h2h[task]
        jev, tez = r["jev"], r["tez"]
        bar(ax, xj, jev, wm, "jev")
        vlab(ax, xj, jev, f3(jev), fs=9)
        bar(ax, xt, tez, wm, "tez")
        vlab(ax, xt, tez, f3(tez), fs=9.5, color=INK, weight="bold")
        vlab(ax, xt, tez, fdelta(tez - jev), fs=8.5, color=TEZ_INK if tez > jev else INK2, dy=15)
        if task in D.probes:
            pv = D.probes[task][0]
            bar(ax, xp, pv, wm, "tez", hatch=HATCH)
            vlab(ax, xp, pv, f3(pv), fs=9.5, color=INK, weight="bold")
            vlab(ax, xp, pv, fdelta(pv - jev), fs=8.5, color=TEZ_INK if pv > jev else INK2, dy=15)
        else:
            ax.annotate("probe\nnot run", (xp, 0), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.2, color=INK3, linespacing=1.1)
        if context:
            for i, k in enumerate(LAYA3):
                xc = x0 + main_w + sep + wc / 2 + i * (wc + gc)
                v = r[k]
                bar(ax, xc, v, wc, k)
                vlab(ax, xc, v, f3(v), fs=7.6, rot=90, dy=2.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{name}\n{sub}" for _, name, sub in ACC_GROUPS], fontsize=10)
    ax.set_xlim(centers[0] - total / 2 - 0.2, centers[-1] + total / 2 + 0.2)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.round(np.arange(0, 1.21, 0.2), 1))
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    handles = [patch("jev", "Jev (published)"), patch("tez", "Tez zero-shot"),
               patch("tez", "Tez probe (trained on labelled rows)", hatch=HATCH)]
    if context:
        handles += [patch(k, NAME[k]) for k in LAYA3]
    head(ax, "Accuracy — the public datasets where Jev numbers exist", handles, ncol=3 if context else 3,
         note="deltas are against Jev's published figure" + (" · Laya's checkpoints on the same rows, for context"
                                                              if context else ""),
         show_title=ctx.show_title)
    foot(ax, "Probe: typed-decisions layer 26 (300 labelled rows per question), Banking77 layer 22 and Emotion layer 26 "
             "(2,000 labelled rows each). laya-typed-decisions is fine-tuned on typed-decisions' train split; "
             "AG News is in Laya's training mix.", dy=-36)


def finding_accuracy(D):
    behind = sum(D.h2h[t]["tez"] < D.h2h[t]["jev"] for t, _, _ in ACC_GROUPS)
    passes = [n for t, n, _ in ACC_GROUPS if t in D.probes and D.probes[t][0] > D.h2h[t]["jev"]]
    return (f"Zero-shot, Tez trails Jev's published figures on {behind} of {len(ACC_GROUPS)} datasets; "
            f"a probe on a frozen 4B passes Jev on {join_and(passes)}")


# ---------------------------------------------------------------- language coverage
def draw_coverage(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    if D.m51:
        total = len(D.m51["langs"])
        keys = [k for k in ("tez", "laya-routed", "laya-multilingual", "laya") if k in D.m51["usable"]]
        rows = [(k, NAME[k], D.m51["usable"][k]) for k in keys]
        rows.append(("pub", "Laya published\n(routed, its own run)", LAYA_PUB["routed_usable_51"]))
    else:
        total = len(D.langs11)
        rows = [(k, NAME[k], D.usable11[k]) for k in ("tez", "laya-multilingual", "laya", "laya-typed-decisions")]
    rows.append(("jev", "Jev (published)", None))
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, n) in zip(y, rows):
        if n is None:
            ax.text(total * 0.02, yi, "no multilingual figure to compare", va="center", fontsize=9.5, color=INK3)
            continue
        if k == "pub":
            ax.barh(yi, n, 0.62, facecolor="none", edgecolor=LAYA_RT, linewidth=1.4, linestyle=(0, (3, 2)), zorder=3)
            hlab(ax, n, yi, f"{n} / {total}", fs=10, color=INK2, weight="bold")
            continue
        bar(ax, yi, n, 0.62, k, horizontal=True)
        inside = n >= total * 0.25
        ax.annotate(f"{n} / {total}", (n, yi), xytext=(-7 if inside else 4, 0), textcoords="offset points",
                    ha="right" if inside else "left", va="center", fontsize=12.5, fontweight="bold",
                    color=text_on(COLOR[k]) if inside else INK, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=10)
    ax.set_xlim(0, total * 1.1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    step = 10 if total > 20 else 2
    ax.set_xticks(range(0, total + 1, step))
    ax.set_xlabel(f"languages usable (> 3× random) of {total} tested", fontsize=10)
    grid(ax, "x")
    strip(ax)
    head(ax, "Language coverage", note="MASSIVE intent, 20 options, 100 rows per language", show_title=ctx.show_title)


def finding_coverage(D):
    if D.m51:
        u, t = D.m51["usable"], len(D.m51["langs"])
        extra = f"; Laya's router {u['laya-routed']} of {t} on the same rows" if "laya-routed" in u else ""
        return f"Tez is usable in {u['tez']} of {t} MASSIVE languages{extra}"
    u = D.usable11
    return (f"Tez reads every language tested ({u['tez']} of {len(D.langs11)} usable), as does laya-multilingual; "
            f"laya and laya-typed-decisions read {u['laya']}")


# ---------------------------------------------------------------- speed per decision (ranges)
def speed_rows(D):
    ms = D.ms
    return [("jev", "Jev (published)", ms["jev"]),
            ("tez", "Tez 12B letters", ms["tez"]),
            ("tez-probe", "Tez probe (4B, 24 blocks)", (D.probe_ms, D.probe_ms)),
            ("tez", "Tez voice loop, per word", (D.voice_after["compute_ms"],) * 2),
            ("laya", "laya", ms["laya"]), ("laya-multilingual", "laya-multilingual", ms["laya-multilingual"]),
            ("laya-typed-decisions", "laya-typed-decisions", ms["laya-typed-decisions"])]


def draw_speed_decision(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    rows = speed_rows(D)
    y = np.arange(len(rows))[::-1]
    jlo, jhi = D.ms["jev"]
    ax.axvspan(jlo, jhi, color=JEV, alpha=0.10, lw=0, zorder=1)
    for yi, (k, lab, (lo, hi)) in zip(y, rows):
        c = COLOR[k]
        if hi > lo:
            ax.plot([lo, hi], [yi, yi], color=c, lw=11, solid_capstyle="butt", zorder=3)
            txt = f"{lo}–{hi} ms"
        else:
            ax.plot([hi], [yi], "o", ms=12, color=c, mec=SURFACE, mew=1.6, zorder=4)
            if k == "tez-probe":
                ax.plot([hi], [yi], "o", ms=5, color=SURFACE, zorder=5)
            txt = f"{hi} ms"
        ax.annotate(txt, (hi, yi), xytext=(9, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=9.5, color=INK if k.startswith("tez") else INK2,
                    fontweight="bold" if k.startswith("tez") else "normal")
        if k == "tez-probe":
            ax.annotate(f"{jlo / D.probe_ms:.1f}–{jhi / D.probe_ms:.1f}× under Jev's published p50", (hi, yi),
                        xytext=(56, 0), textcoords="offset points", va="center", ha="left", fontsize=8.8,
                        color=TEZ_INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 340)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("p50 latency per decision (ms)", fontsize=10)
    grid(ax, "x")
    strip(ax)
    head(ax, "Speed", note="p50 per decision · a bar spans the range over tasks", show_title=ctx.show_title)


def finding_speed_decision(D):
    lo = min(D.ms[k][0] for k in LAYA3)
    hi = max(D.ms[k][1] for k in LAYA3)
    return (f"Laya answers fastest ({lo}–{hi} ms); Tez's probe takes {D.probe_ms} ms and its 12B letters "
            f"{D.ms['tez'][0]}–{D.ms['tez'][1]} ms, against Jev's published {D.ms['jev'][0]}–{D.ms['jev'][1]} ms")


# ---------------------------------------------------------------- calibration vs Jev
def draw_calib_jev(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    items = [("Jev\n(published)", D.ece_shipped["jev"], dict(key="jev")),
             ("Tez\nas shipped", D.ece_shipped["tez"], dict(key="tez", alpha=SHIPPED_ALPHA)),
             ("Tez + one\ntemperature", D.ece_temp["tez"], dict(key="tez")),
             ("Tez\nprobe", D.td["probe_ece"], dict(key="tez", hatch=HATCH))]
    for i, (_, v, st) in enumerate(items):
        bar(ax, i, v, 0.58, **st)
        vlab(ax, i, v, f3(v), fs=10.5, color=INK if i else INK2, weight="bold" if i else "normal")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i[0] for i in items], fontsize=9.5)
    ax.set_xlim(-0.55, len(items) - 0.45)
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("ECE-15  (lower is better)", fontsize=10)
    grid(ax)
    strip(ax)
    head(ax, "Calibration", note="Tez: mean over the head-to-head tasks · probe: typed-decisions only",
         show_title=ctx.show_title)
    s, t = D.ece_shipped, D.ece_temp
    foot(ax, "Laya on the same rows: " + " / ".join(f3(s[k]) for k in LAYA3) + " as shipped → "
         + " / ".join(f3(t[k]) for k in LAYA3) + " after one temperature per task.", dy=-34)


def finding_calib_jev(D):
    return (f"Tez ships slightly better calibrated than Jev's published ECE ({f3(D.ece_shipped['tez'])} vs "
            f"{f3(D.ece_shipped['jev'])}); one temperature per task brings it to {f3(D.ece_temp['tez'])}, "
            f"a probe to {f3(D.td['probe_ece'])}")


# ---------------------------------------------------------------- "And" facts
def and_rows(D):
    lang = (f"{D.m51['usable']['tez']} of {len(D.m51['langs'])} tested usable" if D.m51
            else f"{D.usable11['tez']} of {len(D.langs11)} tested usable")
    weights = "Apache-2.0 (Gemma 4), self-hosted" if D.gemma_apache else "self-hosted"
    return [("Weights", weights, JEV_FACTS["weights"]),
            ("Cost", "$0 self-hosted", JEV_FACTS["cost"]),
            ("Languages", lang, "no multilingual figure to compare"),
            ("Deployment", "local: llama.cpp on one GPU", JEV_FACTS["deployment"]),
            ("Option cap", f"{D.option_cap} per question", JEV_FACTS["options"])]


def draw_and(fig, spec, ctx):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    head(ax, "And", note="blue = Tez · grey = Jev (published)" if ctx.show_title else None,
         show_title=ctx.show_title, fs_title=15, align=False)
    kw = dict(xycoords="axes fraction", textcoords="offset points", va="top")
    y = -4
    for lab, tez, jev in and_rows(ctx.D):
        ax.annotate(lab, (0, 1), xytext=(0, y), fontsize=10, color=INK3, **kw)
        ax.annotate(tez, (0, 1), xytext=(88, y), fontsize=10.5, color=TEZ_INK, fontweight="bold", **kw)
        ax.annotate(jev, (0, 1), xytext=(88, y - 16), fontsize=9.5, color=INK3, **kw)
        y -= 44


# ---------------------------------------------------------------- where Tez leads / where others lead
def _eff(r, k):
    """A Laya checkpoint's value on a workflow row: our rerun, else Laya's published figure."""
    return r["vals"].get(k, r["pub"].get(k))


def leads(D):
    tot = len(D.m51["langs"]) if D.m51 else len(D.langs11)
    u = D.m51["usable"]["tez"] if D.m51 else D.usable11["tez"]
    if D.m51:
        rt = D.m51["usable"].get("laya-routed")
        lang_sub = ((f"laya-routed {rt} / {tot} on the same rows; " if rt is not None else "")
                    + f"Laya published {LAYA_PUB['routed_usable_51']} / {tot}; no Jev figure")
    else:
        lang_sub = (f"MASSIVE, 20 options: macro {f3(D.lang_macro['tez'])} vs laya-multilingual "
                    f"{f3(D.lang_macro['laya-multilingual'])}; no Jev figure")
    fl = [D.flip20[k] for k in LAYA3]
    out = [(f"{u} / {tot}", "languages usable", lang_sub),
           (f3(D.td["probe"]), "typed-decisions, probe on a frozen 4B",
            f"Jev published {f3(D.td['jev'])} · laya-typed-decisions (fine-tuned) {f3(D.td['laya_td'])}"),
           (f"{D.flip20['tez']:.2f}", "answers flipped by reversing 20 options",
            f"Jev published {D.flip20['jev']:.2f} · Laya {min(fl):.2f}–{max(fl):.2f}")]
    gr = next((t for t in (D.apps or {}).get("tasks", []) if t["id"] == "guardrails_jailbreak"), None)
    if gr:
        r = dict(vals={k: gr["systems"][k]["accuracy"] for k in LAYA3 if k in gr["systems"]}, pub=gr["pub"])
        lv = [_eff(r, k) for k in LAYA3 if _eff(r, k) is not None]
        src = "Laya" if len(r["vals"]) == 3 else "Laya (rerun or published)"
        out.append((fnum(gr["systems"]["tez"]["accuracy"]), "LLM guardrails (jailbreak), held out",
                    f"{src} {fnum(min(lv))}–{fnum(max(lv))} on the same {gr['n']} rows; no Jev figure"))
    else:
        pi = D.h2h["prompt-injections"]
        lay = [pi[k] for k in LAYA3]
        out.append((f3(pi["tez"]), "prompt-injections, held out",
                    f"Laya {f3(min(lay))}–{f3(max(lay))} on the same {pi['n']} rows; no Jev figure"))
    out.append(("$0", "self-hosted, Apache-2.0 weights", f"as Laya; Jev: {JEV_FACTS['cost']}, hosted only"))
    return out


def losses(D):
    ag, bk, em = D.h2h["AG News"], D.h2h["Banking77"], D.h2h["DAIR Emotion"]
    lo = min(D.ms[k][0] for k in LAYA3)
    hi = max(D.ms[k][1] for k in LAYA3)
    out = [(f"{lo}–{hi} ms", "laya", "Laya, p50 per decision",
            f"Tez 12B letters {D.ms['tez'][0]}–{D.ms['tez'][1]} ms on the same GPU"),
           (f3(ag["laya-multilingual"]), "laya-multilingual", "AG News",
            f"Tez {f3(ag['tez'])}; Jev published {f3(ag['jev'])}; AG News is in Laya's training mix"),
           (f3(D.xnli["laya-multilingual"]["macro"]), "laya-multilingual", "XNLI macro, 10 languages",
            f"Tez {f3(D.xnli['tez']['macro'])}; NLI is in Laya's training")]
    if D.apps:
        trained = [r for r in workflow_rows(D) if r["src"] == "apps" and r["tag"] == "in Laya's training"]
        rows = [r for r in trained if any((_eff(r, k) or 0) > r["vals"]["tez"] for k in LAYA3)]
        if rows:
            best = max(rows, key=lambda r: max(_eff(r, k) or 0 for k in LAYA3) - r["vals"]["tez"])
            k = max(LAYA3, key=lambda kk: _eff(best, kk) or 0)
            pub_only = k not in best["vals"]
            share = f"all {len(rows)}" if len(rows) == len(trained) else f"{len(rows)} of the {len(trained)}"
            out.append((fnum(_eff(best, k)), k, f"{best['label']}" + (" (Laya published)" if pub_only else ""),
                        f"Tez {fnum(best['vals']['tez'])}; Laya leads {share} workflow themes in its training"))
    out += [(f3(em["laya"]), "laya", "DAIR Emotion",
             f"Tez {f3(em['tez'])} zero-shot, {f3(D.probes['DAIR Emotion'][0])} with a trained probe"),
            (f3(bk["jev"]), "jev", "Banking77, Jev published",
             f"Tez {f3(bk['tez'])} zero-shot, {f3(D.probes['Banking77'][0])} with a trained probe"),
            (f3(D.td["jev"]), "jev", "typed-decisions zero-shot, Jev published",
             f"Tez {f3(D.td['zero'])}; {f3(D.td['four'])} with 4 examples in the prompt")]
    return out


BIG_COL = 128                                              # label column of the big-number blocks, points


def big_rows_height(rows, width_in, marker=False):
    """Inches the big-number rows take at this width (the same wrapping as _big_rows)."""
    sub_in = (width_in * 72 - BIG_COL - 4) / 72
    total = 6.0
    for r in rows:
        lab, sub = (r[2], r[3]) if marker else (r[1], r[2])
        lab_w = wrap(lab, sub_in - (0.22 if marker else 0), 11.2, bold=True)
        sub_w = wrap(sub, sub_in, 8.8)
        total += 16 * (lab_w.count("\n") + 1) + 11.5 * (sub_w.count("\n") + 1) + 17
    return total / 72


def _big_rows(ax, rows, big_color, marker=False):
    kw = dict(xycoords="axes fraction", textcoords="offset points", va="top")
    col = BIG_COL
    sub_in = (ax_width_in(ax) * 72 - col - 4) / 72
    y = -6
    for r in rows:
        if marker:
            big, key, lab, sub = r
        else:
            (big, lab, sub), key = r, None
        ax.annotate(big, (0, 1), xytext=(0, y + 2), fontsize=21, fontweight="bold", color=big_color, **kw)
        lab_w = wrap(lab, sub_in - (0.22 if key else 0), 11.2, bold=True)
        if key:
            ax.annotate("■", (0, 1), xytext=(col, y), fontsize=11, color=COLOR[key], **kw)
        ax.annotate(lab_w, (0, 1), xytext=(col + (15 if key else 0), y), fontsize=11.2, fontweight="bold", color=INK,
                    linespacing=1.15, **kw)
        dy = 16 * (lab_w.count("\n") + 1)
        sub_w = wrap(sub, sub_in, 8.8)
        ax.annotate(sub_w, (0, 1), xytext=(col, y - dy), fontsize=8.8, color=INK3, linespacing=1.3, **kw)
        y -= dy + 11.5 * (sub_w.count("\n") + 1) + 17


def draw_tez_leads(fig, spec, ctx):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    head(ax, "Where Tez leads", note="on the same rows as Laya, or against Jev's published figure"
         if ctx.show_title else None, show_title=ctx.show_title, fs_title=14, align=False)
    _big_rows(ax, leads(ctx.D), TEZ_INK)


def draw_others_lead(fig, spec, ctx):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    head(ax, "Where others lead", note="Tez publishes its losses" if ctx.show_title else None,
         show_title=ctx.show_title, fs_title=14, align=False)
    _big_rows(ax, losses(ctx.D), INK, marker=True)


def finding_others(D):
    s = "Where others lead: Laya on speed, on AG News and XNLI (both in its training)"
    if D.apps:
        rows = [r for r in workflow_rows(D) if r["src"] == "apps" and r["tag"] == "in Laya's training"]
        led = [r for r in rows if any((_eff(r, k) or 0) > r["vals"]["tez"] for k in LAYA3)]
        if led:
            share = f"all {len(led)}" if len(led) == len(rows) else f"{len(led)} of the {len(rows)}"
            s += f", on {share} workflow themes in its training"
    return s + (" and narrowly on DAIR Emotion; Jev's published figures on Banking77 and zero-shot typed-decisions")


# ---------------------------------------------------------------- every workflow, every checkpoint
H2H_ROWS = [("AG News", "Topic — AG News", "in Laya's training"),
            ("DAIR Emotion", "Emotion — DAIR", "held out"),   # Laya's own charts mark Emotion as held out of training
            ("Banking77", "Banking intent — 77 options", None),
            ("SST-5", "Sentiment — SST-5, 5 levels", None),
            ("BoolQ", "Reading — BoolQ", "in Laya's training"),
            ("prompt-injections", "Guardrails — prompt injection", "held out"),
            ("MASSIVE intent, English", "Intent — MASSIVE (en), 20 options", None),
            ("XNLI, English", "Inference — XNLI (en)", "in Laya's training"),
            ("typed-decisions", "typed-decisions", "laya-typed-decisions fine-tuned on it")]


def workflow_rows(D):
    rows = []
    if D.apps:
        for t in D.apps["tasks"]:
            ho = t["held_out"]
            tag = "held out" if ho is True else ("in Laya's training" if ho is False else None)
            vals = {k: float(t["systems"][k]["accuracy"]) for k in KEYS if k in t["systems"]}
            rows.append(dict(label=t["label"], tag=tag, n=t["n"], vals=vals, pub=t["pub"], src="apps"))
    for task, label, tag in H2H_ROWS:
        r = D.h2h[task]
        rows.append(dict(label=label, tag=tag, n=r["n"], vals={k: r[k] for k in KEYS}, pub={}, src="h2h"))
    rows.sort(key=lambda r: -r["vals"]["tez"])
    return rows


def draw_workflows(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    rows = workflow_rows(D)
    h = 0.2
    offs = [-1.5 * h, -0.5 * h, 0.5 * h, 1.5 * h]
    any_pub_only = any_marker = False
    for i, r in enumerate(rows):
        fmt = fnum if r["src"] == "apps" else f3
        for j, k in enumerate(KEYS):
            yy = i + offs[j]
            v = r["vals"].get(k)
            if v is not None:
                bar(ax, yy, v, h * 0.9, k, horizontal=True)
                hlab(ax, v, yy, fmt(v), fs=7.8, color=INK if k == "tez" else INK2,
                     weight="bold" if k == "tez" else "normal", dx=3)
                if k in r["pub"]:
                    any_marker = True
                    ax.plot([r["pub"][k]], [yy], marker="D", ms=5, mfc="none", mec=INK, mew=1.0, zorder=6)
            elif k in r["pub"]:
                any_pub_only = True
                pv = r["pub"][k]
                ax.barh(yy, pv, h * 0.78, facecolor="none", edgecolor=COLOR[k], linewidth=1.3, zorder=3)
                hlab(ax, pv, yy, fnum(pv), fs=7.4, color=INK3, dx=3)
    labels = []
    for r in rows:
        n = "" if r["n"] in (None, 400) else f" ({r['n']:,})"
        labels.append(f"{r['label']}{n}" + (f"\n· {r['tag']}" if r["tag"] else ""))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.5, linespacing=1.15)
    ax.set_ylim(len(rows) - 0.45, -0.6)
    ax.set_xlim(0, 1.16)
    ax.set_xticks(np.round(np.arange(0, 1.01, 0.2), 1))
    ax.set_xlabel("accuracy", fontsize=10)
    grid(ax, "x")
    strip(ax)
    handles = [patch(k, NAME[k] + (" zero-shot" if k == "tez" else "")) for k in KEYS]
    if any_pub_only:
        handles.append(Patch(facecolor="none", edgecolor=INK3, linewidth=1.3,
                             label="outlined: Laya's published figure, not yet rerun"))
    if any_marker:
        handles.append(Line2D([], [], marker="D", ms=5, mfc="none", mec=INK, mew=1.0, ls="",
                              label="Laya's published figure (its own run)"))
    parts = ["400 rows each unless marked", "identical rows per model", "held out = not in Laya's training data"]
    if D.apps:
        parts.insert(0, f"Laya's {len(D.apps['tasks'])} workflow themes + its public suite")
        parts.append("Laya's workflow scores: its own harness on the same laptop's CPU")
    ncol = len(handles) if ax_width_in(ax) > 12 else (4 if ax_width_in(ax) > 8.5 else 2)
    head(ax, "Every workflow, every checkpoint", handles, ncol=min(len(handles), ncol), note=" · ".join(parts),
         show_title=ctx.show_title)


def finding_workflows(D):
    rows = workflow_rows(D)
    led = [r for r in rows if all(r["vals"]["tez"] > (_eff(r, k) if _eff(r, k) is not None else -1) for k in LAYA3)]
    others = [r for r in rows if r not in led]
    held = [r for r in rows if r["tag"] == "held out"]
    s = f"Zero-shot Tez leads every Laya checkpoint on {len(led)} of {len(rows)} tasks"
    if held:
        n_led_held = sum(1 for r in held if r in led)
        if n_led_held == len(held):
            s += ", including " + ("the one" if len(held) == 1 else f"all {len(held)}") + " held out of Laya's training"
        elif n_led_held:
            s += f", including {n_led_held} of the {len(held)} held out of Laya's training"
    trained = [r for r in others if r["tag"] in ("in Laya's training", "laya-typed-decisions fine-tuned on it")]
    s += f"; a Laya checkpoint leads the other {len(others)}"
    if trained:
        s += f", {len(trained)} of them in its training data or fine-tuning"
    return s


# ---------------------------------------------------------------- English vs the rest
def eng_rest(D):
    out = {}
    for fam in ("massive", "xnli"):
        for k in KEYS:
            hk = H2H_KEY[k]
            en = D.S[f"{fam}:en"][hk]["accuracy"]
            oth = [v[hk]["accuracy"] for t, v in D.S.items() if t.startswith(fam + ":") and t != f"{fam}:en" and hk in v]
            out[(fam, k)] = (en, float(np.mean(oth)), len(oth))
    return out


def draw_eng_rest(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    er = eng_rest(D)
    w, g, sg, pitch = 0.088, 0.01, 0.05, 1.12      # bar, gap inside a system, gap between systems, group pitch
    span = 8 * w + 4 * g + 3 * sg
    fams = [("massive", "MASSIVE intent\n20 options"), ("xnli", "XNLI\n3 labels")]
    for fi, (fam, _) in enumerate(fams):
        x = fi * pitch - span / 2
        for k in KEYS:
            en, oth, _ = er[(fam, k)]
            for j, (v, hatch) in enumerate(((en, None), (oth, "///"))):
                xc = x + w / 2
                bar(ax, xc, v, w, k, hatch=hatch)
                vlab(ax, xc, v, f3(v), fs=7.2, rot=90, dy=2, color=INK if k == "tez" else INK2,
                     weight="bold" if k == "tez" else "normal")
                x += w + (g if j == 0 else sg)
    ax.set_xticks([i * pitch for i in range(len(fams))])
    ax.set_xticklabels([f[1] for f in fams], fontsize=9.5)
    ax.set_xlim(-span / 2 - 0.12, (len(fams) - 1) * pitch + span / 2 + 0.12)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.round(np.arange(0, 1.01, 0.2), 1))
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    n_m, n_x = er[("massive", "tez")][2], er[("xnli", "tez")][2]
    handles = [patch(k, NAME[k]) for k in KEYS] + [patch(color=INK3, label="English"),
                                                    patch(color=INK3, hatch="///", label="other languages (mean)")]
    head(ax, "English vs the rest", handles, ncol=3, fs_leg=8.6,
         note=f"hatched = mean of the other languages (MASSIVE {n_m}, XNLI {n_x})", show_title=ctx.show_title)


def finding_eng_rest(D):
    er = eng_rest(D)
    t, l = er[("massive", "tez")], er[("massive", "laya")]
    x = er[("xnli", "laya-multilingual")]
    return (f"Tez holds its accuracy outside English (MASSIVE {f3(t[0])} → {f3(t[1])}) while laya falls from "
            f"{f3(l[0])} to {f3(l[1])}; on XNLI laya-multilingual stays ahead ({f3(x[1])} outside English)")


# ---------------------------------------------------------------- speed per call (speed_per_call.json)
SPC_ORDER = ("tez", "tez-probe", "laya", "laya-multilingual", "laya-typed-decisions", "laya-routed")


def draw_speed_call(fig, spec, ctx):
    D = ctx.D
    sp = D.spc
    ax = fig.add_subplot(spec)
    keys = [k for k in SPC_ORDER if k in sp["systems"]]
    n = len(keys)
    w = 0.8 / n
    x = np.arange(len(sp["q"]))
    jlo, jhi = D.ms["jev"]
    ax.axhspan(jlo, jhi, color=JEV, alpha=0.14, lw=0, zorder=1)
    top = float(np.nanmax([np.nanmax(sp["systems"][k]["call"]) for k in keys]))
    for j, k in enumerate(keys):
        vals = sp["systems"][k]["call"]
        xs = x - 0.4 + w / 2 + j * w
        for xi, v in zip(xs, vals):
            if math.isnan(v):
                continue                                   # not measured: no bar, no number
            bar(ax, xi, v, w * 0.92, "tez" if k == "tez-probe" else k, hatch=HATCH if k == "tez-probe" else None)
            vlab(ax, xi, v, fms(v), fs=7.8, rot=90 if n > 2 else 0, dy=2,
                 color=INK if k == "tez" else INK2, weight="bold" if k == "tez" else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in sp["q"]])
    ax.set_xlabel("questions per call", fontsize=10)
    ax.set_ylabel("p50 latency per call (ms)", fontsize=10)
    ax.set_ylim(0, max(top, jhi) * 1.25)
    grid(ax)
    strip(ax)
    handles = [patch("tez", display(sp["systems"][k]["name"]), hatch=HATCH) if k == "tez-probe"
               else patch(k, NAME[k]) for k in keys]
    handles.append(patch(color=JEV, alpha=0.3, label=f"Jev (published) {jlo}–{jhi} ms, one question"))
    head(ax, "Speed on one RTX 5080 laptop GPU", handles, ncol=min(3, len(handles)), fs_leg=8.8,
         note="p50 per call · every system answered the same questions", show_title=ctx.show_title)


def finding_speed_call(D):
    sp = D.spc
    s = sp["systems"]
    one = {k: v["call"][0] for k, v in s.items() if not math.isnan(v["call"][0])}
    last = {k: v["call"][-1] for k, v in s.items() if not math.isnan(v["call"][-1])}
    if "tez" not in one or "tez" not in last:
        return "p50 latency per call for 1 to 50 questions per call, every system on the same questions"
    f1, fN = min(one, key=one.get), min(last, key=last.get)
    return (f"One question per call: {NAME.get(f1, f1)} is fastest at {fms(one[f1])} ms (Tez {fms(one['tez'])} ms); "
            f"at {sp['q'][-1]} per call {NAME.get(fN, fN)} leads at {fms(last[fN])} ms (Tez {fms(last['tez'])} ms)")


# ---------------------------------------------------------------- batching (ms per question)
def draw_batching(fig, spec, ctx):
    D = ctx.D
    sp = D.spc
    ax = fig.add_subplot(spec)
    keys = [k for k in SPC_ORDER if k in sp["systems"] and sp["systems"][k]["per_q"]]
    x = np.arange(len(sp["q"]))
    for k in keys:
        v = sp["systems"][k]["per_q"]
        ax.plot(x, v, marker="o", color=COLOR[k], lw=2.2, ms=8, mec=SURFACE, mew=1.6, zorder=3,
                ls="--" if k == "tez-probe" else "-")
        for xi in (0, len(x) - 1):
            if not math.isnan(v[xi]):
                vlab(ax, x[xi], v[xi], fms(v[xi]), fs=8.5, dy=7, color=INK if k == "tez" else INK2,
                     weight="bold" if k == "tez" else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in sp["q"]])
    ax.set_xlabel("questions per call", fontsize=10)
    ax.set_ylabel("p50 ms per question", fontsize=10)
    ax.set_ylim(0, float(np.nanmax([np.nanmax(sp["systems"][k]["per_q"]) for k in keys])) * 1.25)
    grid(ax)
    strip(ax)
    handles = [Line2D([], [], color=COLOR[k], marker="o", lw=2, ms=6, ls="--" if k == "tez-probe" else "-",
                      label=display(sp["systems"][k]["name"]) if k.startswith("tez") else NAME[k]) for k in keys]
    head(ax, "Batching pays off", handles, ncol=min(3, len(handles)), fs_leg=8.8,
         note="p50 per call divided by questions per call", show_title=ctx.show_title)


def finding_batching(D):
    s = D.spc["systems"]["tez"]["per_q"] or []
    if len(s) < 2 or math.isnan(s[0]) or math.isnan(s[-1]):
        return "p50 milliseconds per question as more questions share one call"
    verb = "cuts" if s[-1] < s[0] else "does not cut"
    return (f"Batching {verb} Tez's cost per question: {fms(s[0])} ms at one question, {fms(s[-1])} ms at "
            f"{D.spc['q'][-1]} questions per call")


# ---------------------------------------------------------------- calibration, every checkpoint
def draw_calib_all(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    w = 0.36
    for i, k in enumerate(KEYS):
        a, b = D.ece_shipped[k], D.ece_temp[k]
        bar(ax, i - w / 2 - 0.01, a, w, k, alpha=SHIPPED_ALPHA)
        bar(ax, i + w / 2 + 0.01, b, w, k)
        vlab(ax, i - w / 2 - 0.01, a, f3(a), fs=8.5)
        vlab(ax, i + w / 2 + 0.01, b, f3(b), fs=8.5, color=INK if k == "tez" else INK2,
             weight="bold" if k == "tez" else "normal")
    xp = len(KEYS) + 0.05
    bar(ax, xp, D.td["probe_ece"], w, "tez", hatch=HATCH)
    vlab(ax, xp, D.td["probe_ece"], f3(D.td["probe_ece"]), fs=8.5, color=INK, weight="bold")
    jev = D.ece_shipped["jev"]
    ax.axhline(jev, color=INK3, lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"Jev (published) {f3(jev)}", (xp + 0.3, jev), xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color=INK3)
    ax.set_xticks(list(range(len(KEYS))) + [xp])
    ax.set_xticklabels(["Tez", "laya", "laya-\nmultilingual", "laya-typed-\ndecisions", "Tez\nprobe"], fontsize=8.8)
    ax.set_xlim(-0.6, xp + 0.45)
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("mean ECE-15 (lower is better)", fontsize=10)
    grid(ax)
    strip(ax)
    handles = [patch(color=INK3, alpha=SHIPPED_ALPHA, label="as shipped"),
               patch(color=INK3, label="one temperature per task"),
               patch("tez", "Tez probe", hatch=HATCH)]
    head(ax, "Calibration", handles, ncol=3, fs_leg=8.8,
         note="mean over the head-to-head tasks · probe: typed-decisions only", show_title=ctx.show_title)


def finding_calib_all(D):
    t = [D.ece_temp[k] for k in LAYA3]
    return (f"One temperature per task fixes most miscalibration for every model: Tez {f3(D.ece_shipped['tez'])} → "
            f"{f3(D.ece_temp['tez'])}, Laya {f3(min(t))}–{f3(max(t))} after; Tez's probe reads {f3(D.td['probe_ece'])}")


# ---------------------------------------------------------------- typed-decisions ladder (horizontal)
def draw_td(fig, spec, ctx):
    D = ctx.D
    td = D.td
    ax = fig.add_subplot(spec)
    items = [("Tez zero-shot", td["zero"], dict(key="tez")),
             ("Tez + 4 examples", td["four"], dict(key="tez")),
             ("Tez + 50 labels/q", td["labels50"], dict(key="tez", hatch=HATCH)),
             ("Tez probe", td["probe"], dict(key="tez", hatch=HATCH)),
             None,
             ("laya-typed-decisions", td["laya_td"], dict(key="laya-typed-decisions")),
             ("Jev (published)", td["jev"], dict(key="jev")),
             ("laya", td["laya"], dict(key="laya")),
             ("laya-multilingual", td["laya_ml"], dict(key="laya-multilingual")),
             None,
             ("teacher ceiling*", LAYA_PUB["teacher_ceiling"], dict(color=REF)),
             ("majority class*", LAYA_PUB["majority_class"], dict(color=REF))]
    y, ys, labs = 0.0, [], []
    for it in items:
        if it is None:
            y += 0.45
            continue
        lab, v, st = it
        bar(ax, y, v, 0.74, horizontal=True, **st)
        is_tez = st.get("key") == "tez"
        hlab(ax, v, y, f3(v), fs=8.6, color=INK if is_tez else INK2, weight="bold" if is_tez else "normal")
        ys.append(y)
        labs.append(lab)
        y += 1.0
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_ylim(y - 0.45, -0.6)
    ax.set_xlim(0, 0.98)
    ax.set_xticks(np.round(np.arange(0, 0.81, 0.2), 1))
    ax.set_xlabel("accuracy", fontsize=10)
    grid(ax, "x")
    strip(ax)
    head(ax, "typed-decisions", note="2,000 decisions · hatched = trained on labels", show_title=ctx.show_title)
    foot(ax, "4 examples in the prompt; 50 labels per question mixed with the zero-shot answers; probe on 300 labels "
             "per question. laya-typed-decisions is fine-tuned on the train split. * Laya's published baselines, not "
             "our rerun.", dy=-30, extra_in=0.0)


def finding_td(D):
    td = D.td
    return (f"typed-decisions: zero-shot Tez ({f3(td['zero'])}) trails Jev's published {f3(td['jev'])}; 4 examples reach "
            f"{f3(td['four'])}, 50 labels per question {f3(td['labels50'])} (level with the fine-tuned "
            f"laya-typed-decisions) and a probe {f3(td['probe'])}")


# ---------------------------------------------------------------- languages as bars (51 from massive51.json)
def draw_lang_bars(fig, spec, ctx):
    D = ctx.D
    m = D.m51
    ax = fig.add_subplot(spec)
    tz = m["systems"]["tez"]
    langs = sorted([lg for lg in m["langs"] if lg in tz], key=lambda lg: -float(tz[lg]))
    thr = 3 * m["random"]
    x = np.arange(len(langs))
    rt = m["systems"].get("laya-routed")
    for xi, lg in zip(x, langs):
        v = float(tz[lg])
        bar(ax, xi, v, 0.74, color=TEZ if v > thr else BELOW)
        if rt and lg in rt:
            ax.plot([xi - 0.37, xi + 0.37], [float(rt[lg])] * 2, color=LAYA_RT, lw=2.6, solid_capstyle="butt", zorder=4)
    ax.axhline(thr, color=INK3, lw=1.3, ls=(0, (4, 3)), zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=90, fontsize=7.5)
    ax.set_xlim(-0.7, len(x) - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    u = m["usable"]
    handles = [patch("tez", f"Tez (clears 3× random in {u['tez']} of {len(langs)})"),
               patch(color=BELOW, label="Tez, below 3× random")]
    if rt:
        handles.append(Line2D([], [], color=LAYA_RT, lw=2.6,
                              label=f"laya-routed on the same rows ({u.get('laya-routed', '?')} of {len(langs)})"))
    handles.append(Line2D([], [], color=INK3, lw=1.3, ls=(0, (4, 3)), label=f"3× random ({thr:.2f})"))
    head(ax, f"All {len(langs)} languages", handles, ncol=len(handles), fs_leg=9,
         note=f"MASSIVE intent · 20 options · 100 rows per language · Laya published "
              f"{LAYA_PUB['routed_usable_51']} of 51 routed (its own run) · Jev publishes none",
         show_title=ctx.show_title)


def finding_lang_bars(D):
    u, t = D.m51["usable"], len(D.m51["langs"])
    s = f"Tez clears 3× random in {u['tez']} of {t} languages"
    if "laya-routed" in u:
        s += f"; Laya's router, on the same rows, in {u['laya-routed']}"
    return s


# ---------------------------------------------------------------- dumbbells: every language, XNLI
def lang_series(D):
    if D.m51:
        m = D.m51
        keys = [k for k in ("tez", "laya", "laya-multilingual") if k in m["systems"]]
        data = {k: {lg: float(v) for lg, v in m["systems"][k].items()} for k in keys}
        return m["langs"], keys, data, m["random"], len(m["langs"])
    return D.langs11, list(KEYS), {k: D.lang_acc[k] for k in KEYS}, 0.05, len(D.langs11)


def _dumbbell(ax, langs, keys, data, rnd, xlabel):
    others = [k for k in keys if k != "tez"]
    langs = [lg for lg in langs if all(lg in data[k] for k in keys)]
    langs = sorted(langs, key=lambda lg: data["tez"][lg] - max(data[k][lg] for k in others))
    y = np.arange(len(langs))
    for yi, lg in zip(y, langs):
        vals = [data[k][lg] for k in keys]
        ax.hlines(yi, min(vals), max(vals), color=GRID, lw=2.4, zorder=1)
    for k in reversed(keys):
        ax.scatter([data[k][lg] for lg in langs], y, s=64 if k == "tez" else 52, color=COLOR[k], zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
    ax.axvline(rnd, color=INK3, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"random guess ({rnd:.3f})", (rnd, len(langs) - 0.45), xytext=(3, 0), textcoords="offset points",
                fontsize=8.5, color=INK3, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(langs, fontsize=9 if len(langs) > 20 else 10)
    ax.set_ylim(-0.7, len(langs) + 0.1)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.xaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    strip(ax)
    return [Line2D([], [], marker="o", ls="", ms=7.5, color=COLOR[k], mec=SURFACE,
                   label=NAME[k] + (" zero-shot" if k == "tez" else "")) for k in keys]


def draw_dumbbell(fig, spec, ctx):
    langs, keys, data, rnd, _ = lang_series(ctx.D)
    ax = fig.add_subplot(spec)
    handles = _dumbbell(ax, langs, keys, data, rnd, "accuracy  ·  MASSIVE intent, 20 options, 100 rows per language")
    head(ax, "Every language: which system can read it", handles, ncol=2 if len(keys) > 3 else len(keys), fs_leg=9.2,
         note="sorted by Tez's lead over the best Laya checkpoint, largest at the top", show_title=ctx.show_title)


def finding_dumbbell(D):
    langs, keys, data, _, n = lang_series(D)
    tz = [data["tez"][lg] for lg in langs if lg in data["tez"]]
    s = f"Tez reads all {n} languages at {min(tz):.3f}–{max(tz):.3f}" if min(tz) > 0.15 else \
        f"Tez ranges {min(tz):.3f}–{max(tz):.3f} over {n} languages"
    lay = data.get("laya", {})
    low = [lg for lg in langs if lg in lay and lay[lg] <= 0.15]
    if low and len(low) <= 6:
        s += f"; laya falls to 3× random or below in {join_and(low)}"
        if "km" in low:
            s += f" (Khmer {lay['km']:.3f})"
    elif low:
        s += f"; laya falls to 3× random or below in {len(low)} of {n}"
    return s


def draw_xnli(fig, spec, ctx):
    D = ctx.D
    keys = ["tez", "laya", "laya-multilingual"]
    data = {k: dict(zip(D.xnli_langs, D.xnli[k]["values"])) for k in keys}
    ax = fig.add_subplot(spec)
    handles = _dumbbell(ax, D.xnli_langs, keys, data, 1 / 3, "accuracy  ·  XNLI, 3 labels, 100 rows per language")
    head(ax, "XNLI: where Laya's multilingual encoder leads", handles, ncol=3, fs_leg=9.2,
         note="NLI is in Laya's training · sorted by Tez's lead, largest at the top", show_title=ctx.show_title)


def finding_xnli(D):
    x = D.xnli
    return (f"XNLI is the family where Laya's trained encoder leads: laya-multilingual macro "
            f"{f3(x['laya-multilingual']['macro'])} vs Tez {f3(x['tez']['macro'])} over {len(D.xnli_langs)} languages")


# ---------------------------------------------------------------- why --swa-full matters
def draw_swa(fig, spec, ctx):
    D = ctx.D
    b, a = D.voice_before, D.voice_after
    sub = spec.subgridspec(1, 2, wspace=0.55)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    panels = [("compute per streamed word, p50 (ms)", b["compute_ms"], a["compute_ms"], "faster"),
              ("tokens evaluated per word", b["tokens_per_word"], a["tokens_per_word"], "fewer")]
    for ax, (ylab, before, after, word) in zip(axes, panels):
        bar(ax, 0, before, 0.56, "tez", alpha=SHIPPED_ALPHA)
        bar(ax, 1, after, 0.56, "tez")
        vlab(ax, 0, before, f"{before}", fs=10)
        vlab(ax, 1, after, f"{after}", fs=10.5, color=INK, weight="bold")
        vlab(ax, 1, after, f"{before / after:.1f}× {word}", fs=10, color=TEZ_INK, weight="bold", dy=19)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before\n--swa-full", "with\n--swa-full"], fontsize=9.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, before * 1.18)
        ax.set_ylabel(ylab, fontsize=9.5)
        grid(ax)
        strip(ax)
    head(axes[0], "Why --swa-full matters", note="Gemma 4 12B Q8_0 on llama-server · voice loop, transcript last "
         "· 220 spoken commands", show_title=ctx.show_title)
    foot(axes[0], "Without --swa-full Gemma's sliding-window cache cannot be rolled back, so every streamed word "
                  "re-evaluated the whole prompt.", dy=-40, extra_in=ax_width_in(axes[1]) + 0.9)


def finding_swa(D):
    b, a = D.voice_before, D.voice_after
    return (f"--swa-full cut the voice loop from {b['compute_ms']} ms to {a['compute_ms']} ms per streamed word by "
            f"evaluating {a['tokens_per_word']} tokens instead of {b['tokens_per_word']}")


# ---------------------------------------------------------------- option order (fills the batching slot without speed data)
def draw_flip(fig, spec, ctx):
    D = ctx.D
    ax = fig.add_subplot(spec)
    tasks = [("massive:en", "MASSIVE (en)\n20 options"), ("emotion", "DAIR Emotion\n6 options"),
             ("ag_news", "AG News\n4 options")]
    w = 0.19
    for ti, (t, _) in enumerate(tasks):
        for j, k in enumerate(KEYS):
            v = D.flip20[k] if t == "massive:en" else D.S[t][H2H_KEY[k]]["flip_rate"]
            xc = ti - 1.5 * w + j * w
            bar(ax, xc, v, w * 0.9, k)
            vlab(ax, xc, v, f"{v:.2f}", fs=8, color=INK if k == "tez" else INK2, weight="bold" if k == "tez" else "normal")
    jv = D.flip20["jev"]
    ax.plot([-0.45, 0.45], [jv, jv], color=INK3, lw=1.3, ls=(0, (4, 3)))
    ax.annotate(f"Jev (published) {jv:.2f}", (0.45, jv), xytext=(3, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK3)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[1] for t in tasks], fontsize=9)
    ax.set_ylim(0, 0.26)
    ax.set_ylabel("share of answers that change", fontsize=10)
    grid(ax)
    strip(ax)
    head(ax, "Option order", [patch(k, NAME[k]) for k in KEYS], ncol=2, fs_leg=8.8,
         note="options reversed on the first 100 rows (lower is better)", show_title=ctx.show_title)


def finding_flip(D):
    f = D.flip20
    return (f"Reversing 20 options changes Tez's answer least ({f['tez']:.2f}; Jev published {f['jev']:.2f}, "
            f"Laya {min(f[k] for k in LAYA3):.2f}–{max(f[k] for k in LAYA3):.2f})")


# ---------------------------------------------------------------- selective automation
def sel_style(name):
    k = canon(name)
    return COLOR.get(k, INK2), ("--" if k == "tez-probe" else "-"), display(name)


def _sys_order(names):
    rank = {"tez": 0, "tez-probe": 1, "laya": 2, "laya-multilingual": 3, "laya-typed-decisions": 4}
    return sorted(names, key=lambda n: (rank.get(canon(n), 9), str(n)))


def draw_selective(fig, spec, ctx):
    D = ctx.D
    s = D.sel
    ax = fig.add_subplot(spec)
    cov = np.array(s["coverage"]) * (100 if max(s["coverage"]) <= 1.0 else 1)
    names = _sys_order(s["systems"])
    ends = []
    for nm in names:
        c, ls, lab = sel_style(nm)
        v = s["systems"][nm]
        tz = canon(nm).startswith("tez")
        ax.plot(cov, v, ls, color=c, lw=2.4 if tz else 2.0, marker="o", ms=6.5, mec=SURFACE, mew=1.3,
                zorder=4 if tz else 3)
        ends.append([v[-1], lab, tz])
    ends.sort(key=lambda e: -e[0])
    last_y = None
    for val, lab, tz in ends:                                    # direct labels at the right end, nudged apart
        yv = val if last_y is None else min(val, last_y - 0.035)
        ax.annotate(lab, (cov[-1], val), xytext=(cov[-1] + 1.8, yv), textcoords="data", va="center", fontsize=9.5,
                    color=INK if tz else INK2, fontweight="bold" if tz else "normal")
        last_y = yv
    rc = list(np.round(cov, 6))
    if 50.0 in rc:
        i50 = rc.index(50.0)
        ax.axvline(50, color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=1)
        tz = [n for n in names if canon(n) == "tez"]
        if tz:
            v = s["systems"][tz[0]][i50]
            ax.annotate(f"{fnum(v)} at 50%", (50, v), xytext=(-8, -16), textcoords="offset points", ha="right",
                        fontsize=9.5, color=TEZ_INK, fontweight="bold")
    ax.set_xlim(cov.min() - 3, cov.max() + 24)
    lo = min(min(v) for v in s["systems"].values())
    ax.set_ylim(max(0, lo - 0.08), 1.02)
    ax.set_xticks([c for c in cov if abs(c - round(c)) < 1e-6 and int(round(c)) % 10 == 0])
    ax.set_xlabel("coverage: share of decisions answered, most confident first (%)", fontsize=10)
    ax.set_ylabel("accuracy on the answered decisions", fontsize=10)
    grid(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.9)
    strip(ax)
    head(ax, "Selective automation: accuracy vs coverage",
         note="typed-decisions test split · 400 cases · 2,000 decisions · confidence = top probability "
              "· no per-row probe predictions exist, so the probe is not drawn", show_title=ctx.show_title)


def finding_selective(D):
    s = D.sel
    cov = [round(c * (100 if max(s["coverage"]) <= 1 else 1), 6) for c in s["coverage"]]
    if 50.0 not in cov:
        return "Accuracy on the decisions each system answers, most confident first"
    i = cov.index(50.0)
    at = {nm: v[i] for nm, v in s["systems"].items()}
    tz = next((n for n in at if canon(n) == "tez"), None)
    best = max(at, key=at.get)
    parts = [f"Answering its most confident half, Tez's letters reach {fnum(at[tz])}" if tz else ""]
    if best != tz:
        extra = " (fine-tuned on this benchmark)" if canon(best) == "laya-typed-decisions" else ""
        parts.append(f"{display(best)}{extra} {fnum(at[best])}")
    base = [n for n in _sys_order(at) if canon(n) in ("laya", "laya-multilingual")]
    if base:
        parts.append("the base Laya checkpoints " + " and ".join(fnum(at[n]) for n in base))
    return "; ".join(p for p in parts if p)


# ---------------------------------------------------------------- reliability and risk-coverage
def binned_ece(pts):
    nn = np.array([p.get("n", 1) for p in pts], float)
    return float((nn * np.abs(np.array([p["conf"] for p in pts]) - np.array([p["acc"] for p in pts]))).sum() / nn.sum())


def draw_reliability(fig, spec, ctx):
    D = ctx.D
    s = D.sel
    ax = fig.add_subplot(spec)
    ax.plot([0, 1], [0, 1], color=INK3, lw=1.3, ls=(0, (4, 3)), zorder=1)
    handles = [Line2D([], [], color=INK3, lw=1.3, ls=(0, (4, 3)), label="perfect calibration")]
    for nm in _sys_order(s["reliability"]):
        rel = s["reliability"][nm]
        c, ls, lab = sel_style(nm)
        is_tez = canon(nm) == "tez"
        for variant, word in (("shipped", "as shipped"), ("temperature", "one temperature")):
            pts = rel.get(variant) or []
            if not pts:
                continue
            cf = np.array([p["conf"] for p in pts])
            ac = np.array([p["acc"] for p in pts])
            nn = np.array([p.get("n", 1) for p in pts], float)
            ece = ((s.get("ece") or {}).get(nm) or {}).get(variant)
            ece_lab = "ECE-15" if ece is not None else "binned ECE"
            if ece is None:
                ece = binned_ece(pts)
            if is_tez:
                alpha = SHIPPED_ALPHA + 0.14 if variant == "shipped" else 1.0
                ax.plot(cf, ac, ls, color=c, alpha=alpha, lw=2.0, zorder=4)
                ax.scatter(cf, ac, s=16 + 300 * np.sqrt(nn / nn.sum()), color=c, alpha=alpha, edgecolors=SURFACE,
                           linewidths=1.2, zorder=5)
                handles.append(Line2D([], [], color=c, alpha=alpha, marker="o", lw=2.0, ms=7,
                                      label=f"{lab}, {word} ({ece_lab} {fnum(ece)})"))
            else:
                ax.plot(cf, ac, "-", color=c, lw=1.2, marker="o", ms=3.8, mec=SURFACE, mew=0.8, alpha=0.95, zorder=3)
                handles.append(Line2D([], [], color=c, marker="o", lw=1.2, ms=4,
                                      label=f"{lab}, {word} ({ece_lab} {fnum(ece)})"))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confidence (top probability)", fontsize=10)
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.9)
    strip(ax, keep=("bottom", "left"))
    head(ax, "Reliability", handles, ncol=2, fs_leg=8.2,
         note=f"typed-decisions, 2,000 decisions · {s.get('bins_n') or 15} equal-width bins · Tez's marker area "
              "grows with the decisions in the bin", show_title=ctx.show_title)


def draw_risk(fig, spec, ctx):
    D = ctx.D
    s = D.sel
    ax = fig.add_subplot(spec)
    names = _sys_order(s["risk"])
    handles = []
    for nm in names:
        c, ls, lab = sel_style(nm)
        r = s["risk"][nm]
        tz = canon(nm) == "tez"
        au = r.get("aurc")
        line, = ax.plot(np.asarray(r["coverage"], float), r["error"], ls, color=c, lw=2.3 if tz else 1.6,
                        zorder=4 if tz else 3, label=lab + (f" (AURC {fnum(au)})" if au is not None else ""))
        handles.append(line)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, None)
    ax.set_xlabel("coverage (answered most confident first)", fontsize=10)
    ax.set_ylabel("error rate on the answered decisions", fontsize=10)
    grid(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.9)
    strip(ax, keep=("bottom", "left"))
    head(ax, "Risk–coverage", handles, ncol=2, fs_leg=8.2,
         note="typed-decisions, 2,000 decisions · lower is better · AURC = area under the curve",
         show_title=ctx.show_title)


def finding_reliability(D):
    s = D.sel
    name = next((n for n in s["reliability"] if canon(n) == "tez"), None)
    if name is None:
        return "Reliability of each readout on typed-decisions"
    e = (s.get("ece") or {}).get(name)
    rel = s["reliability"][name]
    if e and "temperature" in e:
        a, b, lab = e["shipped"], e["temperature"], "ECE-15"
    elif rel.get("shipped") and rel.get("temperature"):
        a, b, lab = binned_ece(rel["shipped"]), binned_ece(rel["temperature"]), "binned ECE"
    else:
        return "Reliability of Tez's letter readout on typed-decisions"
    return f"Tez's letter readout is over-confident as shipped ({lab} {fnum(a)}); one temperature brings it to {fnum(b)}"


def finding_risk(D):
    s = D.sel
    tz = next((n for n in s["systems"] if canon(n) == "tez"), None)
    cov = [round(c, 6) for c in s["coverage"]]
    if tz and 0.5 in cov and 1.0 in cov:
        half, full = s["systems"][tz][cov.index(0.5)], s["systems"][tz][cov.index(1.0)]
    else:
        S = D.S["typed_decisions"]["tez"]
        half, full = S["acc_at_50_coverage"], S["accuracy"]
    s = (f"Tez's errors sit in its least confident answers: {fnum(1 - half)} error on its most confident half "
         f"vs {fnum(1 - full)} on all decisions")
    au = {n: r.get("aurc") for n, r in s_risk(D).items() if r.get("aurc") is not None}
    tz_n = next((n for n in au if canon(n) == "tez"), None)
    better = [n for n in au if tz_n and canon(n) != "tez" and au[n] < au[tz_n]]
    if better:
        b = min(better, key=au.get)
        extra = " (fine-tuned on this benchmark)" if canon(b) == "laya-typed-decisions" else ""
        s += f"; {display(b)}{extra} ranks its errors better (AURC {fnum(au[b])} vs {fnum(au[tz_n])})"
    return s


def s_risk(D):
    return (D.sel or {}).get("risk") or {}


# ================================================================== registry
# ax: axes size in inches when the panel stands alone; left: room for y labels; head/foot: room above/below the axes.
def P(draw, title, finding, what, source, ax, left=0.85, head_in=0.6, foot_in=0.75, needs=None, probe=False):
    return dict(draw=draw, title=title, finding=finding, what=what, source=source, ax=ax, left=left, head=head_in,
                foot=foot_in, needs=needs, probe=probe)


SRC_H2H = "BENCHMARKS.md §1 (experiments/bench_h2h.py, results/h2h/summary.json)"
PANELS = {
    "accuracy_vs_jev": P(
        draw_accuracy, "Accuracy on the public datasets where Jev numbers exist", finding_accuracy,
        lambda D: "Accuracy on the four public datasets with a published Jev number · 400 rows per dataset; "
                  "typed-decisions 2,000 decisions",
        lambda D: f"Source: {SRC_H2H}; Tez probes BENCHMARKS.md §4b, §4c; Jev: third-party published, as "
                  "collected in BENCHMARKS.md §1.",
        (11.0, 4.6), head_in=1.1, foot_in=1.15, probe=True),
    "language_coverage": P(
        draw_coverage, "Language coverage", finding_coverage,
        lambda D: "Languages where each system clears 3× random on MASSIVE intent (20 options, 100 rows per language)",
        lambda D: ("Source: results/vs_laya/massive51.json; Laya published 45 of 51 from Laya's README." if D.m51
                   else "Source: BENCHMARKS.md §1 Languages (results/h2h/summary.json)."),
        (6.2, 3.6), left=1.95, head_in=0.45),
    "speed_per_decision": P(
        draw_speed_decision, "Speed", finding_speed_decision,
        lambda D: "p50 latency per decision on one RTX 5080 laptop GPU; Jev's range is third-party published, over its API",
        lambda D: "Source: BENCHMARKS.md §1 (ms per decision), §4c (depth-pruned GGUFs: probe on 24 of 32 "
                  "blocks), §3 (voice).",
        (6.6, 3.8), left=1.9, head_in=0.5, probe=True),
    "calibration_vs_jev": P(
        draw_calib_jev, "Calibration", finding_calib_jev,
        lambda D: "Expected calibration error with 15 bins (lower is better)",
        lambda D: "Source: BENCHMARKS.md §1 (mean ECE-15), §4c (probe, layer 26); Jev: third-party published.",
        (5.4, 4.0), head_in=0.5, foot_in=1.15, probe=True),
    "and_facts": P(
        draw_and, "And", lambda D: "Tez runs locally on open weights at no cost; Jev is a hosted, paid API",
        lambda D: "Deployment facts, Tez in blue and Jev in grey",
        lambda D: "Source: README.md (licence), tez/schema.py (option cap), BENCHMARKS.md §1 (languages), "
                  "docs/REPORT.md §1 (Jev's documented terms).",
        (6.0, 3.3), left=0.1, head_in=0.4, foot_in=0.2),
    "where_tez_leads": P(
        draw_tez_leads, "Where Tez leads",
        lambda D: "Where Tez leads: language coverage, option-order stability, held-out jailbreak detection and, with a "
                  "probe, typed-decisions",
        lambda D: "Measured on the same rows as Laya, or against Jev's published figure",
        lambda D: "Source: BENCHMARKS.md §1, §4b, §4c" + ("; results/vs_laya/apps.json" if D.apps else "")
                  + "; README.md; docs/REPORT.md §1.",
        (6.2, 4.4), left=0.1, head_in=0.45, foot_in=0.2, probe=True),
    "where_others_lead": P(
        draw_others_lead, "Where others lead", finding_others,
        lambda D: "Every loss, on the same rows as Laya or against Jev's published figure",
        lambda D: "Source: BENCHMARKS.md §1 (head-to-head, XNLI, latency), §4b (probes)"
                  + ("; results/vs_laya/apps.json" if D.apps else "") + ".",
        (6.4, 5.4), left=0.1, head_in=0.45, foot_in=0.2, probe=True),
    "workflows_every_checkpoint": P(
        draw_workflows, "Every workflow, every checkpoint", finding_workflows,
        lambda D: "Accuracy on every task, zero-shot Tez vs Laya's three checkpoints on identical rows",
        lambda D: ("Source: results/vs_laya/apps.json (workflow themes; Laya published figures from its "
                   "app_benchmark_results.json); " if D.apps else "Source: ") + f"{SRC_H2H}.",
        (8.6, 0), left=3.2, head_in=0.85, foot_in=0.65),
    "english_vs_rest": P(
        draw_eng_rest, "English vs the rest", finding_eng_rest,
        lambda D: "English vs the mean of the other languages, MASSIVE intent and XNLI, 100 rows per language",
        lambda D: "Source: results/h2h/summary.json (BENCHMARKS.md §1 Languages, XNLI); the non-English means are "
                  "computed from its per-language rows.",
        (7.6, 4.2), head_in=1.05),
    "speed_per_call": P(
        draw_speed_call, "Speed on one RTX 5080 laptop GPU", finding_speed_call,
        lambda D: "p50 latency per call for 1, 5, 10 and 50 questions per call",
        lambda D: "Source: results/vs_laya/speed_per_call.json; Jev: third-party published (BENCHMARKS.md §1).",
        (7.4, 4.2), head_in=1.0, needs=lambda D: None if D.spc else "results/vs_laya/speed_per_call.json"),
    "batching": P(
        draw_batching, "Batching pays off", finding_batching,
        lambda D: "p50 milliseconds per question vs questions per call",
        lambda D: "Source: results/vs_laya/speed_per_call.json (p50_ms_per_question).",
        (7.0, 4.2), head_in=1.0,
        needs=lambda D: None if (D.spc and any(v["per_q"] for v in D.spc["systems"].values()))
        else "results/vs_laya/speed_per_call.json"),
    "calibration_shipped_vs_temperature": P(
        draw_calib_all, "Calibration", finding_calib_all,
        lambda D: "Mean ECE-15 over the head-to-head tasks, as shipped and after one temperature per task (lower is better)",
        lambda D: "Source: BENCHMARKS.md §1 (Calibration, order robustness, latency); probe: §4c.",
        (7.2, 4.2), head_in=0.95, foot_in=0.8, probe=True),
    "typed_decisions_ladder": P(
        draw_td, "typed-decisions", finding_td,
        lambda D: "typed-decisions test split: 400 cases, 2,000 decisions, four workflows",
        lambda D: "Source: BENCHMARKS.md §1, §4b (4 examples), §4c (few labels, probe); teacher ceiling and "
                  "majority class: Laya's README (published).",
        (6.2, 4.4), left=1.75, head_in=0.5, foot_in=1.0, probe=True),
    "all_51_languages": P(
        draw_lang_bars, "All 51 languages", finding_lang_bars,
        lambda D: "MASSIVE intent in every language, Tez vs Laya's router on identical rows",
        lambda D: "Source: results/vs_laya/massive51.json; Laya published 45 of 51 from Laya's README.",
        (15.5, 4.4), head_in=0.85, foot_in=0.9, needs=lambda D: None if D.m51 else "results/vs_laya/massive51.json"),
    "swa_full": P(
        draw_swa, "Why --swa-full matters", finding_swa,
        lambda D: "Voice loop per streamed word, before and after llama-server's --swa-full",
        lambda D: "Source: BENCHMARKS.md §3 (table and the sentence on --swa-full).",
        (6.8, 3.8), head_in=0.5, foot_in=1.05),
    "every_language": P(
        draw_dumbbell, "Every language: which system can read it", finding_dumbbell,
        lambda D: "MASSIVE intent per language, 20 options, 100 rows per language",
        lambda D: ("Source: results/vs_laya/massive51.json." if D.m51 else
                   "Source: BENCHMARKS.md §1 Languages (results/h2h/summary.json)."),
        (7.2, 0), left=0.8, head_in=0.85),
    "xnli_languages": P(
        draw_xnli, "XNLI: where Laya's multilingual encoder leads", finding_xnli,
        lambda D: "XNLI per language, 3 labels, 100 rows per language",
        lambda D: "Source: BENCHMARKS.md §1 XNLI (results/h2h/summary.json).",
        (7.2, 4.4), left=0.8, head_in=0.7),
    "option_order": P(
        draw_flip, "Option order", finding_flip,
        lambda D: "Share of answers that change when the options are reversed (first 100 rows)",
        lambda D: "Source: BENCHMARKS.md §1 (option-order flip; results/h2h/summary.json flip_rate).",
        (6.8, 4.0), head_in=1.0),
    "selective": P(
        draw_selective, "Selective automation: accuracy vs coverage", finding_selective,
        lambda D: "Accuracy on the decisions each system answers, most confident first",
        lambda D: f"Source: {D.sel_source}" + (" (typed-decisions rows in results/h2h; ties at a cut counted pro rata)."
                                                if D.sel_source and D.sel_source.endswith(".json") else "."),
        (7.8, 4.6), head_in=0.6, needs=lambda D: None if D.sel else "results/vs_laya/selective.json"),
    "reliability": P(
        draw_reliability, "Reliability", finding_reliability,
        lambda D: "Reliability diagram on typed-decisions: Tez's letter readout, with Laya's checkpoints for context",
        lambda D: f"Source: {D.sel_source} (reliability bins and ECE-15 from the typed-decisions rows in results/h2h).",
        (6.0, 5.2), head_in=1.35, needs=lambda D: None if (D.sel and D.sel["reliability"]) else "results/vs_laya/selective.json"),
    "risk_coverage": P(
        draw_risk, "Risk–coverage", finding_risk,
        lambda D: "Error rate among the answered decisions as coverage grows, typed-decisions",
        lambda D: f"Source: {D.sel_source} (risk-coverage and AURC from the typed-decisions rows in results/h2h).",
        (6.0, 5.2), head_in=1.0, needs=lambda D: None if (D.sel and D.sel["risk"]) else "results/vs_laya/selective.json"),
}


def available(D, key):
    n = PANELS[key]["needs"]
    return (n(D) if n else None) is None


def missing(D, key):
    n = PANELS[key]["needs"]
    return n(D) if n else None


def ax_size(D, key):
    w, h = PANELS[key]["ax"]
    if key == "workflows_every_checkpoint":
        h = 0.5 * len(workflow_rows(D))
    if key == "every_language":
        h = (0.26 if D.m51 else 0.42) * len(lang_series(D)[0])
    if key == "where_tez_leads":
        h = big_rows_height(leads(D), w) + 0.1
    if key == "where_others_lead":
        h = big_rows_height(losses(D), w, marker=True) + 0.1
    if key == "and_facts":
        h = (4 + 44 * len(and_rows(D))) / 72
    return w, h


# ================================================================== layout in inches
# A layout is a tree of ("panel", key, w, h), ("gap", None, w, h), ("rows", [..]) and ("cols", [..]) nodes;
# sizes are inches, so every row, column and gap has a fixed physical size and the page is their sum.
def Pn(key, w, h):
    return ("panel", key, w, h)


def G(w=0.0, h=0.0):
    return ("gap", None, w, h)


def R(*kids):
    return ("rows", list(kids))


def C(*kids):
    return ("cols", list(kids))


def measure(node):
    if node[0] in ("panel", "gap"):
        return node[2], node[3]
    sizes = [measure(k) for k in node[1]]
    if node[0] == "rows":
        return max(s[0] for s in sizes), sum(s[1] for s in sizes)
    return sum(s[0] for s in sizes), max(s[1] for s in sizes)


def place(fig, spec, node, ctx, log):
    kind = node[0]
    if kind == "gap":
        return
    if kind == "panel":
        key = node[1]
        if not available(ctx.D, key):
            log.append(f"panel '{key}' ({missing(ctx.D, key)} missing)")
            return
        PANELS[key]["draw"](fig, spec, ctx)
        return
    kids = node[1]
    sizes = [measure(k) for k in kids]
    if kind == "rows":
        sub = spec.subgridspec(len(kids), 1, height_ratios=[max(s[1], 1e-3) for s in sizes], hspace=0)
        for i, k in enumerate(kids):
            place(fig, sub[i, 0], k, ctx, log)
    else:
        sub = spec.subgridspec(1, len(kids), width_ratios=[max(s[0], 1e-3) for s in sizes], wspace=0)
        for i, k in enumerate(kids):
            place(fig, sub[0, i], k, ctx, log)


MARGIN, TOP0, BOT0 = 0.3, 0.22, 0.18


def text_block_height(lines, width_in):
    h = TOP0
    for text, fs, weight, _ in lines:
        t = wrap(text, width_in, fs, bold=(weight == "bold"))
        h += (t.count("\n") + 1) * fs * 1.22 * 1.08 / 72 + 0.07
    return h


def footer_height(text, width_in, fs):
    t = wrap(text, width_in, fs)
    return BOT0 + (t.count("\n") + 1) * fs * 1.3 / 72 + 0.1


def compose(D, path, tree, lines, sources, top_pad, bottom_pad, show_title=True, fs_src=8.8):
    """Lay the tree out in inches under a header and over a source footer. The first pass measures the room the top
    row's titles, legends and notes need; the page is laid out again with exactly that room."""
    tw, th = measure(tree)
    W = tw + 2 * MARGIN
    text_w = W - 2 * MARGIN
    hh = text_block_height(lines, text_w)
    fh = footer_height(sources, text_w, fs_src)
    for attempt in (0, 1):
        H = hh + top_pad + th + bottom_pad + fh
        fig = plt.figure(figsize=(W, H))
        y = TOP0
        for text, fs, weight, color in lines:
            t = wrap(text, text_w, fs, bold=(weight == "bold"))
            fig.text(MARGIN / W, 1 - y / H, t, fontsize=fs, fontweight=weight, color=color, va="top", ha="left",
                     linespacing=1.22)
            y += (t.count("\n") + 1) * fs * 1.22 * 1.08 / 72 + 0.07
        fig.text(MARGIN / W, BOT0 / H, wrap(sources, text_w, fs_src), fontsize=fs_src, color=INK3, va="bottom",
                 ha="left", linespacing=1.3)
        area = fig.add_gridspec(1, 1, left=MARGIN / W, right=1 - MARGIN / W, top=1 - (hh + top_pad) / H,
                                bottom=(fh + bottom_pad) / H)[0, 0]
        log = []
        place(fig, area, tree, Ctx(D, show_title=show_title), log)
        align_titles(fig)
        need = top_head_in(fig) + 0.12
        if attempt == 0 and abs(need - top_pad) > 0.04:
            plt.close(fig)
            top_pad = need
            continue
        break
    for msg in log:
        print(f"  skipped  {path.name}: {msg}")
        RUN["skipped"].append(f"{path.name}: {msg}")
    save(fig, path)


RUN = {"produced": [], "skipped": [], "composites": {}}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    RUN["produced"].append(path)
    shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    print(f"  wrote    {shown}  ({path.stat().st_size / 1024:.0f} KB)")


def setup_lines(probe, apps=False):
    s = TEZ_LETTERS + (" " + TEZ_PROBE if probe else "")
    return s + (" " + APPS_CPU if apps else "")


def render_panel(D, key, path=None):
    p = PANELS[key]
    if not available(D, key):
        return False
    w, h = ax_size(D, key)
    tree = C(G(p["left"], 0), Pn(key, w, h), G(0.4, 0))
    apps = key in ("workflows_every_checkpoint",) and bool(D.apps)
    lines = [(p["finding"](D), 15, "bold", INK), (p["what"](D), 10.5, "normal", INK2),
             (CONDITIONS + " " + setup_lines(p["probe"], apps), 9, "normal", INK3)]
    compose(D, path or (PANEL_DIR / f"{key}.png"), tree, lines, p["source"](D), p["head"], p["foot"],
            show_title=False, fs_src=8.5)
    return True


def composite_sources(D, keys, name=None):
    if name:
        RUN["composites"][name] = [k for k in keys if available(D, k)]
    seen, out = set(), []
    for k in keys:
        if not available(D, k):
            continue
        s = PANELS[k]["source"](D).replace("Source: ", "").rstrip(".")
        if s not in seen:
            seen.add(s)
            out.append(f"{PANELS[k]['title']}: {s}")
    return "Sources — " + " · ".join(out) + "."


def comp_lines(title, finding, probe=True, apps=False):
    lines = [(title, 25, "bold", INK)] if title else []
    return lines + [(finding, 14.5, "bold", INK), (CONDITIONS, 10.5, "normal", INK3),
                    (setup_lines(probe, apps), 10.5, "normal", INK3)]


# ================================================================== figures
def fig_vs_jev(D):
    tree = R(C(G(0.85), Pn("accuracy_vs_jev", 10.6, 4.4), G(2.3), Pn("language_coverage", 3.9, 4.4)),
             G(0, 1.75),
             C(G(2.05), Pn("speed_per_decision", 3.9, 3.7), G(1.1), Pn("calibration_vs_jev", 4.2, 3.7), G(0.6),
               Pn("and_facts", 5.8, 3.7)))
    keys = ["accuracy_vs_jev", "language_coverage", "speed_per_decision", "calibration_vs_jev", "and_facts"]
    n_use = D.m51["usable"]["tez"] if D.m51 else D.usable11["tez"]
    n_tot = len(D.m51["langs"]) if D.m51 else len(D.langs11)
    finding = finding_accuracy(D) + f"; it reads {n_use} of {n_tot} languages tested and runs locally at no cost."
    compose(D, OUT / "tez_vs_jev.png", tree, comp_lines("Tez  vs  TypeSafe Jev", finding),
            composite_sources(D, keys, "tez_vs_jev.png"), top_pad=1.15, bottom_pad=1.0)


def fig_vs_jev_full(D):
    speed = "speed_per_call" if available(D, "speed_per_call") else "speed_per_decision"
    n_rows = len(workflow_rows(D))
    row1_h = max(4.5, big_rows_height(losses(D), 5.15, marker=True) + 0.15, big_rows_height(leads(D), 5.1) + 0.15)
    if speed == "speed_per_decision":
        row3 = C(G(0.85), Pn("english_vs_rest", 5.1, 3.9), G(2.05), Pn(speed, 3.95, 3.9), G(1.0),
                 Pn("calibration_shipped_vs_temperature", 4.35, 3.9), G(1.75), Pn("typed_decisions_ladder", 3.55, 3.9))
    else:
        row3 = C(G(0.85), Pn("english_vs_rest", 5.3, 3.9), G(1.0), Pn(speed, 5.0, 3.9), G(1.0),
                 Pn("calibration_shipped_vs_temperature", 4.55, 3.9), G(1.75), Pn("typed_decisions_ladder", 3.05, 3.9))
    if available(D, "all_51_languages"):
        row4 = C(G(0.85), Pn("all_51_languages", 14.6, 4.3), G(1.0), Pn("swa_full", 6.05, 4.3))
        row4_keys = ["all_51_languages", "swa_full"]
    else:
        row4 = C(G(0.85), Pn("every_language", 6.5, 4.3), G(1.05), Pn("xnli_languages", 6.5, 4.3), G(1.05),
                 Pn("swa_full", 6.55, 4.3))
        row4_keys = ["every_language", "xnli_languages", "swa_full"]
    tree = R(C(G(0.85), Pn("accuracy_vs_jev", 10.3, row1_h), G(0.75), Pn("where_tez_leads", 5.1, row1_h), G(0.35),
               Pn("where_others_lead", 5.15, row1_h)),
             G(0, 1.95),
             C(G(3.25), Pn("workflows_every_checkpoint", 19.25, 0.44 * n_rows)),
             G(0, 2.0),
             row3,
             G(0, 1.95),
             row4)
    keys = ["accuracy_vs_jev", "where_tez_leads", "where_others_lead", "workflows_every_checkpoint", "english_vs_rest",
            speed, "calibration_shipped_vs_temperature", "typed_decisions_ladder"] + row4_keys
    finding = (f"Tez leads on languages, option order and, with a probe, typed-decisions ({f3(D.td['probe'])}); Laya is "
               "faster and leads AG News and XNLI, both in its training; Jev's published figures lead Banking77 and "
               "zero-shot typed-decisions.")
    compose(D, OUT / "tez_vs_jev_full.png", tree,
            comp_lines("Tez  vs  TypeSafe Jev, with Laya on the same rows", finding, apps=bool(D.apps)),
            composite_sources(D, keys, "tez_vs_jev_full.png"), top_pad=1.15, bottom_pad=1.0)


def fig_benchmark(D):
    speed = "speed_per_call" if available(D, "speed_per_call") else "speed_per_decision"
    PANELS["accuracy_vs_jev_compact"] = dict(PANELS["accuracy_vs_jev"],
                                             draw=lambda f, s, c: draw_accuracy(f, s, c, context=False),
                                             title="Against Jev, same public datasets")
    right = R(Pn(speed, 6.9, 3.0), G(0, 1.75), Pn("accuracy_vs_jev_compact", 6.9, 3.3), G(0, 1.95),
              Pn("calibration_shipped_vs_temperature", 6.9, 3.0))
    if D.m51:
        left = Pn("every_language", 7.4, 13.0)
        keys = ["every_language", speed, "accuracy_vs_jev", "calibration_shipped_vs_temperature"]
    else:
        left = R(Pn("every_language", 7.4, 5.3), G(0, 1.75), Pn("xnli_languages", 7.4, 4.9))
        keys = ["every_language", "xnli_languages", speed, "accuracy_vs_jev", "calibration_shipped_vs_temperature"]
    tree = C(G(0.75), left, G(2.4 if speed == "speed_per_decision" else 1.2), right)
    finding = finding_dumbbell(D) + (f"; XNLI is the exception, where laya-multilingual leads "
                                     f"({f3(D.xnli['laya-multilingual']['macro'])} vs {f3(D.xnli['tez']['macro'])} macro)."
                                     if not D.m51 else ".")
    compose(D, OUT / "tez_benchmark.png", tree,
            comp_lines("Tez benchmark  ·  every system answered byte-identical questions", finding),
            composite_sources(D, keys, "tez_benchmark.png"), top_pad=1.0, bottom_pad=0.95)
    del PANELS["accuracy_vs_jev_compact"]


def fig_benchmark_common(D):
    speed = "speed_per_call" if available(D, "speed_per_call") else "speed_per_decision"
    lower = "batching" if available(D, "batching") else "option_order"
    mid = R(Pn("english_vs_rest", 5.3, 3.6), G(0, 1.95), Pn(lower, 5.3, 3.6))
    right = R(Pn(speed, 4.4, 3.6), G(0, 1.95), Pn("typed_decisions_ladder", 4.4, 3.6))
    wf_h = max(9.15, 0.5 * len(workflow_rows(D)))
    tree = C(G(3.25), Pn("workflows_every_checkpoint", 5.4, wf_h), G(1.05), mid, G(2.1), right)
    keys = ["workflows_every_checkpoint", "english_vs_rest", speed, lower, "typed_decisions_ladder"]
    compose(D, OUT / "tez_benchmark_common.png", tree,
            comp_lines("Tez  ·  consolidated benchmark", finding_workflows(D) + ".", apps=bool(D.apps)),
            composite_sources(D, keys, "tez_benchmark_common.png"), top_pad=1.25, bottom_pad=1.0)


def fig_selective(D):
    if not available(D, "selective"):
        print(f"  skipped  tez_selective.png ({missing(D, 'selective')} missing)")
        RUN["skipped"].append(f"tez_selective.png ({missing(D, 'selective')} missing)")
        return
    RUN["composites"]["tez_selective.png"] = ["selective"]
    render_panel(D, "selective", OUT / "tez_selective.png")


def fig_reliability(D):
    keys = [k for k in ("reliability", "risk_coverage") if available(D, k)]
    if not keys:
        print("  skipped  tez_reliability.png (results/vs_laya/selective.json missing)")
        return
    kids = [G(0.85)]
    for i, k in enumerate(keys):
        kids += ([G(1.2)] if i else []) + [Pn(k, 6.0, 5.2)]
    finding = "; ".join(f(D) for f, k in ((finding_reliability, "reliability"), (finding_risk, "risk_coverage"))
                        if k in keys) + "."
    compose(D, OUT / "tez_reliability.png", C(*kids), comp_lines(None, finding, probe=False),
            composite_sources(D, keys, "tez_reliability.png"), top_pad=1.45, bottom_pad=0.8)


FIGURE_DOC = [
    ("tez_vs_jev.png", "laya_vs_jev.png",
     "Tez against Jev's published figures on one page: accuracy on the four public datasets that have a Jev number "
     "(Tez's probe and Laya's three checkpoints for context), language coverage, speed, calibration and the "
     "deployment facts."),
    ("tez_vs_jev_full.png", "laya_vs_jev_full.png",
     "The full comparison: accuracy against Jev, where Tez leads beside where others lead, every workflow and task on "
     "identical rows, English vs the rest, speed, calibration, the typed-decisions ladder, languages, and why "
     "--swa-full matters (in place of Laya's \"Why preload matters\")."),
    ("tez_benchmark.png", "laya_benchmark.png",
     "Which system can read each language (dumbbells), beside speed, accuracy against Jev and calibration."),
    ("tez_benchmark_common.png", "laya_benchmark_common.png",
     "Consolidated benchmark: every workflow and task, English vs the rest, speed, batching (milliseconds per "
     "question as more questions share one call) and typed-decisions."),
    ("tez_selective.png", "benchmark_comparison.png (bottom-right panel)",
     "Selective automation on typed-decisions: accuracy on the decisions each system answers, most confident first."),
    ("tez_reliability.png", "reliability_eval_zs.png",
     "Reliability diagram and risk-coverage curve of Tez's letter readout on typed-decisions, with Laya's checkpoints "
     "for context."),
]


def compact_sources(srcs):
    """One de-duplicated list of the files and BENCHMARKS.md sections behind a set of panel source lines."""
    text = " ; ".join(srcs)
    tail = []
    if "docs/REPORT.md" in text:
        tail.append("docs/REPORT.md §1")
        text = text.replace("docs/REPORT.md §1", "")
    secs = sorted(set(re.findall(r"§(\d+[a-z]?)", text)), key=lambda s: (int(re.match(r"\d+", s).group()), s))
    out = ["BENCHMARKS.md " + ", ".join("§" + s for s in secs)] if secs else []
    for f in re.findall(r"(?:results|site|experiments|tez)/[\w./*-]*\w|(?<!Laya's )README\.md", text):
        if f not in out:
            out.append(f)
    if "Laya's README" in text:
        out.append("Laya's README (published baselines)")
    if "Jev: third-party" in text:
        out.append("Jev: third-party published")
    return "; ".join(out + tail)


def write_readme(D):
    produced = {p.name for p in RUN["produced"] if p.parent == OUT}
    panels = sorted(p.name for p in RUN["produced"] if p.parent == PANEL_DIR)
    L = ["# Tez against Jev and Laya: Laya's comparison charts, redrawn", "",
         "Generated by `python experiments/make_vs_figures.py`, which also rewrites this file; do not edit it by hand.",
         "",
         "Laya (github.com/NandhaKishorM/laya) publishes its comparison with TypeSafe Jev as a set of charts drawn by "
         "its `research/scripts/make_plots.py`. Each image here redraws one of them in the same layout, surface, "
         "typography, value labels and delta annotations, with Tez in Laya's blue position, Jev as the grey published "
         "reference and Laya's three checkpoints as comparators. Every title states the finding; every image carries "
         "its conditions and a source line.", "",
         f"**Conditions.** {CONDITIONS} {TEZ_LETTERS} {TEZ_PROBE}" + (f" {APPS_CPU}" if D.apps else ""), "",
         "**Colours.** Tez blue (hatched = the probe, trained on labelled rows; tinted = as shipped or before a fix); "
         "laya magenta; laya-multilingual amber; laya-typed-decisions green; laya-routed burnt orange; Jev (published) "
         "grey. The set was checked with a colour-vision-deficiency validator; amber and magenta are light, so every "
         "mark carries its value.", "",
         "**Numbers.** Values from BENCHMARKS.md and site/data/benchmarks.json keep their three decimals; values from "
         "results/vs_laya are printed as the file gives them (up to four decimals). Computed here rather than copied: "
         "the deltas against Jev, the probe's 4.1-4.8x against Jev's published p50 (236/58 and 276/58), the --swa-full "
         "ratios (205/30 and 394/11), the non-English means in \"English vs the rest\" (means of the per-language rows "
         "of results/h2h/summary.json), the counts of usable languages (> 3x random) and the task counts in the "
         "titles.", "",
         "**Laya's own published numbers** (not our rerun) appear only where labelled: 45 of 51 languages usable with "
         "routing, the typed-decisions teacher ceiling (0.735) and majority class (0.461), and the laya_published "
         "column of results/vs_laya/apps.json (hollow diamonds; outlined bars where our rerun of that checkpoint is "
         "not in the file yet).", "",
         "## Figures", "", "| Image | Laya's original | What it shows | Panels | Sources |", "|---|---|---|---|---|"]
    for name, orig, what in FIGURE_DOC:
        if name not in produced:
            continue
        keys = RUN["composites"].get(name, [])
        titles = ", ".join(PANELS[k]["title"] if k in PANELS else "Against Jev, same public datasets" for k in keys)
        srcs = compact_sources([PANELS[k]["source"](D) for k in keys if k in PANELS])
        L.append(f"| [`{name}`]({name}) | {orig} | {what} | {titles} | {srcs} |")
    L += ["", "## Panels, each also saved alone in `panels/`", "", "| Image | Finding (its title) | Source |",
          "|---|---|---|"]
    for fn in panels:
        key = fn[:-4]
        if key in PANELS:
            L.append(f"| [`panels/{fn}`](panels/{fn}) | {PANELS[key]['finding'](D)} | "
                     f"{PANELS[key]['source'](D).replace('Source: ', '').rstrip('.')} |")
    missing_keys = [k for k in PANELS if not available(D, k)]
    if missing_keys or RUN["skipped"]:
        L += ["", "## Not drawn in this run", "",
              "A panel whose data file does not exist yet is left out (never drawn with placeholder numbers); rerun "
              "the script once the file is written."]
        for k in missing_keys:
            L.append(f"- `panels/{k}.png` ({PANELS[k]['title']}): needs `{missing(D, k)}`.")
        if not available(D, "speed_per_call"):
            L.append("- Until `speed_per_call.json` exists, the composites show p50 per decision (BENCHMARKS.md "
                     "section 1) in the speed slot, and `tez_benchmark_common.png` shows option order in place of "
                     "batching.")
        if not available(D, "all_51_languages"):
            L.append("- Until `massive51.json` exists, the language panels use the 11 MASSIVE languages of "
                     "BENCHMARKS.md section 1 and the full composite adds the XNLI dumbbell.")
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote    {(OUT / 'README.md').relative_to(ROOT).as_posix()}")


def main():
    print("data:")
    D = Data()
    OUT.mkdir(parents=True, exist_ok=True)
    print("figures:")
    fig_vs_jev(D)
    fig_vs_jev_full(D)
    fig_benchmark(D)
    fig_benchmark_common(D)
    fig_selective(D)
    fig_reliability(D)
    print("panels:")
    for key in PANELS:
        if key == "selective":
            continue                                   # tez_selective.png is already a single panel
        if key == "option_order" and available(D, "batching"):
            stale = PANEL_DIR / "option_order.png"     # only drawn while speed data is missing
            if stale.exists():
                stale.unlink()
            continue
        if not render_panel(D, key):
            print(f"  skipped  panels/{key}.png ({missing(D, key)} missing)")
    write_readme(D)


if __name__ == "__main__":
    main()
