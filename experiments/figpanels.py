"""The frame shared by the individual result charts in docs/figures/panels/ and by the grids built from them.

make_dashboard.py and make_lab_figure.py each declare their panels as a list of Panel entries: a draw function, a
short title for the grid, a finding (the chart's title, a sentence built from the data), a line on what is plotted,
the conditions and a source line. This module turns one entry into

  - a chart on its own (docs/figures/panels/<key>.png): 9 inches wide at 200 dpi, sized for a README column (about
    900 px as displayed); the finding as its title, the conditions under it, the source in a small footer;
  - one cell of a grid (docs/figures/tez_dashboard.png, docs/figures/tez_lab.png), with the short title.

The look is docs/figures/vs/'s (experiments/make_vs_figures.py): its surface, ink, grid and fonts, Tez blue, Laya's
three checkpoints and Jev in their colours; hatched = a probe trained on labelled rows, tinted = as shipped or before
a fix, outlined = Tez at temperature 1, before its default temperature. Series that are not a system (a policy, a
readout, a model size) are drawn in Tez's blues and neutral greys, the one the finding is about in Tez blue.

docs/figures/panels/README.md lists every chart with its finding and source; write_index() rewrites it from both
generators' panel lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

import make_vs_figures as vs  # noqa: E402  (palette, fonts, rcParams and small drawing helpers)

ROOT = vs.ROOT
PANEL_DIR = ROOT / "docs" / "figures" / "panels"
DPI = vs.DPI

# ------------------------------------------------------------------ palette (vs's, plus neutral series)
SURFACE, INK, INK2, INK3, GRID, REF = vs.SURFACE, vs.INK, vs.INK2, vs.INK3, vs.GRID, vs.REF
TEZ, TEZ_INK = vs.TEZ, vs.TEZ_INK
LAYA_EN, LAYA_ML, LAYA_TD, JEV = vs.LAYA_EN, vs.LAYA_ML, vs.LAYA_TD, vs.JEV
HATCH, SHIPPED_ALPHA, BEFORE_LW = vs.HATCH, vs.SHIPPED_ALPHA, vs.BEFORE_LW
TEZ_DEEP = "#1f3a93"                 # a second Tez series (the 12B beside the 4B, a variant beside the default)
TEZ_LIGHT = "#9db5f7"                # a third, lighter Tez series
NEUTRAL, NEUTRAL2 = "#52514e", "#a9a8a2"   # series that are neither Tez nor a named competitor
BAD = "#b91c1c"                      # DESIGN.md danger-red: only for a result that is harmful
COLOR = {"tez": TEZ, "laya-en": LAYA_EN, "laya-ml": LAYA_ML, "laya-td": LAYA_TD, "jev": JEV}
NAME = {"tez": "Tez", "laya-en": "laya", "laya-ml": "laya-multilingual", "laya-td": "laya-typed-decisions",
        "jev": "Jev (published)"}

strip, grid, vlab, hlab, bar, patch, wrap, text_on = vs.strip, vs.grid, vs.vlab, vs.hlab, vs.bar, vs.patch, vs.wrap, vs.text_on
fms = vs.fms


def f3(v):
    """Three decimals as Python formats the stored value, as docs/figures/vs/ and most of BENCHMARKS.md print them
    (0.7665 -> 0.766, 0.3625 -> 0.362)."""
    return f"{float(v):.3f}"


# ------------------------------------------------------------------ geometry (inches)
FIG_W = 9.0                          # every chart on its own is this wide: one README column at 2x
MARGIN = 0.3
TOP0, BOT0 = 0.22, 0.18
FS_FINDING, FS_WHAT, FS_SETUP, FS_SRC = 14.5, 10.5, 9.0, 8.5
GRID_TITLE_IN = 0.42                 # room for a panel's short title in a grid


def J(rel, default=None):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def pct(v, nd=0):
    return f"{100 * v:.{nd}f}\u00a0%"                  # a no-break space: "40 %" never wraps between its parts


def join_and(items):
    items = list(items)
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def line(key=None, label=None, color=None, marker="o", ls="-", lw=2.0, ms=6, mfc=None):
    c = color or COLOR[key]
    return Line2D([], [], color=c, marker=marker, ls=ls, lw=lw, ms=ms, mfc=mfc or c, mec=c, label=label)


def dot(color, label, marker="o", ms=8, hollow=False):
    return Line2D([], [], color=color, marker=marker, ls="", ms=ms, mfc="none" if hollow else color, mec=color,
                  mew=1.6, label=label)


def refline(ax, y, text, color=INK3, ls=(0, (4, 3)), x=0.99, ha="right", va="bottom", fs=8.5, horizontal=True):
    """A reference value (a published figure, a baseline) as a thin dashed line with its label on the line."""
    if horizontal:
        ax.axhline(y, color=color, lw=1.2, ls=ls, zorder=2)
        ax.annotate(text, (x, y), xycoords=("axes fraction", "data"), xytext=(0, 3 if va == "bottom" else -3),
                    textcoords="offset points", ha=ha, va=va, fontsize=fs, color=color, zorder=6)
    else:
        ax.axvline(y, color=color, lw=1.2, ls=ls, zorder=2)
        ax.annotate(text, (y, x), xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="top", fontsize=fs, color=color, zorder=6)


# ------------------------------------------------------------------ panels
@dataclass
class Panel:
    key: str                                  # file name: docs/figures/panels/<key>.png
    short: str                                # the grid's panel title
    finding: Callable                         # finding(D) -> the chart's title, a sentence built from the data
    what: str | Callable                      # what is plotted (one line)
    setup: str | Callable                     # conditions: model, rows, machine
    source: str | Callable                    # "BENCHMARKS.md §..." and the result files
    draw: Callable                            # draw(ctx, D)
    h: float = 3.6                            # axes height
    left: float = 0.9                         # room for y tick labels
    right: float = 0.3
    needs: tuple = ()                         # result files; a missing one skips the panel
    section: str = ""                         # README index group

    def text(self, attr, D):
        v = getattr(self, attr)
        return v(D) if callable(v) else v

    def missing(self):
        return [n for n in self.needs if not (ROOT / n).exists()]


class Ctx:
    """Where a panel draws: a rectangle in inches on a figure. ax() gives one axes filling it, stack() and row()
    give small multiples (never two y-scales on one plot); legend() puts the legend and a note above the axes and,
    in a grid, the short title above them."""

    def __init__(self, fig, rect, grid_mode=False, title=None):
        self.fig, self.rect, self.grid_mode, self.title = fig, rect, grid_mode, title
        self.axes, self._head_ax, self._legends, self._subs = [], None, [], []

    def _add(self, x, y, w, h, **kw):
        W, H = self.fig.get_size_inches()
        ax = self.fig.add_axes([x / W, y / H, w / W, h / H], **kw)
        self.axes.append(ax)
        return ax

    def ax(self):
        x, y, w, h = self.rect
        a = self._add(x, y, w, h)
        self._head_ax = self._head_ax or a
        return a

    def stack(self, ratios=(1, 1), gap=0.35, sharex=True):
        """Axes stacked top to bottom (heights in proportion to ratios), sharing x; returns them top first."""
        x, y, w, h = self.rect
        avail = h - gap * (len(ratios) - 1)
        out, top = [], y + h
        for i, r in enumerate(ratios):
            hh = avail * r / sum(ratios)
            a = self._add(x, top - hh, w, hh, sharex=out[0] if (sharex and out) else None)
            out.append(a)
            top -= hh + gap
        for a in out[:-1]:
            a.tick_params(labelbottom=False)
        self._head_ax = self._head_ax or out[0]
        return out

    def row(self, ratios=(1, 1), gap=0.6):
        """Axes side by side (widths in proportion to ratios); returns them left first."""
        x, y, w, h = self.rect
        avail = w - gap * (len(ratios) - 1)
        out, left = [], x
        for r in ratios:
            ww = avail * r / sum(ratios)
            out.append(self._add(left, y, ww, h))
            left += ww + gap
        self._head_ax = self._head_ax or out[0]
        return out

    def legend(self, ax, handles=None, ncol=None, note=None, fs=9.3):
        self._legends.append((ax, handles, ncol, note, fs))

    def subtitle(self, ax, text, fs=10):
        """A small bold label right above one of several small multiples (under any legend or note)."""
        self._subs.append((ax, text, fs))

    def finish(self):
        base = {}
        for ax, text, fs in self._subs:
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=fs, fontweight="bold", color=INK2)
            base[id(ax)] = 5 + fs * 1.35 + 4
        for ax, handles, ncol, note, fs in self._legends:
            _head(ax, self.title if self.grid_mode else None, handles, ncol=ncol, note=note, show_title=self.grid_mode,
                  fs_leg=fs, base=base.get(id(ax), 4.0))
        if self.grid_mode and not self._legends and self._head_ax is not None:
            _head(self._head_ax, self.title, show_title=True, base=base.get(id(self._head_ax), 4.0))


def _head(ax, title, handles=None, ncol=None, note=None, show_title=True, fs_leg=9.3, base=4.0):
    """vs.head (Laya's panel head: bold left title, the legend under it, a small grey note, then the plot), starting
    `base` points above the axes so a small multiple's own label stays under it."""
    y = base
    if note:
        note = wrap(note, vs.ax_width_in(ax) + 0.4, 8.5)
        ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=INK3, linespacing=1.3)
        y += 8.5 * 1.36 * (note.count("\n") + 1) + 3
    if handles:
        ncol = ncol or len(handles)
        nrow = -(-len(handles) // ncol)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1),
                  bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, x=-5, y=y, units="points"),
                  frameon=False, ncol=ncol, fontsize=fs_leg, labelcolor=INK2, handlelength=1.4, handleheight=0.9,
                  columnspacing=1.4, borderaxespad=0, borderpad=0, labelspacing=0.45)
        y += nrow * fs_leg * 1.62 + 2
    if show_title and title:
        ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=y + 5)
    return y


# ------------------------------------------------------------------ layout helpers
def _text_h(lines, width_in):
    h = TOP0
    for text, fs, weight, _ in lines:
        t = wrap(text, width_in, fs, bold=(weight == "bold"))
        h += (t.count("\n") + 1) * fs * 1.22 * 1.08 / 72 + 0.07
    return h


def _draw_lines(fig, lines, x_in, top_in, width_in):
    W, H = fig.get_size_inches()
    y = top_in
    for text, fs, weight, color in lines:
        t = wrap(text, width_in, fs, bold=(weight == "bold"))
        fig.text(x_in / W, 1 - y / H, t, fontsize=fs, fontweight=weight, color=color, va="top", ha="left",
                 linespacing=1.22)
        y += (t.count("\n") + 1) * fs * 1.22 * 1.08 / 72 + 0.07


def _foot_h(text, width_in, fs=FS_SRC):
    t = wrap(text, width_in, fs)
    return BOT0 + (t.count("\n") + 1) * fs * 1.3 / 72 + 0.1


def _extent(fig, ctx):
    """Inches the panel's artists reach above the top of its axes and below their bottom."""
    r = fig.canvas.get_renderer()
    boxes = [a.get_tightbbox(r) for a in ctx.axes]
    boxes = [b for b in boxes if b is not None]
    top = max(b.y1 for b in boxes) / fig.dpi
    bot = min(b.y0 for b in boxes) / fig.dpi
    x, y, w, h = ctx.rect
    return max(0.0, top - (y + h)), max(0.0, y - bot)


def _canvas_frame(fig):
    """An invisible rectangle over the whole canvas, so a tight save keeps the full width (every chart the same
    width) and only grows when something spills over."""
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False, edgecolor="none", zorder=-10))


def _save(fig, path, dpi=DPI):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    print(f"  wrote    {shown}  ({path.stat().st_size / 1024:.0f} KB)")


# ------------------------------------------------------------------ one chart on its own
MEASURED = {}                                  # key -> (head, foot) in inches, reused by the grids


def render_panel(p: Panel, D, path=None):
    miss = p.missing()
    if miss:
        print(f"  skipped  panels/{p.key}.png ({', '.join(miss)} missing)")
        return None
    tw = FIG_W - 2 * MARGIN
    source = p.text("source", D)
    source = source if source.startswith("Source") else "Source: " + source
    lines = [(p.finding(D), FS_FINDING, "bold", INK), (p.text("what", D), FS_WHAT, "normal", INK2),
             (p.text("setup", D), FS_SETUP, "normal", INK3)]
    lines = [ln for ln in lines if ln[0]]
    hh, fh = _text_h(lines, tw), _foot_h(source, tw)
    head, foot = 0.35, 0.6
    for _attempt in range(3):
        H = hh + head + p.h + foot + fh
        fig = plt.figure(figsize=(FIG_W, H))
        _canvas_frame(fig)
        _draw_lines(fig, lines, MARGIN, TOP0, tw)
        fig.text(MARGIN / FIG_W, BOT0 / H, wrap(source, tw, FS_SRC), fontsize=FS_SRC, color=INK3, va="bottom",
                 ha="left", linespacing=1.3)
        rect = (MARGIN + p.left, fh + foot, FIG_W - 2 * MARGIN - p.left - p.right, p.h)
        ctx = Ctx(fig, rect)
        p.draw(ctx, D)
        ctx.finish()
        up, down = _extent(fig, ctx)
        need_head, need_foot = up + 0.18, down + 0.2
        if abs(need_head - head) < 0.04 and abs(need_foot - foot) < 0.04:
            break
        plt.close(fig)
        head, foot = need_head, need_foot
    MEASURED[p.key] = (head, foot)
    path = path or PANEL_DIR / f"{p.key}.png"
    _save(fig, path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ a grid of panels
def render_grid(panels, D, png, pdf, title, subtitle, ncol=4, dpi=70):
    """Every available panel in reading order, ncol per row, each cell the size of the chart on its own."""
    avail = [p for p in panels if not p.missing()]
    letters = "abcdefghijklmnopqrstuvwxyz"
    cell_w, gap_x, gap_y = FIG_W - 2 * MARGIN, 0.5, 0.35
    rows = [avail[i:i + ncol] for i in range(0, len(avail), ncol)]
    heights = []
    for r in rows:
        hd = max(MEASURED.get(p.key, (0.6, 0.8))[0] for p in r) + GRID_TITLE_IN
        ft = max(MEASURED.get(p.key, (0.6, 0.8))[1] for p in r)
        heights.append((hd, max(p.h for p in r), ft))
    W = 2 * MARGIN + ncol * cell_w + (ncol - 1) * gap_x
    lines = [(title, 26, "bold", INK), (subtitle, 13, "normal", INK3)]
    hh = _text_h(lines, W - 2 * MARGIN)
    H = hh + sum(a + b + c for a, b, c in heights) + gap_y * (len(rows) - 1) + BOT0
    fig = plt.figure(figsize=(W, H))
    _draw_lines(fig, lines, MARGIN, TOP0, W - 2 * MARGIN)
    y_top = H - hh
    n = 0
    for r, (hd, axh, ft) in zip(rows, heights):
        for j, p in enumerate(r):
            x0 = MARGIN + j * (cell_w + gap_x)
            ax_h = p.h
            rect = (x0 + p.left, y_top - hd - ax_h, cell_w - p.left - p.right, ax_h)
            label = letters[n] if n < len(letters) else str(n + 1)
            ctx = Ctx(fig, rect, grid_mode=True, title=f"{label} · {p.short}")
            p.draw(ctx, D)
            ctx.finish()
            n += 1
        y_top -= hd + axh + ft + gap_y
    for path, kw in ((png, dict(dpi=dpi)), (pdf, {})):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", pad_inches=0.2, **kw)
            print(f"  wrote    {path.relative_to(ROOT).as_posix()}  ({path.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)
    return [p.key for p in avail]


# ------------------------------------------------------------------ index of every chart
INTRO = """# Tez's results, one chart per finding

Generated by `python experiments/make_dashboard.py` and `python experiments/make_lab_figure.py`, which also rewrite
this file; do not edit it by hand. Each chart states its finding as its title, the conditions under it and its source
in a small footer; every number is read from the files named there. The same panels, with short titles, make up the
two grids: [`../tez_dashboard.png`](../tez_dashboard.png) (every measurement) and [`../tez_lab.png`](../tez_lab.png)
(the probe lab). Laya's own comparison charts, redrawn with Tez, are in [`../vs/`](../vs/README.md).

**Colours** follow `docs/figures/vs/`: Tez blue (hatched = a probe trained on labelled rows; tinted = as shipped or
before a fix; outlined = Tez's letters at temperature 1, before its default temperature), laya magenta,
laya-multilingual amber, laya-typed-decisions green, Jev (published, never run here) grey. A series that is not a
system (a policy, a readout, a model size) is drawn in Tez's blues or in neutral grey, the one the finding is about in
Tez blue.
"""


def write_index():
    import make_dashboard
    import make_lab_figure
    out = [INTRO]
    for mod, heading in ((make_dashboard, "Results"), (make_lab_figure, "The probe lab")):
        if not hasattr(mod, "PANELS"):
            continue
        D = mod.load()
        out.append(f"## {heading}")
        section = None
        for p in mod.PANELS:
            if p.missing() or not (PANEL_DIR / f"{p.key}.png").exists():
                continue
            if p.section != section:
                section = p.section
                out.append(f"\n### {section}\n\n| Chart | Finding (its title) | Source |\n|---|---|---|")
            src = p.text("source", D)
            src = src[len("Source: "):] if src.startswith("Source: ") else src
            out.append(f"| [`{p.key}.png`]({p.key}.png) | {p.finding(D)} | {src.rstrip('.')} |")
        out.append("")
    (PANEL_DIR / "README.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  wrote    {(PANEL_DIR / 'README.md').relative_to(ROOT).as_posix()}")


__all__ = ["Panel", "Ctx", "render_panel", "render_grid", "write_index", "J", "pct", "join_and", "line", "dot",
           "refline", "Patch", "Line2D"]
