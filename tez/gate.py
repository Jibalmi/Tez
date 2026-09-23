"""Act / escalate gate by conformal selection (Jin and Candes, 2023), as in the probe lab.

For a decision with confidence c, the conformal p-value of "this decision is wrong" is
    p(c) = (1 + #{wrong calibration decisions with confidence >= c}) / (n_cal + 1).
Benjamini-Hochberg at level alpha over a batch then selects decisions to act on, keeping the expected share
of wrong decisions among the acted ones at most alpha. Decisions arrive one at a time at runtime, so `tez fit`
precomputes, per question and alpha, the confidence cut that BH implies when the held-out calibration
decisions themselves are the batch: act iff confidence >= threshold. The calibration decision being scored is
counted among the wrong ones when it is wrong, which errs on the side of escalating.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def conformal_pvalues(cal_conf: Sequence[float], cal_wrong: Sequence[bool], conf: Sequence[float]) -> np.ndarray:
    cal_conf = np.asarray(cal_conf, float)
    wrong_conf = np.sort(cal_conf[np.asarray(cal_wrong, bool)])
    conf = np.asarray(conf, float)
    above = len(wrong_conf) - np.searchsorted(wrong_conf, conf, side="left")
    return (1.0 + above) / (len(cal_conf) + 1.0)


def bh_select(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean mask of the selected (acted-on) decisions."""
    p = np.asarray(pvalues, float)
    m = len(p)
    sel = np.zeros(m, dtype=bool)
    if m == 0:
        return sel
    order = np.argsort(p, kind="stable")
    ok = np.nonzero(p[order] <= alpha * np.arange(1, m + 1) / m)[0]
    if len(ok):
        sel[order[: ok.max() + 1]] = True
    return sel


def threshold(cal_conf: Sequence[float], cal_wrong: Sequence[bool], alpha: float) -> float | None:
    """Confidence cut implied by BH at level alpha on the calibration set; None = no cut achieves alpha (never act)."""
    conf = np.asarray(cal_conf, float)
    if len(conf) == 0:
        return None
    sel = bh_select(conformal_pvalues(conf, cal_wrong, conf), alpha)
    return float(conf[sel].min()) if sel.any() else None


def thresholds(cal_conf: Sequence[float], cal_wrong: Sequence[bool], alphas: Sequence[float] = DEFAULT_ALPHAS) -> dict:
    """{"0.05": cut or None, ...} for every alpha (JSON-friendly keys)."""
    return {f"{a:g}": threshold(cal_conf, cal_wrong, a) for a in sorted(set(float(a) for a in alphas))}


def lookup(table: Mapping[str, float | None], alpha: float) -> tuple[float | None, float | None]:
    """(cut, alpha used) for the largest precomputed alpha <= the requested one: a runtime alpha between grid
    points is served conservatively. (None, None) when every precomputed alpha is larger."""
    best: tuple[float, float | None] | None = None
    for key, cut in table.items():
        a = float(key)
        if a <= alpha + 1e-12 and (best is None or a > best[0]):
            best = (a, cut)
    return (None, None) if best is None else (best[1], best[0])


def decide(p_max: float, cut: float | None) -> str:
    return "act" if cut is not None and p_max >= cut - 1e-12 else "escalate"


def coverage_and_error(conf: Sequence[float], wrong: Sequence[bool], cut: float | None) -> tuple[float, float | None]:
    """Share of decisions acted on and the error rate among them, for a given cut."""
    conf = np.asarray(conf, float)
    wrong = np.asarray(wrong, bool)
    if cut is None or len(conf) == 0:
        return 0.0, None
    acted = conf >= cut - 1e-12
    return float(acted.mean()), (float(wrong[acted].mean()) if acted.any() else None)
