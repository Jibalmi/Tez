"""From backend scores to Jev-shaped answers.

  letters  softmax over the k option letters' log-probabilities, divided by a per-question temperature
  probe    per-question logistic regression on the last-token state (weights fitted by `tez fit`), numpy only
  blend    w * probe + (1 - w) * letters, w = n / (n + 10) with n the probe's training labels: with few labels
           the zero-shot prior still carries weight, with many the probe dominates
Answer shapes follow Jev exactly: noul {type, noul}; choice {type, choice, probabilities, confidence};
score {type, score, legend, probabilities, confidence}, with confidence = (k * p_max - 1) / (k - 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import Question

BLEND_N0 = 10
T_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 120))


def softmax(z: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Softmax of logits divided by a temperature; -inf entries (options that left a tournament) get 0."""
    z = np.asarray(z, dtype=float) / float(temperature)
    m = np.max(z[np.isfinite(z)]) if np.isfinite(z).any() else 0.0
    e = np.where(np.isfinite(z), np.exp(np.clip(z - m, -745.0, 0.0)), 0.0)
    s = e.sum()
    return e / s if s > 0 else np.full(len(z), 1.0 / len(z))


def temper(p: Sequence[float], temperature: float) -> np.ndarray:
    """Temperature on a probability vector: p^(1/T), renormalised (zeros stay zero)."""
    p = np.asarray(p, dtype=float)
    if temperature == 1.0:
        return p / p.sum()
    with np.errstate(divide="ignore"):
        return softmax(np.log(p), temperature)


def confidence(p: Sequence[float]) -> float:
    """Jev's confidence: (k * p_max - 1) / (k - 1), 0 at uniform, 1 when certain."""
    p = np.asarray(p, dtype=float)
    k = len(p)
    if k < 2:
        return 1.0
    return float(min(1.0, max(0.0, (k * float(p.max()) - 1.0) / (k - 1.0))))


def blend_weight(n: int, n0: int = BLEND_N0) -> float:
    return n / (n + n0) if n > 0 else 0.0


def blend(p_probe: Sequence[float], p_prior: Sequence[float], n: int, n0: int = BLEND_N0) -> np.ndarray:
    """Linear pool of a probe trained on n labels and the zero-shot prior, weight n / (n + n0) on the probe."""
    w = blend_weight(n, n0)
    p = w * np.asarray(p_probe, float) + (1.0 - w) * np.asarray(p_prior, float)
    return p / p.sum()


@dataclass
class Probe:
    """Multinomial logistic probe in softmax form. Rows of W / b belong to `classes` (option indices seen in
    training); options never seen get the 1e-6 floor, as in the probe lab."""

    W: np.ndarray          # (c, d)
    b: np.ndarray          # (c,)
    classes: np.ndarray    # (c,) option indices
    k: int                 # number of options of the question
    n: int                 # labels the probe was trained on (the blend weight uses it)

    @property
    def dim(self) -> int:
        return int(self.W.shape[1])

    def predict(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"embedding has {x.shape[0]} dimensions, the probe expects {self.dim}")
        p = softmax(self.W @ x + self.b)
        out = np.full(self.k, 1e-6)
        out[self.classes] += p
        return out / out.sum()

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.predict(x) for x in np.asarray(X, float)]) if len(X) else np.zeros((0, self.k))

    @classmethod
    def from_sklearn(cls, clf, k: int, n: int) -> "Probe":
        """Binary sklearn models store one weight row for classes_[1]; softmax over [0, z] is the same sigmoid."""
        coef = np.asarray(clf.coef_, float)
        icpt = np.asarray(clf.intercept_, float)
        classes = np.asarray(clf.classes_).astype(int)
        if len(classes) == 2 and coef.shape[0] == 1:
            coef = np.vstack([np.zeros(coef.shape[1]), coef[0]])
            icpt = np.array([0.0, icpt[0]])
        return cls(W=coef, b=icpt, classes=classes, k=k, n=n)


def assemble(question: Question, keys: Sequence[str], p: Sequence[float]) -> dict:
    """The answer in Jev's exact shape for the question type."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    if question.type == "noul":
        return {"type": "noul", "noul": float(p[1])}
    probs = {k: float(v) for k, v in zip(keys, p)}
    if question.type == "choice":
        return {"type": "choice", "choice": keys[int(np.argmax(p))], "probabilities": probs, "confidence": confidence(p)}
    legend = {str(i): c for i, c in enumerate(question.criteria)}
    score = float(np.dot(np.arange(len(p)), p))
    return {"type": "score", "score": score, "legend": legend, "probabilities": probs, "confidence": confidence(p)}


def answer_probabilities(question: Question, answer: dict, keys: Sequence[str]) -> np.ndarray:
    """Inverse of `assemble`: the probability vector in option order (used by `tez eval`)."""
    if question.type == "noul":
        return np.array([1.0 - answer["noul"], answer["noul"]])
    return np.array([answer["probabilities"][k] for k in keys], dtype=float)


# ------------------------------------------------------------------------------------------ metrics
def nll(P: np.ndarray, y: Sequence[int]) -> float:
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def ece(conf: Sequence[float], correct: Sequence[float], bins: int = 15) -> float:
    """Expected calibration error with equal-width bins (ECE-15, as in the benchmarks)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - correct[s].mean())
    return float(e)


def softmax_rows(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise `softmax` for a 2-D array of logits (-inf allowed)."""
    A = np.asarray(Z, float) / float(temperature)
    finite = np.isfinite(A)
    m = np.max(np.where(finite, A, -np.inf), axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    E = np.where(finite, np.exp(np.clip(np.where(finite, A, 0.0) - m, -745.0, 0.0)), 0.0)
    s = E.sum(axis=1, keepdims=True)
    return E / np.where(s > 0, s, 1.0)


def fit_temperature_logits(Z: Sequence[Sequence[float]], y: Sequence[int], min_rows: int = 5, prior_sd: float = 1.0) -> float:
    """Temperature for softmax(z / T), searched on a log-spaced grid (0.05 .. 20). A MAP estimate: mean NLL plus a
    log-normal prior on T (median 1, `prior_sd` in log space) that weighs 1/n. Pure maximum likelihood runs to the
    grid edge whenever every row is already right (a few confident labels), which would make the next mistake
    arbitrarily confident; the prior keeps T near 1 until the data really says otherwise. 1.0 with too few rows."""
    y = np.asarray(y, int)
    if len(y) < min_rows:
        return 1.0
    Z = np.asarray(Z, float)
    best, bt = math.inf, 1.0
    for t in T_GRID:
        v = nll(softmax_rows(Z, t), y) + math.log(t) ** 2 / (2.0 * prior_sd ** 2 * len(y))
        if v < best - 1e-12:
            best, bt = v, float(t)
    return bt


def fit_temperature_probs(P: np.ndarray, y: Sequence[int], min_rows: int = 5, prior_sd: float = 1.0) -> float:
    """The same for probability vectors (probe / blend outputs): p^(1/T) renormalised."""
    with np.errstate(divide="ignore"):
        return fit_temperature_logits(np.log(np.asarray(P, float)), y, min_rows, prior_sd)
