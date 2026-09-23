"""Probabilities, temperature, the numpy probe, the n/(n+10) blend and calibration metrics."""
from __future__ import annotations

import numpy as np
import pytest

from tez.readout import (Probe, blend, blend_weight, confidence, ece, fit_temperature_logits, fit_temperature_probs,
                         softmax, softmax_rows, temper)


def test_softmax_temperature_and_inf():
    z = np.array([2.0, 0.0, -np.inf])
    p = softmax(z, 2.0)
    assert p[2] == 0.0 and np.isclose(p.sum(), 1.0)
    assert np.isclose(p[0] / p[1], np.exp(1.0))
    assert np.allclose(softmax_rows(np.array([[2.0, 0.0, -np.inf], [0.0, 0.0, 0.0]]), 2.0)[0], p)


def test_temper_identity_and_sharpening():
    p = np.array([0.6, 0.3, 0.1, 0.0])
    assert np.allclose(temper(p, 1.0), p)
    sharp = temper(p, 0.5)
    assert sharp[0] > p[0] and sharp[3] == 0.0 and np.isclose(sharp.sum(), 1.0)
    assert np.allclose(sharp[:3], p[:3] ** 2 / (p[:3] ** 2).sum())


@pytest.mark.parametrize("n, w", [(0, 0.0), (10, 0.5), (30, 0.75), (90, 0.9)])
def test_blend_weight(n, w):
    assert blend_weight(n) == pytest.approx(w)


def test_blend_is_linear_pool():
    probe, prior = np.array([0.8, 0.1, 0.1]), np.array([0.2, 0.5, 0.3])
    assert np.allclose(blend(probe, prior, 30), 0.75 * probe + 0.25 * prior)
    assert np.allclose(blend(probe, prior, 0), prior)


def test_probe_matches_sklearn_binary_and_multiclass():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for y, k in ((np.repeat([0, 1], 40), 2), (np.repeat([0, 1, 2, 3], 20), 4)):
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(X, y)
        probe = Probe.from_sklearn(clf, k=k, n=len(y))
        ours = probe.predict_many(X[:10])
        theirs = clf.predict_proba(X[:10])
        assert np.allclose(ours, (theirs + 1e-6) / (1 + k * 1e-6), atol=1e-8)


def test_probe_with_unseen_classes_and_dim_check():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))
    y = np.repeat([0, 2], 20)                                  # option 1 never labelled
    probe = Probe.from_sklearn(LogisticRegression().fit(X, y), k=3, n=40)
    p = probe.predict(X[0])
    assert p.shape == (3,) and np.isclose(p.sum(), 1.0) and p[1] < 1e-5
    with pytest.raises(ValueError, match="dimensions"):
        probe.predict(np.zeros(4))


def test_confidence_is_jev_formula():
    for p in ([0.5, 0.5], [0.9, 0.1], [0.7, 0.2, 0.1], [0.25] * 4):
        k = len(p)
        assert confidence(p) == pytest.approx((k * max(p) - 1) / (k - 1))


def test_temperature_recovers_generating_temperature():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((4000, 4)) * 3.0
    P = softmax_rows(Z, 2.0)
    y = np.array([rng.choice(4, p=p) for p in P])
    assert 1.7 < fit_temperature_logits(Z, y) < 2.4
    assert 1.7 < fit_temperature_probs(softmax_rows(Z, 1.0), y) < 2.4
    assert fit_temperature_logits(Z[:3], y[:3]) == 1.0             # too few rows: identity


def test_temperature_stays_near_one_when_every_row_is_right():
    """34 confident, all-correct rows (the live fit's case): maximum likelihood would run to T = 0.05."""
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((34, 4))
    y = rng.integers(0, 4, 34)
    Z[np.arange(34), y] += 12.0
    assert 0.5 <= fit_temperature_logits(Z, y) <= 1.5
    assert 0.5 <= fit_temperature_probs(softmax_rows(Z), y) <= 1.5


def test_ece():
    rng = np.random.default_rng(4)
    conf = rng.uniform(0.3, 1.0, 20000)
    calibrated = rng.random(20000) < conf
    assert ece(conf, calibrated) < 0.02
    assert ece(np.full(1000, 0.95), np.r_[np.ones(500), np.zeros(500)]) == pytest.approx(0.45)
