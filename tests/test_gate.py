"""Conformal selection thresholds (BH-implied cut per alpha) on synthetic calibration data."""
from __future__ import annotations

import numpy as np
import pytest

from tez.gate import bh_select, conformal_pvalues, coverage_and_error, decide, lookup, threshold, thresholds


def calibrated(n, rng, lo=0.4):
    conf = rng.uniform(lo, 1.0, n)
    wrong = rng.random(n) > conf            # P(correct) = confidence
    return conf, wrong


def test_pvalues():
    cal_conf = np.array([0.9, 0.8, 0.7, 0.6])
    cal_wrong = np.array([False, True, False, True])
    p = conformal_pvalues(cal_conf, cal_wrong, [0.95, 0.75, 0.5])
    assert np.allclose(p, [1 / 5, 2 / 5, 3 / 5])


def test_bh_select():
    sel = bh_select([0.001, 0.01, 0.03, 0.2, 0.9], 0.05)
    assert sel.tolist() == [True, True, True, False, False]
    assert not bh_select([0.5, 0.6], 0.05).any()


def test_separable_calibration_acts_on_the_confident_ones():
    rng = np.random.default_rng(0)
    conf = np.r_[rng.uniform(0.75, 1.0, 150), rng.uniform(0.3, 0.6, 50)]
    wrong = np.r_[np.zeros(150, bool), np.ones(50, bool)]
    cut = threshold(conf, wrong, 0.05)
    assert cut is not None and cut <= conf[:150].min()          # every correct decision is acted on
    acted, err = coverage_and_error(conf, wrong, cut)
    assert acted >= 0.75 and err <= 0.05                         # BH admits wrong ones only up to alpha
    strict = threshold(conf, wrong, 0.01)
    assert strict is not None and coverage_and_error(conf, wrong, strict)[1] <= 0.01


def test_too_few_calibration_rows_never_act():
    conf = np.linspace(0.9, 0.99, 10)
    wrong = np.zeros(10, bool)
    assert threshold(conf, wrong, 0.05) is None           # 1 / (10 + 1) > 0.05: nothing can be certified
    assert threshold(conf, wrong, 0.2) is not None
    assert threshold([], [], 0.1) is None


def test_thresholds_are_monotone_in_alpha():
    conf, wrong = calibrated(1000, np.random.default_rng(1))
    table = thresholds(conf, wrong)
    cuts = [table[k] for k in sorted(table, key=float)]
    finite = [c for c in cuts if c is not None]
    assert finite == sorted(finite, reverse=True)
    assert cuts.index(finite[0]) >= 0 and all(c is None for c in cuts[: cuts.index(finite[0])])


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_error_among_acted_is_controlled_on_fresh_data(alpha):
    errs = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        cut = threshold(*calibrated(1500, rng), alpha)
        conf, wrong = calibrated(20000, rng)
        acted, err = coverage_and_error(conf, wrong, cut)
        assert acted > 0.05
        errs.append(err)
    assert np.mean(errs) <= alpha + 0.01


def test_lookup_uses_largest_alpha_not_above_request():
    table = {"0.05": 0.93, "0.1": 0.85, "0.2": None}
    assert lookup(table, 0.07) == (0.93, 0.05)
    assert lookup(table, 0.1) == (0.85, 0.1)
    assert lookup(table, 0.5) == (None, 0.2)
    assert lookup(table, 0.01) == (None, None)
    assert decide(0.9, 0.85) == "act" and decide(0.8, 0.85) == "escalate" and decide(0.99, None) == "escalate"
