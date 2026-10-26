"""Tests for the inferential-statistics additions in ``afp.metrics``.

Covers the BCa bootstrap CI, the paired sign-flip permutation test, and the
Holm-Bonferroni step-down correction -- the machinery behind the DESIGN Rev-4
"gate on the CI lower bound + paired permutation, Holm-corrected" protocol.
"""

from __future__ import annotations

import numpy as np
import pytest

from afp.metrics import (
    bootstrap_bca_ci,
    bootstrap_ci,
    paired_permutation_test,
    holm_bonferroni,
)


# ---------------------------------------------------------------------------
# bootstrap_bca_ci
# ---------------------------------------------------------------------------
def test_bca_covers_true_mean_symmetric():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=60)
    lo, hi = bootstrap_bca_ci(x, seed=1)
    assert lo < 5.0 < hi
    assert lo < float(np.mean(x)) < hi


def test_bca_single_sample_degenerate():
    lo, hi = bootstrap_bca_ci([3.0])
    assert lo == hi == 3.0


def test_bca_all_identical_samples():
    lo, hi = bootstrap_bca_ci([2.0] * 10, seed=0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_bca_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_bca_ci([])


def test_bca_skewed_shifts_interval_vs_percentile():
    # Strongly right-skewed sample: BCa should differ from the plain
    # percentile interval (that difference is the point of the correction).
    rng = np.random.default_rng(2)
    x = rng.lognormal(mean=0.0, sigma=1.2, size=40)
    bca = bootstrap_bca_ci(x, seed=3)
    perc = bootstrap_ci(x, seed=3)
    assert bca != perc
    # Both still bracket the sample mean.
    m = float(np.mean(x))
    assert bca[0] < m < bca[1]


def test_bca_lower_bound_below_upper():
    rng = np.random.default_rng(4)
    for _ in range(5):
        x = rng.normal(0.0, 2.0, size=25)
        lo, hi = bootstrap_bca_ci(x, seed=5)
        assert lo <= hi


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------
def test_permutation_detects_clear_effect():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.1, size=30)
    x = y + 1.0  # constant +1 shift
    res = paired_permutation_test(x, y, alternative="greater", seed=0)
    assert res["p_value"] < 0.001
    assert res["observed"] == pytest.approx(1.0)
    # A (numerically) constant shift -> an enormous paired Cohen's d.
    assert res["effect_size"] > 100.0


def test_permutation_null_is_uniformish():
    # Under H0 the p-value should not be tiny.
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=40)
    y = rng.normal(0.0, 1.0, size=40)
    res = paired_permutation_test(x, y, alternative="two-sided", seed=1)
    assert res["p_value"] > 0.01


def test_permutation_direction_matters():
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 0.1, size=25)
    x = y + 0.5
    res_gt = paired_permutation_test(x, y, alternative="greater", seed=2)
    res_lt = paired_permutation_test(x, y, alternative="less", seed=2)
    assert res_gt["p_value"] < 0.01
    assert res_lt["p_value"] > 0.99


def test_permutation_effect_size_paired_cohens_d():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.5, 2.0, 3.0])
    d = x - y
    expected = d.mean() / d.std(ddof=1)
    res = paired_permutation_test(x, y, seed=0)
    assert res["effect_size"] == pytest.approx(expected)


def test_permutation_identical_inputs_zero_effect():
    x = np.array([1.0, 2.0, 3.0])
    res = paired_permutation_test(x, x, alternative="two-sided", seed=0)
    assert res["observed"] == 0.0
    assert res["effect_size"] == 0.0
    assert res["p_value"] == 1.0


def test_permutation_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 2.0], [1.0])


def test_permutation_too_few_pairs_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [0.0])


def test_permutation_bad_alternative_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 2.0], [0.0, 0.0], alternative="sideways")


def test_permutation_p_never_zero():
    y = np.zeros(20)
    x = y + 100.0
    res = paired_permutation_test(x, y, alternative="greater", n_perm=500, seed=0)
    assert 0.0 < res["p_value"] <= 1.0 / 500.0 + 1e-6


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------
def test_holm_empty():
    out = holm_bonferroni([])
    assert out == {"adjusted": [], "reject": []}


def test_holm_single_pvalue_unchanged():
    out = holm_bonferroni([0.03])
    assert out["adjusted"][0] == pytest.approx(0.03)
    assert out["reject"] == [True]


def test_holm_known_example():
    # p = (.01, .04, .03) with m=3: sorted (.01,.03,.04) -> adj (.03,.06,.06).
    out = holm_bonferroni([0.01, 0.04, 0.03])
    assert out["adjusted"][0] == pytest.approx(0.03)
    assert out["adjusted"][1] == pytest.approx(0.06)
    assert out["adjusted"][2] == pytest.approx(0.06)
    assert out["reject"] == [True, False, False]


def test_holm_monotone_and_clipped():
    out = holm_bonferroni([0.5, 0.9, 0.7])
    assert all(0.0 <= v <= 1.0 for v in out["adjusted"])
    # Adjusted values respect the ordering of the raw p-values (step-down monotone).
    order = np.argsort([0.5, 0.9, 0.7])
    adj_sorted = np.asarray(out["adjusted"])[order]
    assert np.all(np.diff(adj_sorted) >= -1e-12)


def test_holm_out_of_range_raises():
    with pytest.raises(ValueError):
        holm_bonferroni([0.5, 1.2])


def test_holm_more_powerful_than_bonferroni():
    p = [0.01, 0.02, 0.049]
    out = holm_bonferroni(p)
    bonf = [min(v * len(p), 1.0) for v in p]
    assert all(a <= b + 1e-12 for a, b in zip(out["adjusted"], bonf))


# ---------------------------------------------------------------------------
# integration: scorecards carry the stats family
# ---------------------------------------------------------------------------
def test_scorecard_stats_populated():
    from afp.adapters.mock import MockWorldModel
    from afp.data import make_synthetic_dataset
    from afp.score import score_model

    # NOTE the power floor: a one-tailed sign-flip test on n paired episodes
    # can never produce p < 1/2^n, so the Holm-corrected family (m = 3) needs
    # n >= 6 episodes (3/2^6 ~ .047) to be *able* to reject at alpha = .05.
    episodes, specs = make_synthetic_dataset(n_episodes=8, horizon=6, size=32, seed=0)
    model = MockWorldModel("m", action_gain=1.0, instruction_gain=0.2, noise=0.05,
                           horizon=6, size=32)
    card = score_model(model, episodes, specs, seeds=(0, 1, 2))
    assert card.stats, "stats family should be populated for n_episodes >= 2"
    for name, res in card.stats.items():
        assert 0.0 < res["p_value"] <= 1.0
        assert 0.0 < res["p_holm"] <= 1.0
        assert res["p_holm"] >= res["p_value"] - 1e-12
    # A strongly action-driven mock must clear the reactivity test.
    react = [v for k, v in card.stats.items() if k.startswith("reactivity")]
    assert react and react[0]["p_holm"] < 0.05
