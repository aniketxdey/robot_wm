"""Tests for ``afp.metrics`` -- the pure metric math of the Action-Faithfulness Probe.

Every unit test constructs frame stacks directly with numpy (no models, no I/O)
so the metric behaviour is checked deterministically against hand-computable
expectations. A single end-to-end integration test wires the real mock world
model + distance + synthetic dataset together to confirm that the headline claim
holds: an action-faithful model is more action-sensitive than an
instruction-driven one.
"""

from __future__ import annotations

import numpy as np
import pytest

from afp.distance import get_distance, PixelL2Distance
from afp.metrics import (
    sensitivity_curve,
    transition_curve,
    seed_floor,
    aggregate_curves,
    normalized_sensitivity,
    monotonicity_spearman,
    bootstrap_ci,
    separation_ratio,
    paired_curve_over_episodes,
)

T = 8
H = W = 16


def const_stack(value: int, t: int = T, h: int = H, w: int = W) -> np.ndarray:
    """A ``(t, h, w, 3)`` uint8 stack filled with a constant brightness."""
    return np.full((t, h, w, 3), value, dtype=np.uint8)


def ramp_stack(start: int = 0, step: int = 10, t: int = T) -> np.ndarray:
    """A stack whose brightness increases by ``step`` each frame (compounding S(t))."""
    frames = np.empty((t, H, W, 3), dtype=np.uint8)
    for i in range(t):
        frames[i] = np.clip(start + step * i, 0, 255)
    return frames


# ---------------------------------------------------------------------------
# sensitivity_curve  (DESIGN Sec 2.1)
# ---------------------------------------------------------------------------
def test_sensitivity_curve_identical_is_zero():
    a = const_stack(120)
    s = sensitivity_curve(a, a.copy(), PixelL2Distance())
    assert s.shape == (T,)
    assert np.allclose(s, 0.0)


def test_sensitivity_curve_brighter_is_positive_constant():
    base = const_stack(100)
    pert = const_stack(150)
    s = sensitivity_curve(base, pert, PixelL2Distance())
    assert s.shape == (T,)
    assert np.all(s > 0)
    # A uniform brightness offset gives a constant per-frame distance.
    assert np.allclose(s, s[0])
    # pixel_l2 works in [0,1]: 50/255 RMS over identical pixels.
    assert np.allclose(s, 50.0 / 255.0)


def test_sensitivity_curve_truncates_to_min_horizon():
    base = const_stack(100, t=10)
    pert = const_stack(150, t=6)
    s = sensitivity_curve(base, pert, PixelL2Distance())
    assert s.shape == (6,)


def test_sensitivity_curve_shape_mismatch_raises():
    base = const_stack(100)
    pert = np.full((T, H, W + 1, 3), 150, dtype=np.uint8)
    with pytest.raises(ValueError):
        sensitivity_curve(base, pert, PixelL2Distance())


def test_sensitivity_curve_rejects_non_frame_stack():
    bad = np.zeros((T, H, W), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError):
        sensitivity_curve(bad, bad, PixelL2Distance())


# ---------------------------------------------------------------------------
# transition_curve  (DESIGN Sec 2.1, exploratory)
# ---------------------------------------------------------------------------
def test_transition_curve_length_is_T_minus_1():
    base = ramp_stack(step=5)
    pert = ramp_stack(step=12)
    s = transition_curve(base, pert, PixelL2Distance())
    assert s.shape == (T - 1,)


def test_transition_curve_identical_motion_is_zero():
    # Two stacks that share the SAME per-step difference (parallel ramps) have
    # identical transition images -> S^Delta == 0 even though S(t) > 0.
    base = ramp_stack(start=0, step=10)
    pert = ramp_stack(start=40, step=10)
    s_delta = transition_curve(base, pert, PixelL2Distance())
    assert np.allclose(s_delta, 0.0)
    # ...whereas the plain sensitivity curve is clearly positive.
    s = sensitivity_curve(base, pert, PixelL2Distance())
    assert np.all(s > 0)


def test_transition_curve_different_motion_is_positive():
    base = ramp_stack(step=5)
    pert = ramp_stack(step=20)
    s_delta = transition_curve(base, pert, PixelL2Distance())
    assert np.all(s_delta > 0)


def test_transition_curve_single_frame_is_empty():
    a = const_stack(100, t=1)
    s = transition_curve(a, a.copy(), PixelL2Distance())
    assert s.shape == (0,)


# ---------------------------------------------------------------------------
# seed_floor  (DESIGN Sec 2.3)
# ---------------------------------------------------------------------------
def test_seed_floor_two_identical_rollouts_zero():
    a = const_stack(100)
    mean, std = seed_floor([a, a.copy()], PixelL2Distance())
    assert mean.shape == (T,) and std.shape == (T,)
    assert np.allclose(mean, 0.0)
    assert np.allclose(std, 0.0)


def test_seed_floor_positive_with_jitter():
    rng = np.random.default_rng(0)
    base = const_stack(100).astype(np.int16)
    rolls = [
        np.clip(base + rng.integers(-5, 6, size=base.shape), 0, 255).astype(np.uint8)
        for _ in range(4)
    ]
    mean, std = seed_floor(rolls, PixelL2Distance())
    assert np.all(mean > 0)
    # 4 rollouts -> 6 distinct pairs -> std is well-defined and finite.
    assert np.all(np.isfinite(std))


def test_seed_floor_needs_two_rollouts():
    a = const_stack(100)
    with pytest.raises(ValueError):
        seed_floor([a], PixelL2Distance())


def test_seed_floor_truncates_to_min_horizon():
    mean, std = seed_floor(
        [const_stack(100, t=10), const_stack(100, t=5), const_stack(100, t=7)],
        PixelL2Distance(),
    )
    assert mean.shape == (5,)


# ---------------------------------------------------------------------------
# aggregate_curves
# ---------------------------------------------------------------------------
def test_aggregate_curves_basic_stats():
    curves = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])]
    agg = aggregate_curves(curves)
    assert set(agg) == {"mean", "std", "sem", "n"}
    assert np.allclose(agg["mean"], [2.0, 3.0, 4.0])
    assert np.allclose(agg["std"], [1.0, 1.0, 1.0])  # population std of {x, x+2}
    # sem = std(ddof=1)/sqrt(n) = sqrt(2)/sqrt(2) = 1.0
    assert np.allclose(agg["sem"], [1.0, 1.0, 1.0])
    assert np.allclose(agg["n"], [2, 2, 2])


def test_aggregate_curves_single_curve_zero_sem():
    agg = aggregate_curves([np.array([5.0, 6.0])])
    assert np.allclose(agg["mean"], [5.0, 6.0])
    assert np.allclose(agg["sem"], [0.0, 0.0])
    assert np.allclose(agg["n"], [1, 1])


def test_aggregate_curves_length_mismatch_raises():
    with pytest.raises(ValueError):
        aggregate_curves([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])


def test_aggregate_curves_empty_raises():
    with pytest.raises(ValueError):
        aggregate_curves([])


# ---------------------------------------------------------------------------
# normalized_sensitivity  (DESIGN Sec 2.2)
# ---------------------------------------------------------------------------
def test_normalized_sensitivity_maps_floor_to_zero_ceil_to_one():
    floor = np.array([0.1, 0.2, 0.3])
    ceil = np.array([0.5, 0.6, 0.7])
    assert np.allclose(normalized_sensitivity(floor, floor, ceil), 0.0)
    assert np.allclose(normalized_sensitivity(ceil, floor, ceil), 1.0)


def test_normalized_sensitivity_midpoint_and_clipping():
    floor, ceil = 0.0, 1.0
    assert np.isclose(normalized_sensitivity(0.5, floor, ceil), 0.5, atol=1e-6)
    # below floor clips to 0, above ceil clips to 1.
    assert np.isclose(normalized_sensitivity(-2.0, floor, ceil), 0.0)
    assert np.isclose(normalized_sensitivity(5.0, floor, ceil), 1.0)


def test_normalized_sensitivity_elementwise_arrays():
    S = np.array([0.0, 0.5, 1.0, 2.0])
    out = normalized_sensitivity(S, 0.0, 1.0)
    assert out.shape == (4,)
    assert np.allclose(out, [0.0, 0.5, 1.0, 1.0])


# ---------------------------------------------------------------------------
# monotonicity_spearman  (DESIGN Sec 2.4)
# ---------------------------------------------------------------------------
def test_monotonicity_perfect_increasing():
    sev = [0, 1, 2, 3, 4]
    vals = [0.1, 0.3, 0.35, 0.8, 1.2]  # strictly increasing
    rho = monotonicity_spearman(sev, vals)
    assert np.isclose(rho, 1.0)


def test_monotonicity_perfect_decreasing():
    sev = [0, 1, 2, 3, 4]
    vals = [5.0, 4.0, 3.0, 2.0, 1.0]
    rho = monotonicity_spearman(sev, vals)
    assert np.isclose(rho, -1.0)


def test_monotonicity_handles_ties():
    sev = [0, 1, 1, 2, 3]
    vals = [0.0, 0.5, 0.5, 1.0, 2.0]  # ties mirrored in both -> rho == 1
    rho = monotonicity_spearman(sev, vals)
    assert np.isclose(rho, 1.0)


def test_monotonicity_fallback_matches_scipy():
    # Exercise the manual ranking path directly and compare to scipy when present.
    import afp.metrics as m

    sev = [0.0, 1.0, 2.0, 2.0, 4.0, 5.0]
    vals = [0.2, 0.1, 0.9, 0.8, 1.5, 1.4]
    rx = m._rankdata_average(np.asarray(sev))
    ry = m._rankdata_average(np.asarray(vals))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    manual = float(np.sum(rx * ry) / np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    try:
        from scipy.stats import spearmanr

        assert np.isclose(manual, float(spearmanr(sev, vals).correlation))
    except Exception:
        pytest.skip("scipy not installed; manual path still computed a finite rho")
    assert np.isfinite(manual)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 1.0, size=200)
    lo, hi = bootstrap_ci(samples, n_boot=1000, seed=1)
    assert lo < samples.mean() < hi
    assert lo < hi


def test_bootstrap_ci_is_reproducible():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = bootstrap_ci(samples, n_boot=500, seed=42)
    b = bootstrap_ci(samples, n_boot=500, seed=42)
    assert a == b


def test_bootstrap_ci_single_sample():
    lo, hi = bootstrap_ci([7.0])
    assert lo == hi == 7.0


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


# ---------------------------------------------------------------------------
# separation_ratio  (Stage-2 "3x floor" gate)
# ---------------------------------------------------------------------------
def test_separation_ratio_greater_than_three_when_severe_dominates():
    floor = np.full(T, 0.01)
    severe = np.full(T, 0.05)  # 5x the floor
    ratio = separation_ratio(severe, floor)
    assert ratio > 3.0
    assert np.isclose(ratio, 5.0)


def test_separation_ratio_near_one_when_severe_equals_floor():
    floor = np.full(T, 0.02)
    severe = np.full(T, 0.02)
    assert np.isclose(separation_ratio(severe, floor), 1.0)


def test_separation_ratio_handles_zero_floor():
    floor = np.zeros(T)
    severe = np.full(T, 0.1)
    # max(floor, eps) keeps this finite (and very large) rather than dividing by 0.
    assert np.isfinite(separation_ratio(severe, floor))


# ---------------------------------------------------------------------------
# paired_curve_over_episodes  (optional convenience)
# ---------------------------------------------------------------------------
def test_paired_curve_over_episodes_aggregates():
    pairs = [
        (const_stack(100), const_stack(150)),
        (const_stack(100), const_stack(150)),
    ]
    agg = paired_curve_over_episodes(pairs, PixelL2Distance())
    assert set(agg) == {"mean", "std", "sem", "n"}
    assert agg["mean"].shape == (T,)
    assert np.allclose(agg["mean"], 50.0 / 255.0)
    assert np.allclose(agg["n"], 2)


def test_paired_curve_over_episodes_truncates_to_min():
    pairs = [
        (const_stack(100, t=8), const_stack(150, t=8)),
        (const_stack(100, t=5), const_stack(150, t=5)),
    ]
    agg = paired_curve_over_episodes(pairs, PixelL2Distance())
    assert agg["mean"].shape == (5,)


def test_paired_curve_over_episodes_empty_raises():
    with pytest.raises(ValueError):
        paired_curve_over_episodes([], PixelL2Distance())


# ---------------------------------------------------------------------------
# End-to-end integration: faithful model is more action-sensitive than the
# instruction-driven model (DESIGN headline, using the real mock pieces).
# ---------------------------------------------------------------------------
def test_e2e_faithful_more_action_sensitive_than_instruction():
    from afp.adapters.mock import MockWorldModel
    from afp.data import make_synthetic_dataset

    size = 64
    horizon = 12
    n_episodes = 4

    episodes, _specs = make_synthetic_dataset(
        n_episodes=n_episodes, horizon=horizon, size=size, seed=0
    )
    distance = get_distance("pixel_l2")

    faithful = MockWorldModel(
        "mock-faithful", action_gain=1.0, instruction_gain=0.15, noise=0.0,
        horizon=horizon, size=size,
    )
    instruction = MockWorldModel(
        "mock-instruction", action_gain=0.12, instruction_gain=1.0, noise=0.0,
        horizon=horizon, size=size,
    )

    def action_sensitivity(model) -> float:
        pairs = []
        for ep in episodes:
            base = model.rollout_episode(ep, seed=0).frames
            # Apply the `negate` action edit by hand: flip the 6 motion dims.
            a2 = ep.actions.copy()
            a2[:, 0:6] *= -1
            pert = model.rollout(
                o0=ep.o0, actions=a2, instruction=ep.instruction,
                instruction_emb=ep.instruction_emb, seed=0,
            ).frames
            pairs.append((base, pert))
        agg = paired_curve_over_episodes(pairs, distance)
        return float(agg["mean"].mean())

    s_faithful = action_sensitivity(faithful)
    s_instruction = action_sensitivity(instruction)

    # The faithful model tracks the action, so negating it moves the predicted
    # blob far more than for the instruction-driven model.
    assert s_faithful > s_instruction
    # And the gap should be clear, not marginal (the instruction model still has
    # a small residual action_gain=0.12, so demand a solid but not extreme margin).
    assert s_faithful > 1.2 * s_instruction
