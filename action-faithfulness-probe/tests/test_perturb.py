"""Tests for ``afp.perturb`` -- the action / instruction perturbation operators."""

from __future__ import annotations

import numpy as np
import pytest

from afp.rollout import ACTION_DIM, MOTION_DIMS, GRIPPER_DIM
from afp import perturb
from afp.perturb import (
    identity,
    null_shuffle,
    gauss,
    scale,
    reverse,
    negate,
    swap,
    zero,
    random_action,
    ACTION_OPERATORS,
    SEVERITY_RANK,
    gauss_ladder,
    gauss_ladder_severity,
    instr_identity,
    paraphrase,
    cf_instruction,
    INSTRUCTION_OPERATORS,
)


T = 12
E = 16


def make_actions(seed: int = 0) -> np.ndarray:
    """A (T, 7) chunk with non-trivial motion and a varying gripper."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.3, size=(T, ACTION_DIM)).astype(np.float64)
    # Ensure dx, dy have clear, non-zero motion so motion-edits are observable.
    a[:, 0] = 0.2
    a[:, 1] = -0.15
    a[:, GRIPPER_DIM] = (np.arange(T) % 2).astype(np.float64)  # 0/1 gripper toggling
    return a


def make_emb(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, size=E)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Purity + shape preservation
# ---------------------------------------------------------------------------
ZERO_ARG_OPS = [identity, null_shuffle, gauss, scale, reverse, negate, zero, random_action]


@pytest.mark.parametrize("op", ZERO_ARG_OPS)
def test_purity_input_unchanged(op):
    a = make_actions()
    before = a.copy()
    rng = np.random.default_rng(123)
    _ = op(a, rng=rng)
    assert np.array_equal(a, before), f"{op.__name__} mutated its input"


@pytest.mark.parametrize("op", ZERO_ARG_OPS)
def test_shape_and_newarray(op):
    a = make_actions()
    out = op(a, rng=np.random.default_rng(0))
    assert out.shape == (T, ACTION_DIM)
    assert out is not a, f"{op.__name__} returned the same object"


def test_swap_purity_shape():
    a = make_actions(0)
    donor = make_actions(7)
    before = a.copy()
    out = swap(a, donor=donor)
    assert np.array_equal(a, before)
    assert out.shape == (T, ACTION_DIM)
    assert out is not a and out is not donor


# ---------------------------------------------------------------------------
# Operator-specific semantics
# ---------------------------------------------------------------------------
def test_identity_is_true_copy():
    a = make_actions()
    out = identity(a)
    assert np.array_equal(out, a)
    assert out is not a


def test_negate_flips_motion_not_gripper():
    a = make_actions()
    out = negate(a)
    assert np.allclose(out[:, list(MOTION_DIMS)], -a[:, list(MOTION_DIMS)])
    assert np.array_equal(out[:, GRIPPER_DIM], a[:, GRIPPER_DIM])


def test_null_shuffle_leaves_active_dims_untouched():
    a = make_actions()
    out = null_shuffle(a, rng=np.random.default_rng(0))
    for d in (0, 1, GRIPPER_DIM):
        assert np.array_equal(out[:, d], a[:, d]), f"dim {d} was modified"
    # at least one inert dim should actually change
    inert_changed = any(not np.allclose(out[:, d], a[:, d]) for d in (2, 3, 4, 5))
    assert inert_changed


def test_zero_clears_motion_keeps_gripper():
    a = make_actions()
    out = zero(a)
    assert np.allclose(out[:, list(MOTION_DIMS)], 0.0)
    assert np.array_equal(out[:, GRIPPER_DIM], a[:, GRIPPER_DIM])


def test_scale_multiplies_motion():
    a = make_actions()
    out = scale(a, k=2.0)
    assert np.allclose(out[:, list(MOTION_DIMS)], 2.0 * a[:, list(MOTION_DIMS)])
    assert np.array_equal(out[:, GRIPPER_DIM], a[:, GRIPPER_DIM])


def test_reverse_reverses_time():
    a = make_actions()
    out = reverse(a)
    assert np.array_equal(out, a[::-1])


def test_swap_trims_and_pads():
    a = make_actions()
    long_donor = np.ones((T + 5, ACTION_DIM))
    out = swap(a, donor=long_donor)
    assert out.shape == (T, ACTION_DIM)
    assert np.allclose(out, 1.0)

    short_donor = np.full((T - 4, ACTION_DIM), 2.0)
    out2 = swap(a, donor=short_donor)
    assert out2.shape == (T, ACTION_DIM)
    assert np.allclose(out2[: T - 4], 2.0)
    assert np.allclose(out2[T - 4:], 0.0)


def test_swap_requires_donor():
    with pytest.raises(ValueError):
        swap(make_actions())


def test_registry_contents():
    assert "swap" not in ACTION_OPERATORS  # needs a donor
    for name in ["identity", "null_shuffle", "gauss", "scale", "reverse", "negate", "zero", "random_action"]:
        assert name in ACTION_OPERATORS
        out = ACTION_OPERATORS[name](make_actions(), rng=np.random.default_rng(0))
        assert out.shape == (T, ACTION_DIM)


# ---------------------------------------------------------------------------
# Gauss ladder monotonicity
# ---------------------------------------------------------------------------
def test_gauss_ladder_increasing_delta():
    sigmas = [0.02, 0.05, 0.1, 0.2, 0.4]
    ops = gauss_ladder(sigmas)
    sev = gauss_ladder_severity(sigmas)
    assert set(ops) == set(sev)

    a = make_actions()
    motion = list(MOTION_DIMS)
    mean_deltas = []
    for s in sigmas:
        name = f"gauss@{float(s)}"
        # average over several seeds to get a stable estimate
        ds = []
        for seed in range(32):
            out = ops[name](a, rng=np.random.default_rng(seed))
            ds.append(np.mean(np.abs(out[:, motion] - a[:, motion])))
        mean_deltas.append(np.mean(ds))

    for lo, hi in zip(mean_deltas, mean_deltas[1:]):
        assert hi > lo, f"mean |delta| not increasing: {mean_deltas}"


def test_severity_rank_ordering():
    assert SEVERITY_RANK["identity"] == 0
    assert SEVERITY_RANK["null_shuffle"] == 0
    assert SEVERITY_RANK["random_action"] == max(SEVERITY_RANK.values())
    assert SEVERITY_RANK["negate"] > SEVERITY_RANK["gauss"]
    assert perturb.severity_of("gauss@0.1") == 1


# ---------------------------------------------------------------------------
# Instruction operators
# ---------------------------------------------------------------------------
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_instr_identity_copy():
    e = make_emb()
    out = instr_identity(e)
    assert np.array_equal(out, e)
    assert out is not e


def test_instr_purity():
    e = make_emb()
    before = e.copy()
    paraphrase(e, rng=np.random.default_rng(0))
    cf_instruction(e, rng=np.random.default_rng(0))
    assert np.array_equal(e, before)


def test_paraphrase_close_cf_far():
    e = make_emb()
    for seed in range(16):
        p = paraphrase(e, rng=np.random.default_rng(seed))
        assert p.shape == e.shape
        assert _cos(p, e) > 0.95
        assert abs(np.linalg.norm(p) - 1.0) < 1e-9

        c = cf_instruction(e, rng=np.random.default_rng(seed))
        assert c.shape == e.shape
        assert _cos(c, e) < 0.5
        assert abs(np.linalg.norm(c) - 1.0) < 1e-9


def test_cf_instruction_target_emb():
    e = make_emb(1)
    target = make_emb(99) * 3.0  # not unit length
    c = cf_instruction(e, target_emb=target)
    assert abs(np.linalg.norm(c) - 1.0) < 1e-9
    assert _cos(c, target) > 0.999


def test_instruction_registry():
    for name in ["instr_identity", "paraphrase", "cf_instruction"]:
        assert name in INSTRUCTION_OPERATORS


# ---------------------------------------------------------------------------
# End-to-end with the mock world model
# ---------------------------------------------------------------------------
def test_end_to_end_null_below_negate():
    from afp.adapters.mock import MockWorldModel
    from afp.distance import get_distance

    model = MockWorldModel("m", action_gain=1.0, instruction_gain=0.15, size=64, horizon=12)
    dist = get_distance("pixel_l2")

    a = make_actions()
    emb = make_emb()
    o0 = np.zeros((64, 64, 3), dtype=np.uint8)

    base = model.rollout(o0, a, "task", emb, seed=0).frames
    null_frames = model.rollout(o0, null_shuffle(a, rng=np.random.default_rng(0)), "task", emb, seed=0).frames
    neg_frames = model.rollout(o0, negate(a), "task", emb, seed=0).frames

    d_null = dist(base, null_frames)
    d_neg = dist(base, neg_frames)

    # null is physically inert => ~ exactly the floor (here 0 with fixed seed)
    assert float(np.mean(d_null)) < 1e-9
    # negate moves the blob strongly => large divergence
    assert float(np.mean(d_neg)) > float(np.mean(d_null))
    assert float(np.mean(d_neg)) > 1e-3
