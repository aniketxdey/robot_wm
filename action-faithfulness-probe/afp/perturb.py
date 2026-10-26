"""Perturbation operators for the Action-Faithfulness Probe.

AFP measures how an action-conditioned video world model's predicted future
diverges when we edit its *input*. This module supplies the edits.

There are two channels:

ACTION channel
    Operators that take an action chunk ``(T, 7)`` and return a NEW chunk of the
    same shape. The 7 dims are ``(dx, dy, dz, droll, dpitch, dyaw, gripper)``;
    see :data:`afp.rollout.MOTION_DIMS` (the first 6) and
    :data:`afp.rollout.GRIPPER_DIM` (the 7th). These operators span a *physical
    severity ladder*: from physically-inert edits (the metric calibration null)
    up to a maximally-inconsistent random action.

INSTRUCTION channel
    Operators that take an instruction embedding ``(E,)`` and return a NEW
    embedding of the same shape. These calibrate the instruction side: a
    paraphrase preserves semantics (instruction-channel FLOOR) while a
    counterfactual flips them (instruction-channel CEILING).

Design notes
------------
* Every operator is a **pure function**: it never mutates its input and always
  returns a freshly allocated array. This makes operators safe to compose and
  trivially parallelizable.
* The signature for every action operator is
  ``op(actions, rng=None, **params) -> np.ndarray``. Operators that do not need
  randomness simply ignore ``rng``; this keeps the registry uniform so callers
  can invoke any operator the same way.
* NULL operators (the metric-calibration floor). There are two:

  - :func:`null_micro` is the **embodiment-agnostic** null (DESIGN Rev 4): a
    *sub-quantization* perturbation -- Gaussian noise smaller than one action
    token bin / the control-noise floor -- applied to ALL motion dims. Any
    faithful model should be invariant to it, so ``S ~ determinism floor`` on a
    real robot too. This is the null the Stage-2 gate should use. (Grounding:
    counterfactual invariance, Veitch et al. 2021; action-group identity /
    zero-action drift, "World Models as Group Actions" arXiv:2605.24578.)
  - :func:`null_shuffle` is a **mock-only illustrative** null: it perturbs dims
    ``(2, 3, 4, 5)`` which are INERT *for the synthetic mock* (see
    ``afp/adapters/mock.py``) but are physically ACTIVE on a real robot (``dz``
    is vertical motion!). It is therefore an *invalid placebo* off the mock and
    must NOT be used as the floor for real checkpoints -- kept only for the
    no-GPU demo. Prefer :func:`null_micro` everywhere else.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from afp.rollout import ACTION_DIM, MOTION_DIMS, GRIPPER_DIM

__all__ = [
    "identity",
    "null_micro",
    "null_shuffle",
    "gauss",
    "scale",
    "reverse",
    "negate",
    "swap",
    "zero",
    "random_action",
    "ACTION_OPERATORS",
    "SEVERITY_RANK",
    "gauss_ladder",
    "gauss_ladder_severity",
    "severity_of",
    "instr_identity",
    "paraphrase",
    "cf_instruction",
    "interpolate_instruction",
    "instruction_ladder",
    "INSTRUCTION_OPERATORS",
]

# Tuple of motion-dim indices, handy for fancy indexing.
_MOTION = list(MOTION_DIMS)


def _prepare(actions: np.ndarray) -> np.ndarray:
    """Validate and return a fresh, contiguous float64 copy of an action chunk."""
    a = np.array(actions, dtype=np.float64, copy=True)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions must be (T, {ACTION_DIM}), got shape {a.shape}"
        )
    return a


def _as_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Coerce ``rng`` into a Generator (default_rng if None)."""
    if rng is None:
        return np.random.default_rng()
    return rng


# ---------------------------------------------------------------------------
# Action operators
# ---------------------------------------------------------------------------
def identity(actions: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return an exact copy of ``actions`` (no edit).

    The reference channel: feeding ``identity(actions)`` back into the model
    yields the unperturbed rollout, so the only divergence from the original is
    the model's own determinism floor.
    """
    return _prepare(actions)


def null_micro(
    actions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    sigma: float = 1e-3,
) -> np.ndarray:
    """Embodiment-agnostic NULL: a sub-quantization micro-edit of the motion dims.

    Adds tiny Gaussian noise (``sigma`` defaults to ``1e-3``, well below a typical
    action-token bin width / control-noise floor) to all MOTION dims. A faithful
    world model should be *invariant* to a perturbation this small, so the
    resulting sensitivity ``S`` should sit at the model's determinism/sampling
    floor -- on a real robot, not just the mock. This is the recommended metric-
    calibration null and the basis for the Stage-2 null-floor gate (DESIGN Rev 4,
    Sec 2.3). Grounding: counterfactual invariance (Veitch et al. 2021) and the
    action-group identity / zero-action drift ("World Models as Group Actions",
    arXiv:2605.24578).

    Unlike :func:`null_shuffle`, this makes no assumption about which dims are
    inert, so it is valid for any embodiment.
    """
    a = _prepare(actions)
    rng = _as_rng(rng)
    noise = rng.normal(0.0, sigma, size=(a.shape[0], len(_MOTION)))
    a[:, _MOTION] = a[:, _MOTION] + noise
    return a


def null_shuffle(
    actions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    inert_dims: Sequence[int] = (2, 3, 4, 5),
    jitter: float = 1.0,
) -> np.ndarray:
    """Mock-only illustrative NULL: perturb dims ``(2,3,4,5)``, leave ``0,1,6``.

    .. warning::
        These dims are inert *only for the synthetic mock*. On a real robot
        ``dz``/rotations are physically active, so this is an **invalid placebo**
        off the mock -- use :func:`null_micro` as the real floor. Kept for the
        no-GPU demo only (DESIGN Rev 4, Sec 2.3).


    Both a temporal shuffle (permute rows within each inert column) and additive
    jitter are applied to the inert dims so the edit is numerically substantial,
    yet — because the mock world model ignores dims ``(2, 3, 4, 5)`` — it is
    physically a no-op. Expected sensitivity ``S ~ floor``. This is the
    metric-calibration null (AGRA-style null).

    Parameters
    ----------
    inert_dims : sequence of int
        Action dims that do not move the predicted future. Defaults to the
        mock's inert set ``(2, 3, 4, 5)``.
    jitter : float
        Std of optional Gaussian jitter added to the inert dims after shuffling.
    """
    a = _prepare(actions)
    rng = _as_rng(rng)
    T = a.shape[0]
    for d in inert_dims:
        if d < 0 or d >= ACTION_DIM:
            raise ValueError(f"inert dim {d} out of range for ACTION_DIM={ACTION_DIM}")
        if T > 1:
            a[:, d] = a[rng.permutation(T), d]
        if jitter > 0:
            a[:, d] = a[:, d] + rng.normal(0.0, jitter, size=T)
    return a


def gauss(
    actions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    sigma: float = 0.1,
) -> np.ndarray:
    """Add ``N(0, sigma)`` noise to the MOTION dims only (gripper untouched)."""
    a = _prepare(actions)
    rng = _as_rng(rng)
    noise = rng.normal(0.0, sigma, size=(a.shape[0], len(_MOTION)))
    a[:, _MOTION] = a[:, _MOTION] + noise
    return a


def scale(
    actions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    k: float = 1.5,
) -> np.ndarray:
    """Multiply the MOTION dims by ``k`` (gripper untouched)."""
    a = _prepare(actions)
    a[:, _MOTION] = a[:, _MOTION] * k
    return a


def reverse(actions: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Reverse the temporal order of the whole chunk (all dims)."""
    a = _prepare(actions)
    return a[::-1].copy()


def negate(actions: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Negate the MOTION dims; leave the gripper dim as-is.

    A directionally-opposite command: a severe, clearly action-inconsistent edit.
    """
    a = _prepare(actions)
    a[:, _MOTION] = -a[:, _MOTION]
    return a


def swap(
    actions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    donor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Replace ``actions`` with another episode's chunk ``donor``.

    ``donor`` is trimmed (if longer) or zero-padded (if shorter) to match
    ``actions``' temporal length ``T``, then returned as a NEW ``(T, 7)`` array.
    Because the donor is a coherent action from a *different* episode, this is a
    severe but plausible (in-distribution) edit.

    Parameters
    ----------
    donor : np.ndarray
        Required. The substitute action chunk ``(T_d, 7)``.
    """
    a = _prepare(actions)
    if donor is None:
        raise ValueError("swap requires a `donor` action chunk (keyword arg)")
    d = _prepare(donor)
    T = a.shape[0]
    out = np.zeros((T, ACTION_DIM), dtype=np.float64)
    n = min(T, d.shape[0])
    out[:n] = d[:n]
    return out


def zero(actions: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Set the MOTION dims to 0 (keep the gripper dim as-is).

    Commands "hold position": the blob stops translating but the gripper state is
    preserved.
    """
    a = _prepare(actions)
    a[:, _MOTION] = 0.0
    return a


def random_action(
    actions: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    sigma: float = 0.5,
) -> np.ndarray:
    """Replace the MOTION dims with ``N(0, sigma)`` noise (gripper untouched).

    A maximally-inconsistent command unrelated to the original — the severe end
    of the physical severity ladder.
    """
    a = _prepare(actions)
    rng = _as_rng(rng)
    a[:, _MOTION] = rng.normal(0.0, sigma, size=(a.shape[0], len(_MOTION)))
    return a


#: Registry of action operators keyed by name. ``swap`` is intentionally omitted
#: because it requires a ``donor``; import :func:`swap` directly to use it.
ACTION_OPERATORS: Dict[str, Callable[..., np.ndarray]] = {
    "identity": identity,
    "null_micro": null_micro,
    "null_shuffle": null_shuffle,
    "gauss": gauss,
    "scale": scale,
    "reverse": reverse,
    "negate": negate,
    "zero": zero,
    "random_action": random_action,
}


#: Increasing physical-severity ordering used by the monotonicity test.
#:
#: The rank is an ordinal estimate of how much an edit should move the predicted
#: future (NOT a guarantee of exact magnitude). Ties (e.g. ``identity`` and
#: ``null_shuffle``) are edits expected to land at the same level:
#:
#:   0  identity      -- no edit (reference / floor)
#:   0  null_shuffle  -- edits only inert dims => physically null (~ floor)
#:   0  zero          -- removes motion; for a moving chunk this is a real change,
#:                       but it is the mildest "destroy the signal" edit, so we
#:                       place it just above the true nulls (see note below)
#:   1  gauss         -- small additive motion noise
#:   2  scale         -- amplifies the existing motion
#:   3  reverse       -- same motions, reordered in time
#:   4  negate        -- directionally opposite motion
#:   4  swap          -- a coherent but unrelated episode's motion
#:   5  random_action -- motion replaced by large noise (maximally inconsistent)
#:
#: Note: ``zero`` is given rank 1 alongside small edits because, unlike the
#: nulls, it does change the physically-active dims; keep it distinct from the
#: true floor operators when interpreting monotonicity.
SEVERITY_RANK: Dict[str, int] = {
    "identity": 0,
    "null_micro": 0,
    "null_shuffle": 0,
    "zero": 1,
    "gauss": 1,
    "scale": 2,
    "reverse": 3,
    "negate": 4,
    "swap": 4,
    "random_action": 5,
}


def severity_of(name: str) -> int:
    """Return the physical-severity rank of an operator name.

    Understands the dynamic ``gauss@{sigma}`` names produced by
    :func:`gauss_ladder` (see :func:`gauss_ladder_severity`).
    """
    if name in SEVERITY_RANK:
        return SEVERITY_RANK[name]
    if name.startswith("gauss@"):
        # All ladder rungs are "small additive noise"; finer ordering is by sigma.
        return 1
    raise KeyError(f"unknown operator name '{name}'")


def gauss_ladder(sigmas: Sequence[float]) -> Dict[str, Callable[..., np.ndarray]]:
    """Build a family of Gaussian-noise ops, one per sigma, for the ladder test.

    Returns a dict mapping ``"gauss@{sigma}"`` to a pure operator that adds
    ``N(0, sigma)`` to the MOTION dims. Because mean ``|delta|`` of zero-mean
    Gaussian noise grows monotonically with sigma, feeding these through a model
    with ``action_gain > 0`` yields a monotonically increasing sensitivity ladder.

    Example
    -------
    >>> ops = gauss_ladder([0.05, 0.1, 0.2])
    >>> sorted(ops)
    ['gauss@0.05', 'gauss@0.1', 'gauss@0.2']
    """
    ops: Dict[str, Callable[..., np.ndarray]] = {}
    for s in sigmas:
        s_val = float(s)

        def _op(actions, rng=None, _sigma=s_val):
            return gauss(actions, rng=rng, sigma=_sigma)

        _op.__name__ = f"gauss@{s_val}"
        _op.__doc__ = f"Add N(0, {s_val}) to MOTION dims (gauss ladder rung)."
        ops[f"gauss@{s_val}"] = _op
    return ops


def gauss_ladder_severity(sigmas: Sequence[float]) -> Dict[str, float]:
    """Companion severity helper for :func:`gauss_ladder`.

    Maps each ``"gauss@{sigma}"`` name to a continuous severity score equal to
    the sigma itself, giving the strictly-increasing ordering the monotonicity
    test expects.
    """
    return {f"gauss@{float(s)}": float(s) for s in sigmas}


# ---------------------------------------------------------------------------
# Instruction operators
# ---------------------------------------------------------------------------
def _prepare_emb(emb: np.ndarray) -> np.ndarray:
    """Validate and return a fresh float64 copy of an instruction embedding."""
    e = np.array(emb, dtype=np.float64, copy=True)
    if e.ndim != 1:
        raise ValueError(f"instruction_emb must be 1-D (E,), got shape {e.shape}")
    return e


def _normalize(e: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(e))
    if n < 1e-12:
        return e
    return e / n


def instr_identity(emb: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return an exact copy of the instruction embedding (no edit)."""
    return _prepare_emb(emb)


def paraphrase(
    emb: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    eps: float = 0.05,
) -> np.ndarray:
    """Nudge the embedding by small Gaussian noise, renormalized to unit length.

    Semantics are ~preserved (cosine to the original stays high), so this is the
    instruction-channel FLOOR: a small, "same meaning, different words" divergence.
    """
    e = _prepare_emb(emb)
    rng = _as_rng(rng)
    e = e + rng.normal(0.0, eps, size=e.shape)
    return _normalize(e)


def cf_instruction(
    emb: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    target_emb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a unit embedding with *changed* semantics (counterfactual).

    The result is roughly orthogonal-to-opposite of ``emb`` (low cosine), so this
    is the instruction-channel CEILING: a large, "different meaning" divergence.

    Parameters
    ----------
    target_emb : np.ndarray, optional
        If supplied, use this counterfactual embedding (renormalized to unit)
        instead of synthesizing a random one. Useful for paired counterfactuals.
    """
    e = _prepare_emb(emb)
    if target_emb is not None:
        return _normalize(_prepare_emb(target_emb))

    rng = _as_rng(rng)
    u = _normalize(e)
    # Draw a random direction, then remove its component along ``u`` so the result
    # is orthogonal to the original (cosine ~ 0). Retry if we drew (near) parallel.
    for _ in range(8):
        r = rng.normal(0.0, 1.0, size=e.shape)
        r = r - float(np.dot(r, u)) * u
        n = float(np.linalg.norm(r))
        if n > 1e-8:
            return r / n
    # Degenerate fallback (e.g. E == 1): just flip the sign.
    return -u


def interpolate_instruction(
    emb: np.ndarray,
    target_emb: np.ndarray,
    frac: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Spherical-ish interpolation from ``emb`` toward ``target_emb`` by ``frac``.

    Returns a unit embedding ``frac`` of the way from the original meaning to the
    counterfactual ``target_emb`` (``frac=0`` -> original, ``frac=1`` -> target).
    This is the instruction-channel analogue of the action ``gauss`` ladder: it
    lets us sweep instruction *dose* and build a dose-response curve so the
    instruction axis can be compared to the action axis by SHAPE, not by an
    arbitrary single operating point (DESIGN Rev 4, Sec 2.2).
    """
    e = _prepare_emb(emb)
    t = _prepare_emb(target_emb)
    if t.shape != e.shape:
        raise ValueError(f"target_emb shape {t.shape} != emb shape {e.shape}")
    f = float(np.clip(frac, 0.0, 1.0))
    return _normalize((1.0 - f) * e + f * t)


def instruction_ladder(
    emb: np.ndarray,
    target_emb: np.ndarray,
    fracs: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> Dict[str, np.ndarray]:
    """Build an instruction dose ladder ``{"instr@{frac}": embedding}``.

    Each rung interpolates the original meaning toward the counterfactual
    ``target_emb`` by an increasing fraction, giving a strictly-increasing
    semantic-dose sequence for the instruction-channel dose-response sweep.
    """
    return {f"instr@{float(f)}": interpolate_instruction(emb, target_emb, f) for f in fracs}


#: Registry of instruction operators keyed by name.
INSTRUCTION_OPERATORS: Dict[str, Callable[..., np.ndarray]] = {
    "instr_identity": instr_identity,
    "paraphrase": paraphrase,
    "cf_instruction": cf_instruction,
}
