"""Pure metric math for the Action-Faithfulness Probe (AFP).

This module is *pure*: it does no model inference, no I/O, and no orchestration.
It consumes frame stacks (``(T, H, W, 3)`` uint8, the ``Rollout.frames``
convention from :mod:`afp.rollout`) plus a ``Distance`` callable
(:mod:`afp.distance`) and turns them into the horizon-resolved quantities the
DESIGN calls for. Everything here is numpy; ``scipy`` is used opportunistically
for Spearman's rho with a hand-rolled fallback so the harness still runs in the
light (numpy-only) install.

Mapping to ``DESIGN.md`` (Section references):

* ``sensitivity_curve``      -> Sec 2.1, the Action-Sensitivity Curve
  ``S_P(t) = D( M(o0,l,a)[t], M(o0,l,P(a))[t] )``. The headline horizon-resolved
  signal: how far the perturbed rollout has diverged from the base rollout at
  each step ``t``.
* ``transition_curve``       -> Sec 2.1, the transition-space variant
  ``S^Delta_P(t) = D( base[t]-base[t-1], pert[t]-pert[t-1] )`` (inspired by
  dWorldEval). **Exploratory** -- see the function docstring.
* ``seed_floor``             -> Sec 2.3 / Stage-1 gate. The empirical determinism
  / seed-noise floor ``epsilon(t)``: identical inputs, different seeds should
  give ``S ~ epsilon`` rather than exactly 0 (GPU sampling is not bit-exact).
* ``aggregate_curves``       -> per-episode -> population curve with mean / std /
  sem / n at each ``t``.
* ``normalized_sensitivity`` -> Sec 2.2, per-channel calibrated sensitivity
  ``sens_c(t) = clip( (S_c - floor_c) / (ceil_c - floor_c), 0, 1 )`` so the
  action vs. instruction channels are comparable on a unitless [0, 1] scale.
* ``monotonicity_spearman``  -> Sec 2.4 / Stage-2 gate. Spearman rho between
  perturbation-severity rank and mean divergence (the validity check borrowed
  from UWM-CF).
* ``bootstrap_ci``           -> percentile bootstrap confidence intervals for any
  scalar statistic over episodes (used to put error bars on the gates).
* ``separation_ratio``       -> Stage-2 "separation" gate: how many multiples of
  the seed floor a severe perturbation clears (the "3x floor" check).

Conventions shared by every function here
-----------------------------------------
* **Horizon alignment.** When two frame stacks (or two curves) differ in length
  ``T``, they are truncated to the shorter one before comparison. Real adapters
  emit different native horizons, so silent truncation to ``min(T)`` is the
  documented behaviour rather than an error.
* **Shape errors are loud.** A mismatch in the spatial / channel dims
  ``(H, W, 3)`` is a genuine bug (incompatible renders) and raises ``ValueError``
  with a descriptive message.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "sensitivity_curve",
    "transition_curve",
    "seed_floor",
    "aggregate_curves",
    "normalized_sensitivity",
    "dose_response",
    "monotonicity_spearman",
    "bootstrap_ci",
    "bootstrap_bca_ci",
    "paired_permutation_test",
    "holm_bonferroni",
    "separation_ratio",
    "paired_curve_over_episodes",
]


# ---------------------------------------------------------------------------
# Internal helpers: shape validation + horizon alignment.
# ---------------------------------------------------------------------------
def _as_frames(x: np.ndarray, name: str) -> np.ndarray:
    """Coerce to ndarray and assert the ``(T, H, W, 3)`` frame-stack contract."""
    arr = np.asarray(x)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"{name} must be a frame stack of shape (T, H, W, 3); got {arr.shape}"
        )
    return arr


def _align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate two frame stacks and truncate both to the shared horizon ``min(T)``.

    Raises
    ------
    ValueError
        If either input is not ``(T, H, W, 3)``, or if the spatial/channel dims
        ``(H, W, 3)`` disagree (a real incompatibility, not a horizon mismatch).
    """
    a = _as_frames(a, "frames_a")
    b = _as_frames(b, "frames_b")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(
            "spatial/channel shape mismatch between frame stacks: "
            f"{a.shape[1:]} vs {b.shape[1:]} (H, W, 3 must match exactly)"
        )
    t = min(a.shape[0], b.shape[0])
    return a[:t], b[:t]


# ---------------------------------------------------------------------------
# Sec 2.1 -- Action-Sensitivity Curve S(t) and its transition-space variant.
# ---------------------------------------------------------------------------
def sensitivity_curve(
    base_frames: np.ndarray,
    pert_frames: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Action-Sensitivity Curve ``S(t)`` (DESIGN Sec 2.1).

    Per-frame perceptual distance between the *base* (unperturbed-action) rollout
    and the *perturbed*-action rollout::

        S_P(t) = D( M(o0, l, a)[t],  M(o0, l, P(a))[t] )   for t = 1..T

    A genuinely action-faithful model produces an ``S(t)`` that rises with ``t``
    (action edits compound over the rollout) and is well above the determinism
    floor (:func:`seed_floor`); an action-insensitive model produces an ``S(t)``
    pinned near that floor.

    Parameters
    ----------
    base_frames, pert_frames : np.ndarray
        ``(T, H, W, 3)`` uint8 rollouts. If the two horizons differ they are
        truncated to ``min(T)``.
    distance : callable
        A :class:`afp.distance.Distance` (or any callable) mapping two
        ``(T, H, W, 3)`` stacks to a per-frame ``(T,)`` distance vector.

    Returns
    -------
    np.ndarray
        ``S(t)`` of shape ``(T,)`` (``float``), with ``T = min`` of the inputs.
    """
    a, b = _align_pair(base_frames, pert_frames)
    s = np.asarray(distance(a, b), dtype=np.float64).reshape(-1)
    if s.shape[0] != a.shape[0]:
        raise ValueError(
            f"distance returned {s.shape[0]} values for {a.shape[0]} frames"
        )
    return s


def transition_curve(
    base_frames: np.ndarray,
    pert_frames: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Transition-space sensitivity ``S^Delta(t)`` (DESIGN Sec 2.1, EXPLORATORY).

    Instead of comparing frames directly, compare the consecutive-frame
    *difference images* (the per-step motion/appearance change) of the base vs.
    perturbed rollouts::

        d_base[t] = base[t] - base[t-1]
        d_pert[t] = pert[t] - pert[t-1]
        S^Delta_P(t) = D( d_base[t], d_pert[t] )   for t = 1..T-1

    so the output has length ``T - 1``.

    .. warning::

        **Exploratory, not load-bearing.** Difference images ``o_t - o_{t-1}``
        are *out-of-distribution* for LPIPS / DreamSim backbones, which are
        trained on natural RGB; and dWorldEval (the inspiration) has no public
        code, so this variant is unvalidated. Always cross-check ``S^Delta``
        against a **pixel-L2-of-differences baseline** (i.e. call this with
        ``get_distance('pixel_l2')``) before reporting it, and never let a
        headline claim rest on it alone.

    Implementation note (DESIGN Rev 4)
    ----------------------------------
    Difference images ``o_t - o_{t-1}`` lie in ``[-1, 1]`` in normalized frame
    space. We remap them to ``[0, 1]`` via ``(d + 1) / 2`` before handing them to
    ``distance``. This matters for the perceptual backends: ``LPIPSDistance``
    rescales inputs to ``[-1, 1]`` internally (so signed diffs would land in
    ``[-3, 1]``, outside LPIPS's contract) and ``DreamSimDistance`` casts to
    ``uint8`` (truncating negatives). Remapping keeps every backend in-range; for
    ``pixel_l2`` it only rescales the curve by a constant. ``pixel_l2`` remains
    the recommended backend for ``S^Delta``; prefer motion-explicit metrics
    (flow EPE / FVMD) or the Delta of frame-level ``S(t)`` for load-bearing claims.

    Parameters
    ----------
    base_frames, pert_frames : np.ndarray
        ``(T, H, W, 3)`` uint8 rollouts (truncated to ``min(T)`` if unequal).
    distance : callable
        Per-frame distance backend (see :func:`sensitivity_curve`).

    Returns
    -------
    np.ndarray
        ``S^Delta(t)`` of shape ``(T - 1,)`` (``float``). For ``T < 2`` returns
        an empty ``(0,)`` array (a single frame has no transitions).
    """
    a, b = _align_pair(base_frames, pert_frames)
    if a.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    a01 = a.astype(np.float64) / 255.0
    b01 = b.astype(np.float64) / 255.0
    # Diffs are in [-1, 1]; remap to [0, 1] so perceptual backends stay in-range.
    d_base = (a01[1:] - a01[:-1] + 1.0) / 2.0
    d_pert = (b01[1:] - b01[:-1] + 1.0) / 2.0
    s = np.asarray(distance(d_base, d_pert), dtype=np.float64).reshape(-1)
    if s.shape[0] != d_base.shape[0]:
        raise ValueError(
            f"distance returned {s.shape[0]} values for {d_base.shape[0]} transitions"
        )
    return s


# ---------------------------------------------------------------------------
# Sec 2.3 -- the determinism / seed-noise floor epsilon(t).
# ---------------------------------------------------------------------------
def seed_floor(
    rollouts_same_input: Sequence[np.ndarray],
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical determinism floor ``epsilon(t)`` (DESIGN Sec 2.3 / Stage-1 gate).

    Given ``>= 2`` rollouts produced from **identical inputs but different
    seeds**, measure how much the model's own stochasticity (e.g. non-bit-exact
    GPU sampling) moves the frames. This sets the noise floor against which every
    real ``S(t)`` must be judged: the Stage-1 gate is ``S_identity(t) <
    epsilon(t)`` for this *measured* ``epsilon``, not ``== 0``.

    The floor is the **per-frame distance averaged over all distinct rollout
    pairs**::

        epsilon(t) = mean_{i<j} D( R_i[t], R_j[t] )
        sigma(t)   = std_{i<j}  D( R_i[t], R_j[t] )

    Parameters
    ----------
    rollouts_same_input : sequence of np.ndarray
        At least two ``(T, H, W, 3)`` rollouts from identical inputs / differing
        seeds. Stacks of differing length are truncated to the common ``min(T)``.
    distance : callable
        Per-frame distance backend.

    Returns
    -------
    (mean, std) : tuple of np.ndarray
        Each of shape ``(T,)``: the mean and standard deviation of the pairwise
        per-frame distance at each horizon ``t``. With exactly two rollouts there
        is a single pair, so ``std`` is all zeros.

    Raises
    ------
    ValueError
        If fewer than two rollouts are supplied.
    """
    rolls = list(rollouts_same_input)
    if len(rolls) < 2:
        raise ValueError(
            f"seed_floor needs >= 2 rollouts from identical inputs; got {len(rolls)}"
        )
    rolls = [_as_frames(r, f"rollouts_same_input[{i}]") for i, r in enumerate(rolls)]
    # Validate mutual (H, W, 3) compatibility and find the shared horizon.
    hwc = rolls[0].shape[1:]
    for i, r in enumerate(rolls[1:], start=1):
        if r.shape[1:] != hwc:
            raise ValueError(
                f"rollout {i} spatial/channel shape {r.shape[1:]} != {hwc}"
            )
    t = min(r.shape[0] for r in rolls)
    rolls = [r[:t] for r in rolls]

    pair_curves: List[np.ndarray] = []
    n = len(rolls)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.asarray(distance(rolls[i], rolls[j]), dtype=np.float64).reshape(-1)
            pair_curves.append(d)
    stacked = np.stack(pair_curves, axis=0)  # (n_pairs, T)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    return mean, std


# ---------------------------------------------------------------------------
# Per-episode -> population curve aggregation.
# ---------------------------------------------------------------------------
def aggregate_curves(curves: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Stack equal-length per-episode curves into a population summary.

    Parameters
    ----------
    curves : sequence of np.ndarray
        One ``(T,)`` curve per episode (e.g. each an ``S(t)`` from
        :func:`sensitivity_curve`). All curves must share the same length ``T``.

    Returns
    -------
    dict
        ``{'mean', 'std', 'sem', 'n'}``, each a ``(T,)`` array:

        * ``mean`` -- per-``t`` mean across episodes.
        * ``std``  -- per-``t`` population standard deviation (``ddof=0``).
        * ``sem``  -- per-``t`` standard error of the mean
          (``std(ddof=1) / sqrt(n)``; zeros when ``n == 1``).
        * ``n``    -- per-``t`` episode count (constant; returned as an array so
          the dict is uniformly array-valued and ready for plotting / CIs).

    Raises
    ------
    ValueError
        If no curves are given or their lengths differ.
    """
    cs = [np.asarray(c, dtype=np.float64).reshape(-1) for c in curves]
    if not cs:
        raise ValueError("aggregate_curves requires at least one curve")
    lengths = {c.shape[0] for c in cs}
    if len(lengths) != 1:
        raise ValueError(f"curves must share one length; got lengths {sorted(lengths)}")
    arr = np.stack(cs, axis=0)  # (n, T)
    n = arr.shape[0]
    t = arr.shape[1]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    if n > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        sem = np.zeros(t, dtype=np.float64)
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "n": np.full(t, n, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Sec 2.2 -- per-channel normalized (calibrated) sensitivity.
# ---------------------------------------------------------------------------
def normalized_sensitivity(
    S_c: np.ndarray,
    floor_c: np.ndarray,
    ceil_c: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel normalized sensitivity ``sens_c(t)`` (DESIGN Sec 2.2).

    Calibrate a raw sensitivity ``S_c`` against *that channel's own* floor (an
    edit that should change nothing) and ceiling (a maximal, meaning-changing
    edit), so the action and instruction channels become comparable on a
    unitless ``[0, 1]`` scale::

        sens_c(t) = clip( (S_c(t) - floor_c(t)) / (ceil_c(t) - floor_c(t) + eps), 0, 1 )

    This removes the "violent action edit vs. gentle paraphrase" artifact that
    plagues a naive cross-channel ratio: ``floor`` maps to ``0`` and ``ceil``
    maps to ``1`` within each channel.

    Works elementwise on ``(T,)`` arrays *or* on scalars (any broadcastable mix).

    Parameters
    ----------
    S_c : array-like or scalar
        Raw sensitivity (e.g. ``S(t)`` for the action channel).
    floor_c : array-like or scalar
        Channel floor (null/paraphrase edit). Maps to 0.
    ceil_c : array-like or scalar
        Channel ceiling (negate/counterfactual edit). Maps to 1.
    eps : float, default 1e-8
        Guards against a zero ``ceil - floor`` range.

    Returns
    -------
    np.ndarray
        ``sens_c`` clipped to ``[0, 1]``, broadcast to the common shape of the
        inputs.
    """
    s = np.asarray(S_c, dtype=np.float64)
    floor = np.asarray(floor_c, dtype=np.float64)
    ceil = np.asarray(ceil_c, dtype=np.float64)
    out = (s - floor) / (ceil - floor + eps)
    return np.clip(out, 0.0, 1.0)


def dose_response(
    doses: Sequence[float],
    responses: Sequence[float],
    floor: float = 0.0,
) -> Dict[str, float]:
    """Summarize a per-channel dose-response sweep by SHAPE (DESIGN Rev 4, Sec 2.2).

    Given increasing perturbation *doses* (e.g. gauss sigmas, or instruction
    interpolation fractions) and the scalar divergence ``responses`` they elicit,
    return shape descriptors that are comparable *across heterogeneous channels*
    without a shared magnitude unit:

    * ``auc``   -- trapezoidal area under the (dose-normalized) response curve,
      with the response measured above ``floor`` and the dose axis rescaled to
      ``[0, 1]`` so action (sigma) and instruction (fraction) sweeps are on the
      same x-scale.
    * ``slope`` -- least-squares slope of response-above-floor vs. normalized dose
      (local channel gain).
    * ``max``   -- maximum response above floor (the realized ceiling of the sweep).
    * ``norm_auc`` -- ``auc`` divided by ``max`` (pure shape, in ``[0, 1]``): how
      front-loaded vs. saturating the channel's response is, independent of level.

    Comparing ``slope`` / ``norm_auc`` across channels answers "how steeply does
    this channel drive the future" as SHAPE, sidestepping the unit mismatch that
    makes a single-operating-point cross-channel ratio an artifact.
    """
    d = np.asarray(doses, dtype=np.float64).reshape(-1)
    r = np.asarray(responses, dtype=np.float64).reshape(-1) - float(floor)
    if d.shape[0] != r.shape[0] or d.shape[0] < 2:
        raise ValueError("dose_response needs >=2 matched (dose, response) points")
    order = np.argsort(d)
    d, r = d[order], r[order]
    span = float(d[-1] - d[0])
    dn = (d - d[0]) / span if span > 1e-12 else np.linspace(0.0, 1.0, len(d))
    auc = float(np.trapezoid(r, dn)) if hasattr(np, "trapezoid") else float(np.trapz(r, dn))
    A = np.vstack([dn, np.ones_like(dn)]).T
    slope = float(np.linalg.lstsq(A, r, rcond=None)[0][0])
    rmax = float(np.max(r)) if r.size else 0.0
    return {
        "auc": auc,
        "slope": slope,
        "max": rmax,
        "norm_auc": float(auc / rmax) if abs(rmax) > 1e-12 else 0.0,
    }


# ---------------------------------------------------------------------------
# Sec 2.4 -- monotonicity (Spearman rho) of severity vs. divergence.
# ---------------------------------------------------------------------------
def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Assign average ranks to ``x`` (ties share the mean of their rank span).

    A small standalone implementation of ``scipy.stats.rankdata(method='average')``
    so the Spearman fallback works without scipy.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # ranks are 1-based; tied group [i..j] gets the average rank.
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def monotonicity_spearman(
    severity_ranks: Sequence[float],
    values: Sequence[float],
) -> float:
    """Spearman rho between perturbation severity and divergence (DESIGN Sec 2.4).

    The validity check borrowed from UWM-CF: along a graded perturbation ladder
    of increasing physical severity (``identity < small-noise < scale < reverse
    < negate < random``), mean divergence should be monotone non-decreasing. We
    quantify this with Spearman's rank correlation ``rho`` between the severity
    rank and the observed mean divergence. The Stage-2 gate wants ``rho >= 0.7``
    (kill below ``0.4``).

    Uses ``scipy.stats.spearmanr`` when available; otherwise falls back to
    average-rank ranking followed by a Pearson correlation on the ranks. Ties are
    handled via average ranks in both paths.

    Parameters
    ----------
    severity_ranks : sequence of float
        The ordered severity index of each perturbation (need not be integers;
        any monotone encoding works since only ranks matter).
    values : sequence of float
        The corresponding mean divergence (e.g. ``mean_t S(t)``) per perturbation.

    Returns
    -------
    float
        Spearman ``rho`` in ``[-1, 1]``. Returns ``nan`` if either input is
        constant (correlation undefined) or has fewer than two points.
    """
    sr = np.asarray(severity_ranks, dtype=np.float64).reshape(-1)
    vv = np.asarray(values, dtype=np.float64).reshape(-1)
    if sr.shape[0] != vv.shape[0]:
        raise ValueError(
            f"severity_ranks and values must match length; got {sr.shape[0]} vs {vv.shape[0]}"
        )
    if sr.shape[0] < 2:
        return float("nan")

    try:
        from scipy.stats import spearmanr  # type: ignore

        rho = spearmanr(sr, vv).correlation
        return float(rho)
    except Exception:
        # Manual fallback: Pearson correlation of the average ranks.
        rx = _rankdata_average(sr)
        ry = _rankdata_average(vv)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
        if denom == 0.0:
            return float("nan")
        return float(np.sum(rx * ry) / denom)


# ---------------------------------------------------------------------------
# Percentile bootstrap CI.
# ---------------------------------------------------------------------------
def bootstrap_ci(
    samples: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for a scalar ``statistic``.

    Resample ``samples`` with replacement ``n_boot`` times, recompute
    ``statistic`` on each resample, and return the empirical ``[alpha/2,
    1 - alpha/2]`` percentiles. Used to put error bars on the Stage-2 gate
    numbers (mean ``S``, separation ratio, etc.).

    Parameters
    ----------
    samples : sequence of float
        The observed values (e.g. per-episode mean divergence).
    n_boot : int, default 2000
        Number of bootstrap resamples.
    alpha : float, default 0.05
        Two-sided miscoverage; the default gives a 95% CI.
    statistic : callable, default ``np.mean``
        Maps a 1-D resample to a scalar.
    seed : int, default 0
        Seed for the resampling RNG (reproducible CIs).

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper CI bounds. For a single sample both bounds equal
        ``statistic([x])``.

    Raises
    ------
    ValueError
        If ``samples`` is empty.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.shape[0] == 0:
        raise ValueError("bootstrap_ci requires at least one sample")
    if x.shape[0] == 1:
        v = float(statistic(x))
        return v, v
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        stats[b] = float(statistic(x[idx[b]]))
    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_bca_ci(
    samples: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI (Efron 1987).

    The DESIGN Rev-4 Stage-2 gates fire on the **lower bound** of a BCa interval
    rather than a point estimate or a plain percentile interval (Dror et al.,
    ACL 2018; Bestgen, LREC 2022). BCa corrects the percentile interval for

    * **median bias** ``z0``: the normal quantile of the fraction of bootstrap
      replicates below the observed statistic, and
    * **acceleration** ``a``: the skewness of the jackknife (leave-one-out)
      distribution of the statistic,

    so coverage stays honest for skewed, small-n samples -- exactly the regime
    of per-episode divergence means (n ~ 20-50, right-skewed).

    Falls back to the plain percentile interval when the correction is
    degenerate (all replicates identical, or an all-equal jackknife).

    Parameters mirror :func:`bootstrap_ci`.

    Returns
    -------
    (lo, hi) : tuple of float
        The BCa ``[alpha/2, 1 - alpha/2]`` interval.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.shape[0] == 0:
        raise ValueError("bootstrap_bca_ci requires at least one sample")
    if x.shape[0] == 1:
        v = float(statistic(x))
        return v, v

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    theta_hat = float(statistic(x))

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot[b] = float(statistic(x[idx[b]]))

    # Bias correction z0.
    prop = float(np.mean(boot < theta_hat))
    # Degenerate distributions (all replicates on one side) break the normal
    # quantile; clamp into (0, 1) and fall back to percentile if fully one-sided.
    if prop <= 0.0 or prop >= 1.0:
        lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
        hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
        return lo, hi
    z0 = _norm_ppf(prop)

    # Acceleration a from the jackknife.
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = float(statistic(np.delete(x, i)))
    jm = jack.mean()
    num = float(np.sum((jm - jack) ** 3))
    den = float(np.sum((jm - jack) ** 2)) ** 1.5
    a = num / (6.0 * den) if den > 1e-24 else 0.0

    z_lo = _norm_ppf(alpha / 2.0)
    z_hi = _norm_ppf(1.0 - alpha / 2.0)

    def _adj(z: float) -> float:
        adj = z0 + (z0 + z) / max(1.0 - a * (z0 + z), 1e-12)
        return float(np.clip(_norm_cdf(adj), 0.0, 1.0))

    lo = float(np.percentile(boot, 100.0 * _adj(z_lo)))
    hi = float(np.percentile(boot, 100.0 * _adj(z_hi)))
    return (lo, hi) if lo <= hi else (hi, lo)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via erf (no scipy needed)."""
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (Acklam's rational approximation, ~1e-9 abs err)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    # Coefficients for the central and tail regions.
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return float((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                     / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return float((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                     / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return float(-(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))


def paired_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = 10000,
    alternative: str = "greater",
    seed: int = 0,
) -> Dict[str, float]:
    """Paired sign-flip permutation test on per-episode differences.

    The recommended significance test for paired benchmark comparisons (Dror et
    al., ACL 2018): under H0 (no difference) each per-episode difference
    ``d_i = x_i - y_i`` is symmetric about 0, so randomly flipping the sign of
    each ``d_i`` yields the exact null distribution of ``mean(d)``. Used for the
    Stage-2 reactivity/separation gates (e.g. ``x`` = per-episode severe-edit
    divergence, ``y`` = per-episode seed floor).

    Parameters
    ----------
    x, y : sequences of float
        Paired observations (same length, same episode order).
    n_perm : int, default 10000
        Number of sign-flip resamples.
    alternative : {"greater", "less", "two-sided"}
        Direction of H1 for ``mean(x - y)``. Default "greater" (one-tailed),
        which is the pre-registered direction for every AFP gate: a real edit
        should *increase* divergence, never decrease it. State this in reports.
    seed : int
        RNG seed (reproducible p-values).

    Returns
    -------
    dict
        ``{"p_value", "observed", "effect_size"}`` where ``observed`` is
        ``mean(x - y)`` and ``effect_size`` is Cohen's d for paired samples
        (``mean(d) / std(d, ddof=1)``; ``inf`` if the differences are constant
        and nonzero, ``0`` if identically zero).
    """
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if xv.shape[0] != yv.shape[0]:
        raise ValueError(f"paired inputs must match length; got {xv.shape[0]} vs {yv.shape[0]}")
    if xv.shape[0] < 2:
        raise ValueError("paired_permutation_test needs >= 2 pairs")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"unknown alternative '{alternative}'")

    d = xv - yv
    obs = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd > 1e-24:
        effect = obs / sd
    else:
        effect = 0.0 if abs(obs) < 1e-24 else float("inf") * np.sign(obs)

    rng = np.random.default_rng(seed)
    n = d.shape[0]
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    null = (signs * d[None, :]).mean(axis=1)
    # Add-one smoothing keeps p > 0 (the observed labeling is itself a permutation).
    if alternative == "greater":
        p = (1.0 + float(np.sum(null >= obs))) / (n_perm + 1.0)
    elif alternative == "less":
        p = (1.0 + float(np.sum(null <= obs))) / (n_perm + 1.0)
    else:
        p = (1.0 + float(np.sum(np.abs(null) >= abs(obs)))) / (n_perm + 1.0)
    return {"p_value": float(p), "observed": obs, "effect_size": float(effect)}


def holm_bonferroni(
    pvalues: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, list]:
    """Holm-Bonferroni step-down correction for a family of hypothesis tests.

    AFP runs several gates (reactivity, null-floor, separation, and one family
    per model on a leaderboard); reporting them uncorrected inflates the
    family-wise error rate. Holm's method is uniformly more powerful than plain
    Bonferroni and needs no independence assumption (Dror et al., ACL 2018).

    Parameters
    ----------
    pvalues : sequence of float
        Raw p-values, in any order.
    alpha : float, default 0.05
        Family-wise error rate.

    Returns
    -------
    dict
        ``{"adjusted": [...], "reject": [...]}`` in the ORIGINAL input order.
        ``adjusted[i]`` is the Holm-adjusted p-value (monotone, clipped to 1);
        ``reject[i]`` is ``bool(adjusted[i] <= alpha)``.
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return {"adjusted": [], "reject": []}
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, i in enumerate(order):
        val = min((m - rank) * p[i], 1.0)
        running_max = max(running_max, val)
        adj[i] = running_max
    return {
        "adjusted": adj.tolist(),
        "reject": [bool(v <= alpha) for v in adj],
    }


# ---------------------------------------------------------------------------
# Stage-2 separation gate.
# ---------------------------------------------------------------------------
def separation_ratio(
    S_severe: np.ndarray,
    floor: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Separation ratio for the Stage-2 "3x floor" gate (DESIGN Sec 2.3 / gates).

    How many multiples of the determinism floor a *severe* perturbation clears::

        separation = mean(S_severe) / mean( max(floor, eps) )

    The Stage-2 separation gate requires this to exceed ``3`` (a severe action
    edit must produce divergence at least 3x the seed-noise floor); failing it on
    every model is a signal to switch distance backends rather than blame a model.

    Parameters
    ----------
    S_severe : array-like
        The severe-perturbation sensitivity (e.g. ``S_negate(t)``); reduced by
        its mean over ``t``.
    floor : array-like
        The determinism floor ``epsilon(t)`` from :func:`seed_floor`; clamped to
        ``>= eps`` before averaging to avoid divide-by-zero on a perfectly
        deterministic model.
    eps : float, default 1e-8
        Lower clamp on the floor.

    Returns
    -------
    float
        The (unitless) separation ratio.
    """
    s = np.asarray(S_severe, dtype=np.float64).reshape(-1)
    f = np.asarray(floor, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("S_severe must be non-empty")
    if f.size == 0:
        raise ValueError("floor must be non-empty")
    denom = np.mean(np.maximum(f, eps))
    return float(np.mean(s) / denom)


# ---------------------------------------------------------------------------
# Optional convenience: per-episode pairs -> aggregated S(t).
# ---------------------------------------------------------------------------
def paired_curve_over_episodes(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Convenience: ``S(t)`` for many ``(base, pert)`` pairs -> aggregated curve.

    Computes :func:`sensitivity_curve` for each ``(base_frames, pert_frames)``
    pair and feeds the lot to :func:`aggregate_curves`. Because episodes may have
    differing native horizons, all per-episode curves are truncated to the common
    ``min(T)`` before aggregation (matching the truncate-to-``min(T)`` policy used
    everywhere else in this module).

    Parameters
    ----------
    pairs : sequence of (base_frames, pert_frames)
        One ``(base, pert)`` rollout pair per episode.
    distance : callable
        Per-frame distance backend.

    Returns
    -------
    dict
        ``{'mean', 'std', 'sem', 'n'}`` from :func:`aggregate_curves`.

    Raises
    ------
    ValueError
        If ``pairs`` is empty.
    """
    if len(pairs) == 0:
        raise ValueError("paired_curve_over_episodes requires at least one pair")
    curves = [sensitivity_curve(base, pert, distance) for base, pert in pairs]
    t = min(c.shape[0] for c in curves)
    curves = [c[:t] for c in curves]
    return aggregate_curves(curves)
