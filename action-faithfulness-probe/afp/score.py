"""Orchestration: assemble the AFP pipeline into a per-model :class:`Scorecard`.

This module is the glue between the frozen pieces:

    rollout  (afp.rollout)      -- run a world model on (o0, l, a)
    perturb  (afp.perturb)      -- edit the action / instruction
    distance (afp.distance)     -- per-frame perceptual distance
    metrics  (afp.metrics)      -- S(t), seed floor, aggregation, normalization, stats

For each model we produce a :class:`Scorecard` that places it on the 2D
**sensitivity plane** ``(sens_action, sens_instruction)`` of DESIGN Section 2.2,
records the Stage-2 GO/KILL gates (Section 8), and is JSON-serializable.

--------------------------------------------------------------------------
Calibration: dose-response channel gains  (READ THIS -- DESIGN Rev 4, Sec 2.2)
--------------------------------------------------------------------------
A design review found that comparing a *single* action operating point against a
*single* instruction operating point silently reintroduces the
"violent-vs-gentle" artifact across channels: there is no shared magnitude unit
between an action-space edit and a language/embedding edit, so any fixed pair of
operating points encodes an arbitrary severity choice. (The earlier shared-ceiling
scheme also tied the instruction score to an *action*-scaled ceiling, which
distorts an instruction-driven, low-action-gain model.)

The literature-grounded fix (causal-mediation standardized effects; the
controllability-fidelity CFG sweep; perturbation dose-response methods) is to
**avoid a shared magnitude and compare the SHAPE of a per-channel dose-response
curve**:

  * Sweep each channel's perturbation *dose* on a ladder
    (action: the ``gauss(sigma)`` ladder; instruction: ``interpolate_instruction``
    from the true meaning toward the counterfactual at fractions in (0, 1]).
  * Record the divergence ``S`` at each dose, above that channel's no-change floor
    (action floor = seed/determinism floor; instruction floor = ``paraphrase``).
  * Fit the **local gain** = slope of (S above floor) vs. *normalized* dose
    (dose axis rescaled to [0, 1]). Action and instruction gains are then both in
    the SAME units -- delta-distance per unit normalized dose -- and directly
    comparable. This is the channel's identifiable response in the linear regime,
    so it does not depend on whether a maximal edit saturates ``pixel_l2``.

The headline plane is ``(gain_action, gain_instruction)``. The reported
``sens_action`` / ``sens_instruction`` curves are the per-horizon *dominance
shares* ``gain_c(t) / (gain_action(t) + gain_instruction(t))`` in [0, 1] (a
unit-free readout of which channel drives the future at each step).

De-circularity (DESIGN Rev 4): the dose ladders and the slope estimator are a
**fixed protocol** -- nothing here is tuned to recover the mock's known answer.
The raw ``S(t)`` curves and the gains are calibration-free; the synthetic mocks
only *validate* (not calibrate) the metric, which is accepted methodology
(Adebayo et al. 2018, "Sanity Checks"; M4/SynScore 2023).

VERDICT
    Classified from the two gains by *channel dominance* with a small margin; a
    model whose *both* gains sit at/near the noise floor is "inert":

        max(gain_action, gain_instruction) < INERT_GAIN     -> "inert"
        share_action  - share_instruction  > MARGIN          -> "action-faithful"
        share_instruction - share_action   > MARGIN          -> "instruction-driven"
        otherwise                                            -> "balanced"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from afp import metrics, perturb
from afp.data import InstructionPerturbationSpec
from afp.distance import get_distance
from afp.rollout import Episode, WorldModelAdapter

# Verdict decision thresholds (DESIGN Rev 4 dose-response plane).
# A model is "inert" if even its strongest channel barely responds: peak gain
# below this many delta-distance units per unit normalized dose.
INERT_GAIN = 1e-3
VERDICT_MARGIN = 0.10

# Stage-2 GO/KILL gate thresholds (DESIGN Section 8).
MONOTONICITY_MIN = 0.7
SEPARATION_MIN = 3.0


# ===========================================================================
# Scorecard
# ===========================================================================
@dataclass
class Scorecard:
    """Per-model AFP result: the headline 2D plane point, curves, stats, gates.

    All array-valued fields are stored as plain Python lists (or numpy arrays
    that :meth:`to_dict` converts to lists) so the whole object is JSON round-trips.
    """

    model_id: str
    horizon: int
    fps: int
    distance_name: str

    # Aggregated action-perturbation S(t) curves: name -> {"mean","std","sem","n"}.
    action_curves: Dict[str, Dict[str, list]] = field(default_factory=dict)
    # Transition-space S^Delta(t) curves: name -> {"mean","std","sem","n"} (len T-1).
    transition_curves: Dict[str, Dict[str, list]] = field(default_factory=dict)
    # Instruction-channel curves: floor/operating/ceiling -> {"mean","std","sem","n"}.
    instruction_curves: Dict[str, Dict[str, list]] = field(default_factory=dict)
    # Seed/determinism floor: {"mean": (T,), "std": (T,)}.
    seed_floor: Dict[str, list] = field(default_factory=dict)

    # Per-channel normalized sensitivity curves (len T), in [0, 1].
    sens_action: list = field(default_factory=list)
    sens_instruction: list = field(default_factory=list)

    # Scalar summaries.
    sens_action_final: float = 0.0
    sens_action_mean: float = 0.0
    sens_instruction_final: float = 0.0
    sens_instruction_mean: float = 0.0

    # Dose-response channel gains (DESIGN Rev 4, Sec 2.2): local slope of S vs
    # normalized perturbation dose, in common units (delta-distance per unit dose),
    # so the action and instruction channels are directly comparable by SHAPE.
    gain_action: float = 0.0
    gain_instruction: float = 0.0
    dose_response: Dict[str, Dict[str, float]] = field(default_factory=dict)

    monotonicity_rho: float = 0.0
    separation_ratio: float = 0.0

    # Bootstrap CIs for the headline scalars: name -> [lo, hi].
    bootstrap_cis: Dict[str, list] = field(default_factory=dict)

    # Paired permutation tests (one-tailed, sign-flip; Dror et al. 2018) with
    # Holm-Bonferroni-adjusted p-values and paired Cohen's d effect sizes:
    # test name -> {"p_value", "p_holm", "effect_size", "observed"}.
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    verdict: str = "inert"
    gates: Dict[str, bool] = field(default_factory=dict)

    # Provenance / calibration record.
    calibration: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain ``dict`` with every numpy array coerced to lists."""

        def clean(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            return obj

        return clean(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Scorecard":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scorecard":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ===========================================================================
# Internal helpers (defensive wrappers around the concurrently-built modules)
# ===========================================================================
def _apply_action_op(fn, actions: np.ndarray, rng, **params) -> np.ndarray:
    """Call a perturb operator, tolerating ``rng=``-vs-positional signatures."""
    try:
        return np.asarray(fn(actions, rng=rng, **params), dtype=np.float64)
    except TypeError:
        try:
            return np.asarray(fn(actions, rng, **params), dtype=np.float64)
        except TypeError:
            return np.asarray(fn(actions, **params), dtype=np.float64)


def _gauss_ladder_entries(sigma_ladder: Sequence[float]) -> List[Tuple[str, object]]:
    """Normalize ``perturb.gauss_ladder`` output to a list of (name, callable).

    ``gauss_ladder`` is documented as ``gauss_ladder(sigmas)`` but its exact
    return container is built concurrently; accept dict / list-of-(name,fn) /
    list-of-callables, and fall back to wrapping ``ACTION_OPERATORS['gauss']``.
    """
    entries: List[Tuple[str, object]] = []
    gl = None
    try:
        gl = perturb.gauss_ladder(tuple(sigma_ladder))
    except Exception:
        gl = None

    if isinstance(gl, dict):
        entries = list(gl.items())
    elif isinstance(gl, (list, tuple)) and gl:
        if isinstance(gl[0], tuple) and len(gl[0]) == 2:
            entries = [(str(n), fn) for n, fn in gl]
        else:  # list of callables -> name by sigma
            entries = [(f"gauss_{s}", fn) for s, fn in zip(sigma_ladder, gl)]

    if not entries:  # fallback: build from the base gauss operator
        gop = perturb.ACTION_OPERATORS["gauss"]
        entries = [
            (f"gauss_{s}", (lambda a, rng=None, _s=s: _apply_action_op(gop, a, rng, sigma=_s)))
            for s in sigma_ladder
        ]
    return entries


def _build_action_ops(sigma_ladder: Sequence[float]) -> "List[Tuple[str, object]]":
    """Default action-perturbation suite (DESIGN Section 3), ordered by severity.

    Each entry is ``(name, fn)`` where ``fn(actions, rng, donor) -> actions``.
    ``donor`` (the next episode's action chunk) is only used by ``swap``.
    """
    ops = perturb.ACTION_OPERATORS

    def wrap(opfn):
        return lambda a, rng=None, donor=None, _f=opfn: _apply_action_op(_f, a, rng)

    entries: List[Tuple[str, object]] = []
    entries.append(("identity", wrap(ops["identity"])))
    # null_micro: embodiment-agnostic sub-quantization null (the gate floor).
    # null_shuffle: mock-only illustrative null (kept for the demo, see perturb.py).
    if "null_micro" in ops:
        entries.append(("null_micro", wrap(ops["null_micro"])))
    entries.append(("null_shuffle", wrap(ops["null_shuffle"])))
    for name, fn in _gauss_ladder_entries(sigma_ladder):
        entries.append((name, (lambda a, rng=None, donor=None, _f=fn: _apply_action_op(_f, a, rng))))
    entries.append(("scale", wrap(ops["scale"])))
    entries.append(("reverse", wrap(ops["reverse"])))
    entries.append(("negate", wrap(ops["negate"])))
    entries.append(("swap", _make_swap()))
    entries.append(("random_action", wrap(ops["random_action"])))
    return entries


def _make_swap():
    """Wrap ``perturb.swap``; ``swap(actions, rng=None, donor=None)`` needs donor as kw."""
    def _swap(a, rng=None, donor=None):
        try:
            return np.asarray(perturb.swap(a, rng=rng, donor=donor), dtype=np.float64)
        except TypeError:
            return np.asarray(perturb.swap(a, donor=donor), dtype=np.float64)
    return _swap


def _seed_floor(stacks: List[np.ndarray], distance) -> Tuple[np.ndarray, np.ndarray]:
    """Call ``metrics.seed_floor`` defensively; return (mean(T,), std(T,))."""
    mean, std = metrics.seed_floor(stacks, distance)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def _transition(base: np.ndarray, pert: np.ndarray, distance) -> np.ndarray:
    try:
        return np.asarray(metrics.transition_curve(base, pert, distance), dtype=np.float64)
    except TypeError:
        return np.asarray(metrics.transition_curve(base, pert, distance=distance), dtype=np.float64)


def _bootstrap(samples: np.ndarray) -> Tuple[float, float]:
    """BCa bootstrap CI (DESIGN Rev 4: gate on the BCa lower bound), with
    percentile-bootstrap and raw-percentile fallbacks."""
    s = np.asarray(samples, dtype=np.float64)
    try:
        lo, hi = metrics.bootstrap_bca_ci(s)
        return float(lo), float(hi)
    except Exception:
        pass
    try:
        lo, hi = metrics.bootstrap_ci(s)
        return float(lo), float(hi)
    except Exception:
        if s.size == 0:
            return 0.0, 0.0
        return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _curve_dict(curves: List[np.ndarray]) -> Dict[str, list]:
    """Aggregate a list of per-episode curves to {"mean","std","sem","n"} lists."""
    agg = metrics.aggregate_curves([np.asarray(c, dtype=np.float64) for c in curves])
    n_arr = np.asarray(agg["n"], dtype=np.float64).reshape(-1)  # metrics returns (T,) array
    return {
        "mean": np.asarray(agg["mean"], dtype=np.float64).tolist(),
        "std": np.asarray(agg["std"], dtype=np.float64).tolist(),
        "sem": np.asarray(agg["sem"], dtype=np.float64).tolist(),
        "n": int(n_arr[0]) if n_arr.size else len(curves),
    }


def _mean_curve(curve_dict: Dict[str, list]) -> np.ndarray:
    return np.asarray(curve_dict["mean"], dtype=np.float64)


def _channel_gain(
    dose_curves: List[np.ndarray],
    doses: Sequence[float],
    floor_curve: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Per-horizon dose-response gain for one channel (DESIGN Rev 4, Sec 2.2).

    ``dose_curves[i]`` is the aggregated ``S(t)`` at dose ``doses[i]`` (e.g. a
    gauss sigma or an instruction interpolation fraction). Returns

    * ``gain_t`` -- a ``(T,)`` array: at each horizon ``t``, the least-squares
      slope of ``(S(t) - floor(t))`` vs. the dose axis *rescaled to [0, 1]*. Units
      are delta-distance per unit normalized dose, identical across channels.
    * ``summary`` -- :func:`afp.metrics.dose_response` of the horizon-averaged
      response (auc / slope / max / norm_auc).
    """
    d = np.asarray(doses, dtype=np.float64).reshape(-1)
    R = np.stack([np.asarray(c, dtype=np.float64) for c in dose_curves], axis=0)
    R = np.clip(R - np.asarray(floor_curve, dtype=np.float64)[None, :], 0.0, None)  # (D, T)
    span = float(d.max() - d.min())
    dn = (d - d.min()) / span if span > 1e-12 else np.linspace(0.0, 1.0, len(d))
    dn_c = dn - dn.mean()
    denom = float(np.sum(dn_c ** 2)) + 1e-12
    gain_t = np.clip((dn_c[:, None] * (R - R.mean(axis=0, keepdims=True))).sum(axis=0) / denom, 0.0, None)
    resp = R.mean(axis=1)  # horizon-averaged response per dose (floor already removed)
    try:
        summary = metrics.dose_response(d, resp, floor=0.0)
    except Exception:
        summary = {"auc": 0.0, "slope": float(np.mean(gain_t)), "max": float(resp.max() if resp.size else 0.0),
                   "norm_auc": 0.0}
    return gain_t, summary


def _instruction_emb_pair(
    episode: Episode,
    spec: Optional[InstructionPerturbationSpec],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (paraphrase_emb, counterfactual_emb) for the instruction channel.

    Prefer the dataset's matched specs (DESIGN: the human-checked paraphrase floor
    and counterfactual operating point); fall back to ``perturb`` operators when a
    spec row is missing. The ceiling is *not* an embedding here -- it is the shared
    action-channel ceiling (see module docstring).
    """
    base = np.asarray(episode.instruction_emb, dtype=np.float64)

    if spec is not None and getattr(spec, "paraphrase_emb", None):
        para = np.asarray(spec.paraphrase_emb, dtype=np.float64)
    else:
        para = _apply_action_op(perturb.paraphrase, base, rng, eps=0.05)

    if spec is not None and getattr(spec, "counterfactual_emb", None):
        cf = np.asarray(spec.counterfactual_emb, dtype=np.float64)
    else:
        cf = _apply_action_op(perturb.cf_instruction, base, rng)

    return para, cf


# ===========================================================================
# Public API
# ===========================================================================
def score_model(
    adapter: WorldModelAdapter,
    episodes: List[Episode],
    specs: Optional[List[InstructionPerturbationSpec]] = None,
    distance_name: str = "pixel_l2",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    action_perturbations=None,
    sigma_ladder: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    instruction_fracs: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> Scorecard:
    """Run the full AFP pipeline for one adapter and return a :class:`Scorecard`.

    Parameters
    ----------
    adapter
        A :class:`~afp.rollout.WorldModelAdapter`.
    episodes
        Probe instances (start frame + instruction + ground-truth action chunk).
    specs
        Matched instruction-perturbation specs (paraphrase + counterfactual embeddings),
        keyed by ``episode_id``. If ``None`` / missing rows, falls back to ``perturb``.
    distance_name
        A backend name understood by :func:`afp.distance.get_distance`.
    seeds
        Sampling seeds. ``seeds[0]`` is the *paired* seed used for base and all
        perturbed rollouts (common-random-number sampling, DESIGN Section 3); the
        full set drives the seed/determinism floor.
    action_perturbations
        Optional explicit ``[(name, fn), ...]`` action-op suite; defaults to the
        DESIGN Section-3 suite (identity, null_shuffle, gauss ladder, scale,
        reverse, negate, swap, random_action).
    sigma_ladder
        Sigmas for the gauss action dose ladder (also the monotonicity ladder).
    instruction_fracs
        Interpolation fractions (true meaning -> counterfactual) for the
        instruction dose ladder. ``1.0`` is the full counterfactual.
    """
    distance = get_distance(distance_name)
    seeds = tuple(int(s) for s in seeds)
    base_seed = seeds[0]
    spec_by_id = {s.episode_id: s for s in (specs or [])}
    instr_fracs = tuple(float(f) for f in instruction_fracs)

    if action_perturbations is None:
        action_ops = _build_action_ops(sigma_ladder)
    else:
        action_ops = [(n, (lambda a, rng=None, donor=None, _f=f: _apply_action_op(_f, a, rng)))
                      for n, f in action_perturbations]

    gauss_names = [n for n, _ in action_ops if str(n).startswith("gauss")]
    op_names = [n for n, _ in action_ops]

    # Per-op, per-episode curves.
    action_S: Dict[str, List[np.ndarray]] = {n: [] for n in op_names}
    action_Sdelta: Dict[str, List[np.ndarray]] = {n: [] for n in op_names}
    floor_means: List[np.ndarray] = []
    floor_stds: List[np.ndarray] = []
    # Instruction channel: paraphrase floor + a dose ladder toward the
    # counterfactual. "counterfactual" aliases the full-dose (frac=1.0) rung.
    instr_dose_names = [f"instr@{f}" for f in instr_fracs]
    instr_S: Dict[str, List[np.ndarray]] = {"paraphrase": [], "counterfactual": []}
    for nm in instr_dose_names:
        instr_S[nm] = []

    n_ep = len(episodes)
    for i, ep in enumerate(episodes):
        rng = np.random.default_rng(1000 + i)
        donor = episodes[(i + 1) % n_ep].actions  # next episode's actions, for swap

        # Base rollout at the paired seed.
        base = adapter.rollout_episode(ep, seed=base_seed).frames

        # Seed/determinism floor: identity across all seeds.
        stacks = [adapter.rollout_episode(ep, seed=s).frames for s in seeds]
        fmean, fstd = _seed_floor(stacks, distance)
        floor_means.append(fmean)
        floor_stds.append(fstd)

        # Action channel: each op, paired seed.
        for name, fn in action_ops:
            pert_actions = fn(ep.actions, rng=rng, donor=donor)
            pert = adapter.rollout(
                o0=ep.o0, actions=pert_actions, instruction=ep.instruction,
                instruction_emb=ep.instruction_emb, seed=base_seed,
            ).frames
            action_S[name].append(np.asarray(metrics.sensitivity_curve(base, pert, distance), dtype=np.float64))
            action_Sdelta[name].append(_transition(base, pert, distance))

        # Instruction channel: paraphrase (floor) + a dose ladder interpolating
        # the true meaning toward the counterfactual (DESIGN Rev 4 dose-response).
        para_emb, cf_emb = _instruction_emb_pair(ep, spec_by_id.get(ep.episode_id), rng)

        def _instr_rollout(emb):
            return adapter.rollout(
                o0=ep.o0, actions=ep.actions, instruction=ep.instruction,
                instruction_emb=emb, seed=base_seed,
            ).frames

        instr_S["paraphrase"].append(
            np.asarray(metrics.sensitivity_curve(base, _instr_rollout(para_emb), distance), dtype=np.float64))
        for f, nm in zip(instr_fracs, instr_dose_names):
            emb_f = perturb.interpolate_instruction(ep.instruction_emb, cf_emb, frac=f)
            curve = np.asarray(metrics.sensitivity_curve(base, _instr_rollout(emb_f), distance), dtype=np.float64)
            instr_S[nm].append(curve)
            if abs(f - 1.0) < 1e-9:  # full dose == the counterfactual
                instr_S["counterfactual"].append(curve)
        if not instr_S["counterfactual"]:  # no frac==1.0 rung -> use the literal cf
            instr_S["counterfactual"].append(
                np.asarray(metrics.sensitivity_curve(base, _instr_rollout(cf_emb), distance), dtype=np.float64))

    # -- aggregate ---------------------------------------------------------
    action_curves = {n: _curve_dict(action_S[n]) for n in op_names if action_S[n]}
    transition_curves = {n: _curve_dict(action_Sdelta[n]) for n in op_names if action_Sdelta[n]}
    instruction_curves = {k: _curve_dict(v) for k, v in instr_S.items() if v}

    floor_mean = np.mean(np.stack(floor_means, axis=0), axis=0)
    floor_std = np.mean(np.stack(floor_stds, axis=0), axis=0)
    seed_floor_d = {"mean": floor_mean.tolist(), "std": floor_std.tolist()}

    # -- per-channel dose-response gains (DESIGN Rev 4, Sec 2.2) -----------
    # Action dose ladder = the gauss sigmas; instruction dose ladder = the
    # interpolation fractions. Each channel's gain is the slope of S above its
    # own no-change floor vs normalized dose -> common units -> comparable.
    T = floor_mean.shape[0]
    if gauss_names:
        a_doses = [float(str(n).replace("gauss@", "").replace("gauss_", "")) for n in gauss_names]
        a_curves = [_mean_curve(action_curves[n]) for n in gauss_names]
        action_gain_t, action_dr = _channel_gain(a_curves, a_doses, floor_mean)
    else:  # custom suite without a gauss ladder: fall back to a single severe op
        sev = "negate" if "negate" in action_curves else op_names[-1]
        action_gain_t = np.clip(_mean_curve(action_curves[sev]) - floor_mean, 0.0, None)
        action_dr = {"auc": 0.0, "slope": float(np.mean(action_gain_t)), "max": 0.0, "norm_auc": 0.0}

    S_para = _mean_curve(instruction_curves["paraphrase"])
    i_curves = [_mean_curve(instruction_curves[nm]) for nm in instr_dose_names if nm in instruction_curves]
    i_doses = [f for f, nm in zip(instr_fracs, instr_dose_names) if nm in instruction_curves]
    if len(i_curves) >= 2:
        instr_gain_t, instr_dr = _channel_gain(i_curves, i_doses, S_para)
    else:  # degenerate ladder: use counterfactual above paraphrase
        instr_gain_t = np.clip(_mean_curve(instruction_curves["counterfactual"]) - S_para, 0.0, None)
        instr_dr = {"auc": 0.0, "slope": float(np.mean(instr_gain_t)), "max": 0.0, "norm_auc": 0.0}

    gain_action = float(np.mean(action_gain_t))
    gain_instruction = float(np.mean(instr_gain_t))

    # Per-horizon dominance share in [0,1] (the reported sens curves).
    denom_t = action_gain_t + instr_gain_t + 1e-12
    sens_action = np.clip(action_gain_t / denom_t, 0.0, 1.0)
    sens_instruction = np.clip(instr_gain_t / denom_t, 0.0, 1.0)

    # Action ceiling retained for the separation / severe-edit gate.
    ceil_candidates = [action_curves[n] for n in ("negate", "random_action") if n in action_curves]
    if ceil_candidates:
        S_ceil_action = np.max(np.stack([_mean_curve(c) for c in ceil_candidates], axis=0), axis=0)
    else:
        S_ceil_action = _mean_curve(action_curves[op_names[-1]])

    # -- scalar summaries --------------------------------------------------
    sa_final = float(sens_action[-1])
    sa_mean = float(np.mean(sens_action))
    si_final = float(sens_instruction[-1])
    si_mean = float(np.mean(sens_instruction))

    # -- monotonicity (DESIGN 2.4): severity rank vs mean S ----------------
    ladder = []
    for name in op_names:
        if name in action_curves:
            rank = _severity(name)
            ladder.append((rank, float(np.mean(_mean_curve(action_curves[name])))))
    ladder.sort(key=lambda r: r[0])
    ranks = np.asarray([r for r, _ in ladder], dtype=np.float64)
    vals = np.asarray([v for _, v in ladder], dtype=np.float64)
    try:
        rho = float(metrics.monotonicity_spearman(ranks, vals))
    except Exception:
        rho = 0.0
    if not np.isfinite(rho):
        rho = 0.0

    # -- separation ratio: most-severe S(T) vs seed floor(T) ---------------
    severe_name = "random_action" if "random_action" in action_curves else (
        "negate" if "negate" in action_curves else op_names[-1])
    S_severe = _mean_curve(action_curves[severe_name])
    try:
        sep = float(metrics.separation_ratio(S_severe, floor_mean))
    except Exception:
        sep = float(S_severe[-1] / max(floor_mean[-1], 1e-9))
    if not np.isfinite(sep):
        sep = 0.0

    # -- bootstrap CIs on the headline final-horizon quantities ------------
    # Reactivity operating point = negate (a maximal *meaningful* action edit).
    react_name = "negate" if "negate" in action_S else severe_name
    floor_final = float(floor_mean[-1])
    severe_final_samples = np.asarray([c[-1] for c in action_S[severe_name]], dtype=np.float64)
    react_final_samples = np.asarray([c[-1] for c in action_S[react_name]], dtype=np.float64)
    cf_final_samples = np.asarray([c[-1] for c in instr_S["counterfactual"]], dtype=np.float64)
    severe_ci = _bootstrap(severe_final_samples)
    react_ci = _bootstrap(react_final_samples)
    bootstrap_cis = {
        f"S_{severe_name}_final": list(severe_ci),
        f"S_{react_name}_final": list(react_ci),
        "S_counterfactual_final": list(_bootstrap(cf_final_samples)),
        "seed_floor_final": [floor_final, floor_final],
    }

    # -- paired permutation tests + Holm correction (statistical reporting) --
    # One family of one-tailed sign-flip tests per model, paired per episode
    # against that episode's own seed floor (common-random-number pairing).
    # Direction pre-registered as "greater": a real edit must INCREASE
    # divergence. Holm-Bonferroni controls the family-wise error rate across
    # the three confirmatory tests (Dror et al., ACL 2018).
    floor_final_samples = np.asarray([fm[-1] for fm in floor_means], dtype=np.float64)
    stats_tests: Dict[str, Dict[str, float]] = {}
    if len(floor_final_samples) >= 2:
        test_defs = [
            # H1: a maximal *meaningful* action edit raises divergence above floor.
            (f"reactivity[{react_name}>floor]", react_final_samples, floor_final_samples),
            # H1: the most severe edit clears 3x the floor (the separation gate).
            (f"separation[{severe_name}>3x_floor]", severe_final_samples, 3.0 * floor_final_samples),
            # H1: a counterfactual instruction moves the prediction above floor
            # (instruction-channel reactivity; used for the plane's y-axis validity).
            ("cf_instruction>floor", cf_final_samples, floor_final_samples),
        ]
        raw_p: List[float] = []
        for name, xs, ys in test_defs:
            n_pair = min(len(xs), len(ys))
            res = metrics.paired_permutation_test(xs[:n_pair], ys[:n_pair], alternative="greater")
            stats_tests[name] = {
                "p_value": res["p_value"],
                "observed": res["observed"],
                "effect_size": res["effect_size"],
            }
            raw_p.append(res["p_value"])
        holm = metrics.holm_bonferroni(raw_p)
        for (name, _, _), p_adj in zip(test_defs, holm["adjusted"]):
            stats_tests[name]["p_holm"] = float(p_adj)

    # -- gates (DESIGN Section 8 Stage-2; Rev 4: CI-based + reactivity) -----
    # Null floor: an embodiment-agnostic sub-quantization edit (null_micro) must
    # sit within ~1 sigma of the determinism floor (falls back to null_shuffle).
    null_key = "null_micro" if "null_micro" in action_curves else (
        "null_shuffle" if "null_shuffle" in action_curves else None)
    null_mean = _mean_curve(action_curves[null_key]) if null_key else floor_mean
    null_floor_ok = bool(float(np.mean(null_mean)) < float(np.mean(floor_mean) + np.mean(floor_std)))
    monotonicity_ok = bool(rho >= MONOTONICITY_MIN)
    # Reactivity (Adebayo-style model-sensitivity sanity check): a maximal action
    # edit must move the output above the floor (CI lower bound clears the floor).
    reactivity_ok = bool(react_ci[0] > floor_final)
    # Separation: gate on the CI *lower bound*, not the point estimate.
    separation_ok = bool(severe_ci[0] >= SEPARATION_MIN * max(floor_final, 1e-9))
    gates = {
        "reactivity_ok": reactivity_ok,
        "null_floor_ok": null_floor_ok,
        "monotonicity_ok": monotonicity_ok,
        "separation_ok": separation_ok,
    }

    verdict = _decide_verdict(gain_action, gain_instruction)

    return Scorecard(
        model_id=getattr(adapter, "model_id", "model"),
        horizon=int(getattr(adapter, "horizon", len(floor_mean))),
        fps=int(getattr(adapter, "fps", 0)),
        distance_name=distance_name,
        action_curves=action_curves,
        transition_curves=transition_curves,
        instruction_curves=instruction_curves,
        seed_floor=seed_floor_d,
        sens_action=sens_action.tolist(),
        sens_instruction=sens_instruction.tolist(),
        sens_action_final=sa_final,
        sens_action_mean=sa_mean,
        sens_instruction_final=si_final,
        sens_instruction_mean=si_mean,
        gain_action=gain_action,
        gain_instruction=gain_instruction,
        dose_response={"action": action_dr, "instruction": instr_dr},
        monotonicity_rho=rho,
        separation_ratio=sep,
        bootstrap_cis=bootstrap_cis,
        stats=stats_tests,
        verdict=verdict,
        gates=gates,
        calibration={
            "method": "dose-response gain (DESIGN Rev 4, Sec 2.2)",
            "action_dose": "gauss(sigma) ladder",
            "action_floor": "seed_floor",
            "instruction_dose": "interpolate paraphrase->counterfactual",
            "instruction_floor": "paraphrase",
            "plane_axes": "(gain_action, gain_instruction) = slope of S vs normalized dose",
            "sens_curves": "per-horizon dominance share gain_c/(sum)",
            "null_gate_op": str(null_key),
            "reactivity_op": react_name,
            "severe_op": severe_name,
            "ci_method": "BCa bootstrap (Efron 1987), 95%, gate on lower bound",
            "test_method": "paired sign-flip permutation, one-tailed (greater), Holm-corrected",
            "note": "operating protocol fixed a priori; not tuned to recover the mock answer",
        },
        meta={"n_episodes": n_ep, "seeds": list(seeds)},
    )


def score_suite(
    adapters: Sequence[WorldModelAdapter],
    episodes: List[Episode],
    specs: Optional[List[InstructionPerturbationSpec]] = None,
    **kw,
) -> List[Scorecard]:
    """Score a list of adapters; returns one :class:`Scorecard` each."""
    return [score_model(a, episodes, specs, **kw) for a in adapters]


# ---------------------------------------------------------------------------
def _severity(name: str) -> float:
    """Severity score for an op name, using ``perturb``'s ranks (DESIGN 2.4).

    Prefers ``perturb.SEVERITY_RANK`` for base ops; for the gauss-ladder rungs
    (named ``gauss@{sigma}`` / ``gauss_{sigma}``) it adds a small sigma-ordered
    offset on top of the base ``gauss`` rank so the rungs stay below ``scale`` but
    preserve their within-ladder ordering (a strictly increasing noise ladder is
    the strongest part of the monotonicity signal). Falls back to
    ``perturb.severity_of`` and finally to a mid rank.
    """
    sr = getattr(perturb, "SEVERITY_RANK", {}) or {}
    if name in sr:
        return float(sr[name])
    sname = str(name)
    if sname.startswith("gauss@") or sname.startswith("gauss_"):
        try:
            sigma = float(sname.replace("gauss@", "").replace("gauss_", ""))
        except ValueError:
            sigma = 0.5
        base = float(sr.get("gauss", 1.0))
        return base + min(sigma, 0.9)  # keep within (gauss, scale)
    try:
        return float(perturb.severity_of(name))
    except Exception:
        return 5.0


def _decide_verdict(gain_action: float, gain_instruction: float) -> str:
    """Dominance decision on the dose-response plane (DESIGN Rev 4).

    Uses the raw channel gains (common units) for the inert check, then the
    dominance share for the action/instruction/balanced call.
    """
    ga, gi = float(gain_action), float(gain_instruction)
    if max(ga, gi) < INERT_GAIN:
        return "inert"
    total = ga + gi + 1e-12
    share_a, share_i = ga / total, gi / total
    if share_a - share_i > VERDICT_MARGIN:
        return "action-faithful"
    if share_i - share_a > VERDICT_MARGIN:
        return "instruction-driven"
    return "balanced"
