"""Experiment 1: oracle-recovery (metric identifiability) on synthetic world models.

Question
--------
Does the AFP dose-response readout recover the *known* causal structure of a
world model? We instantiate a grid of mock world models whose ground-truth
action gain ``g_a`` and instruction gain ``g_i`` are set by construction
(`afp.adapters.mock.MockWorldModel`), score every model with the exact
pre-registered AFP protocol (nothing tuned per model), and test:

H1a (action-axis recovery)      : recovered ``gain_action`` increases
                                  monotonically with true ``g_a`` at every
                                  fixed ``g_i``  (Spearman rho ~ 1).
H1b (instruction-axis recovery) : recovered ``gain_instruction`` increases
                                  monotonically with true ``g_i`` at every
                                  fixed ``g_a``.
H2  (no cross-channel leakage)  : recovered ``gain_action`` is ~flat in true
                                  ``g_i`` at fixed ``g_a`` (|rho| small), and
                                  vice versa.
H3  (inert detection)           : the ``g_a = g_i = 0`` model is classified
                                  "inert".

This is the sanity-check methodology of Adebayo et al. (NeurIPS 2018) applied
to a benchmark metric: oracles with known sensitivity *validate* the probe;
they are never used to tune it (DESIGN Rev 4, Sec 2.2 de-circularization).

Outputs
-------
results/synthetic/exp1_oracle_recovery.json   -- all recovered scorecard scalars
                                                 + the H1/H2/H3 test numbers.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from afp.adapters.mock import MockWorldModel
from afp.data import make_synthetic_dataset
from afp.metrics import monotonicity_spearman
from afp.score import score_model

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic")

# Pre-registered protocol (identical for every model; never tuned per model).
N_EPISODES = 16
HORIZON = 12
SIZE = 64
SEEDS = (0, 1, 2, 3, 4)
DATA_SEED = 0
NOISE = 0.04

ACTION_GAINS = (0.0, 0.12, 0.3, 0.6, 1.0)
INSTRUCTION_GAINS = (0.0, 0.15, 0.4, 1.0)


def run() -> dict:
    episodes, specs = make_synthetic_dataset(
        n_episodes=N_EPISODES, horizon=HORIZON, size=SIZE, seed=DATA_SEED
    )

    rows = []
    t0 = time.time()
    for g_a, g_i in itertools.product(ACTION_GAINS, INSTRUCTION_GAINS):
        model = MockWorldModel(
            f"oracle_a{g_a}_i{g_i}", action_gain=g_a, instruction_gain=g_i,
            noise=NOISE, horizon=HORIZON, size=SIZE,
        )
        card = score_model(model, episodes, specs, seeds=SEEDS)
        rows.append({
            "true_action_gain": g_a,
            "true_instruction_gain": g_i,
            "recovered_gain_action": card.gain_action,
            "recovered_gain_instruction": card.gain_instruction,
            "sens_action_share": card.sens_action_final,
            "sens_instruction_share": card.sens_instruction_final,
            "monotonicity_rho": card.monotonicity_rho,
            "separation_ratio": card.separation_ratio,
            "verdict": card.verdict,
            "gates": card.gates,
            "stats": card.stats,
        })
        print(f"  scored g_a={g_a:<5} g_i={g_i:<5} -> "
              f"gain_a={card.gain_action:.4f} gain_i={card.gain_instruction:.4f} "
              f"verdict={card.verdict}")
    elapsed = time.time() - t0

    # ---- H1a / H2 (leakage into the action axis) --------------------------
    h1a, h2_into_action = [], []
    for g_i in INSTRUCTION_GAINS:
        sub = [r for r in rows if r["true_instruction_gain"] == g_i]
        rho = monotonicity_spearman(
            [r["true_action_gain"] for r in sub],
            [r["recovered_gain_action"] for r in sub])
        h1a.append({"fixed_instruction_gain": g_i, "spearman_rho": rho})
    for g_a in ACTION_GAINS:
        sub = [r for r in rows if r["true_action_gain"] == g_a]
        rho = monotonicity_spearman(
            [r["true_instruction_gain"] for r in sub],
            [r["recovered_gain_action"] for r in sub])
        h2_into_action.append({"fixed_action_gain": g_a, "spearman_rho": rho})

    # ---- H1b / H2 (leakage into the instruction axis) ---------------------
    h1b, h2_into_instr = [], []
    for g_a in ACTION_GAINS:
        sub = [r for r in rows if r["true_action_gain"] == g_a]
        rho = monotonicity_spearman(
            [r["true_instruction_gain"] for r in sub],
            [r["recovered_gain_instruction"] for r in sub])
        h1b.append({"fixed_action_gain": g_a, "spearman_rho": rho})
    for g_i in INSTRUCTION_GAINS:
        sub = [r for r in rows if r["true_instruction_gain"] == g_i]
        rho = monotonicity_spearman(
            [r["true_action_gain"] for r in sub],
            [r["recovered_gain_instruction"] for r in sub])
        h2_into_instr.append({"fixed_instruction_gain": g_i, "spearman_rho": rho})

    # ---- H3: the doubly-inert oracle -------------------------------------
    inert_row = next(r for r in rows
                     if r["true_action_gain"] == 0.0 and r["true_instruction_gain"] == 0.0)

    # ---- global rank recovery (pooled, both axes) -------------------------
    global_rho_action = monotonicity_spearman(
        [r["true_action_gain"] for r in rows],
        [r["recovered_gain_action"] for r in rows])
    global_rho_instr = monotonicity_spearman(
        [r["true_instruction_gain"] for r in rows],
        [r["recovered_gain_instruction"] for r in rows])

    result = {
        "protocol": {
            "n_episodes": N_EPISODES, "horizon": HORIZON, "size": SIZE,
            "seeds": list(SEEDS), "noise": NOISE, "data_seed": DATA_SEED,
            "action_gains": list(ACTION_GAINS),
            "instruction_gains": list(INSTRUCTION_GAINS),
            "distance": "pixel_l2",
        },
        "rows": rows,
        "h1a_action_axis_recovery": h1a,
        "h1b_instruction_axis_recovery": h1b,
        "h2_leakage_instr_into_action": h2_into_action,
        "h2_leakage_action_into_instr": h2_into_instr,
        "h3_inert_verdict": inert_row["verdict"],
        "global_spearman_action": global_rho_action,
        "global_spearman_instruction": global_rho_instr,
        "elapsed_seconds": elapsed,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "exp1_oracle_recovery.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nwrote {out_path}  ({elapsed:.1f}s)")

    print("\n== summary ==")
    print(f"H1a per-row rho (recovered gain_a vs true g_a): "
          f"{[round(x['spearman_rho'], 3) for x in h1a]}")
    print(f"H1b per-col rho (recovered gain_i vs true g_i): "
          f"{[round(x['spearman_rho'], 3) for x in h1b]}")
    print(f"H2 leakage g_i -> gain_a: "
          f"{[round(x['spearman_rho'], 3) if np.isfinite(x['spearman_rho']) else None for x in h2_into_action]}")
    print(f"H2 leakage g_a -> gain_i: "
          f"{[round(x['spearman_rho'], 3) if np.isfinite(x['spearman_rho']) else None for x in h2_into_instr]}")
    print(f"H3 inert verdict: {inert_row['verdict']}")
    print(f"global rho: action={global_rho_action:.3f} instruction={global_rho_instr:.3f}")
    return result


if __name__ == "__main__":
    run()
