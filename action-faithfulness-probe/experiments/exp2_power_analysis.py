"""Experiment 2: statistical power of the AFP Stage-2 gates vs probe-set size.

Question
--------
How many probe episodes does AFP need before the pre-registered reactivity /
separation gates reliably detect a *weak* action channel above the stochastic
floor -- and what does that imply for the GPU budget on the real checkpoint?

Design
------
For each (true action gain ``g_a``, per-step noise ``noise``, ``n_episodes``)
cell we run ``R`` independent replicates (fresh probe sets, fresh model
identity) of the full pre-registered scoring protocol and record whether

  * the reactivity test rejects (Holm-adjusted one-tailed sign-flip p < .05), and
  * the CI-based reactivity gate fires (BCa lower bound > seed floor),

then report the rejection *rate* per cell (empirical power). ``g_a = 0`` rows
give the empirical false-positive rate (should sit near/below alpha).

The noise parameter maps onto the real-world seed/determinism floor of a
sampled diffusion world model; sweeping it tells us how power degrades as the
floor rises, which is the quantity we cannot know until the first GPU run.

Cost model
----------
Every scored episode costs ``(1 base + n_seeds floor + n_action_ops +
n_instruction_rollouts)`` model rollouts; the JSON output includes the exact
rollout count per protocol so the A100 budget in the handoff document is
arithmetic, not guesswork.

Outputs
-------
results/synthetic/exp2_power_analysis.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from afp.adapters.mock import MockWorldModel
from afp.data import make_synthetic_dataset
from afp.score import score_model

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic")

# Pre-registered protocol constants (match Experiment 1 / the shipped CLI).
HORIZON = 12
SIZE = 64
SEEDS = (0, 1, 2, 3, 4)
SIGMA_LADDER = (0.05, 0.1, 0.2, 0.4)
INSTR_FRACS = (0.25, 0.5, 0.75, 1.0)
ALPHA = 0.05

# Sweep.
ACTION_GAINS = (0.0, 0.03, 0.06, 0.12, 0.3)   # 0.0 row = false-positive rate
NOISES = (0.04, 0.3)                            # low / high stochastic floor
N_EPISODES_GRID = (4, 8, 16, 32)
REPLICATES = 10

# Rollouts per scored episode under this protocol:
#   1 base + len(SEEDS) floor + (3 fixed + 4 gauss + 4 severe + 1 swap) action
#   + (1 paraphrase + 4 fracs) instruction
N_ACTION_OPS = 3 + len(SIGMA_LADDER) + 5   # identity,null_micro,null_shuffle + ladder + scale,reverse,negate,swap,random
N_INSTR_ROLLOUTS = 1 + len(INSTR_FRACS)
ROLLOUTS_PER_EPISODE = 1 + len(SEEDS) + N_ACTION_OPS + N_INSTR_ROLLOUTS


def run() -> dict:
    cells = []
    t0 = time.time()
    for noise in NOISES:
        for g_a in ACTION_GAINS:
            for n_ep in N_EPISODES_GRID:
                reject_perm, reject_gate = 0, 0
                p_values = []
                for rep in range(REPLICATES):
                    episodes, specs = make_synthetic_dataset(
                        n_episodes=n_ep, horizon=HORIZON, size=SIZE,
                        seed=10_000 + rep,
                    )
                    model = MockWorldModel(
                        f"pw_a{g_a}_n{noise}_r{rep}", action_gain=g_a,
                        instruction_gain=0.3, noise=noise,
                        horizon=HORIZON, size=SIZE,
                    )
                    card = score_model(
                        model, episodes, specs, seeds=SEEDS,
                        sigma_ladder=SIGMA_LADDER, instruction_fracs=INSTR_FRACS,
                    )
                    react = [v for k, v in card.stats.items() if k.startswith("reactivity")]
                    if react:
                        p_values.append(react[0]["p_holm"])
                        if react[0]["p_holm"] < ALPHA:
                            reject_perm += 1
                    if card.gates.get("reactivity_ok", False):
                        reject_gate += 1
                cells.append({
                    "noise": noise,
                    "true_action_gain": g_a,
                    "n_episodes": n_ep,
                    "replicates": REPLICATES,
                    "power_permutation": reject_perm / REPLICATES,
                    "power_ci_gate": reject_gate / REPLICATES,
                    "median_p_holm": float(np.median(p_values)) if p_values else None,
                })
                print(f"  noise={noise:<5} g_a={g_a:<5} n={n_ep:<3} -> "
                      f"power(perm)={reject_perm/REPLICATES:.1f} "
                      f"power(CI)={reject_gate/REPLICATES:.1f}")
    elapsed = time.time() - t0

    # ---- cost mapping for the real checkpoint -----------------------------
    # A100 per-rollout latency for Cosmos-Predict2-2B-SAC (DESIGN Sec 4.4):
    # ~100-150 s (H100 measured 80 s; A100 unlisted, slower). Use 120 s central.
    per_rollout_s = 120.0
    a100_hourly = 1.19
    cost_rows = []
    for n_ep in N_EPISODES_GRID + (50,):
        n_roll = n_ep * ROLLOUTS_PER_EPISODE
        gpu_h = n_roll * per_rollout_s / 3600.0
        cost_rows.append({
            "n_episodes": n_ep,
            "rollouts_per_model": n_roll,
            "a100_hours_per_model": round(gpu_h, 1),
            "usd_per_model_at_1.19": round(gpu_h * a100_hourly, 2),
        })

    result = {
        "protocol": {
            "horizon": HORIZON, "size": SIZE, "seeds": list(SEEDS),
            "sigma_ladder": list(SIGMA_LADDER), "instruction_fracs": list(INSTR_FRACS),
            "alpha": ALPHA, "replicates": REPLICATES,
            "rollouts_per_episode": ROLLOUTS_PER_EPISODE,
            "instruction_gain_fixed": 0.3,
            "distance": "pixel_l2",
        },
        "cells": cells,
        "cost_model": {
            "per_rollout_seconds_a100": per_rollout_s,
            "a100_usd_per_hour": a100_hourly,
            "rows": cost_rows,
        },
        "elapsed_seconds": elapsed,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "exp2_power_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}  ({elapsed:.1f}s)")
    print("\ncost model (per real model):")
    for r in cost_rows:
        print(f"  n_ep={r['n_episodes']:<3} rollouts={r['rollouts_per_model']:<5} "
              f"A100-h={r['a100_hours_per_model']:<6} ${r['usd_per_model_at_1.19']}")
    return result


if __name__ == "__main__":
    run()
