"""Reporting: turn a list of :class:`~afp.score.Scorecard` into human artifacts.

Produces the DESIGN Section-6 deliverables:

    results/leaderboard.md   -- ranked markdown table
    results/plane.md         -- the 2D (sens_action, sens_instruction) plane + ASCII scatter
    results/scorecards.json  -- the raw scorecards
    results/curves/          -- S(t) PNGs (matplotlib) or CSVs (fallback)

Nothing here hard-fails on a missing optional dependency: if ``matplotlib`` is
absent, :func:`plot_curves` writes CSVs instead so the harness still runs on a
bare CPU box with only numpy installed.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence

import numpy as np

from afp.score import Scorecard

GATE_KEYS = ("reactivity_ok", "null_floor_ok", "monotonicity_ok", "separation_ok")


# ===========================================================================
# Markdown leaderboard
# ===========================================================================
def write_leaderboard(scorecards: Sequence[Scorecard], path: str) -> str:
    """Write a markdown leaderboard sorted by ``sens_action_final`` descending."""
    cards = sorted(scorecards, key=lambda c: c.sens_action_final, reverse=True)
    lines: List[str] = []
    lines.append("# AFP Leaderboard")
    lines.append("")
    lines.append(
        "Action-Faithfulness Probe results. Sorted by action sensitivity at the "
        "final horizon (higher = the predicted future tracks the *action* more)."
    )
    lines.append("")
    lines.append(
        "| model | verdict | gain_action | gain_instruction | sens_action (share) | "
        "sens_instruction (share) | monotonicity rho | sep. ratio | gates |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in cards:
        gates_pass = sum(bool(c.gates.get(k, False)) for k in GATE_KEYS)
        gate_str = f"{gates_pass}/{len(GATE_KEYS)}"
        lines.append(
            f"| `{c.model_id}` | {c.verdict} | {c.gain_action:.4f} | "
            f"{c.gain_instruction:.4f} | {c.sens_action_final:.3f} | "
            f"{c.sens_instruction_final:.3f} | {c.monotonicity_rho:.3f} | "
            f"{c.separation_ratio:.2f} | {gate_str} |"
        )
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append(
        "- **gain_action / gain_instruction** -- the headline dose-response gains "
        "(DESIGN Rev 4, Sec 2.2): the slope of S vs *normalized perturbation dose* for each "
        "channel (action dose = gauss sigma ladder; instruction dose = interpolation toward the "
        "counterfactual). Both are in the same units (delta-distance per unit normalized dose), "
        "so the (gain_action, gain_instruction) plane is the directly comparable readout."
    )
    lines.append(
        "- **sens_action / sens_instruction (share)** -- per-horizon dominance share "
        "`gain_c / (gain_action + gain_instruction)` in [0, 1]: which channel drives the "
        "predicted future. (These two sum to ~1; use the raw gains for the 2D plane.)"
    )
    lines.append(
        "- **verdict** -- dominance on the dose-response plane: *action-faithful* "
        "(action gain dominates), *instruction-driven* (instruction gain dominates), "
        "*balanced* (comparable), *inert* (both gains near the noise floor -- under-conditioned)."
    )
    lines.append("- **monotonicity rho** -- Spearman of S vs perturbation severity (gate >= 0.7).")
    lines.append("- **sep. ratio** -- severe-edit S(T) / seed-floor(T); the gate uses the "
                 "bootstrap-CI lower bound >= 3x (DESIGN Rev 4).")
    lines.append(
        "- **gates** -- Stage-2 GO/KILL checks passed (reactivity, null-floor, monotonicity, "
        "separation)."
    )
    lines.append("")
    text = "\n".join(lines)
    _write(path, text)
    return text


# ===========================================================================
# 2D sensitivity plane
# ===========================================================================
def write_plane(scorecards: Sequence[Scorecard], path: str) -> str:
    """Write the 2D sensitivity plane (per-model point + an ASCII quadrant scatter)."""
    lines: List[str] = []
    lines.append("# AFP Sensitivity Plane")
    lines.append("")
    lines.append(
        "Each model is a point `(gain_action, gain_instruction)` -- the per-channel "
        "dose-response gains (DESIGN Rev 4, Sec 2.2), in common units. Bottom-right = "
        "action-faithful (the goal); top-left = instruction-driven (the failure mode we hunt); "
        "top-right = reads both; near-origin = inert. The ASCII scatter normalizes each axis by "
        "the largest gain in the suite so the spread is visible."
    )
    lines.append("")
    lines.append("| model | gain_action | gain_instruction | sens_action (share) | "
                 "sens_instruction (share) | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for c in sorted(scorecards, key=lambda c: c.gain_action, reverse=True):
        lines.append(
            f"| `{c.model_id}` | {c.gain_action:.4f} | {c.gain_instruction:.4f} | "
            f"{c.sens_action_final:.3f} | {c.sens_instruction_final:.3f} | {c.verdict} |"
        )
    lines.append("")
    lines.append("## Quadrant scatter (gains, normalized to the suite max)")
    lines.append("")
    lines.append("```")
    lines.extend(_ascii_scatter(scorecards))
    lines.append("```")
    lines.append("")
    lines.append(
        "x-axis = gain_action (0 left -> max right); y-axis = gain_instruction "
        "(max top -> 0 bottom). Markers are the model index in the legend below."
    )
    lines.append("")
    for i, c in enumerate(sorted(scorecards, key=lambda c: c.gain_action, reverse=True)):
        lines.append(f"- `{i}` = `{c.model_id}` ({c.verdict})")
    lines.append("")
    text = "\n".join(lines)
    _write(path, text)
    return text


def _ascii_scatter(scorecards: Sequence[Scorecard], width: int = 41, height: int = 21) -> List[str]:
    """A compact ASCII scatter of the dose-response gain plane (normalized to suite max)."""
    cards = sorted(scorecards, key=lambda c: c.gain_action, reverse=True)
    gmax = max([1e-9] + [c.gain_action for c in cards] + [c.gain_instruction for c in cards])
    grid = [[" "] * width for _ in range(height)]
    midr, midc = height // 2, width // 2
    for r in range(height):
        grid[r][midc] = "|"
    for col in range(width):
        grid[midr][col] = "-"
    grid[midr][midc] = "+"

    def cell(x, y):
        c = int(round(np.clip(x, 0, 1) * (width - 1)))
        r = int(round((1.0 - np.clip(y, 0, 1)) * (height - 1)))
        return r, c

    for i, sc in enumerate(cards):
        r, c = cell(sc.gain_action / gmax, sc.gain_instruction / gmax)
        grid[r][c] = str(i % 10)

    out = ["  instr^"]
    for r in range(height):
        out.append("   " + "".join(grid[r]))
    out.append("   " + "".join("0" if k == 0 else (">" if k == width - 1 else " ")
                                for k in range(width)) + "  action")
    return out


# ===========================================================================
# Plots (matplotlib) or CSV fallback
# ===========================================================================
def plot_curves(scorecards: Sequence[Scorecard], outdir: str) -> List[str]:
    """Save per-model S(t) plots + a plane scatter; CSV fallback if no matplotlib.

    Returns the list of written file paths. Never raises on a missing plotting dep.
    """
    os.makedirs(outdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _plot_curves_csv(scorecards, outdir)

    written: List[str] = []
    for c in scorecards:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, cd in c.action_curves.items():
            mean = np.asarray(cd["mean"], dtype=float)
            t = np.arange(1, len(mean) + 1)
            ax.plot(t, mean, label=name, alpha=0.85)
        floor = np.asarray(c.seed_floor.get("mean", []), dtype=float)
        if floor.size:
            t = np.arange(1, len(floor) + 1)
            std = np.asarray(c.seed_floor.get("std", np.zeros_like(floor)), dtype=float)
            ax.plot(t, floor, "k--", lw=2, label="seed floor")
            ax.fill_between(t, floor - std, floor + std, color="k", alpha=0.12)
        ax.set_title(f"{c.model_id}: action S(t) ({c.distance_name})")
        ax.set_xlabel("horizon t")
        ax.set_ylabel("S(t)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = os.path.join(outdir, f"{_safe(c.model_id)}_curves.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)

    # Dose-response gain plane scatter (raw gains, common units).
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for c in scorecards:
        ax.scatter(c.gain_action, c.gain_instruction, s=60)
        ax.annotate(c.model_id, (c.gain_action, c.gain_instruction),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("gain_action (delta-dist / normalized dose)")
    ax.set_ylabel("gain_instruction (delta-dist / normalized dose)")
    ax.set_title("AFP dose-response gain plane")
    fig.tight_layout()
    p = os.path.join(outdir, "plane.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)
    return written


def _plot_curves_csv(scorecards: Sequence[Scorecard], outdir: str) -> List[str]:
    """CSV fallback when matplotlib is unavailable."""
    written: List[str] = []
    for c in scorecards:
        p = os.path.join(outdir, f"{_safe(c.model_id)}_curves.csv")
        names = list(c.action_curves.keys())
        floor = np.asarray(c.seed_floor.get("mean", []), dtype=float)
        T = len(floor) if floor.size else (
            len(next(iter(c.action_curves.values()))["mean"]) if c.action_curves else 0)
        with open(p, "w") as f:
            header = ["t"] + names + ["seed_floor"]
            f.write(",".join(header) + "\n")
            for ti in range(T):
                row = [str(ti + 1)]
                for n in names:
                    mean = c.action_curves[n]["mean"]
                    row.append(f"{mean[ti]:.6f}" if ti < len(mean) else "")
                row.append(f"{floor[ti]:.6f}" if ti < len(floor) else "")
                f.write(",".join(row) + "\n")
        written.append(p)

    p = os.path.join(outdir, "plane.csv")
    with open(p, "w") as f:
        f.write("model,gain_action,gain_instruction,sens_action_share,sens_instruction_share,verdict\n")
        for c in scorecards:
            f.write(f"{c.model_id},{c.gain_action:.6f},{c.gain_instruction:.6f},"
                    f"{c.sens_action_final:.6f},{c.sens_instruction_final:.6f},{c.verdict}\n")
    written.append(p)
    return written


# ===========================================================================
def write_scorecards_json(scorecards: Sequence[Scorecard], path: str) -> str:
    """Serialize all scorecards to a single JSON file."""
    payload = [c.to_dict() for c in scorecards]
    text = json.dumps(payload, indent=2)
    _write(path, text)
    return text


# -- small fs helpers -------------------------------------------------------
def _write(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
