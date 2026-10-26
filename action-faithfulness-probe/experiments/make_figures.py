"""Publication figures for the AFP synthetic-validation experiments.

Reads the JSON outputs of exp1/exp2 (+ the committed demo scorecards) and
writes colorblind-safe, journal-sized figures (PDF vector + 300-dpi PNG) to
``results/figures/``. Style follows the scientific-visualization checklist:
Okabe-Ito palette, no top/right spines, panel labels, error bars with the type
stated in the caption, direct labeling over legends where practical.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
SYN = os.path.join(RES, "synthetic")
FIG = os.path.join(RES, "figures")

# Okabe-Ito colorblind-safe palette.
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})


def _save(fig, name: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def fig1_oracle_recovery(exp1: dict) -> None:
    """(A) recovered action gain vs true g_a; (B) recovered instruction gain vs true g_i."""
    rows = exp1["rows"]
    g_is = exp1["protocol"]["instruction_gains"]
    g_as = exp1["protocol"]["action_gains"]
    colors = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"], OI["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)

    ax = axes[0]
    for k, g_i in enumerate(g_is):
        sub = sorted((r for r in rows if r["true_instruction_gain"] == g_i),
                     key=lambda r: r["true_action_gain"])
        x = [r["true_action_gain"] for r in sub]
        y = [r["recovered_gain_action"] for r in sub]
        ax.plot(x, y, marker=MARKERS[k], color=colors[k], label=f"$g_i$ = {g_i}")
    ax.set_xlabel("true action gain $g_a$ (oracle)")
    ax.set_ylabel("recovered gain$_{action}$\n($\\Delta$dist / unit dose)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(A) action-axis recovery", loc="left", fontweight="bold")

    ax = axes[1]
    for k, g_a in enumerate(g_as):
        sub = sorted((r for r in rows if r["true_action_gain"] == g_a),
                     key=lambda r: r["true_instruction_gain"])
        x = [r["true_instruction_gain"] for r in sub]
        y = [r["recovered_gain_instruction"] for r in sub]
        ax.plot(x, y, marker=MARKERS[k], color=colors[k], label=f"$g_a$ = {g_a}")
    ax.set_xlabel("true instruction gain $g_i$ (oracle)")
    ax.set_ylabel("recovered gain$_{instruction}$\n($\\Delta$dist / unit dose)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(B) instruction-axis recovery", loc="left", fontweight="bold")

    _save(fig, "fig1_oracle_recovery")


def fig2_plane(exp1: dict) -> None:
    """The recovered dose-response plane, colored by ground-truth dominant channel."""
    rows = exp1["rows"]
    fig, ax = plt.subplots(figsize=(3.4, 3.1), constrained_layout=True)

    style = {
        "action": (OI["blue"], "o", "action-dominant oracle"),
        "instruction": (OI["vermillion"], "s", "instruction-dominant oracle"),
        "balanced": (OI["green"], "^", "balanced oracle"),
        "inert": (OI["black"], "x", "inert oracle"),
    }

    seen = set()
    for r in rows:
        ga, gi = r["true_action_gain"], r["true_instruction_gain"]
        if ga == 0 and gi == 0:
            key = "inert"
        elif ga > gi:
            key = "action"
        elif gi > ga:
            key = "instruction"
        else:
            key = "balanced"
        c, m, lab = style[key]
        ax.scatter(r["recovered_gain_action"], r["recovered_gain_instruction"],
                   c=c, marker=m, s=26, label=lab if key not in seen else None,
                   zorder=3)
        seen.add(key)

    lim = max(max(r["recovered_gain_action"] for r in rows),
              max(r["recovered_gain_instruction"] for r in rows)) * 1.15
    ax.plot([0, lim], [0, lim], color="0.75", lw=0.8, ls="--", zorder=1)
    ax.text(lim * 0.97, lim * 0.88, "parity", color="0.5", fontsize=6, ha="right")
    ax.set_xlim(-0.002, lim)
    ax.set_ylim(-0.002, lim)
    ax.set_xlabel("recovered gain$_{action}$")
    ax.set_ylabel("recovered gain$_{instruction}$")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    ax.set_title("dose-response sensitivity plane (20 oracles)", loc="left",
                 fontweight="bold")
    _save(fig, "fig2_sensitivity_plane")


def fig3_power(exp2: dict) -> None:
    """Empirical power of the reactivity gates vs probe-set size."""
    cells = exp2["cells"]
    noises = sorted({c["noise"] for c in cells})
    gains = sorted({c["true_action_gain"] for c in cells if c["true_action_gain"] > 0})
    colors = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True,
                             constrained_layout=True)
    for j, noise in enumerate(noises):
        ax = axes[j]
        for k, g in enumerate(gains):
            sub = sorted((c for c in cells
                          if c["noise"] == noise and c["true_action_gain"] == g),
                         key=lambda c: c["n_episodes"])
            x = [c["n_episodes"] for c in sub]
            ax.plot(x, [c["power_permutation"] for c in sub],
                    marker=MARKERS[k], color=colors[k], label=f"$g_a$ = {g}")
            ax.plot(x, [c["power_ci_gate"] for c in sub],
                    marker=MARKERS[k], color=colors[k], ls=":", alpha=0.55)
        # false-positive rate (g_a = 0)
        fp = sorted((c for c in cells
                     if c["noise"] == noise and c["true_action_gain"] == 0),
                    key=lambda c: c["n_episodes"])
        ax.plot([c["n_episodes"] for c in fp], [c["power_permutation"] for c in fp],
                marker="x", color=OI["black"], label="$g_a$ = 0 (FPR)")
        ax.axhline(0.8, color="0.8", lw=0.8, ls="--")
        ax.axhline(0.05, color="0.8", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32])
        ax.set_xticklabels(["4", "8", "16", "32"])
        ax.set_xlabel("probe episodes $n$")
        panel = chr(ord("A") + j)
        ax.set_title(f"({panel}) seed-noise floor = {noise}", loc="left",
                     fontweight="bold")
        if j == 0:
            ax.set_ylabel("empirical power\n(reject rate over 10 replicates)")
            ax.legend(frameon=False, loc="center right")
    fig.text(0.995, -0.05,
             "solid: Holm-corrected sign-flip test   dotted: BCa CI gate",
             ha="right", fontsize=6, color="0.35")
    _save(fig, "fig3_power_analysis")


def fig4_curves(scorecards_path: str) -> None:
    """Exemplar horizon-resolved S(t) curves: faithful vs instruction-driven mock."""
    with open(scorecards_path) as f:
        cards = json.load(f)
    by_id = {c["model_id"]: c for c in cards}
    picks = [mid for mid in ("mock-faithful", "mock-instruction") if mid in by_id]
    if len(picks) < 2:
        print("  (skipping fig4: demo scorecards not found)")
        return

    show_ops = [("negate", OI["vermillion"]), ("gauss@0.2", OI["orange"]),
                ("null_micro", OI["sky"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True,
                             constrained_layout=True)
    for j, mid in enumerate(picks):
        ax = axes[j]
        c = by_id[mid]
        for name, col in show_ops:
            if name not in c["action_curves"]:
                continue
            cd = c["action_curves"][name]
            m = np.asarray(cd["mean"], float)
            sem = np.asarray(cd["sem"], float)
            t = np.arange(1, len(m) + 1)
            ax.plot(t, m, color=col, label=name)
            ax.fill_between(t, m - sem, m + sem, color=col, alpha=0.18, lw=0)
        cf = c["instruction_curves"].get("counterfactual")
        if cf:
            m = np.asarray(cf["mean"], float)
            t = np.arange(1, len(m) + 1)
            ax.plot(t, m, color=OI["purple"], ls="-.", label="cf-instruction")
        fl = np.asarray(c["seed_floor"]["mean"], float)
        t = np.arange(1, len(fl) + 1)
        ax.plot(t, fl, color=OI["black"], ls="--", label="seed floor")
        panel = chr(ord("A") + j)
        ax.set_title(f"({panel}) {mid}", loc="left", fontweight="bold")
        ax.set_xlabel("prediction horizon $t$ (frames)")
        if j == 0:
            ax.set_ylabel("$S(t)$ (pixel RMS, [0,1] scale)")
        ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig4_sensitivity_curves")


def main() -> None:
    with open(os.path.join(SYN, "exp1_oracle_recovery.json")) as f:
        exp1 = json.load(f)
    with open(os.path.join(SYN, "exp2_power_analysis.json")) as f:
        exp2 = json.load(f)
    print("writing figures ->", os.path.abspath(FIG))
    fig1_oracle_recovery(exp1)
    fig2_plane(exp1)
    fig3_power(exp2)
    fig4_curves(os.path.join(RES, "scorecards.json"))


if __name__ == "__main__":
    main()
