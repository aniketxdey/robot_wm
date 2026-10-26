"""Render the README headline image from the committed mock demo.

Rolls out one committed probe episode through ``mock-faithful`` and
``mock-instruction`` under the original and the negated action chunk, with the
same start frame, instruction, and seed in both rows.

    python docs/make_headline.py        # writes docs/assets/headline.png
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from afp.adapters.mock import make_mock_suite
from afp.data import load_episodes
from afp.perturb import negate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "assets", "headline.png")

EPISODE = 0
SEED = 0
SHOW_T = (0, 3, 6, 9, 12)
MODELS = ("mock-faithful", "mock-instruction")
ROWS = (("original actions", "#0072B2", False), ("negated actions", "#D55E00", True))


def _frame(o0: np.ndarray, frames: np.ndarray, t: int) -> np.ndarray:
    return o0 if t == 0 else frames[t - 1]


def main() -> None:
    ep = load_episodes(os.path.join(ROOT, "data", "episodes.jsonl"))[EPISODE]
    suite = {m.model_id: m for m in make_mock_suite()}

    fig = plt.figure(figsize=(11, 3.3))
    outer = fig.add_gridspec(1, 2, wspace=0.08, left=0.06, right=0.99, top=0.86, bottom=0.09)

    for col, model_id in enumerate(MODELS):
        model = suite[model_id]
        inner = outer[col].subgridspec(2, len(SHOW_T), wspace=0.06, hspace=0.08)
        for row, (label, color, flip) in enumerate(ROWS):
            actions = negate(ep.actions) if flip else ep.actions
            frames = model.rollout(ep.o0, actions, ep.instruction, ep.instruction_emb, seed=SEED).frames
            for k, t in enumerate(SHOW_T):
                ax = fig.add_subplot(inner[row, k])
                ax.imshow(_frame(ep.o0, frames, t), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(color); sp.set_linewidth(2.5)
                if row == 0:
                    ax.set_title(f"t = {t}", fontsize=10)
                if k == 0 and col == 0:
                    ax.set_ylabel(label, fontsize=10, color=color)
        x = outer[col].get_position(fig)
        fig.text((x.x0 + x.x1) / 2, 0.95, model_id, ha="center", fontsize=13, fontweight="bold")

    fig.text(0.525, 0.01,
             f"Synthetic mock world models with known gains. Committed probe episode {EPISODE}, seed {SEED}; "
             "start frame, instruction, and seed are identical in both rows.",
             ha="center", fontsize=9, color="#555555")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
