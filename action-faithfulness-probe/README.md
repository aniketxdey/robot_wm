# AFP — Action-Faithfulness Probe

A **training-free, judge-free, horizon-resolved** benchmark that measures whether an
action-conditioned video world model's predicted future *actually responds to the
action it is given* — rather than just repainting the task from the language
instruction.

> **The gap.** Released video world models produce plausible futures that track the
> high-level *instruction* `l`, but are often insensitive to the low-level *action*
> `a`. AFP quantifies this **causal conditioning gap** by perturbing `a` (and, as a
> calibration channel, `l`) while holding the start frame `o0` fixed, then measuring
> how the predicted frames diverge as a function of the prediction horizon `t`.

See [`../DESIGN.md`](../DESIGN.md) for the full motivation, literature positioning,
metric definitions, and the staged GO/KILL decision tree.

---

## Why it's different

- **Counterfactual, paired sampling.** We compare a base rollout against a rollout
  under a perturbed action *with the same seed*, so the difference isolates the
  action's causal effect (not sampling noise).
- **Horizon-resolved.** Everything is a curve `S(t)`, not a single scalar — you see
  *where* in the rollout faithfulness decays.
- **Calibrated, two-channel.** Action sensitivity is reported alongside an
  instruction-sensitivity calibration channel, placing each model on a 2D
  **sensitivity plane** (action-faithful vs. instruction-driven).
- **Runs on a released checkpoint in an afternoon.** No training, no human judges,
  no learned critic.

## The metrics (`afp/metrics.py`)

| symbol | name | meaning |
|---|---|---|
| `S(t)` | Action-Sensitivity Curve | per-frame perceptual distance between base and action-perturbed rollouts |
| `Sᐩ(t)` | Transition-Sensitivity (exploratory) | distance between consecutive-frame *difference* images (flagged OOD for LPIPS) |
| `ε(t)` | Seed / determinism floor | divergence between identical-input rollouts across seeds — the noise floor |
| `sens_action(t)`, `sens_instruction(t)` | per-channel normalized sensitivity | operating-point edit as a fraction of that channel's floor→ceiling range, in `[0,1]` |
| `ρ` | monotonicity | Spearman correlation of `S` vs. perturbation severity (should be high) |
| sep. ratio | separation | severe-edit `S(T)` / seed-floor — must clear `3×` |
| gain_action, gain_instruction | dose-response gains | slope of `S` above floor vs. normalized dose, per channel (common units — the headline plane) |

**Inference.** Gates fire on **BCa bootstrap** 95% lower bounds; each scorecard also carries a
Holm-corrected family of one-tailed **paired sign-flip permutation tests** (reactivity,
separation, instruction reactivity) with paired Cohen's d effect sizes (`Scorecard.stats`).

## Install

```bash
cd action-faithfulness-probe
pip install -e ".[dev]"          # numpy + matplotlib + scipy + pytest
# optional perceptual backends:
pip install -e ".[lpips]"        # LPIPS (torch + torchvision + lpips)
pip install -e ".[dreamsim]"     # DreamSim
```

Core (harness + mock world model) needs **only numpy**. Distance defaults to
`pixel_l2` so the whole thing runs on CPU with no GPU and no downloads.

## Quickstart — the no-GPU demo

The repo ships a synthetic probe set and three **mock world models with known,
tunable action/instruction sensitivity**, so you can validate that the metric
recovers ground truth before spending a dollar of GPU:

```bash
afp gen-data --out data/ --n 24
afp run --models mock --distance pixel_l2 --data data/ --out results/ --seeds 5
```

Output (`results/leaderboard.md`, `results/plane.md`, `results/scorecards.json`,
`results/curves/*.png`). The three mocks separate exactly as designed:

| model | verdict | sens_action | sens_instruction | ρ | sep. |
|---|---|---|---|---|---|
| `mock-faithful` | action-faithful | **0.52** | 0.26 | 0.96 | 61× |
| `mock-balanced` | balanced/instr. | 0.46 | 0.83 | 0.97 | 49× |
| `mock-instruction` | instruction-driven | **0.17** | 1.00 | 0.94 | 20× |

`sens_action` is monotone in the mock's latent `action_gain` (1.0 → 0.6 → 0.12) and
`sens_instruction` is inverted — i.e. **the metric recovers the ground-truth causal
structure**. (The continuous sensitivities are the real output; the `verdict` label
is a simple quadrant threshold and is the coarsest part of the readout.)

## Scoring a real checkpoint

Real adapters live in `afp/adapters/` and are **guarded**: they import without
torch/weights and refuse to run until you set them up on a GPU box. The intended
primary is NVIDIA **Cosmos-Predict2-2B-Sample-Action-Conditioned** (genuine
video+action→video, 12 frames @ 4 fps, ~32.5 GB VRAM, A100-class).

```bash
# on a GPU box, after cloning the repo + downloading weights + accepting licenses:
export AFP_COSMOS_REPO=/path/to/cosmos-predict2
export AFP_COSMOS_CKPT=/path/to/cosmos_sac_2b.pt
afp run --models cosmos-sac --distance lpips --data data/ --out results_cosmos/
```

`afp backends` prints which distance backends are constructible in your env.

| adapter | model | env vars to activate |
|---|---|---|
| `cosmos_sac` | Cosmos-Predict2-2B-Sample-Action-Conditioned (primary) | `AFP_COSMOS_REPO`, `AFP_COSMOS_CKPT` |
| `irasim` | IRASim (Bridge, best spread fit) | `AFP_IRASIM_REPO`, `AFP_IRASIM_CKPT` |
| `ctrl_world` | Ctrl-World (DROID) | `AFP_CTRLWORLD_REPO`, `AFP_CTRLWORLD_CKPT` (+`AFP_CTRLWORLD_SVD`) |

> **You provide:** a GPU/cloud account, the model weights (HF license acceptance),
> and the Bridge dataset (~31 GB external download). Until those are set, each
> adapter's `available()` is `False` and `rollout()` raises the exact setup steps.

## Add your own world model

Implement one method:

```python
from afp.rollout import WorldModelAdapter, Rollout

class MyWM(WorldModelAdapter):
    model_id, horizon, fps = "my-wm", 12, 4
    def rollout(self, o0, actions, instruction, instruction_emb, seed=0) -> Rollout:
        frames = my_model(o0, actions, instruction, seed)   # (T,H,W,3) uint8
        return Rollout(frames=frames, model_id=self.model_id, episode_id="", seed=seed)
```

Then `afp.score_model(MyWM(), episodes, specs)`.

## Repo layout

```
afp/
  rollout.py        # frozen interface: Episode, Rollout, WorldModelAdapter
  distance.py       # pixel_l2 (default), lpips, dreamsim backends
  data.py           # probe schema + reproducible synthetic generator
  perturb.py        # action + instruction perturbation operators
  metrics.py        # S(t), Sᐩ(t), seed floor, normalized sens, monotonicity, bootstrap
  score.py          # Scorecard + orchestration (score_model / score_suite)
  report.py         # leaderboard.md, plane.md, curve PNGs
  cli.py            # `afp gen-data | run | backends`
  adapters/
    mock.py         # deterministic synthetic WMs with known sensitivity
    cosmos_sac.py   # Cosmos-Predict2-2B-Sample-Action-Conditioned (primary)
    irasim.py       # IRASim
    ctrl_world.py   # Ctrl-World
    _action_io.py   # (T,7) <-> Bridge annotation JSON, video I/O
experiments/        # exp1 oracle recovery, exp2 power analysis, figure builder
tests/              # 123 tests, CPU-only
data/, results/     # committed demo artifacts (+ results/synthetic, results/figures)
```

## Synthetic validation experiments

```bash
python experiments/exp1_oracle_recovery.py   # 20 oracles with known gains (~10 s)
python experiments/exp2_power_analysis.py    # 400-replicate power study (~3.5 min)
python experiments/make_figures.py           # publication figures (PDF + PNG)
```

Headline: perfect rank recovery of both channel gains (Spearman ρ = 1.0 per stratum),
17/20 dominance verdicts (all errors conservative), 0/80 false positives, and power ≥ 0.9
for a 3%-of-scale action gain from 16 probe episodes. See `../PAPER.md` for the full
write-up and the GPU handoff protocol.

## Tests

```bash
python -m pytest -q          # 123 tests, all CPU, no downloads
```

## Reproducibility

Embeddings and the synthetic probe set use a process-stable hash (not Python's
`PYTHONHASHSEED`-randomized `hash`), and all sampling is seeded — every verdict is
bit-reproducible across machines.

## License

MIT.
