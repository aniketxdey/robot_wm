![Two synthetic mock world models rolled out from the same start frame with original and negated actions. The action-faithful mock's end effector reverses direction; the instruction-driven mock's barely changes.](docs/assets/headline.png)

# AFP: Action-Faithfulness Probe

AFP tests whether an action-conditioned video world model's predicted future depends on the action it is given, or whether the model mostly repaints the outcome implied by the language instruction. It runs paired rollouts from the same start frame and seed, edits only the action chunk (or only the instruction), and measures how far the predicted frames diverge at each step of the horizon. It needs no training, no human or VLM judges, and no access to model internals.

The image above shows the effect the probe measures. Both mocks start from the same frame with the same instruction and seed; negating the actions sends `mock-faithful`'s end effector the other way, while `mock-instruction` follows nearly the same path. These are synthetic models with gains set by construction. No released checkpoint has been scored yet; the adapters for Cosmos-Predict2, IRASim, and Ctrl-World are written and tested up to the point where they need GPU weights.

The longer documents are [`afp/DESIGN.md`](afp/DESIGN.md) (design rationale, metric derivations, staged GO/KILL gates, and revision history) and [`../PAPER.md`](../PAPER.md) (the paper draft, including the runbook for the real-checkpoint study).

## Method

A world model under test is treated as a black box that maps a start frame `o0`, an instruction `l`, and an action chunk `a` of shape `(T, 7)` to `T` predicted frames. The seven action dimensions are `dx, dy, dz, droll, dpitch, dyaw, gripper`, matching the Bridge annotations used by the target checkpoints.

1. **Paired rollouts.** Each perturbed rollout reuses the base rollout's seed, so the frame difference comes from the input edit rather than sampling noise. A separate set of same-input rollouts across seeds gives the **seed floor** `ε(t)`, which every claim is measured against.
2. **Perturbation ladders.** The action channel runs from nulls (`identity`, and `null_micro`, a Gaussian edit below one action-token bin) through a Gaussian ladder with σ ∈ {0.05, 0.1, 0.2, 0.4} to `scale`, `reverse`, `negate`, `swap`, and `random_action`. The instruction channel uses a paraphrase as its floor and interpolates the instruction embedding toward a counterfactual at fractions {0.25, 0.5, 0.75, 1.0}. Operators live in [`afp/perturb.py`](afp/perturb.py).
3. **Sensitivity curves.** For each operator, `S(t)` is the per-frame distance between the base and perturbed rollouts. The default distance is pixel RMS; LPIPS and DreamSim are optional backends.
4. **Dose-response gains.** Action and instruction edits have no shared magnitude unit, so a ratio of two fixed operating points would encode an arbitrary severity choice. AFP instead fits, per channel, the slope of divergence above that channel's floor against dose normalized to [0, 1] within the channel's ladder. The two gains are in the same units and place each model on a 2D plane.
5. **Verdict, gates, and tests.** The verdict compares the two gains: a model is `inert` if both are below 1e-3, otherwise `action-faithful` or `instruction-driven` if one channel's share of the total gain exceeds the other's by more than 0.10, and `balanced` if not. Four gates decide whether the probe's readout can be trusted on that model, and a Holm-corrected family of one-tailed paired sign-flip permutation tests (with paired Cohen's d) accompanies each scorecard.

| Gate | Passes when |
| --- | --- |
| Reactivity | the 95% BCa lower bound of `S_negate(T)` exceeds the seed floor |
| Null floor | mean `S_null_micro` is below the mean floor plus one standard deviation |
| Monotonicity | Spearman ρ between operator severity rank and mean divergence is at least 0.7 |
| Separation | the 95% BCa lower bound of `S_random_action(T)` is at least 3× the seed floor |

The metric functions are in [`afp/metrics.py`](afp/metrics.py) and the orchestration, thresholds, and verdict logic in [`afp/score.py`](afp/score.py).

## Setup

AFP requires Python 3.9 or later. The core harness and the mock world models depend only on numpy; the `dev` extra adds matplotlib, scipy, and pytest for plots, statistics, and tests.

```bash
cd action-faithfulness-probe
pip install -e ".[dev]"
pip install -e ".[lpips]"      # optional: LPIPS backend (torch, torchvision, lpips)
pip install -e ".[dreamsim]"   # optional: DreamSim backend
```

`afp backends` reports which distance backends can be constructed. It builds each backend to check, so with the DreamSim extra installed it downloads model weights into `./models/` on first use. That directory is gitignored.

## Mock demo

The repository ships a 24-episode synthetic probe set and three mock world models whose action and instruction gains are set by construction. The mock renders an end-effector blob whose velocity is the action gain times the commanded `(dx, dy)`, plus the instruction gain times a drift read from the instruction embedding, plus seeded jitter.

```bash
afp gen-data --out data/ --n 24
afp run --models mock --distance pixel_l2 --data data/ --out results/ --seeds 5
```

This writes `results/leaderboard.md`, `results/plane.md`, `results/scorecards.json`, and per-model curve plots under `results/curves/`. The two commands took about 3 seconds on a MacBook Air CPU, and the output matched the committed `data/` and `results/leaderboard.md` exactly.

| Model | True `g_a`, `g_i` | Verdict | `gain_action` | `gain_instruction` | Action share | Monotonicity ρ | Separation | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mock-faithful` | 1.0, 0.15 | action-faithful | 0.0543 | 0.0225 | 0.663 | 0.961 | 61.8× | 4/4 |
| `mock-balanced` | 0.6, 0.6 | instruction-driven | 0.0485 | 0.0637 | 0.406 | 0.965 | 49.4× | 4/4 |
| `mock-instruction` | 0.12, 1.0 | instruction-driven | 0.0131 | 0.0647 | 0.185 | 0.951 | 18.4× | 4/4 |

The action gain falls and the instruction gain rises across the three mocks in the order their true gains imply. `mock-balanced` has equal true gains but is called instruction-driven, because its recovered instruction gain is larger than its action gain by more than the 0.10 share margin. Equal mock gains therefore do not produce equal recovered gains under these dose ladders, so the gains are best read as a ranking rather than a calibrated ratio.

## Synthetic validation

Two experiments test the probe on mocks whose ground truth is known. The thresholds and ladders are fixed in advance; the oracles check the protocol and are not used to tune it.

```bash
python experiments/exp1_oracle_recovery.py   # 20 oracles on a g_a × g_i grid
python experiments/exp2_power_analysis.py    # 400 replicated scoring runs
python experiments/make_figures.py           # rebuilds results/figures/ from the JSON outputs
```

The outputs are committed under [`results/synthetic/`](results/synthetic/) and [`results/figures/`](results/figures/). The recorded runtimes are 11 seconds for Experiment 1 and 198 seconds for Experiment 2.

**Experiment 1** scores 20 oracles with `g_a ∈ {0, 0.12, 0.3, 0.6, 1.0}` and `g_i ∈ {0, 0.15, 0.4, 1.0}` (16 episodes, 5 seeds, pixel RMS). The recovered action gain has Spearman ρ = 1.00 against the true `g_a` at every fixed `g_i` (ρ = 0.985 pooled), and the instruction axis recovers the same way (ρ = 1.00 per stratum, 0.977 pooled). 17 of the 20 verdicts match the configured dominant channel. The three misses are action-dominant cells near the margin, (0.3, 0.15), (0.6, 0.4), and (1.0, 0.4), called `balanced`; none is inverted. The doubly inert oracle is called `inert`, and all 16 oracles with nonzero action gain pass the four gates. Recovery is concave at large gains because the pixel metric saturates, and a strong competing channel compresses the recovered gain slightly: at `g_a = 1.0`, the recovered action gain drops from 0.0559 to 0.0515 as `g_i` goes from 0 to 1.

![Recovered action and instruction gains against the true oracle gains](results/figures/fig1_oracle_recovery.png)

**Experiment 2** estimates power for the Holm-corrected reactivity test and the BCa gate across true action gains {0, 0.03, 0.06, 0.12, 0.3}, two seed-noise levels (0.04 and a pessimistic 0.3), and 4 to 32 episodes, with 10 replicates per cell. Neither decider fired in any of the 80 replicates with zero action gain. At low noise, the permutation test reaches power 1.0 at 8 episodes even for `g_a = 0.03`. At high noise, `g_a = 0.03` needs 16 episodes for power 0.9 and 32 for 1.0. With 4 episodes the permutation family cannot reject at all, because a sign-flip test on `n` pairs cannot produce p below 2⁻ⁿ. These results set the planned operating point of 16 episodes and 5 seeds for real checkpoints. They describe the mock's Gaussian jitter, not a diffusion sampler's actual noise floor.

![Power of the reactivity deciders against probe-set size at low and high seed noise](results/figures/fig3_power_analysis.png)

The averaged horizon curves for the committed demo show the same ordering as the headline image: `negate` sits far above the counterfactual instruction for `mock-faithful`, the ordering inverts for `mock-instruction`, and `null_micro` stays on the seed floor for both.

![Mean S(t) curves for mock-faithful and mock-instruction](results/figures/fig4_sensitivity_curves.png)

## Scoring a real checkpoint

The real adapters in [`afp/adapters/`](afp/adapters/) wrap each model's own inference script in a subprocess. They import without torch or weights, report `available() == False` until their environment variables point at an existing repository and checkpoint, and raise their setup instructions if called before then. `afp run` skips unavailable adapters and never substitutes a mock.

| `--models` id | Checkpoint | Required environment variables |
| --- | --- | --- |
| `cosmos-sac` | NVIDIA Cosmos-Predict2-2B-Sample-Action-Conditioned (Bridge, 12 frames at 4 fps) | `AFP_COSMOS_REPO`, `AFP_COSMOS_CKPT`, and `AFP_COSMOS_INPUT_VIDEO` at rollout time |
| `irasim` | IRASim (Bridge) | `AFP_IRASIM_REPO`, `AFP_IRASIM_CKPT` |
| `ctrl-world` | Ctrl-World (DROID) | `AFP_CTRLWORLD_REPO`, `AFP_CTRLWORLD_CKPT`, `AFP_CTRLWORLD_SVD` |

Each adapter also accepts optional overrides such as `*_PYTHON`, `*_ENTRY`, and `*_EXTRA_ARGS`; the module docstrings list them. Running any of these needs a GPU (the design targets an A100 80 GB for Cosmos), accepted model licenses, and the external Bridge data, none of which are in this repository. A probe set built from real Bridge clips also has to be assembled first; [`afp/adapters/_action_io.py`](afp/adapters/_action_io.py) converts between `(T, 7)` action arrays and Bridge annotation JSON.

```bash
export AFP_COSMOS_REPO=/path/to/cosmos-predict2
export AFP_COSMOS_CKPT=/path/to/checkpoint
export AFP_COSMOS_INPUT_VIDEO=/path/to/bridge/videos/test/13/rgb.mp4
afp run --models cosmos-sac --distance pixel_l2 --data data_bridge/ --out results_cosmos/ --seeds 5
```

These commands have not been run, since no checkpoint has been scored. The staged plan, pinned inference settings, and GPU cost estimates are in Section 5 of [`../PAPER.md`](../PAPER.md). The cost estimates assume 120 seconds per rollout; that figure has not been measured.

## Adding a world model

Any model that can produce frames from a start frame, an action chunk, and an instruction can be scored by subclassing `WorldModelAdapter`:

```python
from afp import Rollout, WorldModelAdapter, load_episodes, score_model

class MyWM(WorldModelAdapter):
    model_id, horizon, fps = "my-wm", 12, 4
    instruction_input = "text"  # or "embedding" if the model reads instruction_emb

    def rollout(self, o0, actions, instruction, instruction_emb, seed=0) -> Rollout:
        frames = my_model(o0, actions, instruction, seed)  # (T, H, W, 3) uint8
        return Rollout(frames=frames, model_id=self.model_id, episode_id="", seed=seed)

card = score_model(MyWM(), load_episodes("data/episodes.jsonl"))
```

`instruction_input` tells the scorer which channel to perturb. Editing only the embedding of a model that reads text would probe nothing.

## Repository layout

```
afp/
  rollout.py      Episode, Rollout, and the WorldModelAdapter interface
  perturb.py      action and instruction perturbation operators
  distance.py     pixel_l2 (default), lpips, and dreamsim backends
  metrics.py      S(t), seed floor, dose-response, BCa bootstrap, permutation tests, Holm
  score.py        score_model / score_suite, verdicts, gates, Scorecard
  report.py       leaderboard.md, plane.md, curve plots
  data.py         probe-set schema and the synthetic generator
  cli.py          afp gen-data | run | backends
  adapters/       mock suite and guarded Cosmos-SAC, IRASim, Ctrl-World adapters
  DESIGN.md       design document
experiments/      oracle recovery, power analysis, figure builder
docs/             headline image and the script that renders it
data/, results/   committed demo probe set and outputs
tests/            CPU-only test suite
```

The headline image is rendered from the committed demo by `python docs/make_headline.py`.

## Tests

```bash
python -m pytest
```

The 123 tests run on CPU in about 3 seconds and need no downloads. Embeddings and the synthetic probe set use a process-stable hash rather than Python's randomized `hash`, and all sampling is seeded, so the mock results are reproducible across runs.

## License

`pyproject.toml` declares the MIT license; the repository does not include a separate license file.
