# From Zero to One Hundred: A Guided Tour of Robot World-Model Evaluation

**And where the Action-Faithfulness Probe (AFP) sits in it.**

A self-contained primer that starts where a CS student who has taken an ML
course and written some PyTorch is, and ends where someone who can defend a
research project on world-model evaluation in front of a professor needs to be.
It is written *through the lens of this repository*: every abstract idea is
tied back to a concrete file in `action-faithfulness-probe/`, so by the end you
should understand both the field and your own code as one continuous story.

> How to use this document. Read Part 0 first (it is the map). Then read
> linearly. The "Checkpoint" boxes tell you what you should be able to explain
> to someone else before moving on -- if you cannot, re-read that part. Citations
> are tagged with a confidence level; `[verified]` means it was checked against
> the primary source, `[high/med/low]` is my confidence that the claim is true
> as stated. Treat anything below `[high]` as "read the source before you repeat
> it to a professor."

---

## Part 0 -- The map (read this first)

### 0.1 The one-sentence version

A *world model* is a learned simulator that predicts what happens next given
what you do; this project asks the deceptively hard question **"does it actually
listen to what you do?"** -- and builds a cheap, training-free instrument to
measure that for any released video world model.

### 0.2 The five questions this document answers

1. **What is a world model**, and why does robotics want one? (Part I)
2. **Why is evaluating one so hard**, and why do the obvious metrics lie? (Part II)
3. **What is the "causal" turn** -- counterfactuals, interventions, faithfulness --
   and why is it the right frame? (Part III)
4. **What does the 2025-2026 literature actually say**, paper by paper, and where
   is the genuine open gap? (Part IV)
5. **What did *we* build**, how does the code work, and where does it sit in the
   field -- including its honest weaknesses? (Parts V-VI)

### 0.3 The dependency graph of ideas

```
generative models (VAE/GAN/diffusion/transformer)
        |
        v
world models (latent RL  ->  pixel/video, action-conditioned)
        |
        v
the evaluation problem  --->  visual-fidelity metrics (and why they fail)
        |                              |
        v                              v
the causal turn (counterfactual action fidelity)  <--- this is the key pivot
        |
        v
2026 benchmarks (MiraBench, World-in-World, dWorldEval, group-actions, ...)
        |
        v
THIS REPO: a training-free, horizon-resolved, judge-free action-sensitivity probe
```

### 0.4 Where the code lives (so you can follow along)

| Path | What it is | Part it illuminates |
|---|---|---|
| `afp/rollout.py` | the data contracts (`Episode`, `Rollout`) + the `WorldModelAdapter` interface | I, V |
| `afp/distance.py` | perceptual distance backends (pixel L2, LPIPS, DreamSim) | II, V |
| `afp/perturb.py` | the action + instruction perturbation operators (the "knob") | III, V |
| `afp/metrics.py` | the math: `S(t)`, seed floor, dose-response gains | III, V |
| `afp/score.py` | orchestration: runs the probe, builds the `Scorecard`, decides the verdict | IV, V |
| `afp/report.py` | leaderboard / plane / plots | V |
| `afp/adapters/` | `mock.py` (synthetic oracles) + guarded real adapters (Cosmos-SAC, IRASim, Ctrl-World) | IV, V |
| `afp/cli.py` | the `afp gen-data` / `afp run` command line | V |
| `afp/DESIGN.md` | the executable design doc + staged GO/KILL gates + full changelog | IV, VI |

> Checkpoint 0. You can draw the dependency graph above from memory and name
> what each top-level file in `afp/` is responsible for.

---

## Part I -- Foundations: what a world model is, and what "listens to actions" means

### 1.1 The one idea under everything

Every system in this field learns some version of

```
next  ~=  f(history, action)
```

a function that predicts *what happens next* given *what you did*. Systems differ
on two axes only: **where the prediction lives** (a compact latent code vs. raw
pixels) and **what it is for** (planning/control vs. simulation/data-generation).
Hold those two axes in mind and the whole literature organizes itself.

### 1.2 The six generative-model building blocks (the vocabulary)

You cannot read this literature without these. One line each `[High]` unless noted:

- **VAE** -- learns a compressed probabilistic latent code (encoder+decoder, reconstruction+KL). *Kingma & Welling, ICLR 2014.* This is the "compress an image into `z`" step.
- **GAN** -- generator vs. discriminator minimax; the pre-diffusion image generator. *Goodfellow et al., NeurIPS 2014.*
- **Autoregressive Transformer** -- predict the next token from all previous tokens via self-attention. *Vaswani et al., NeurIPS 2017.* The backbone of "tokenize frames, predict the next."
- **Diffusion model** -- generate by learning to reverse a gradual noising process. *Sohl-Dickstein et al., ICML 2015; Ho, Jain & Abbeel (DDPM), NeurIPS 2020.* Today's dominant image/video generator.
- **Latent diffusion** -- run diffusion inside a VAE latent (Stable Diffusion). *Rombach et al., CVPR 2022.*
- **Classifier-free guidance (CFG)** -- a sampling knob that controls *how strongly* a generation obeys its conditioning (text/action), by interpolating conditional vs. unconditional predictions. *Ho & Salimans, NeurIPS 2021 DGM workshop.* **This knob is conceptually identical to "how action-faithful is the model" -- remember it for Part III.**
- (bonus) **VQ-VAE** -- discrete tokenization of images. *van den Oord et al., NeurIPS 2017.* Underlies IRIS, Genie, GAIA-1.

### 1.3 The conceptual fork that prevents 90% of confusion

There are **two different things** called a "world model":

- **(a) Model-based RL world models -- latent, reward-oriented.** A compact internal
  simulator that predicts future *latent* states + rewards to plan or train a
  policy "in imagination." The pixels are incidental. *Evaluated by downstream
  task return / sample efficiency.* Examples: Ha & Schmidhuber (2018), PlaNet
  (ICML 2019), Dreamer/DreamerV2 (ICLR 2020/2021), **DreamerV3 (Nature 2025)**,
  IRIS (ICLR 2023). `[High]`
- **(b) Generative video world models -- pixel-space, action/text-conditioned.** The
  deliverable is the *video itself*, used as a neural simulator, playable engine,
  or synthetic-data generator. *Evaluated by generative + controllability metrics
  (FVD, LPIPS, "did the requested action actually happen").* Examples: GAIA-1
  (Wayve, 2023), **GameNGen (ICLR 2025)**, **Genie (ICML 2024 Oral)**, **NWM
  (CVPR 2025 Oral)**, **NVIDIA Cosmos (Jan 2025)**. `[High]`

Why this fork matters for *your* project: AFP targets category (b) -- released
*video* world models -- and its whole premise is that category (b)'s standard
metrics (the realism metrics of Part II) do **not** measure whether the model
obeys actions. The frontier (Genie/NWM/Cosmos) is precisely the *collision* of
(a) and (b): pixel-space models realistic enough *and* controllable enough to
plan robot actions with -- which is exactly why "does it obey the action?" became
an urgent, unsolved evaluation question.

### 1.4 The lineage in one breath (so names stop being scary)

2018 **World Models** (VAE+RNN, "train in a dream") -> 2019 **PlaNet** (latent
planning, the RSSM) -> 2020 **Dreamer** (learn behaviors in imagination) -> 2021
**DreamerV2** (discrete latents, human-level Atari) -> 2022-23 **IRIS**
(transformer over image tokens) and the **video-diffusion trio** (Video Diffusion
Models NeurIPS 2022; Imagen Video & Make-A-Video) -> 2023 **GAIA-1** (9B driving
world model) -> 2024 **GameNGen** (real-time neural DOOM) and **Genie** (learns a
*latent action* from unlabeled video -- huge, because action labels are the scarce
resource in robotics) -> 2025 **NWM** (navigation world model used for planning)
and **Cosmos** (open "world foundation models" you fine-tune for your robot).
Worth knowing as contrast: **MuZero** (value-equivalent latent model, Nature
2020), **DIAMOND** (diffusion world model for RL, NeurIPS 2024), and **JEPA/
V-JEPA** (predict in representation space, not pixels -- the philosophical rival
to generative video). `[High; Genie 2/Sora/GAIA-1/Cosmos are tech-reports/blogs, not peer-reviewed -- don't cite a venue]`

### 1.5 Action vs. instruction -- the distinction your whole project rests on

A modern video world model is conditioned on (at least) two very different
things: a **low-level action** (per-step end-effector motion, e.g. the 7-DoF
`(dx,dy,dz,droll,dpitch,dyaw,gripper)` vector in `afp/rollout.py`) and a
**high-level instruction** (a language goal like "put the spoon in the towel").
The *causal conditioning gap* is the worry that a model trained mostly on
successful demonstrations learns the shortcut "instruction => the demonstrated
outcome" and treats the per-step action as ignorable side-information -- so it
produces beautiful task completions that are **not faithful to the specific
actions**. Your repo encodes this distinction structurally: the `Episode` carries
both `actions` and `instruction_emb`, and the probe perturbs each *channel*
separately (Part V). No prior benchmark contrasts the two *within a single model*
-- that is the gap (Part IV).

> Checkpoint I. Explain the (a)/(b) fork and why it changes how you evaluate.
> State the causal conditioning gap in one sentence, and why "trained on
> successes" is what causes it.

---

## Part II -- The evaluation problem: why the obvious metrics lie

You have a predicted video and (maybe) a real one. How do you score it? The
naive answer -- "measure how close the pixels are" -- is where the field spent
years learning hard lessons. This part is the intellectual justification for
`afp/distance.py` and for why AFP does *not* lean on these metrics.

### 2.1 The reference-based fidelity ladder (and its rot)

- **PSNR / MSE** -- pixel-wise error in decibels. *Fatal flaw: it rewards blur.*
  Averaging several plausible futures lowers MSE but looks unrealistic -- the
  "regression-to-the-mean" problem -- which is catastrophic for *predictive*
  models that face genuinely multi-modal futures. `[High]`
- **SSIM** -- compares local luminance/contrast/structure, built because MSE
  ignores structure. *Wang et al., IEEE TIP 2004.* Better than PSNR, but
  implementation-sensitive (window size, MS-SSIM vs single-scale) and still
  reference-based + frame-by-frame (no temporal modeling). `[High]`
- **LPIPS** -- distance between *deep CNN features*, which predicts human
  similarity judgments far better than PSNR/SSIM. *Zhang, Isola, Efros, Shechtman,
  Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,"
  CVPR 2018.* **Input contract: images must be normalized to `[-1, 1]`** -- get
  this wrong and you silently compute nonsense. This exact contract is encoded in
  `afp/distance.py::LPIPSDistance` and is the reason the transition metric needs a
  `[0,1]` remap (Part V). *Fragility:* LPIPS is a neural net, so it is
  adversarially attackable (E-LPIPS, Kettunen et al. 2019) and not guaranteed to
  behave on out-of-distribution inputs like difference images. `[High]`
- **DreamSim** -- an *ensemble* of fine-tuned DINO+CLIP+OpenCLIP ViT features
  (cosine distance), trained on human triplet judgments (NIGHTS), capturing
  *mid-level* similarity (layout, pose, identity) that LPIPS misses. *Fu et al.,
  NeurIPS 2023.* *Fragility:* still adversarially breakable (LipSim, ICLR 2024)
  and biased toward semantic/foreground content -- it can *miss* small but
  physically meaningful errors (a slightly mispositioned object). `[High]`

### 2.2 The distributional metric everyone reports -- and why it is nearly blind to dynamics

**FVD (Frechet Video Distance)** is the video analogue of FID: push real and
generated clips through an **I3D** action-recognition net, fit a Gaussian to each
set's features, and take the Frechet distance. *Unterthiner et al., arXiv 2018 /
ICLR 2019 workshop.* It is the de-facto headline number in video generation.

The problem -- and this is load-bearing for AFP's motivation -- is that FVD is
**near-blind to exactly the temporal/dynamical errors a world model makes**:

- *Ge et al., "On the Content Bias in Frechet Video Distance," CVPR 2024:* FVD
  barely moves under large *temporal* corruption; you can halve FVD by resampling
  motionless frames; the bias traces to I3D's content-biased training. `[High]`
- *STREAM (ICLR 2024)* and *JEDi (ICLR 2025)*: FVD over-weights the spatial axis,
  is hard-capped at 16 frames, is statistically unstable (needs huge samples), and
  diverges from human judgment.
- *FVMD (2024)* fixes this only by building the metric on **explicit motion
  features** (keypoint velocity/acceleration).

The conceptual takeaway: FVD answers a *distributional/associational* question
("does the set of generated videos look like the set of real ones?"), not a
*per-intervention* question ("did this action produce the right change?"). That
mismatch is the entire reason AFP measures a per-perturbation, horizon-resolved
`S(t)` instead of reporting an FVD.

### 2.3 The three ways to evaluate, and what AFP chooses

- **Reference-based** (PSNR/SSIM/LPIPS to a ground-truth continuation): penalizes
  *correct-but-different* futures -- wrong for action-conditioned rollouts that
  legitimately diverge. World-in-World and dWorldEval are here.
- **Distributional** (FVD/KVD): dynamics-blind, as above.
- **Judge-based** (a VLM or human rates the video): powerful and semantic, but
  expensive and not training-free. **MiraBench** is here (InternVL3-78B judges
  validated by 16k human judgments).

AFP deliberately takes a **fourth path**: *reference-free, judge-free, internal-
comparison.* It never needs the real future and never needs a judge -- it perturbs
the action and measures how the model's *own* prediction changes. That is the
methodological niche, and it is *cheap* by construction. The cost: AFP measures
*sensitivity* (does the output respond to the input?), not *correctness* (is the
response physically right?) -- a tradeoff you must state plainly to a professor.

> Checkpoint II. Why does PSNR reward blur? Why is FVD nearly blind to temporal
> corruption (name the CVPR 2024 result)? Name the four evaluation modes and say
> which one AFP uses and what that costs you.

---

## Part III -- The causal turn: from "looks right" to "responds right"

This is the intellectual heart. The shift the field is making -- and that AFP
implements -- is from *associational* metrics (how do generated and real videos
compare?) to *interventional* ones (if I change the action, does the right thing
change?). Every concept here maps to a specific function in `afp/`.

### 3.1 Pearl's ladder of causation (the frame)

Three rungs of increasing power *(Pearl, Causality 2009; Pearl & Mackenzie, The
Book of Why, 2018)*: (1) **association** `P(Y|X)` -- "seeing"; (2) **intervention**
`P(Y | do(X=x))` -- "doing", forcibly setting `X` and breaking its normal causes;
(3) **counterfactual** -- "what would this specific case have done had `X` been
different?". `[Background]`

The punchline: **a world model you can intervene on is an interventional engine.**
Setting the action and holding `o0` fixed *is* the `do`-operator. So FVD
(rung 1, associational) is literally the *wrong rung* for action faithfulness,
which is a rung-2/3 question. AFP's `rollout(o0, P(actions), ...)` in
`afp/score.py` is a `do(action = edited)` query, by construction.

### 3.2 Counterfactual sensitivity and the Action-Sensitivity Curve

The core AFP quantity is a counterfactual contrast: roll out under the true
action `a` and under an edited action `P(a)` *with the same seed* (so the only
difference is the edit), and measure the per-frame divergence over the horizon.
That is `S(t)` -- `afp/metrics.py::sensitivity_curve`. A model that ignores
actions has `S(t) ~ 0`; a faithful one has `S(t)` that grows with the horizon and
with the edit magnitude.

### 3.3 Placebos / negative controls -- and the "invalid placebo" bug

To know that a nonzero `S(t)` is real and not noise, you need a **placebo**: an
edit that *should* change nothing. This is the negative-control logic of *Lipsitch
et al., Epidemiology 2010*: a valid negative control must (a) share the same noise
structure as the real comparison and (b) genuinely have **no** true effect. **A
placebo with a real effect invalidates the study** -- because a nonzero response
can no longer be attributed to a bug rather than a real pathway. `[High]`

This is *exactly* the `null_shuffle` -> `null_micro` story in `afp/perturb.py`:
perturbing "inert" dims was an invalid placebo (those dims are real motion on a
robot), so it was replaced by a sub-quantization micro-edit that is genuinely
no-effect for *any* embodiment. The formalization is **counterfactual invariance**
*(Veitch et al., NeurIPS 2021 -- "Counterfactual Invariance to Spurious
Correlations")*: a faithful model's relevant output should be invariant under an
edit that should not matter. `[High]`

### 3.4 The cross-channel comparison problem -- and dose-response as the answer

How do you compare "how much the *action* drives the video" to "how much the
*instruction* drives it" when the two edits have **no shared unit**? Two ideas
from causal/statistical methodology:

- **Standardized / normalized effects** from mediation analysis *(Preacher &
  Kelley, Psychological Methods 2011)* -- put effects on a common scale before
  comparing. *(Caveat the subagent verified: their `kappa^2` index is superseded
  as non-monotone by Wen & Fan 2015 -- use proportion-style measures, not
  `kappa^2`.)* `[High]`
- **Dose-response curves** (standard pharmacology; Bradford Hill's "biological
  gradient"): instead of one perturbation magnitude, *sweep* the dose and compare
  the *shape/slope* of the response. This compares channels **without** a shared
  magnitude unit. `[Background]`

This is precisely AFP's Rev-4 redesign: `afp/metrics.py::dose_response` fits the
slope of `S` vs. *normalized dose* for each channel (action dose = gauss-sigma
ladder; instruction dose = interpolation fraction toward the counterfactual), and
`afp/score.py` uses those two slopes -- now in common units -- as the plane axes.
The honest residue (Part V's "wart", and the open problem in 6.3) is that even
dose-response leaves the *parameterization* of each dose axis as a judgment call.

### 3.5 Validate the validator (sanity checks) + statistical rigor

- **Sanity checks** *(Adebayo et al., NeurIPS 2018)*: an attribution/probe that is
  *insensitive to the model* is measuring an artifact, not the model. The
  transferable rule: your probe must give different answers for a competent model
  vs. a broken/randomized one. AFP's **reactivity gate** is exactly this
  model-sensitivity check; and the synthetic-oracle suite (`mock.py`) is the
  positive/negative control. Necessary, not sufficient. `[High]`
- **Statistical testing** *(Dror et al., ACL 2018; BCa bootstrap, Efron 1987)*:
  compare systems with **paired** tests and report **confidence intervals**, not
  point estimates; resample at the **independent unit** (whole episodes, not
  frames). AFP's separation gate fires only when the **bootstrap-CI lower bound**
  clears 3x the seed floor -- this principle, in code. `[High]`

> Checkpoint III. Map each to a file/function: do-operator, counterfactual
> contrast, placebo, counterfactual invariance, dose-response, model-sensitivity
> sanity check, CI-based gating. If you can do this, you understand both the
> causal-inference canon and your own repo as one thing.

---

## Part IV -- The 2026 map: the closest work, verified, and the gap

Now the frontier. Each entry below was verified against primary sources; treat
the confidence tags seriously when speaking to a professor.

### 4.1 The closest benchmarks and instruments

- **MiraBench** *(arXiv:2605.29360; NeurIPS submission, anonymous)* `[verified existence; venue = submission]` -- **the closest prior work.** A hierarchical
  benchmark for action-conditioned reliability: Physics Adherence /
  **Action-Following Fidelity** / **Optimism-Bias Detection**. ~906 videos,
  16,704 human judgments, VLM judges (InternVL3-78B), 12 model configs incl.
  Cosmos-Predict2. Its Optimism-Bias level (does the model predict success under
  *failure-inducing* actions?) is, in substance, the causal conditioning gap and
  is the **closest existing reference-free, perturbation-based instrument.** *Gap
  it leaves:* judge-dependent (not training-free/afternoon-scale), not
  horizon-resolved, and no controlled within-model `Delta a` vs `Delta l`
  contrast. This is AFP's wedge -- and the natural validation is to *correlate*
  AFP's judge-free scores with MiraBench's expensive ones on shared models.
- **World-in-World** *(arXiv:2510.18135; ICLR 2026 Oral)* `[verified]` -- a
  closed-loop world-model *utility* platform; defines controllability as
  `1 - LPIPS(GT, pred | same actions)`. **Correction (verified): its manipulation
  suite is RLBench, not LIBERO.** This is *ground-truth-referenced and
  single-channel* -- distinct from AFP's reference-free, two-channel sensitivity.
- **dWorldEval** *(arXiv:2604.22152; preprint, no public code)* `[verified ID + numbers]` -- introduces **Delta-LPIPS** (transition-space LPIPS) "to evaluate
  action controllability"; reports r~0.9 correlation with real success on
  LIBERO+RoboTwin. AFP borrows the transition-space idea (`transition_curve`,
  marked exploratory) but adds the action-vs-instruction contrast it lacks.
- **The position paper** *(arXiv:2606.15032)* `[verified]` -- "How Should World
  Models Be Evaluated?"; an L0-L7 ladder foregrounding "counterfactual action
  fidelity" + "full-horizon outcome fidelity". It *proposes* the L4 interventional
  test; AFP is a concrete training-free implementation of it. Cite as conceptual
  parent, not competitor.

### 4.2 The scoop risk, assessed honestly

- **"World Models as Group Actions"** *(arXiv:2605.24578)* `[existence verified; treat exact ID as medium]` -- **the single biggest novelty risk.** It independently
  formalizes action faithfulness via the group axioms (identity / inverse /
  composition), uses the **zero-action null**, and treats stochastic-rollout
  divergence as a robustness floor -- overlapping AFP's premise and two of its
  mechanisms. **But the verified scope is narrower than feared: it operates in the
  *navigation* domain and uses the framing as a *training-time regularizer* with a
  recovered-state estimator** -- *not* a training-free, post-hoc probe of released
  manipulation/VLA video models, and *not* a within-model action-vs-language
  comparison. So your differentiation survives, but you must read it in full and
  state the difference explicitly. (This is also why `afp/perturb.py::null_micro`
  cites it: the zero-action/identity null is shared intellectual ground.)

### 4.3 The checkpoints AFP targets (the adapters)

Verified, with one correction:

- **Cosmos-Predict2-2B-Sample-Action-Conditioned** (NVIDIA, HuggingFace) `[verified]` -- the primary: video+action -> 12 future frames @ 4fps, 7-DoF Bridge
  action. Wrapped by `afp/adapters/cosmos_sac.py` (guarded).
- **IRASim** -- **correction: `arXiv:2406.14540`, ByteDance, ICCV 2025 (a 2024
  paper, not 2026).** `[verified]` Action-conditioned robot video. `afp/adapters/irasim.py`.
- **Ctrl-World** (DROID domain) `[verified existence]` -- `afp/adapters/ctrl_world.py`.

These adapters import cleanly but `available()` returns False until weights/env
are set (Part V, 5.6) -- which is *why no real numbers exist yet* (Part VI).

### 4.4 The gap, stated as a sentence you can defend

> As of mid-2026, no released benchmark provides a **training-free, reference-free,
> judge-free, horizon-resolved** probe that, **within a single checkpoint**,
> contrasts sensitivity to the **low-level action** against sensitivity to the
> **high-level instruction**. MiraBench is judge-based and not horizon-resolved;
> World-in-World and dWorldEval are ground-truth-referenced and single-channel;
> the group-actions paper is a navigation training-regularizer. AFP occupies that
> intersection.

> Checkpoint IV. Name the closest benchmark and AFP's three-part wedge against it.
> Name the biggest scoop risk and the *verified* reason your work still differs.
> State the gap sentence from memory.

---


## Part V -- The Action-Faithfulness Probe, explained through the repository

This is the part no textbook can give you: a line-of-sight from the field's
ideas to *your* code. Read it with the files open.

### 5.1 The core abstraction: an adapter and two data contracts

Everything starts with a promise: *any* world model, no matter how it is
implemented internally, can be described as a function

```
(start frame o0, instruction text + embedding, action chunk a[1..T])  ->  predicted frames ohat[1..T]
```

That promise is the `WorldModelAdapter` abstract base class, and the two
dataclasses that flow in and out of it:

```40:57:action-faithfulness-probe/afp/rollout.py
@dataclass
class Episode:
    """A single probe instance: start frame + instruction + a ground-truth action chunk.

    ``instruction_emb`` is a small fixed-dim numeric embedding of ``instruction``.
    Mock / latent adapters consume the embedding directly; real adapters consume
    the ``instruction`` text and may ignore the embedding. ``true_future`` is the
    held-out ground-truth continuation when available (e.g. the Bridge clip), used
    only for optional fidelity (not for the core sensitivity metrics).
    """

    episode_id: str
    o0: np.ndarray                      # (H, W, 3) uint8
    instruction: str
    instruction_emb: np.ndarray         # (E,) float32
    actions: np.ndarray                 # (T, A) float32
    true_future: Optional[np.ndarray] = None   # (T, H, W, 3) uint8 or None
    meta: dict = field(default_factory=dict)
```

```114:124:action-faithfulness-probe/afp/rollout.py
    @abstractmethod
    def rollout(
        self,
        o0: np.ndarray,
        actions: np.ndarray,
        instruction: str,
        instruction_emb: np.ndarray,
        seed: int = 0,
    ) -> Rollout:
        """Predict the future frames for one (start frame, action chunk, instruction)."""
        raise NotImplementedError
```

Why this matters conceptually: the entire field's "training-free, any released
checkpoint" ambition (Part IV) reduces, in engineering terms, to *"can you wrap
it in a `rollout()`?"* The probe never looks inside the model. It only perturbs
inputs and watches outputs -- a **black-box, observation-space** method. The
`true_future` field is deliberately optional and *not used by the core metric*:
this is what makes AFP **reference-free** (no ground-truth video needed), unlike
the controllability metrics in World-in-World or dWorldEval (Part IV) that
compute distance to a real continuation.

Note the action layout encoded as constants -- this is the Bridge / Cosmos
7-DoF convention, and it is referenced everywhere downstream:

```34:37:action-faithfulness-probe/afp/rollout.py
# 7-DoF Bridge action layout (indices into the action vector).
ACTION_DIM = 7
MOTION_DIMS = (0, 1, 2, 3, 4, 5)  # dx, dy, dz, droll, dpitch, dyaw
GRIPPER_DIM = 6
```

> Checkpoint 5.1. Explain why measuring action faithfulness does *not* require
> the ground-truth future, and what that buys you (cost, generality) versus what
> it costs you (you can measure *sensitivity* but not *correctness*).

### 5.2 The synthetic oracle: why a fake model is the most important model

Here is the deepest methodological point in the whole project, and it comes
straight from Part III's "sanity checks" literature (Adebayo et al., 2018): **to
trust a measuring instrument, first point it at something whose answer you
already know.** That is what `MockWorldModel` is -- a world model with a *dial*
for exactly how much it listens to actions versus instructions:

```131:138:action-faithfulness-probe/afp/adapters/mock.py
        vel = (
            self.action_gain * self._act_scale * actions[:, 0:2]
            + self.instruction_gain * self._instr_scale * drift[None, :]
            + jitter
        )
        start = np.array([self.size / 2.0, self.size / 2.0])
        positions = start[None, :] + np.cumsum(vel, axis=0)
```

The blob's velocity is a weighted sum of (a) the commanded action, scaled by
`action_gain`, (b) a drift direction read from the instruction embedding, scaled
by `instruction_gain`, and (c) seed jitter. So we can *construct* models with
known ground truth and check the probe recovers it:

```155:159:action-faithfulness-probe/afp/adapters/mock.py
    return [
        MockWorldModel("mock-faithful", action_gain=1.0, instruction_gain=0.15, noise=0.04),
        MockWorldModel("mock-balanced", action_gain=0.6, instruction_gain=0.6, noise=0.04),
        MockWorldModel("mock-instruction", action_gain=0.12, instruction_gain=1.0, noise=0.04),
    ]
```

`mock-faithful` *should* be measured as action-faithful; `mock-instruction`
*should* come out instruction-driven. If the probe got this backwards, the probe
is broken -- and we would never have known on a real model, because real models
have no ground-truth "gain". This is the validation logic of the field's
synthetic-oracle tradition (Part III).

The `jitter` term is not decoration: it emulates the fact that diffusion world
models are **stochastic samplers**. Run the same model twice and you get
slightly different videos. That sets up the single most important calibration in
the whole probe -- the *seed floor* (5.5).

> Checkpoint 5.2. Why can't you validate an action-faithfulness metric purely on
> real checkpoints? (Answer: you would have no ground truth for "how faithful is
> faithful", so a wrong metric and a wrong model are indistinguishable.)

### 5.3 The knob: perturbation operators

If the adapter is the thing under test and the mock is the calibration target,
the perturbation operators are the *experimental intervention* (Part III's
"do-operator" in code). They take an action chunk and return an edited one. The
key design rule, stated in the module docstring, is that every operator is a
**pure function** (no mutation, fresh array) so interventions compose cleanly.

The operators span a *severity ladder* used by the monotonicity test:

```268:278:action-faithfulness-probe/afp/perturb.py
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
```

**The null problem -- a worked example of the causal-inference content in Part
III.** A "null" perturbation is a *placebo*: an edit that *should* change
nothing, used to measure the metric's noise floor. The original null
(`null_shuffle`) perturbed action dims `(2,3,4,5)` because those are inert *in
the mock*. But on a real robot `dz` is vertical motion -- so that "placebo" has a
real causal effect. In causal-inference terms (Veitch et al., 2021; negative-
control theory) that is an **invalid placebo**, and a placebo with a real effect
invalidates the calibration. The fix is the embodiment-agnostic `null_micro`:

```157:178:action-faithfulness-probe/afp/perturb.py
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
    """
```

On the instruction side, the same "dose ladder" idea (Part III's dose-response)
appears: instead of a single counterfactual, we interpolate from the true
meaning toward the counterfactual at increasing fractions, so we can sweep
*semantic dose*:

```405:421:action-faithfulness-probe/afp/perturb.py
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
```

> Checkpoint 5.3. Define "invalid placebo" and explain, using `null_shuffle` vs
> `null_micro`, why one is valid on a real robot and the other is not.

### 5.4 The ruler: perceptual distance backends

To turn "the videos look different" into a number, you need a *distance on
frames*. This is where Part II's perceptual-metric literature enters the code.
Three backends, in increasing perceptual sophistication and dependency weight:

```119:123:action-faithfulness-probe/afp/distance.py
_BACKENDS: Dict[str, Callable[..., Distance]] = {
    "pixel_l2": PixelL2Distance,
    "lpips": LPIPSDistance,
    "dreamsim": DreamSimDistance,
}
```

`pixel_l2` is the default precisely because it has no failure modes of its own
to debate -- it is a transparent ruler, runs on a CPU with no downloads, and lets
the whole harness be validated end-to-end before anyone argues about LPIPS. The
LPIPS backend encodes the field's hard-won convention that LPIPS expects inputs
in `[-1, 1]` (Part II):

```77:83:action-faithfulness-probe/afp/distance.py
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        torch = self._torch
        fa = _to_float01(a) * 2.0 - 1.0
        fb = _to_float01(b) * 2.0 - 1.0
        # (T,H,W,3) -> (T,3,H,W)
        ta = torch.from_numpy(fa.transpose(0, 3, 1, 2)).float().to(self._device)
```

This `[-1,1]` detail is *exactly* the bug that bit the transition-space metric
(5.5): feeding signed difference images to LPIPS without remapping pushes inputs
out of the range LPIPS was trained on. The fix lives in `metrics.py` and is a
microcosm of why Part II insists "know your metric's input contract."

> Checkpoint 5.4. Why is the *default* distance the dumbest one (`pixel_l2`)?
> (Answer: validating the pipeline must not be entangled with debating the
> metric; a transparent ruler isolates harness bugs from metric bugs.)

### 5.5 The math: from a distance to an Action-Sensitivity Curve

Now the centerpiece. The **Action-Sensitivity Curve** `S(t)` is the per-frame
distance between the rollout under the true action and the rollout under a
perturbed action -- a horizon-resolved measure of "how much did editing the
action change the predicted future, step by step." Everything in `metrics.py` is
a *pure function* of frames + a distance callable (no model, no I/O), which is
what makes it unit-testable. Three load-bearing pieces:

1. **The seed floor** (`seed_floor`): the same input, different sampling seeds,
   gives a non-zero baseline divergence. This is the noise floor every real
   number must be compared against -- the diffusion-sampler analogue of
   measurement error. Grounded in the reproducibility-of-diffusion literature
   (Part III) and the mock's `jitter`.

2. **The transition variant `S^Delta`** (`transition_curve`): inspired by
   dWorldEval's Delta-LPIPS (Part IV). It compares *frame-to-frame differences*
   rather than frames. It is explicitly marked **exploratory** because difference
   images are out-of-distribution for perceptual nets -- and the code carries the
   `[0,1]` remap fix that the LPIPS contract (5.4) demands:

```370:381:action-faithfulness-probe/afp/metrics.py
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
```

3. **The dose-response gain** (`dose_response`): the resolution to the field's
   hardest measurement problem (Part III + Part IV). You cannot directly compare
   "how much an action edit moves the video" to "how much an instruction edit
   moves the video" because the two edits have no shared unit. The dose-response
   answer is to sweep each channel's *dose* and compare the *slope* of the
   response, which is unit-free per channel:

```393:420:action-faithfulness-probe/afp/metrics.py
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
```

> Checkpoint 5.5. Explain the seed floor to a skeptic who says "your metric just
> measures sampling noise." Then explain why a dose-response *slope* is more
> defensible than comparing a single action edit to a single instruction edit.

### 5.6 The orchestration and the verdict

`score.py` is where the field's whole pipeline runs for one model: roll out the
base, roll out every perturbation at a paired seed (common-random-number
sampling, so the only difference between base and perturbed is the *edit*, not
the noise), aggregate into curves, compute the two channel gains, and decide a
verdict. The plane axes are the two dose-response gains, which (by construction)
share units:

```452:467:action-faithfulness-probe/afp/score.py
    A literature-grounded fix (causal-mediation standardized effects; the
    controllability-fidelity CFG sweep; perturbation dose-response methods) is to
    **avoid a shared magnitude and compare the SHAPE of a per-channel dose-response
    curve**:
```

The verdict is a dominance decision on the gain plane, with an explicit "inert"
case for models that respond to nothing (the under-conditioning failure the
survey warns about):

```283:298:action-faithfulness-probe/afp/score.py
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
```

**The gates** are the project's scientific guardrails (DESIGN Section 8). They
embody Part III's statistical-rigor literature: a *reactivity* gate (an
Adebayo-style sanity check that the model reacts to actions at all), a *null-
floor* gate (the placebo sits at the noise floor), a *monotonicity* gate (bigger
edits -> bigger divergence, Spearman rho >= 0.7), and a *separation* gate that
fires only when the **bootstrap confidence-interval lower bound** clears 3x the
floor -- not a point estimate:

```572:583:action-faithfulness-probe/afp/score.py
    gates = {
        "reactivity_ok": reactivity_ok,
        "null_floor_ok": null_floor_ok,
        "monotonicity_ok": monotonicity_ok,
        "separation_ok": separation_ok,
    }

    verdict = _decide_verdict(gain_action, gain_instruction)
```

> Checkpoint 5.6. What is "common-random-number" sampling and why is it essential
> here? Name the four gates and the one statistical principle each encodes.

### 5.7 What the demo actually shows (and an honest wart)

Running `afp run --models mock` produces this leaderboard (committed under
`results/`):

| model | verdict | gain_action | gain_instruction | sens_action (share) | rho | sep |
|---|---|---|---|---|---|---|
| `mock-faithful` | action-faithful | 0.0543 | 0.0225 | 0.663 | 0.96 | 61.8 |
| `mock-balanced` | instruction-driven | 0.0485 | 0.0637 | 0.406 | 0.96 | 49.4 |
| `mock-instruction` | instruction-driven | 0.0131 | 0.0647 | 0.185 | 0.95 | 18.4 |

Read it like an expert: the probe **recovers the ground-truth ordering** -- the
action share falls monotonically from the faithful model (0.66) to the balanced
one (0.41) to the instruction-driven one (0.19), and all four gates pass for
every model. `mock-faithful` is the *only* model whose action gain exceeds its
instruction gain (0.054 > 0.023). That is exactly the structure we built in.

The **honest wart**: `mock-balanced` was constructed with *equal* internal gains
(0.6/0.6) yet is labeled "instruction-driven", because the mock's action dose
(a Gaussian sigma on `dx,dy`) and instruction dose (an interpolation fraction)
are **not physically commensurable** -- equal internal gains do not map to equal
dose-response slopes. We deliberately did *not* widen the verdict margin to force
a "balanced" label, because tuning a threshold to recover the answer you already
know is the **circular calibration** Part III warns against. This wart *is* the
open research problem (6.3) -- and being able to articulate it cleanly is itself
a sign you understand the field.

> Checkpoint 5.7. The demo is on synthetic models. State precisely what that does
> and does not prove. (It proves the metric is not broken / recovers known
> structure; it proves nothing about any real checkpoint.)

---

## Part VI -- Where this work sits, honestly

### 6.1 The contribution, as a table

The defensible claim is never "we invented measuring action faithfulness" -- the
field is already there (Part IV). The claim is a *specific conjunction of
properties* that, as of mid-2026, no single prior work holds at once:

| Property | dWorldEval | AGRA | nwm | MiraBench | World-in-World | group-actions | **AFP (this)** |
|---|---|---|---|---|---|---|---|
| Training-free on any released ckpt | partial | no | nav only | no | partial | ? | **yes** |
| Judge-free (no VLM/human corpus) | yes | yes | yes | **no** | yes | yes | **yes** |
| Reference-free (no ground-truth video) | no | n/a | no | no | **no** | ? | **yes** |
| Horizon-resolved `S(t)` curve | no | no | partial | **no** | no | partial | **yes** |
| Within-model action-vs-instruction contrast | no | no | no | **no** | no | no | **yes** |
| Runs on 1 GPU in ~a day | no | no | yes | no | no | ? | **yes (design)** |

The novel objects are precisely three: (i) the horizon-resolved Action-
Sensitivity Curve `S(t)`; (ii) the calibrated, *within-model* action-vs-
instruction comparison (the dose-response plane); (iii) the empirical finding of
*which* released world models are faithful vs. instruction-driven, measured
cheaply. Items borrowed (and cited as such): Delta-LPIPS, the null-as-floor idea,
the monotonicity validity check.

### 6.2 Honest positioning against the closest work

- **MiraBench** (the closest benchmark) already argues "visual fidelity is a poor
  proxy for action fidelity" and probes optimism bias. **Do not claim to have
  discovered the gap.** Our wedge is that MiraBench is judge-dependent (VLM judges
  validated by a 16k-judgment human corpus), not horizon-resolved, and has no
  controlled within-model `Delta a` vs `Delta l` contrast. Frame AFP as a *cheap,
  judge-free screen you run before paying for a MiraBench-style evaluation*, and
  propose correlating AFP scores with MiraBench scores on the models both cover
  (e.g. Cosmos-Predict2) as the validation.
- **World-in-World** (ICLR 2026 Oral) defines controllability against ground truth
  (`1 - LPIPS(GT, pred | same actions)`) -- single-channel and reference-*ed*. AFP
  is reference-*free* and two-channel. Complementary, not competing.
- **The position paper (2606.15032)** calls for "counterfactual action fidelity"
  and "full-horizon outcome fidelity" but stops at *proposing* the test. Cite it
  as the conceptual parent; position AFP as a concrete, training-free
  implementation of its L4 interventional test.
- **"World Models as Group Actions" (2605.24578) -- the one to watch.** It
  independently formalizes action faithfulness via the identity/inverse/
  composition group axioms, uses the zero-action null, and treats stochastic-
  rollout divergence as a robustness floor. This *overlaps our premise and two of
  our mechanisms*. The surviving differentiator is the *application*: training-
  free, reference-free, horizon-resolved, action-vs-instruction scoring of
  released checkpoints, plus the dose-response plane. **This paper must be read in
  full and explicitly differentiated before any submission** -- it is the single
  largest novelty risk. (Confidence on its exact arXiv ID is medium; verify.)

### 6.3 The one genuine open research problem

Cross-channel comparability (the 5.7 wart) is not a bug -- it is an unsolved
methodological question the literature also has not closed: **how do you compare
the influence of two interventions that live in incommensurable spaces (a 7-DoF
continuous action vs. a language embedding)?** The dose-response slope is the
best current answer (it removes the per-channel unit), but the *cross-channel*
comparison still rests on how each dose axis is parameterized. This is exactly
the kind of well-scoped, real-but-tractable problem a professor can sink their
teeth into -- and a strong reason to bring them in rather than paper over it.

### 6.4 What is actually done vs. not (be precise with your advisor)

**Done:** a tested (101 passing tests), pip-installable, training-free harness;
a synthetic-oracle validation showing the metric recovers known structure; three
guarded real-model adapters that import cleanly but refuse to run without
weights; a literature-grounded design doc with staged GO/KILL gates (`DESIGN.md`).

**Not done (the decisive gap):** *zero real checkpoints have been run.* Every
number in the repo is from synthetic mocks with hand-set gains. The empirical
contribution the paper is about -- which released world models are action-
faithful -- does not exist yet, because it needs an A100-class GPU, the model
weights/licenses, and the Bridge dataset. This is the work that turns a harness
into a paper.

### 6.5 The path to publication (calibrated)

From `DESIGN.md` Section 8, with the staged gates:

- **Workshop (robot-learning / world-models):** reachable once 2-3 real
  checkpoints are scored and produce a real `S(t)` finding -- *even a negative
  one* ("all tested models are instruction-driven at long horizon") is
  publishable, because it corroborates the field's central worry with a cheap new
  instrument. Estimated ~0.6 conditional on getting compute.
- **Main conference:** needs the model spread *plus* the MiraBench correlation
  *plus* a clean differentiation from the group-actions paper. ~0.25-0.3.

The risk is concentrated in two places the design names explicitly: Stage 1 (is
the action input *causally wired* in the checkpoint at all?) and Stage 3
(assembling a spread of action-conditioned models that reveals contrast).

### 6.6 How to take this to a professor

You are at the *ideal* moment to approach one -- not despite having no real
results, but because of it:

1. **You have a working, tested artifact.** This silently answers "can this
   student execute?" -- the question that sinks most undergrad cold-emails.
2. **You have a literature-grounded, honestly-scoped proposal** with explicit
   kill criteria. That signals research maturity, not naivety.
3. **You have not yet spent the money.** A professor can still shape the
   bets (which models, plane vs. `S(t)`-only, how to handle 6.3) *before* compute
   is burned -- which is precisely when their input is most valuable.

What they provide that you cannot self-source: **GPU/cluster access** (the
literal blocker), a **read on the scoop risk** (they will know whether the
group-actions paper or MiraBench already eats your novelty), and **mentorship on
the open problem in 6.3**.

The pitch, in three sentences: *"I built a training-free probe that measures
whether a video world model actually responds to its action inputs, validated it
on synthetic models with known ground truth, and wrote it up as an executable
design with explicit GO/KILL gates. I need compute and guidance to run it on
real checkpoints like Cosmos-Predict2 and IRASim. Here is the repo and the
design doc."*

> Checkpoint 6. You can (a) state the contribution as a conjunction without
> overclaiming, (b) name the single biggest scoop risk and how you differ, (c)
> articulate the open problem in 6.3, and (d) deliver the three-sentence pitch.

---

## Appendix A -- Annotated bibliography (verified, with confidence)

Confidence: `[V]` verified against the primary source this round; `[H]` high
(well-established or checked); `[M]` medium (core solid, a detail to re-check);
`[L]` low / flagged uncertainty. Where no peer-reviewed venue exists I say so.

**Foundations -- generative models**
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014. `[H]` -- VAE.
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014. `[H]` -- GAN.
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. `[H]` -- Transformer.
- van den Oord et al., *Neural Discrete Representation Learning (VQ-VAE)*, NeurIPS 2017. `[H]`
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020. `[H]` -- DDPM.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022. `[H]`
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS 2021 DGM workshop. `[H]`

**Foundations -- world models**
- Ha & Schmidhuber, *World Models* / *Recurrent World Models Facilitate Policy Evolution*, NeurIPS 2018. `[H]`
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels (PlaNet)*, ICML 2019. `[H]`
- Hafner et al., *Dream to Control (Dreamer)*, ICLR 2020. `[H]`
- Hafner et al., *Mastering Atari with Discrete World Models (DreamerV2)*, ICLR 2021. `[H]`
- Hafner et al., *Mastering Diverse Control Tasks through World Models (DreamerV3)*, Nature 2025 (vol. 640, pp. 647-653). `[V]`
- Kaiser et al., *Model-Based RL for Atari (SimPLe)*, ICLR 2020. `[H]`
- Micheli, Alonso & Fleuret, *Transformers are Sample-Efficient World Models (IRIS)*, ICLR 2023. `[V]`
- Ho et al., *Video Diffusion Models*, NeurIPS 2022. `[H]`
- Hu et al. (Wayve), *GAIA-1: A Generative World Model for Autonomous Driving*, arXiv:2309.17080, 2023 (tech report, no venue). `[V]`
- Valevski et al., *Diffusion Models Are Real-Time Game Engines (GameNGen)*, ICLR 2025. `[V]`
- Bruce et al. (DeepMind), *Genie: Generative Interactive Environments*, ICML 2024 (Oral). `[V]`
- DeepMind, *Genie 2* (Dec 2024 announcement, no paper). `[M]`
- Bar et al., *Navigation World Models (NWM)*, CVPR 2025 (Oral); arXiv:2412.03572. `[V]`
- NVIDIA, *Cosmos World Foundation Model Platform for Physical AI*, Jan 2025 (platform report). `[V]`
- Schrittwieser et al., *MuZero*, Nature 2020. `[H]` -- value-equivalent contrast.
- Alonso et al., *Diffusion for World Modeling (DIAMOND)*, NeurIPS 2024. `[H]`

**Evaluation metrics**
- Wang et al., *Image Quality Assessment (SSIM)*, IEEE TIP 2004. `[V]`
- Zhang et al., *The Unreasonable Effectiveness of Deep Features (LPIPS)*, CVPR 2018. `[V]`
- Kettunen et al., *E-LPIPS*, arXiv:1906.03973, 2019. `[V]` -- LPIPS adversarial fragility + fix.
- Ghazanfari et al., *LipSim: A Provably Robust Perceptual Similarity Metric*, ICLR 2024. `[V]` -- DreamSim is attackable.
- Fu et al., *DreamSim*, NeurIPS 2023 (Spotlight). `[V]`
- Unterthiner et al., *FVD: A New Metric for Video Generation*, ICLR 2019 workshop / arXiv:1812.01717. `[V]`
- Ge et al., *On the Content Bias in Frechet Video Distance*, CVPR 2024. `[V]` -- FVD is dynamics-blind. (This is the paper often misremembered as "On the Robustness of FVD".)
- Kim et al., *STREAM*, ICLR 2024. `[V]`; Luo et al., *Beyond FVD (JEDi)*, ICLR 2025. `[V]`; Liu et al., *FVMD*, arXiv:2407.16124, 2024. `[V]`

**Causal / statistical methodology**
- Pearl, *Causality*, CUP 2009; Pearl & Mackenzie, *The Book of Why*, 2018. `[H]`
- Veitch et al., *Counterfactual Invariance to Spurious Correlations*, NeurIPS 2021 (arXiv:2106.00545). `[V]`
- Lipsitch, Tchetgen Tchetgen & Cohen, *Negative Controls*, Epidemiology 2010. `[V]`
- Preacher & Kelley, *Effect Size Measures for Mediation Models*, Psychological Methods 2011. `[V]` -- **but `kappa^2` is superseded** (Wen & Fan, 2015). `[V]`
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018. `[V]`
- Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018. `[V]`; Efron, *Better Bootstrap Confidence Intervals (BCa)*, JASA 1987. `[H]`

**The 2026 frontier (verify IDs before submission)**
- *MiraBench*, arXiv:2605.29360 (NeurIPS submission, anonymous). `[V existence]`
- *World-in-World*, arXiv:2510.18135 (ICLR 2026 Oral). `[V]` -- manipulation suite is **RLBench**, not LIBERO.
- *dWorldEval*, arXiv:2604.22152 (preprint, no code). `[V]`
- *How Should World Models Be Evaluated?* (position), arXiv:2606.15032. `[V]`
- *World Models as Group Actions*, arXiv:2605.24578. `[M on ID]` -- navigation, training regularizer.
- *IRASim*, arXiv:2406.14540 (ByteDance, ICCV 2025). `[V]`
- *Unitary World Model (UWM-JEPA)*, arXiv:2605.25313. `[V]` -- monotone degradation under action-perturbation severity. (Name is "Unitary," not "Unified".)
- *Cosmos-Predict2-2B-Sample-Action-Conditioned* (HuggingFace, NVIDIA). `[V]`

> Honesty notes carried from verification: "AGRA / Matched Sensitivity Ratio"
> could **not** be independently verified -- do not cite until sourced. Several
> 2026 arXiv IDs are medium-confidence; re-verify before any submission. The
> "LPIPS specifically fails on *residual/difference* images" claim is reasoned
> inference from the input contract + known OOD fragility, not a measured result.

---

## Appendix B -- Glossary (the 30 words you must own)

- **World model** -- a learned predictor of future observations given actions.
- **Action-conditioned** -- the prediction takes a per-step action as input.
- **Causal conditioning gap** -- a model that obeys the instruction's *outcome*
  while ignoring the per-step *action*.
- **Action faithfulness / sensitivity** -- the predicted future changes correctly
  when you change the action.
- **`S(t)` (Action-Sensitivity Curve)** -- per-horizon divergence between rollouts
  under true vs. perturbed actions.
- **Seed floor** -- the divergence from sampling noise alone (same input, different
  seed); the baseline every real signal is compared against.
- **Perturbation operator** -- a function that edits the action (or instruction)
  input; the experimental intervention.
- **Null / placebo** -- an edit that *should* change nothing; calibrates the noise
  floor. **Invalid placebo** -- a "null" that secretly has a real effect.
- **Dose-response** -- sweeping perturbation strength and reading the response
  *shape/slope*; lets you compare channels with no shared unit.
- **Dominance share / gain plane** -- the two dose-response slopes (action,
  instruction) in common units; AFP's headline readout.
- **Reference-free** -- needs no ground-truth future. **Judge-free** -- needs no
  VLM/human rater. **Training-free** -- runs on a released checkpoint as-is.
- **Counterfactual** -- "what would the output have been had the action differed?"
- **Intervention / `do`-operator** -- forcibly setting an input, breaking its
  normal causes.
- **Horizon-resolved** -- reported as a function of prediction time `t`, not a
  single scalar.
- **LPIPS / DreamSim / FVD** -- learned perceptual distance / ensemble perceptual
  distance / distributional video distance (and their failure modes, Part II).
- **CFG (classifier-free guidance)** -- the knob controlling how strongly a
  generation obeys its conditioning.
- **RSSM** -- recurrent state-space model (the Dreamer-family latent dynamics core).
- **Adapter** -- AFP's uniform `rollout()` wrapper letting any model be scored.
- **Gate** -- a GO/KILL check (reactivity, null-floor, monotonicity, separation).

---

## Appendix C -- A suggested reading order (your personal 0->100 curriculum)

1. **Orient (this doc):** Part 0, then skim Part VI so you know the destination.
2. **Foundations (1 day):** the Ha & Schmidhuber *World Models* paper + the DDPM
   paper; skim the DreamerV3 Nature abstract and the Genie abstract. Goal: feel
   the (a)/(b) fork in your bones.
3. **The eval problem (1 day):** the LPIPS paper (read the intro + method) and the
   FVD content-bias paper (Ge et al. CVPR 2024). Goal: internalize *why realism
   metrics are not faithfulness metrics*.
4. **The causal turn (1 day):** Pearl's ladder (Book of Why, ch. 1) + the Adebayo
   sanity-checks paper + skim Veitch et al. Goal: see your probe as an
   intervention with a placebo and a sanity check.
5. **The frontier (2 days):** MiraBench, World-in-World, the position paper, and
   -- carefully -- *World Models as Group Actions*. For each, write one sentence:
   "how does AFP differ?" Goal: be able to defend novelty unprompted.
6. **The repo (1 day):** read Part V with `afp/` open; run `afp run --models mock`
   and read `results/`. Goal: connect every concept to a function.
7. **Synthesize:** re-read Part VI; rehearse the three-sentence pitch and the
   answers to the six Checkpoints. Then book the professor meeting.

*If you can teach Parts II-IV to a friend and answer every Checkpoint, you are at
100 -- ready to walk a professor through the repo and the field around it.*



