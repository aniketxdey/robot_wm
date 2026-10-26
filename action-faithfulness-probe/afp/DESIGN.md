# Action-Faithfulness Probe for Video World Models

**An executable design document.**
Status: validated against the 2026 literature and live checkpoints (June 2026).
Revision 4 (after building a reference harness + a problem-driven literature pass): re-grounded
the action-vs-instruction plane as a per-channel **dose-response** comparison, replaced the
invalid-placebo null with the **action-group identity / sub-quantization** null measured against
the **seed floor**, fixed the `S^Delta` perceptual-range bug + pointed to motion metrics,
upgraded the gates to **CI-lower-bound + reactivity sanity check**, de-circularized calibration,
and flagged the key **concurrent work** (*World Models as Group Actions*, 2605.24578). See
changelog (Sec 14). Revision 3 (two-track review): scoped the horizon claim to the primary
checkpoint's real 12-frame output, corrected inference-time / data-download / spread-set facts,
relaxed the determinism gate, restored World-in-World's ICLR-2026-Oral venue, added arXiv IDs,
and added the decision-centric position paper (2606.15032) as a novelty boundary (Sec 13).
Notation note: this file uses ASCII for math (`Delta`, `||.||`, `S(t)`) for portability.

---

## 0. TL;DR (read this first)

**The one-sentence project.** A *training-free, controlled, horizon-resolved* benchmark
that scores any released video/world model on whether its predicted future actually
*responds to the action you feed it* (action faithfulness) rather than just hallucinating
a plausible task completion from the instruction. You feed a checkpoint a start frame `o0`,
an instruction `l`, and an action chunk `a`; you re-feed it a *perturbed* action `a'`; you
measure how much, and how fast, the predicted rollout diverges. A faithful model diverges
in proportion to the physical size of the perturbation; an "instruction-driven" model barely
moves.

**Why it is still open (as of June 2026).** Several 2026 preprints measure *pieces* of this
(dWorldEval's `Delta-LPIPS`; AGRA's Matched Sensitivity Ratio; MiraBench's Action-Following
Fidelity and Optimism-Bias Detection), but **none** of them is, at the same time, (a)
training-free *and* judge-free *and* applicable to arbitrary released checkpoints, (b)
horizon-resolved (a sensitivity *curve* `S(t)`, not a single level score), and (c) a
*within-model* action-vs-instruction contrast. MiraBench (3 weeks old) is the closest prior
work and overlaps the thesis on Optimism-Bias; the surviving wedge is precisely those three
properties. That narrow intersection is the contribution -- see Section 1.3 for the honest
side-by-side.

**Why it is feasible (corrected).** The genuinely action-*conditioned* released checkpoint is
NVIDIA `Cosmos-Predict2-2B-Sample-Action-Conditioned` ("video + action -> future video,"
post-trained on the **Bridge** dataset), driven by a turnkey entrypoint
`examples/video2world_action.py --input_annotation <7-dim action json>`. It needs **~32.5 GB
VRAM** (an A100 at ~$1.19/hr, *not* a 4090) and **no MuJoCo/LIBERO/EGL stack** -- but note two
review-corrected facts: (i) the Bridge data is **not bundled in the repo**; it is a documented
**~31 GB external download** (`bridge_train_data.tar.gz` from IRASim's public bucket); and (ii)
the checkpoint emits **only 12 future frames at 640x480, 4 fps (~3 s)** -- so "horizon-resolved"
means `t = 1..12` natively (longer horizons need autoregressive chaining; see Sec 2.1).
Whole-project cloud bill: **~$300-550** (revised up; see Sec 7). (The "6.8 GB / RTX 4090 /
$150 / LIBERO" claim from Rev 1 was wrong: that number belongs to `Cosmos-Policy-LIBERO`, a
*policy that emits actions*. See Section 4.)

**The two real risks** are now (1) assembling **>= 3 action-conditioned WMs** so the
leaderboard shows a spread -- harder than Rev 2 implied, because two previously-listed
candidates turned out to be the wrong type (see Sec 4.3) and the genuine ones span different
domains/resolutions/horizons (comparability work); and (2) proving the metric clears sampling
noise (Stage 2). The LIBERO/EGL install risk is **avoided** on the Bridge route.

**Overall calibrated probability of a publishable workshop artifact: ~60-65%.** Probability of
a strong main-conference result: ~25-30%. See Section 8 for the staged breakdown and exact
GO/KILL gates.

---

## 1. Motivation, validated

### 1.1 The phenomenon (the "causal conditioning gap")

Video world models for robotics are trained to predict future frames given an initial
observation, a task instruction, and (sometimes) an action signal. Because the training
loss is dominated by pixel reconstruction of demonstrations that *succeed at the task*, a
model can minimize loss by learning the shortcut **"instruction => the demonstrated
outcome"** while treating the per-step action as weak, ignorable side information. The model
then produces beautiful, plausible task completions that are **not faithful to the specific
low-level actions** it was given. We call the difference between "responds to the
instruction `l`" and "responds to the action `a`" the *causal conditioning gap*.

Why this matters: a world model is only useful for planning / model-predictive control / RL
if perturbing the candidate action actually perturbs the predicted outcome. If it does not,
every downstream planner is optimizing over a function that is flat in the variable it is
supposed to control. This is the exact failure mode the survey (`2605.00080`, Secs. 5.3,
8.1) flags as under-measured.

### 1.2 What the 2026 literature already established (so we do not re-derive it)

These are load-bearing, verified prior results. Cite them; do not reinvent them.

- **dWorldEval (arXiv:2604.22152; April 2026 preprint, no confirmed venue; no public code)**
  introduced `Delta-LPIPS`: instead of comparing absolute predicted frames to ground truth,
  compare the *transition* (difference image `o_{t+Delta} - o_t` over a fixed stride) under
  LPIPS, "to explicitly evaluate action controllability" (verified quote). It compares against
  WorldEval, Ctrl-World, and WorldGym on LIBERO + RoboTwin, and reports its success-rate
  estimates correlate with real execution at Pearson r~0.9. Reported expert-subset Delta-LPIPS
  (lower = better): WorldEval 0.423, WorldGym 0.347, Ctrl-World 0.334, dWorldEval 0.315;
  failure-subset: WorldEval 0.701, WorldGym 0.650, Ctrl-World 0.416, dWorldEval 0.352 (all 8
  numbers verified verbatim). **Takeaway we borrow:** transition-space metrics separate models
  that absolute-state metrics collapse together. **Gap left open:** no code, no
  action-vs-instruction contrast.

- **AGRA / "Making Foresight Actionable" (2026 preprint)** introduced a **Matched Sensitivity
  Ratio (MSR)** in *latent* space. The load-bearing qualitative result we borrow is that a
  well-designed *null* edit lands near ratio ~1.0 (a clean no-sensitivity reference) while
  meaningful edits sit far above it. *Review caveat:* AGRA's MSR is constructed around
  *region/segment* perturbations, not a literal "action vs. null" ratio -- so cite it for the
  null~1.0 *principle*, not as a drop-in for our metric. **Gap left open:** latent-space only
  (needs internal access), architecture-specific, not training-free, not horizon-resolved.

- **nwm (navigation world model; facebookresearch/nwm, CVPR 2025, public code)** is a
  goal/action-conditioned navigation WM evaluated by rolling out under different action/goal
  inputs at multiple time horizons. **Takeaway we borrow:** the *practice* of measuring
  divergence of predicted futures under input changes across horizons, with released code to
  imitate. *Review caveat:* Meta's official nwm does **not** name a "Counterfactual Divergence"
  metric -- but a community **fork** (`harsh-sutariya/nwm`) does add a "Counterfactual
  Divergence / DreamSim @ 8-16s" protocol. So the Rev-1 attribution was wrong about *which*
  repo; the idea exists but in a fork, not the canonical paper. Domain is navigation
  (ego-motion), not manipulation, and it does not contrast action vs. language.

- **UWM-JEPA counterfactual probe (arXiv 2605.25313)** reports that prediction quality
  **degrades monotonically as action-perturbation severity increases** (their public framing
  uses accuracy / probe-R^2 / Hilbert-Schmidt-style measures). **Takeaway we borrow:** the
  *monotone severity-vs-divergence ordering* is a validated, publishable validity signal that
  an action-faithful model must reproduce. *Caveat:* the six-number hidden-state ladder cited
  in Rev 1 (0.0082 < ... < 0.0132) was fabricated and has been removed; only the qualitative
  monotonicity claim is supported.

- **LIBERO-CF (public, vla-va.github.io)** released counterfactual *instructions* for VLA
  policies -- but it perturbs language, not actions, and targets policies, not WMs.

- **MiraBench (arXiv 2605.29360; NeurIPS submission, anonymous; ~3 weeks old) -- THE CLOSEST
  PRIOR WORK.** A hierarchical benchmark for *action-conditioned reliability* with three levels:
  (1) Physics Adherence, (2) **Action-Following Fidelity** (do predictions respect
  task-relevant action inputs?), (3) **Optimism-Bias Detection** (does the model predict
  success under *failure-inducing* actions?). Level 3 is, in substance, the causal conditioning
  gap. Verified specifics: a human-annotated corpus of **906 videos / 16,704 judgments**,
  used to supervise **VLM judges** (Task-Completion-Rate judged by **InternVL3-78B** with
  multi-frame voting; >85% agreement with humans); **12 model configurations** evaluated
  (vector-conditioned incl. **Cosmos-Predict2** and DreamDojo, plus text-conditioned systems)
  on the **GR-1** humanoid and **Lingchu** bimanual corpora. Headline finding: *"visual
  fidelity is a poor proxy for action fidelity"* -- adjacent to our thesis.
  **Why a gap still survives (this is the wedge -- be precise, not dismissive):** MiraBench is
  (i) **judge-dependent** (requires VLM judges validated by a 16k-judgment human corpus; not
  training-free/afternoon-scale on a new checkpoint), (ii) **not horizon-resolved** (per-level
  scores, not an `S(t)` curve; Level 3 only notes that late-phase frames are most salient),
  and (iii) has **no single within-model action-vs-instruction ratio** (it compares
  vector- vs text-conditioned models *across* the suite, not a controlled `Delta a` vs
  `Delta l` contrast *within* one checkpoint). Our contribution lives exactly in (i)-(iii).

- **"How Should World Models Be Evaluated? A Decision-Making-Centric Position"
  (arXiv:2606.15032) -- a NOVELTY BOUNDARY surfaced in review.** This is a *position/protocol*
  paper (no implementation, no results). It argues that for control-oriented WMs the decisive
  question is "what would happen, in task-relevant terms, if the agent took these actions from
  this history?", organizes evaluation on an **L0-L7 ladder**, and proposes a protocol
  foregrounding **"counterfactual action fidelity"** and **"full-horizon outcome fidelity"**
  (verified quotes). **Implication for us (two-sided, be honest):** (a) it *validates* the axis
  -- we are no longer claiming to have *identified* that counterfactual action fidelity is the
  right thing to measure; (b) but it stops at *proposing* the L4 interventional test, whereas
  we *implement* a concrete, training-free, judge-free, horizon-resolved instance of it and run
  it on released checkpoints. Frame our work as a **concrete L4 instantiation** of this paper's
  call -- cite it as motivation, not as a competitor with overlapping results.

### 1.3 The precise, defensible contribution (the open intersection)

Lay these four properties on a table; no single prior work has all four:

| Property | dWorldEval | AGRA-MSR | nwm | UWM-CF | **MiraBench** | **This work** |
|---|---|---|---|---|---|---|
| Training-free, any released ckpt | partial | no | yes(nav) | no | no | **yes** |
| Judge-free (no VLM/human corpus) | yes | yes | yes | yes | **no** | **yes** |
| Observation space (no internal hooks) | yes | no | yes | no | yes | **yes** |
| Horizon-resolved `S(t)` curve | no | no | partial | no | **no** | **yes** |
| Within-model action-vs-instruction contrast | no | no | no | no | **no** | **yes** |
| Light enough for 1 GPU / ~1 day | no | no | yes | no | **no** | **yes** |

The contribution is the **conjunction**: a horizon-resolved, observation-space, training-free
**and judge-free** *within-model* action-vs-instruction faithfulness scorecard that runs on a
single rented GPU and ships as a pip-installable harness with a leaderboard. No single prior
work -- MiraBench included -- holds all of these simultaneously as of June 2026.

**Honest novelty caveat (do not hide this from reviewers; pre-empt the MiraBench question).**
`Delta-LPIPS` and the shuffle-null idea are *borrowed*. MiraBench already argues "visual
fidelity is a poor proxy for action fidelity" and already probes optimism bias -- so we must
**not** claim to have discovered the causal conditioning gap. The novel objects are (i) the
horizon-resolved **Action-Sensitivity Curve `S(t)`**, (ii) the **calibrated, within-model**
action-vs-instruction comparison (Section 2.2), and (iii) the empirical finding of *which
released action-conditioned WMs are faithful vs. instruction-driven, measured judge-free in a
day*. Frame the paper as a **lightweight diagnostic that complements MiraBench** (a cheap,
judge-free, horizon-resolved screen you can run on any new checkpoint before paying for a
full MiraBench-style human/VLM evaluation), **not** a new theory of the gap. A natural,
reviewer-friendly validation is to show our judge-free scores *correlate* with MiraBench's
expensive scores on the models both cover (e.g., Cosmos-Predict2). This framing is the most
robust to "incremental / scooped by MiraBench" rejection.

**Second novelty boundary (position paper 2606.15032).** A June-2026 position paper already
calls for "counterfactual action fidelity" and "full-horizon outcome fidelity" as the right
evaluation axis (its L4 interventional test). We must therefore **cite it as the conceptual
parent** and position our work as the *first lightweight, training-free, judge-free
implementation* of that L4 test on released checkpoints -- not as the originator of the idea.
This is still publishable (a protocol paper + an implemented benchmark are different
contributions), but the "we noticed action fidelity is under-measured" rhetoric must go.

---

## 2. The metrics (precise definitions)

Let a world model `M` map `(o0, l, a_{1:T}) -> ohat_{1:T}` (predicted future frames). We control
the action input `a` and observe predicted frames `ohat`.

### 2.1 Action-Sensitivity Curve `S(t)`

For a perturbation operator `P` applied to the action sequence, define the per-horizon
divergence of the predicted rollout:

```
S_P(t) = D( M(o0, l, a)[t] ,  M(o0, l, P(a))[t] )        for t = 1..T
```

where `D` is a perceptual distance on frames (primary: LPIPS; secondary: DreamSim; sanity:
pixel L2). We additionally report a transition-space variant inspired by dWorldEval:

```
S^Delta_P(t) = D( M(...)[t] - M(...)[t-1] ,  M(...,P(a))[t] - M(...,P(a))[t-1] )
```

**Review caveat on `S^Delta` (treat as exploratory, not load-bearing).** Difference images
`o_t - o_{t-1}` are out-of-distribution for LPIPS/DreamSim backbones (trained on natural RGB).
dWorldEval has no public code, so this variant is unvalidated. Always cross-check `S^Delta`
against a pixel-L2-of-differences baseline before reporting it; do not let a headline rest on
it alone.

> **Rev 4 (two concrete corrections).** (i) *Range bug:* signed diffs in `[-1,1]` fed to LPIPS
> (which internally maps inputs to `[-1,1]`) land in `[-3,1]`, and DreamSim's `uint8` cast
> truncates negatives -- remap diffs to `[0,1]` via `(d+1)/2` before any perceptual backend, and
> default `S^Delta` to `pixel_l2`. (ii) *Better alternatives exist:* the field measures *change /
> dynamics* with **motion-explicit** metrics -- optical-flow EPE, FVMD (keypoint-velocity
> features; Liu et al., ICML 2024 workshop), or STREAM's temporal axis (Kim et al., ICLR 2024) --
> not perceptual nets on residuals. (Note FVD itself is known to under-weight temporal
> corruption.) Prefer analyzing the **Delta of frame-level `S(t)`** (inputs stay in-distribution)
> or a motion metric; a DINOv2 feature distance is a more domain-robust backbone than LPIPS but
> is still *not* validated on residual images. Keep `S^Delta` exploratory.

**Horizon scope (review correction -- important).** The primary checkpoint
(`...Sample-Action-Conditioned`) emits **only 12 frames at 4 fps (~3 s)**, so `S(t)` is a curve
over `t = 1..12` natively. "Horizon-resolved" is therefore *short-horizon* unless you build
**autoregressive chaining** (feed predicted frames back as conditioning) -- which compounds
error and multiplies cost, so treat long-horizon `S(t)` as an *extension*, not the base claim.
Spread models differ in native length/fps (e.g., Ctrl-World ~15-step multi-view; Boundless
81 frames), so **resample/align to a common time grid before any cross-model `S(t)`
comparison.**

`S(t)` is a curve, not a scalar. A faithful model's `S(t)` should be (i) non-decreasing in `t`
over the available horizon (action edits compound) and (ii) monotone in perturbation
magnitude (bigger `||a' - a||` => larger `S(t)`).

### 2.2 Action-vs-instruction comparison (calibrated)

**The flaw with a naive ratio (do not ship this).** A raw
`ARR(t) = S_action(t) / S_instruction(t)` divides the effect of an action edit by the effect
of an instruction edit, but the two perturbations have **no shared magnitude scale**. A large
ratio could simply mean you chose a violent action edit and a gentle paraphrase. The null
floor (Section 2.3) calibrates the action channel against noise, but nothing in the raw ratio
calibrates the *cross-channel* comparison. This is the most attackable part of the headline,
so we calibrate each channel against **its own** floor and ceiling before comparing.

**Per-channel normalized sensitivity.** For each channel `c in {action, instruction}` define a
floor (an edit that should change nothing) and a ceiling (a maximal, meaning-changing edit):

```
sens_c(t) = ( S_c(t) - S_c^floor(t) ) / ( S_c^ceil(t) - S_c^floor(t) ),   clipped to [0, 1]
```

- Action channel: `floor` = null/shuffle action edit; `ceil` = a maximal physical perturbation
  (`negate`/`swap`/`random`).
- Instruction channel: `floor` = a semantics-preserving paraphrase (should not change physics);
  `ceil` = a counterfactual instruction (different target object/goal -- *should* change the
  predicted outcome if the model reads language at all).

Each `sens_c(t)` is now a unitless fraction of *that channel's own achievable range*, so the
"violent vs gentle" artifact is removed.

**Headline = the 2D sensitivity plane, not a scalar.** Report each model as a point/curve in
`( sens_action(t), sens_instruction(t) )`:

- high `sens_action`, low `sens_instruction` => **action-faithful** (the goal region).
- low `sens_action`, high `sens_instruction` => **instruction-driven** (the failure we hunt).
- both high => reads both channels; both low => inert / under-conditioned (suspect a wiring bug
  -- see Stage 1).

**Derived summary (optional, with explicit budgets).** Only after normalization, a scalar
`ARR_norm(t) = sens_action(t) / max(sens_instruction(t), eps)` may be reported as a convenience,
**always** accompanied by a statement of the perturbation budgets used for each ceiling. Never
present the raw, un-normalized ratio as the headline.

> **Rev 4 (literature-grounded redesign of the plane).** A design review found that comparing a
> *single* action operating point against a *single* instruction operating point silently
> reintroduces the "violent-vs-gentle" artifact *across* channels: there is no shared magnitude
> unit between an action-space edit and a language/embedding edit, so any fixed pair of operating
> points encodes an arbitrary severity choice. A literature pass (causal mediation's
> *completely standardized indirect effect*, Preacher & Kelley 2011; the controllability-vs-fidelity
> CFG sweep, Ho & Salimans 2022; perturbation dose-response methods, e.g. GPerturb, Nat. Commun.
> 2025) shows the accepted way out is to **avoid a shared magnitude and instead compare the
> *shape* of a per-channel dose-response curve, not its level**:
>
> 1. **Per-channel dose-response sweep.** For each channel, sweep perturbation strength
>    (action: the `gauss(sigma)` ladder already in Section 3; instruction: an interpolation
>    ladder from `paraphrase` -> `cf-instruction`) and record `S` at each dose. Report the
>    **normalized slope / AUC / monotonicity** of `S` vs dose, which is unit-free per channel.
>    The plane axes become "how *steeply* does this channel drive the future", compared as shape.
> 2. **Matched-relative operating point.** If a single scalar per axis is still wanted, take each
>    channel's response at the *same fraction of its own floor->ceiling range* (e.g. the dose that
>    reaches 50% of that channel's own ceiling), not a fixed absolute edit. This is the
>    cross-channel analogue of per-channel normalization.
> 3. **De-circularization (Problem 6).** The operating point / ceiling MUST be fixed *a priori*
>    (a pre-registered protocol), or chosen on a **held-out** model split -- never tuned so the
>    metric "recovers" the known answer on the same synthetic models used to validate it.
>    Validating a faithfulness metric against synthetic oracles with *known* sensitivity is itself
>    accepted methodology (Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018;
>    M4/SynScore, NeurIPS 2023) -- the circularity is only in *tuning* on the validation set, not
>    in *using* oracles. Report the **raw `S(t)` ordering** (which is calibration-free) as the
>    primary evidence and treat the calibrated plane as the secondary, protocol-dependent readout.
>
> **No existing benchmark builds an action-vs-instruction sensitivity plane** (World-in-World and
> iWorld-Bench measure single-channel controllability against ground truth; see Section 11) --
> so the plane, done as a dose-response comparison, is the contribution rather than a liability.

### 2.3 Null-perturbation floor (calibration)

A correctly-behaving metric must give, for an action edit that *should* not change the outcome:

```
S_null(t) ~ S_floor(t)   (near the metric's sampling/determinism noise floor)
```

If `S_null(t)` is large, the metric is measuring sampling noise, not action faithfulness --
this is a kill signal for the metric, not the model (see Stage 2 gate).

> **Rev 4 (the null was an invalid placebo -- corrected).** The earlier null ("shuffle action
> dims the embodiment treats as exchangeable") assumed dims like `dz, droll, dpitch, dyaw` are
> inert. That is true only for our toy mock; on a *real* robot `dz` is vertical motion, so the
> "null" carries a real treatment effect -- a textbook **invalid placebo** (a placebo with a
> nonzero causal effect is evidence of mis-specification, not a floor; Veitch et al.,
> *Counterfactual Invariance*, 2021). The embodiment-agnostic, citable fix is a hierarchy:
>
> 1. **Action-group identity null (conceptual).** Treat the world model as an action group: a
>    null action is the **identity / zero-displacement** action `e=0`, which *should* leave the
>    state unchanged ("zero-action drift" should be ~floor). This is exactly the invariance
>    reference formalized in *World Models as Group Actions* (arXiv:2605.24578, 2026), which
>    independently defines action faithfulness via the identity, inverse (`a^{-1}=-a`), and
>    composition conditions.
> 2. **Sub-quantization micro-perturbation (practical, embodiment-agnostic).** A perturbation
>    *below the action's quantization / control-noise floor* (e.g. < one action token bin; modern
>    action models discretize each dim into ~256 bins) should produce only floor-level output
>    change. This frames the null as a counterfactual-invariance test that holds for any
>    embodiment, unlike "inert dims".
> 3. **Sampling/determinism floor (the denominator).** Same action, different sampling seed gives
>    the intrinsic stochastic floor of a generative world model -- an accepted noise reference
>    (the "Group-Action Robustness" rollout-stability measure in arXiv:2605.24578; diffusion
>    seed-reproducibility, Zhang et al., arXiv:2310.05264, ICLR 2024). This is the *right*
>    denominator for the perceptual-divergence numbers.
>
> **Drop** temporal reordering as a "null" (valid only if the action group is abelian, which world
> models generally violate) and **drop** "permute symmetric dims" unless a symmetry is proven for
> the embodiment. Implementation: replace `null_shuffle`-as-floor with a `null_micro`
> (sub-quantization Gaussian on all motion dims) measured against the seed floor.

### 2.4 Monotonicity score (borrowed from UWM-CF)

Define a graded perturbation ladder of increasing physical severity
(`identity < small-noise < scale < reverse < negate < random`) and check that mean `S` is
monotone non-decreasing along the ladder. Report Spearman rho between perturbation
severity rank and mean divergence. UWM-CF reported that prediction quality degrades
monotonically with action-perturbation severity (qualitative, verified); reproducing that
ordering is our validity check. (No specific numeric ladder is attributed to UWM-CF.)

> **Rev 4 (reactivity as a sanity check).** The Stage-1 "does the model react to `a` at all?"
> test is the world-model analogue of the **model-sensitivity sanity check** of Adebayo et al.,
> *Sanity Checks for Saliency Maps* (NeurIPS 2018): a probe that is independent of the thing it
> claims to measure is invalid. Frame the reactivity gate as a *necessary* condition (cite
> Adebayo), not a proof of validity (the sanity-check methodology has known limits; Yona &
> Greenfeld 2021). Statistically, declare an effect "real" via the **lower bound of a paired
> bootstrap CI**, not a point estimate (BCa + paired permutation; Dror et al., ACL 2018;
> Bestgen, LREC 2022) -- see the Stage-2 gate.

---

## 3. Perturbation operators (the experimental knob)

All operators act on the action chunk `a_{1:T}` and leave `o0` and `l` fixed. Choose them so
their *physical* magnitude is known, which lets you test the monotonicity property.

| Operator | Definition | Expected effect | Role |
|---|---|---|---|
| `identity` | `a' = a` | `S = 0` exactly | sanity (must be 0) |
| `null/shuffle` | permute exchangeable dims or re-pad no-op | `S ~ floor` | metric calibration |
| `gauss(sigma)` | `a' = a + N(0, sigma^2)`, ladder of sigma | `S` grows with sigma | monotonicity ladder |
| `scale(k)` | `a' = k * a` | larger motion magnitude | monotonicity ladder |
| `reverse` | reverse temporal order | wrong trajectory | severe |
| `negate` | `a' = -a` | opposite motion | severe (UWM's hardest) |
| `swap` | replace with another episode's action | physically inconsistent | severe (AGRA's swap) |
| `zero` | `a' = 0` (or no-op token) | model should freeze / drift slowly | reference |

Instruction-side operators (the instruction channel of Section 2.2):
- `paraphrase`: semantics-preserving rewordings (use LIBERO-CF style or LLM-generated,
  human-checked). This defines the instruction-channel **floor** -- a faithful model should
  give *small* `S_instruction` here.
- `cf-instruction`: semantics-changing counterfactual (different target object/goal). This
  defines the instruction-channel **ceiling**.

**Critical methodological requirement: control for stochasticity.** Diffusion world models
are sampled. `S` must be measured *above sampling noise*. Protocol: fix the seed across the
`a` vs `a'` pair (paired / common-random-number sampling) AND, separately, estimate the
seed-induced floor by running `identity` across `N>=5` seeds. Report `S` net of this floor.

**Determinism caveat (review correction).** Bit-exact GPU determinism is **not** guaranteed in
PyTorch even with a fixed seed, so `S_identity` will be *near* zero, not exactly zero. Pin a
single GPU model, set bf16 (fp16/fp32 are unsupported by this checkpoint), and enable
deterministic algorithms (`CUBLAS_WORKSPACE_CONFIG`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`). The Stage-1 gate is therefore `S_identity(t) < epsilon` for a
*measured* determinism floor `epsilon`, not `= 0`. Also fix and report the guidance scale
(the action checkpoint's reference command uses `--guidance 0`) and the number of denoising
steps -- both modulate how strongly the model honors conditioning and are action-sensitivity
confounds if left free.

---

## 4. Target checkpoints (verified, June 2026)

**Hard requirement: action must be a controllable *input*.** A faithfulness probe feeds the
model a perturbed action `a'` and reads the changed future. That rules out any checkpoint that
*emits* actions. This is the error Rev 1 made (below), so the verdicts here are now sorted by
*conditioning direction*, then VRAM.

### 4.1 The Rev-1 mistake, stated plainly

`Cosmos-Policy-LIBERO-Predict2-2B` is **a policy**: it "learns to directly *generate* robot
actions encoded as latent frames" (Cosmos Policy paper, arXiv:2601.16163). Its future-state /
value outputs exist to score the model's *own* self-generated action proposals in best-of-N
planning. The `get_action(..., generate_future_state_and_value_in_parallel=True)` entrypoint is
the **action-generation** path, not an action-conditioning path. Its 6.8 GB footprint is real
but **does not give us a world model we can feed a counterfactual action into**. Do not anchor
on it.

### 4.2 PRIMARY (Fork-1 = Bridge, selected): true action-conditioned

| Checkpoint | Backbone | Action input | VRAM | Output | Domain | Entry / code |
|---|---|---|---|---|---|---|
| **Cosmos-Predict2-2B-Sample-Action-Conditioned** | Cosmos-Predict2-2B DiT | **yes**: 7-dim (x,y,z,roll,pitch,yaw + gripper open/close) per step, via `--input_annotation` json | **32.54 GB** (A100; OOMs <=32 GB w/o offload) | **12 frames, 640x480, 4 fps (~3 s)** | **Bridge** (real-robot tabletop) | `examples/video2world_action.py`; nvidia-cosmos/cosmos-predict2 |

Why this is the primary: it is a *bona fide* action-conditioned video WM with a turnkey
action-input entrypoint, and **it needs no LIBERO/MuJoCo/EGL**. The action chunk lives in
`--input_annotation`, so applying the Section-3 perturbation operators is literally editing a
JSON of 7-vectors. Review-corrected operational facts:
- **Data is a download, not bundled.** Fetch `bridge_train_data.tar.gz` (~31 GB) from IRASim's
  public bucket into `datasets/`. Watch a path mismatch: the inference example points at
  `datasets/bridge/videos/test/13/rgb.mp4` + `.../annotation/test/13.json` (singular
  `annotation`, a `test/` split) while the tarball lays out `datasets/bridge/{annotations,videos}/`
  -- reconcile at Stage 0.
- **Offload flags are version-dependent.** In `cosmos-predict2` the memory flags are
  `--offload_guardrail` / `--offload_prompt_refiner`; the `--offload-text-encoder` /
  `--offload-tokenizer` form is from `cosmos-predict2.5`. A 40 GB A100 runs it comfortably; a
  24 GB card needs offload **and** reduced resolution (fragile -- avoid for a benchmark).
- **Precision: bf16 only** (fp16/fp32 not officially supported).

### 4.3 SPREAD set (need >= 2 more so the leaderboard is a benchmark, not a measurement)

Review-corrected: two Rev-2 candidates are the **wrong type** and were removed (same
category error the doc warns about). Confirmed action-*input* WMs, ranked by how realistically
an undergrad gets a working `(o0, l, a) -> frames` rollout:

| Candidate | Action input? | Domain / output | Verdict |
|---|---|---|---|
| **IRASim** (arXiv:2406.14540, ByteDance, ICCV 2025) | yes (7-DoF trajectory -> video) | Bridge / RT-1 / Language-Table | **best fit** (Bridge overlap); public code+weights; large (~95-129 GB) downloads, Latte codebase |
| **Ctrl-World** | yes (frame-level action conditioning) | **DROID** (192x320, multi-view, 15-step chunks) | public (ICLR 2026), ~8 GB weights; *also appears as a baseline in dWorldEval* |
| **Boundless-World-Model** | yes (action trajectories) | Wan2.2-TI2V-5B; 480x640, 81 frames | public, inference released 2026-05; newer/less battle-tested |
| **DreamDojo** | yes (action-conditioned post-training) | 640x480; G1/GR-1/AgiBot/YAM | public 2B+14B; complex multi-embodiment action layout -> adapter work |

**Removed (NOT action-input -- do not use as primaries):**
- **GR00T-Dreams / DreamGen** -- text/instruction-conditioned (14B, ~56 GB); actions are
  *extracted post-hoc by an IDM*, i.e., an **output**, not a conditioning input.
- **StarVLA / WM4A** -- a VLA **policy** with an MLP action head; it *emits* actions (same
  category error as Cosmos-Policy-LIBERO).

Selection rule at Stage 0: pick 2-3 that are (a) publicly downloadable, (b) expose `a` as an
overwriteable input, and (c) ideally overlap the primary's domain (IRASim shares Bridge, so it
is the safest second model). Resample all to a common time grid before cross-model `S(t)`.

### 4.4 LIBERO route (only if Fork 1 were `libero_inject`, NOT selected)

Repurposing `Cosmos-Policy-LIBERO` as a conditioned WM would require *latent-frame injection*:
overwrite the action latents in the diffusion sequence and denoise only the future-state
frames. The framework supports this in principle (the released **ALOHA planning model** is the
proof that conditioning-on-supplied-actions works in Cosmos Policy), but **no LIBERO planning
checkpoint is released**, so this is unconfirmed engineering, not a turnkey path. Kept here only
as a documented fallback.

**Compute reality (review-corrected).** Primary at 32.54 GB => A100 80 GB (~$1.19/hr) or 40 GB
with offload. Per-generation time (HF runtime table): **~80 s on H100 PCIe/NVL**, ~50 s on
H200, **~25 s only on GB200/B200**; A100 is unlisted and will be >= H100, so **expect
~100-150 s per rollout on A100**. (The Rev-2 "~25-45 s on H100" figure was the GB200 number.)
The sweep (3-4 models x 8 perturbations x ~50 episodes x 5 seeds ~ 6k-8k rollouts) is the cost
driver -- see Section 7.

---

## 5. Infrastructure (verified)

**On the selected Bridge route, LIBERO/MuJoCo/EGL is NOT needed.** The primary checkpoint
consumes Bridge videos + 7-dim action annotations that ship with `cosmos-predict2`. The
ground-truth future is the held-out Bridge clip. The subsection below applies *only* if you
later add a LIBERO-domain model to the spread set.

- **Metrics:** `pip install lpips` (AlexNet/VGG backbone), DreamSim (`pip install dreamsim`),
  pixel L2 trivial. All CPU-or-GPU, negligible cost.
- **Inference stack:** `nvidia-cosmos/cosmos-predict2` (Conda or Docker; the repo documents
  both). Use the Docker image to avoid CUDA/driver drift.
- **Harness scaffolding to reuse, not rebuild:** `World-in-World` (**ICLR 2026 Oral**;
  arXiv:2510.18135; closed-loop WM-utility platform with a *unified action API* over text /
  viewpoint / low-level controls, LIBERO + RLBench) for any later closed-loop/utility tie-in --
  its finding that *"visual quality alone does not guarantee task success -- controllability
  matters more"* directly supports our motivation. `ACT-Bench` is a useful *API-shape* template
  (`from act_bench import ...; compute_score`) but is **install-from-source** (not on PyPI) and
  **driving-domain** (nuScenes, trajectory-following), so borrow its interface, not its metric.

### 5.1 LIBERO (ONLY for an added LIBERO-domain model -- otherwise skip)

- **Simulator: LIBERO** (`Lifelong-Robot-Learning/LIBERO`, pip `libero==0.1.1`). Python 3.8.13,
  `robosuite==1.4.0`, `mujoco>=3.0`. **This is the highest-friction component if you opt in.**
  Known failure modes and fixes (all verified from issue tracker):
  - `egl_probe` build failure => `pip install cmake==3.31.6` then install `egl_probe` from the
    `fix_windows_build` branch, or `sudo apt-get install build-essential`.
  - Headless rendering: `export MUJOCO_GL=egl` (GPU) or fall back to `MUJOCO_GL=osmesa` (CPU)
    if EGL device display fails.
  - Editable install under pip>=25.3 (PEP 660) may not expose `libero.*` packages; use a repo
    version with the PEP660 fix (PR #137) or `pip install -e .` from a patched fork.
  - **Escape hatch:** use a CUDA+OpenGL Docker image (e.g. `cudagl` devel tags). Cosmos-Policy
    ships a Docker workflow (`uv run --extra cu128 --group libero --python 3.10 ...`) that
    already wires LIBERO -- prefer it over a hand-rolled env.
- **LIBERO seed frames / ground truth:** LIBERO RLDS demonstration datasets provide
  `(o0, l, a)` triples and the *true* future. (Note: for an added LIBERO model you still need a
  checkpoint that *accepts* `a` as input -- the Cosmos-Policy LIBERO checkpoint does not; see
  Section 4.4.)

---

## 6. Repository / deliverable shape

```
action-faithfulness-probe/
  README.md                 # leaderboard + 1-command repro
  pyproject.toml            # pip-installable: `pip install afp`
  afp/
    perturb.py              # Section 3 operators, magnitude-tagged (action + instruction)
    metrics.py              # S(t), S^Delta, per-channel sens_c (Sec 2.2), null floor, Spearman
    rollout.py              # uniform (o0,l,a)->ohat adapter interface
    adapters/
      cosmos_sac.py         # Cosmos-Predict2-2B-Sample-Action-Conditioned (PRIMARY)
      irasim.py             # spread model
      ctrl_world.py         # spread model
    score.py                # afp.compute_score(checkpoint, suite) -> scorecard
  data/
    episodes.jsonl          # frozen (o0,l,a) probe set from Bridge test split
    paraphrases.jsonl       # instruction perturbations (human-checked)
  results/
    leaderboard.md
    plane.md                # 2D (sens_action, sens_instruction) per model
    curves/                 # S(t) plots per model x perturbation
```

The **adapter interface** is the real product: any future checkpoint implementing
`rollout(o0, l, a) -> frames` gets a scorecard for free. This is what makes it a *benchmark*
rather than a one-off measurement, and it is the main reusable artifact reviewers reward.

---

## 7. Budget and timeline

**Compute budget (cloud, verified June 2026 pricing).**

| Item | Assumption | Cost |
|---|---|---|
| Dev/debug GPU time | A100 80GB @ $1.19/hr x ~50 hr | ~$60 |
| Main sweep | ~6-8k rollouts x ~120 s on A100 = ~200-270 GPU-hr @ $1.19/hr | ~$240-320 |
| Spread-model bring-up (IRASim/Ctrl-World/etc., + big data downloads) | A100, ~40 GPU-hr | ~$50 |
| Buffer / re-runs | | ~$50-80 |
| **Total** | | **~$300-550** |

Review-corrected (the Rev-2 "$200-400 / 150-200 GPU-hr" understated it: 150-200 GPU-hr
implicitly assumed H100-class per-rollout speed, but the sweep is sized on A100 at ~120 s/rollout).
Options to control it: run the sweep on H100 (~$2/hr but ~80 s/rollout -> similar total, less
wall-clock), cap at ~6k rollouts, or accept ~$300-550. Still small; compute is not a gating
risk. A department A100 -> cost ~$0.

**Timeline (single part-time undergrad, calendar weeks).**

| Phase | Weeks | Output |
|---|---|---|
| Stage 0: env + one action-conditioned rollout (Bridge) | 1 | one `(o0,l,a)->ohat` via `video2world_action.py` |
| Stage 1: perturb the action json + identity/null sanity | 0.5-1 | `S_identity=0`, `S_negate>0`, `S_null~floor` |
| Stage 2: metric validity (monotonicity, seed floor) | 1-2 | Spearman rho, seed-noise floor |
| Stage 3: bring up 2-3 spread models + sens-plane + `S(t)` | 2-4 | leaderboard, 2D plane, curves |
| Stage 4: write-up + harness release (+ MiraBench corr.) | 2 | workshop paper + pip package |

~7-10 weeks part-time to a workshop submission. On the Bridge route, Stage 0 is *easier* (no
EGL); the variance moves to Stage 3 (assembling 2-3 working action-conditioned spread models).

---

## 8. Staged decision tree (calibrated probabilities + exact GO/KILL gates)

Each stage lists: the gate metric, the **GO** condition (continue), the **KILL/PIVOT**
condition (stop or change), and a calibrated success probability *conditional on reaching
that stage*. Probabilities were re-estimated after the June 2026 validation (revisions noted).

### Stage 0 -- Environment & first action-conditioned rollout
**Goal:** run `examples/video2world_action.py` on the primary
(`Cosmos-Predict2-2B-Sample-Action-Conditioned`) over one Bridge test clip and save `ohat`.
**P(success | started) ~ 0.85** (revised UP: no LIBERO/EGL on the Bridge route; turnkey
entrypoint + repo Docker; only friction is A100 access + checkpoint download/license).
- **GO:** you get `ohat` frames out within ~1 week of effort.
- **KILL/PIVOT:** if A100 access or the checkpoint/license blocks you for >1 week, pivot the
  primary to a spread model that ships pre-rendered action-conditioned data (IRASim). Do NOT
  fall back to the LIBERO sim unless you have a confirmed action-*input* LIBERO checkpoint.

### Stage 1 -- Action is a controllable input + trivial sanity
**Goal:** edit the 7-dim action json (`--input_annotation`) and confirm reactivity.
**P(success | reached) ~ 0.90** (UP: the primary is *built* to take `a` as input, so the
"is `a` even wired in?" risk that dominated the LIBERO route is largely gone).
- **GO:** `S_identity(t) < epsilon` for all `t` (paired seed + deterministic flags; `epsilon` =
  measured determinism floor, since GPU sampling is not bit-exact) AND `S_negate(t)` clearly
  exceeds that floor for some `t` (the model reacts *at all* to a maximal action edit).
- **KILL/PIVOT:** if even `negate`/`swap` produce `S(t)` indistinguishable from the seed-noise
  floor at *every* horizon, the checkpoint conditions on `a` only cosmetically. **Swap in
  another spread model** where `a` is a first-class input. (A genuinely action-insensitive
  released WM is itself a publishable finding -- but rule out a wiring bug via the negate test
  first.)

### Stage 2 -- Metric validity (the real scientific gate)
**Goal:** show the metric measures faithfulness, not noise.
**P(success | reached) ~ 0.70** (this is the crux stage).
- **GO (all three must hold):**
  1. **Null floor:** `S_null(t)` within ~1 metric-noise-sigma of `S_identity`'s seed floor
     (analogue of AGRA's MSR_null ~ 1.0). Quantitatively: `mean_t S_null < S_floor + 1*sd_floor`.
  2. **Monotonicity:** Spearman rho between perturbation-severity rank and mean `S` >= **0.7**
     (UWM-CF achieved a clean monotone ordering; we demand a strong but not perfect rho to
     allow sampling noise).
  3. **Separation:** at the longest horizon, `S_severe(T)` exceeds the seed floor by at least
     **3x** (effect dwarfs noise). **Rev 4:** the gate fires only if the **lower bound** of the
     paired-bootstrap CI (BCa) on `S_severe(T)` exceeds `3 x` the floor -- gate on the CI bound,
     not the point estimate (Dror et al., ACL 2018; Bestgen, LREC 2022). Also add a **reactivity
     gate** (Stage-1 sanity check, Adebayo et al. 2018): the action operating-point edit must
     produce divergence whose CI lower bound clears the seed floor.
- **KILL/PIVOT:** if monotonicity rho < 0.4 OR the severe perturbation cannot clear 3x the seed
  floor on *any* model, the observation-space metric is too noisy. **Pivot the metric** to (a)
  transition-space `S^Delta` (often higher SNR per dWorldEval), (b) DreamSim instead of LPIPS,
  or (c) more seeds. If after these fixes rho is still < 0.4 across all models, the
  pixel-space approach is dead -- escalate to latent-space (AGRA-style), which costs the
  training-free property. **Decision rule:** allow 2 weeks of metric engineering here before
  declaring the pixel-space route dead.

### Stage 3 -- The empirical result (`S(t)` + sensitivity plane across models)
**Goal:** produce the leaderboard, the 2D `(sens_action, sens_instruction)` plane, and the
headline finding across 3-4 action-conditioned WMs.
**P(interesting result | reached) ~ 0.75.**
- **GO / "interesting" (any one suffices):**
  - **Spread:** models occupy clearly different regions of the sensitivity plane (some
    action-faithful: high `sens_action`/low `sens_instruction`; some instruction-driven: the
    reverse) -- the strongest story.
  - **Universal gap:** *all* tested models sit in `sens_action < sens_instruction` at long
    horizon -- a provocative negative result corroborating MiraBench's "visual fidelity is a
    poor proxy for action fidelity" with a cheap, judge-free, horizon-resolved instrument.
  - **Horizon structure:** `S(t)` reveals models faithful early but decohering late (or vice
    versa) -- novel because no prior action-faithfulness work resolved horizon.
- **KILL/PIVOT (the "boring" outcome):** all models cluster at the same point of the plane AND
  `S(t)` is flat/featureless. Probability of this dull outcome ~ 0.25. If it happens, **widen
  the model set** (add IRASim / Wan2.2 5B / DreamDojo) and **harden the instruction ceiling**
  (stronger counterfactual instructions) to expose contrast -- usually one axis breaks the tie.

### Stage 4 -- Write-up & release
**Goal:** workshop paper + pip-installable harness + leaderboard, framed as a judge-free
horizon-resolved *complement* to MiraBench (Section 1.3 wedge).
**P(workshop accept | reached with an interesting result) ~ 0.65-0.75** at a robot-learning /
world-models workshop; **P(main-conference accept) ~ 0.25-0.3** (needs the strong spread plus
a MiraBench-correlation validation and/or a downstream-utility tie-in).

### Compounded outcome (calibrated)
- P(reach a publishable workshop result) ~ 0.85 x 0.90 x 0.70 x 0.75 ~ **0.40 to a *clean*
  positive**; because **the "universal gap" negative result is itself publishable** (and now
  triangulates MiraBench), the effective P(workshop-publishable artifact) is **~0.60-0.65**.
- P(main-conference-strength result) ~ **0.25-0.30**.

These match the TL;DR. The asymmetry to exploit: *a flat/null finding is not a failure here* --
it is direct evidence for the survey's central hypothesis, which is why the floor on
publishability is high.

---

## 9. Consolidated KILL/PIVOT thresholds (quick reference)

| Stage | Metric | KILL/PIVOT trigger | Action |
|---|---|---|---|
| 0 | env works | A100/license blocks >1 wk | switch primary to IRASim (pre-rendered) |
| 1 | reactivity | `S_negate ~ floor` at all `t` (floor = measured determinism `epsilon`) | swap to another action-input WM |
| 2 | null floor | `mean S_null > floor + 1 sd` | re-pad/redefine null; fix metric |
| 2 | monotonicity | Spearman rho < 0.4 after fixes | abandon pixel-space; consider latent |
| 2 | separation | `S_severe(T) < 3x` floor on all models | switch to `S^Delta` / DreamSim |
| 3 | contrast | flat `S(t)`, single plane cluster, all models | widen models + harder cf-instructions |
| any | timeline | a single stage exceeds 2x its budget | escalate / descope to workshop-only |

---

## 10. What to do *right now* (first concrete actions)

1. Spin up an A100 80GB (RunPod ~$1.19/hr) with the `nvidia-cosmos/cosmos-predict2` Docker
   image; accept the model license and download `Cosmos-Predict2-2B-Sample-Action-Conditioned`.
2. Run `examples/video2world_action.py` on a Bridge test clip + its `--input_annotation`
   action json; save `ohat`. **This single artifact clears Stage 0** (no LIBERO/EGL needed).
3. Programmatically edit the action json: `identity` (copy) and `negate` (flip the 6 motion
   dims). Re-run with a fixed seed; confirm `S_identity = 0`, `S_negate > 0`. **Clears Stage 1.**
4. Only then build `afp/metrics.py`, run the monotonicity ladder + seed-noise floor (Stage 2),
   and define the per-channel `sens_c` floors/ceilings (Section 2.2).

Do not write metric code, paper outlines, or the harness until Stage 0+1 are green. The
project's risk is concentrated in Stage 1 (is `a` causally wired?) and Stage 3 (assembling a
spread of action-conditioned models).

---

## 11. Primary references (with verification status)

- Hou et al., *World Model for Robot Learning: A Comprehensive Survey*, arXiv:2605.00080
  (Secs. 2.1.1, 5.3, 7.1, 8.1, 8.6). [verified]
- dWorldEval (**arXiv:2604.22152**): `Delta-LPIPS`, transition-space fidelity; baselines
  WorldEval/Ctrl-World/WorldGym on LIBERO+RoboTwin; r~0.9 to real success. **April 2026
  preprint, no confirmed venue; no public code.** [arXiv ID added; all 8 numbers + definition
  verified verbatim]
- AGRA / *Making Foresight Actionable* (2026 preprint): Matched Sensitivity Ratio; the
  load-bearing claim we use is null ~1.0. [qualitative claim verified; MSR is a
  region/segment-perturbation ratio, not a literal action-vs-null ratio -- borrow the principle]
- nwm (facebookresearch/nwm, **CVPR 2025**, public code): goal/action-conditioned navigation
  WM evaluated across horizons. [real + code verified; the canonical paper does NOT name a
  "Counterfactual Divergence / DreamSim @ 8-16s" metric -- that lives in a community fork
  `harsh-sutariya/nwm`, so the Rev-1 attribution pointed at the wrong repo]
- UWM-JEPA = **Unitary World Model** (arXiv 2605.25313; "Unitary", not "Unified"): prediction
  quality degrades monotonically with action-perturbation severity. [name + qualitative claim
  verified; fabricated 6-number ladder retracted]
- LIBERO-CF (vla-va.github.io): counterfactual instructions for VLA policies (public). [verified]
- **MiraBench (arXiv 2605.29360; NeurIPS submission, anonymous):** 3-level action-conditioned
  reliability (Physics Adherence / Action-Following Fidelity / Optimism-Bias Detection);
  906 videos, 16,704 human judgments; VLM judges (TCR via InternVL3-78B, multi-frame voting,
  >85% human agreement); 12 model configs incl. Cosmos-Predict2 + DreamDojo; GR-1 + Lingchu
  corpora. [every number/name/headline verified verbatim; venue is a submission, not accepted]
- **"How Should World Models Be Evaluated? A Decision-Making-Centric Position"
  (arXiv:2606.15032):** position/protocol paper; L0-L7 ladder; foregrounds "counterfactual
  action fidelity" + "full-horizon outcome fidelity." [verified; cite as conceptual parent /
  novelty boundary -- we implement its L4 test]
- World-in-World (**ICLR 2026 Oral; arXiv:2510.18135; OpenReview yDmb7xAfeb**): closed-loop
  WM-utility platform, unified action API; manipulation suite is **RLBench** (the Rev-3
  "LIBERO/RLBench" was wrong -- re-verified RLBench, not LIBERO). [venue VERIFIED -- the Rev-2
  "retracted, treat as preprint" note was wrong and is reversed]
- NVIDIA Cosmos Policy (NVlabs/cosmos-policy; arXiv:2601.16163; OpenReview wPEIStHxYH, under
  review): a *policy* that generates actions via latent-frame injection (LIBERO 98.5%). The
  `Cosmos-Policy-LIBERO-Predict2-2B` checkpoint (6.8 GB) is **action-emitting, not
  action-conditioned** -- not usable as our primary. [corrected]
- **`Cosmos-Predict2-2B-Sample-Action-Conditioned`** (nvidia-cosmos/cosmos-predict2; HF):
  video+action -> future, Bridge dataset, 7-dim action, ~32.5 GB, entry
  `examples/video2world_action.py`. **This is the primary.** [verified]
- LIBERO (`Lifelong-Robot-Learning/LIBERO`, pip `libero==0.1.1`) -- only if a LIBERO-domain
  action-input model is added. [verified]

**Rev 4 additions (from the problem-driven literature pass):**
- **(!) Concurrent-work flag -- *World Models as Group Actions* (arXiv:2605.24578, 2026):**
  independently formalizes action faithfulness via group identity / inverse / composition, uses
  the **identity (zero) action** as the invariance null ("zero-action drift"), and treats
  **stochastic-rollout divergence** as a robustness floor (GAR). It both validates our premise
  *and* overlaps it -- the single most important paper to position against. **Action item: verify
  and differentiate** (our wedge: training-free, *no ground truth*, counterfactual, horizon-
  resolved, action-vs-instruction plane on any released checkpoint). [full-text fetched; treat
  exact ID as lower-confidence until re-verified]
- World-in-World (ICLR 2026 Oral, arXiv:2510.18135) defines controllability as
  `1 - LPIPS(GT, pred | same actions)` -- **ground-truth-referenced, single-channel**; distinct
  from our counterfactual, GT-free, two-channel sensitivity. [verified via search]
- iWorld-Bench (arXiv:2605.03941) / WBench (arXiv:2605.25874): unified-action-API controllability
  benchmarks; report trajectory-accuracy vs ground truth and the text-vs-action controllability
  trade-off. Confirm the "action signals beat text for control" thesis; neither builds a
  sensitivity plane. [verified via search]
- "Evaluating Robot Policies in a World Model" (arXiv:2506.00613): action-conditioned video model
  for policy eval; metrics = agreement between GT and generated video under the *same* actions
  (GT-referenced). [verified via search]
- Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018): model-sensitivity sanity check
  -> framing for our reactivity gate (necessary, not sufficient). [verified]
- Veitch et al., *Counterfactual Invariance to Spurious Correlations* (arXiv:2106.00545, 2021):
  formal placebo/invariance basis for the corrected null. [verified]
- Zhang et al., *Emergence of Reproducibility and Consistency in Diffusion Models*
  (arXiv:2310.05264, ICLR 2024): seed-floor / determinism reference for stochastic WMs. [verified]
- Statistical rigor: Dror et al. *Hitchhiker's Guide* (ACL 2018), Bestgen (LREC 2022) -- bootstrap
  CIs, paired tests, Holm correction; gate on the CI lower bound. [verified]
- LPIPS robustness/OOD context: Zhang et al. (CVPR 2018, the `[-1,1]` RGB contract); E-LPIPS
  (arXiv:1906.03973); LipSim (ICLR 2024, DreamSim 2AFC collapses under small perturbations).
  Motion metrics: FVMD (ICML 2024 wkshp), STREAM (ICLR 2024). [verified]
- *Caveats:* "AGRA / Matched Sensitivity Ratio" could **not** be independently verified -- treat
  as unconfirmed and stop citing until a real source is found. Several 2026 arXiv IDs
  (MiraBench 2605.29360, the group-actions paper, dWorldEval 2604.22152) are **lower-confidence**;
  re-verify venue/ID before any submission. The "LPIPS specifically fails on *residual* images"
  claim is **reasoned inference** from the input contract + OOD fragility, not a measured result.

---

## 12. Changelog (Rev 1 -> Rev 2)

1. **(Critical) Primary checkpoint corrected.** Rev 1 anchored on `Cosmos-Policy-LIBERO`
   (6.8 GB) and conflated its action-*emitting* `get_action(...)` path with action
   *conditioning*. Rev 2 anchors on `Cosmos-Predict2-2B-Sample-Action-Conditioned` (Bridge,
   ~32.5 GB, A100). Cost story revised $150 -> $200-400; LIBERO/EGL risk removed on this route.
2. **(High) MiraBench promoted** from one table row to the closest prior work, with a precise
   judge-free / horizon-resolved / within-model wedge and a recommended correlation validation.
3. **(Medium) Citations corrected:** dWorldEval venue -> preprint; nwm metric-name + DreamSim
   protocol retracted; UWM-CF 6-number ladder retracted (kept qualitative monotonicity only).
4. **(Medium) `ARR` calibrated:** replaced the raw ratio with per-channel normalized
   sensitivities and a 2D sensitivity plane (Section 2.2).
5. **(Low) Venue claims downgraded** to preprint/submission throughout.

## 13. Changelog (Rev 2 -> Rev 3, after a two-track design review)

Two parallel reviews -- a stack/infrastructure viability pass and a full literature
re-verification (every citation re-checked against arXiv/OpenReview/HF/GitHub).

**Stack corrections:**
1. **Horizon scoped.** Primary checkpoint emits only **12 frames @ 640x480 @ 4 fps (~3 s)**;
   `S(t)` is `t=1..12` natively, long horizon needs autoregressive chaining (extension).
2. **Inference time fixed:** ~80 s/H100, ~100-150 s/A100 (the Rev-2 25-45 s was the GB200 row).
3. **Bridge data is a ~31 GB external download**, not bundled in the repo (+ a path mismatch to
   reconcile at Stage 0). "No simulator needed" still holds.
4. **Spread set corrected:** removed GR00T-Dreams (text-conditioned) and StarVLA/WM4A
   (action-emitting policy); confirmed IRASim / Ctrl-World(DROID) / Boundless / DreamDojo.
5. **Determinism gate relaxed** `S_identity=0` -> `< epsilon`; added bf16-only, fixed guidance
   (`--guidance 0`), pinned denoising steps, deterministic-algorithm flags.
6. **`S^Delta` flagged exploratory** (diff images are OOD for LPIPS; dWorldEval has no code).
7. **Budget re-bracketed** $200-400 -> **$300-550** (~200-270 A100-GPU-hr).

**Literature corrections:**
8. **World-in-World un-retracted:** it IS an ICLR 2026 Oral (arXiv:2510.18135) -- the Rev-2
   retraction was wrong; its "controllability > visual quality" finding supports our motivation.
9. **Added position paper arXiv:2606.15032** as the conceptual parent (counterfactual action
   fidelity / full-horizon) -- we implement its L4 test; drop "we identified the gap" rhetoric.
10. **dWorldEval arXiv:2604.22152 added**; all 8 Delta-LPIPS numbers re-confirmed verbatim.
11. **nwm fork nuance:** the "Counterfactual Divergence" idea lives in a fork, not Meta's repo.
12. **AGRA MSR** clarified as a region-perturbation ratio (borrow the null~1.0 principle only).

## 14. Changelog (Rev 3 -> Rev 4, after a reference build + problem-driven literature pass)

A working reference harness was built (synthetic mock world models with *known* sensitivity,
perturbation operators, metrics, scoring, and guarded real adapters). A skeptical judge review
surfaced six issues; a three-track literature pass (web + papers) checked whether each is a
project-killer, a known-fix, or a dev artifact. Verdict: **none are kill-risks.** Changes:

1. **(HIGH) Plane re-grounded as dose-response (Sec 2.2).** A single action operating point vs a
   single instruction operating point reintroduces "violent-vs-gentle" *across* channels. Fix:
   per-channel **dose-response sweep**, compare normalized **slope/AUC/monotonicity (shape, not
   level)**; if a scalar is needed, use the dose at a matched *fraction of each channel's own
   range*. (Mediation's standardized indirect effect; CFG controllability-fidelity sweep.)
2. **(HIGH/MED) De-circularized calibration (Sec 2.2).** Operating point/ceiling fixed a priori or
   on a held-out model split; raw `S(t)` ordering (calibration-free) is the primary evidence;
   synthetic oracles validate but are never tuned on. (Adebayo 2018; M4/SynScore 2023.)
3. **(MED) Null fixed -> action-group identity + sub-quantization (Sec 2.3).** The old "inert dims"
   null was an invalid placebo (`dz` is real motion on a robot). New null = zero/identity action
   + sub-quantization micro-edit, measured against the seed/determinism floor. (Group-actions
   2605.24578; counterfactual invariance, Veitch 2021; diffusion seed-floor, Zhang 2023.)
4. **(MED) `S^Delta` range bug + alternatives (Sec 2.1).** Remap diffs to `[0,1]` before perceptual
   backends; default to pixel_l2; prefer motion-explicit metrics (flow EPE / FVMD / STREAM) or the
   Delta of frame-level `S(t)`; keep `S^Delta` exploratory.
5. **(MED) Gates upgraded (Sec 2.4, Stage 2).** Separation gates on the **bootstrap CI lower
   bound** (not a point estimate); added an explicit **reactivity** gate framed as Adebayo's
   model-sensitivity sanity check. (Dror et al. 2018; Bestgen 2022.)
6. **(MED) Engineering (harness).** CLI must register real-adapter ids and **skip-if-unavailable**
   (filter on `available()`); pure dev, not a research issue.
7. **(!) Concurrent work flagged (Sec 11).** *World Models as Group Actions* (2605.24578) overlaps
   the premise (identity null, stochastic floor, "action faithfulness"). Our surviving wedge:
   training-free, **no ground truth**, counterfactual, horizon-resolved, **action-vs-instruction
   plane** on any released checkpoint -- which no surveyed benchmark builds. Verify + differentiate.
8. **Citation hygiene.** "AGRA / Matched Sensitivity Ratio" is **unverified** -- stop citing until
   sourced; several 2026 IDs are lower-confidence; the residual-LPIPS failure is inference.

