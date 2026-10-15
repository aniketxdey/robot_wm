"""Build a PDF literature review on action-faithful video world models."""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER

OUT = "/Users/aniketdey/Documents/robotwm/lit_review_action_faithful_wm.pdf"

# ---- styles ----
ss = getSampleStyleSheet()
S_TITLE = ParagraphStyle("Title", parent=ss["Title"], fontSize=13.5, leading=15.5,
                          spaceAfter=2, alignment=TA_CENTER)
S_SUB = ParagraphStyle("Sub", parent=ss["BodyText"], fontSize=9.5, leading=11,
                        spaceAfter=8, alignment=TA_CENTER, textColor=colors.grey)
S_H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=11.5, leading=13,
                       spaceBefore=8, spaceAfter=3, textColor=colors.black)
S_H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.2, leading=12,
                       spaceBefore=5, spaceAfter=2, textColor=colors.black)
S_BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.2, leading=11.4,
                         alignment=TA_JUSTIFY, spaceAfter=3)
S_TIGHT = ParagraphStyle("Tight", parent=S_BODY, spaceAfter=1, leftIndent=10,
                          bulletIndent=0)
S_FORMULA = ParagraphStyle("Formula", parent=S_BODY, fontName="Helvetica-Oblique",
                            leftIndent=18, spaceAfter=4, fontSize=9.5)
S_CELL = ParagraphStyle("Cell", parent=S_BODY, fontSize=8.2, leading=9.8,
                         alignment=TA_LEFT, spaceAfter=0)
S_CELLB = ParagraphStyle("CellB", parent=S_CELL, fontName="Helvetica-Bold")

flow = []  # the document

def H1(t): flow.append(Paragraph(t, S_H1))
def H2(t): flow.append(Paragraph(t, S_H2))
def P(t):  flow.append(Paragraph(t, S_BODY))
def F(t):  flow.append(Paragraph(t, S_FORMULA))
def SP(h=4): flow.append(Spacer(1, h))

# =========================== TITLE ============================
flow.append(Paragraph(
    "Action-Faithful Video World Models for Robotics",
    S_TITLE))
flow.append(Paragraph(
    "A focused literature review for an action-counterfactual sensitivity benchmark "
    "(Option A). Compiled from Hou et al., <i>World Model for Robot Learning: A "
    "Comprehensive Survey</i> (arXiv:2605.00080) and the works it cites.",
    S_SUB))

# =========================== PAGE 1: DEFINITIONS ============================
H1("1. Definitions, in build-up order")
P("Each term below uses only the terms defined above it. Read top-to-bottom.")

defs = [
 ("Observation o<sub>t</sub> / state x<sub>t</sub>",
  "Sensory input at time t (typically RGB image, often multi-view). The "
  "<i>state</i> is the idealized variable that fully characterizes the system; "
  "in practice we use observations as a proxy. Throughout, x<sub>t</sub> &asymp; o<sub>t</sub>."),
 ("Low-level action a<sub>t</sub>",
  "Per-step motor command (joint torques, end-effector velocities, gripper "
  "open/close). Dense (~10&ndash;30&nbsp;Hz). One value per control timestep."),
 ("High-level action / instruction l",
  "Sparse, episode-level task specification, usually a natural-language "
  "string. A single l is compatible with infinitely many a-trajectories "
  "(e.g. \"pick up the red block\" admits many grasps and paths)."),
 ("Trajectory &tau;",
  "&tau; = (o<sub>0</sub>, a<sub>0</sub>, o<sub>1</sub>, a<sub>1</sub>, &hellip;, "
  "o<sub>T</sub>). <i>Real</i> if produced by a physical / simulated environment; "
  "<i>imagined</i> if produced by a world model."),
 ("Policy &pi;<sub>&theta;</sub>(a<sub>t</sub> | o<sub>t</sub>, l)",
  "Parametric map from observation and instruction to motor command. "
  "Modern VLA policies internalize l directly."),
 ("Inverse Dynamics Model (IDM) &phi;(a<sub>t</sub> | o<sub>t</sub>, o<sub>t+1</sub>)",
  "Recovers the action that explains a transition. Used by several benchmarks "
  "to score whether a generated future is action-realizable."),
 ("World model (general)",
  "p<sub>&phi;</sub>(x<sub>t+1:t+H</sub> | x<sub>t</sub>, a<sub>t:t+H-1</sub>, l). "
  "A predictive model of agent&ndash;environment dynamics. Survey Eq. 1."),
 ("Video world model",
  "World model with x<sub>t</sub>=o<sub>t</sub> in pixel space; output is a "
  "future video. Most current robotic WMs are video WMs."),
 ("Action-conditioned world model (AC-WM)",
  "A WM whose output meaningfully depends on a. The survey calls this the "
  "&quot;most important subclass&quot; because action-inconsistent futures are "
  "useless for closed-loop control (lines 270&ndash;271)."),
 ("Latent (JEPA-style) world model",
  "Predicts in embedding space rather than pixels: &zcirc;<sub>t+1</sub> = "
  "f(z<sub>t</sub>, a<sub>t</sub>) with z<sub>t</sub> = E(o<sub>t</sub>). "
  "Cheaper but harder to inspect."),
 ("Open-loop rollout",
  "Given o<sub>0</sub> and a<sub>0:H-1</sub>, generate &ocirc;<sub>1:H</sub> "
  "in one shot. No feedback from generated frames into action selection."),
 ("Closed-loop rollout",
  "WM is plugged into a control loop; at each step the policy reacts to the "
  "WM's prediction. Compounding errors dominate."),
 ("Action faithfulness",
  "Property that &ocirc;<sub>1:H</sub> changes <i>correctly</i> when a is "
  "perturbed&mdash;i.e., the model has internalized p(&middot; | a, &middot;) "
  "rather than p(&middot; | l, &middot;) alone."),
 ("Causal conditioning gap",
  "The survey's &sect;8.1 term for the failure mode in which a WM produces "
  "intention-consistent (driven by l) but action-inconsistent futures."),
 ("Counterfactual sensitivity",
  "Quantitative measure of action faithfulness: how rollout output varies "
  "under controlled perturbations of a, with o<sub>0</sub> and l fixed. "
  "<b>This is the variable Option A measures.</b>"),
]
for term, body in defs:
    P("<b>" + term + ".</b> " + body)

H1("2. Key formulas")
P("Numbered for reference in the paper reviews and gap analysis below.")

F("(F1) WM factorization (Survey Eq.&nbsp;1): "
  "p(x<sub>t+1:t+H</sub> | x<sub>t</sub>, a<sub>t:t+H-1</sub>, l).")
F("(F2) Imagined-rollout joint: p(&tau;) = &pi;<sub>&theta;</sub>(a<sub>0</sub>|o<sub>0</sub>,l) "
  "&prod;<sub>t</sub> p<sub>&phi;</sub>(o<sub>t+1</sub>|o<sub>&le;t</sub>, a<sub>&le;t</sub>, l) "
  "&pi;<sub>&theta;</sub>(a<sub>t+1</sub>|o<sub>t+1</sub>,l).")
F("(F3) GRPO objective for WM-as-simulator RL (Survey Eq.&nbsp;18): "
  "L<sub>RL</sub>(&theta;) = &minus;E<sub>t</sub>[ min(r<sub>t</sub>(&theta;)&Acirc;<sub>t</sub>, "
  "clip(r<sub>t</sub>(&theta;),1&minus;&epsilon;,1+&epsilon;)&Acirc;<sub>t</sub>) ].")
F("(F4) <b>Pixel action sensitivity (proposed):</b> "
  "S(t; a, a&prime;) = E<sub>o<sub>0</sub>,l</sub>[ || &ocirc;<sub>t</sub>(o<sub>0</sub>, a, l) "
  "&minus; &ocirc;<sub>t</sub>(o<sub>0</sub>, a&prime;, l) ||<sub>2</sub> ]. "
  "Larger ⇒ more action-faithful when (a&prime;&minus;a) is a meaningful perturbation.")
F("(F5) <b>Counterfactual rank fidelity:</b> "
  "&rho;<sub>CF</sub> = Spearman( rank<sub>sim</sub>{c(&tau;<sub>k</sub><sup>real</sup>)}, "
  "rank<sub>WM</sub>{c(&tau;<sub>k</sub><sup>WM</sup>)} ) over candidate actions "
  "{a<sub>k</sub>} with cost c(&middot;). Generalizes WorldEval to per-action "
  "counterfactuals.")
F("(F6) <b>IDM-recoverability</b> (WorldSimBench, WoW-World-Eval style): "
  "R = E[ || a &minus; &phi;(&ocirc;<sub>t</sub>, &ocirc;<sub>t+1</sub>) ||<sub>2</sub> ].")
F("(F7) <b>a-vs-l reliance gap (proposed):</b> "
  "&Delta;<sub>a</sub> = E ||&ocirc;(a,l) &minus; &ocirc;(a&prime;,l)|| (random a&prime;); "
  "&Delta;<sub>l</sub> = E ||&ocirc;(a,l) &minus; &ocirc;(a,l&prime;)|| (paraphrased l&prime;). "
  "<b>Action-Reliance Ratio</b> ARR = &Delta;<sub>a</sub>/&Delta;<sub>l</sub>; "
  "ARR&gt;1 ⇒ action-faithful, ARR&lt;1 ⇒ instruction-driven.")

# =========================== PAGE 2-3: PAPER REVIEWS ============================
H1("3. Paper reviews (only what matters for Option A)")

H2("3.1 Foundations &mdash; concept anchors")
P("<b>Ha &amp; Schmidhuber (2018), &quot;Recurrent World Models Facilitate Policy "
  "Evolution.&quot;</b> Origin of the modern term. Trains a recurrent latent dynamics "
  "model (V&ndash;M&ndash;C: vision, memory, controller) and shows that policies "
  "trained <i>inside imagined rollouts</i> of the learned dynamics transfer to the "
  "real environment. <b>Take-away:</b> the legitimacy of evaluating policies in "
  "imagined rollouts depends entirely on whether the WM tracks action consequences "
  "&mdash; the unstated assumption your benchmark interrogates.")
P("<b>Hafner et&nbsp;al., DreamerV3 (2023).</b> Canonical model-based RL with a "
  "Recurrent State-Space Model (RSSM); shows scaling/robustness across hundreds of "
  "tasks. <b>Take-away:</b> establishes the latent-state baseline; recent video WMs "
  "are pixel-space generalizations, but they have not inherited Dreamer's careful "
  "action-conditioning evaluation.")
P("<b>Yang et&nbsp;al., UniSim (ICLR 2024).</b> First widely-cited paper to frame "
  "video generation as an interactive simulator: p(video<sub>t+1</sub> | "
  "video<sub>t</sub>, action). <b>Take-away:</b> defines the genre your benchmark "
  "evaluates, but offers only appearance- and task-success-style metrics.")
P("<b>Assran et&nbsp;al., V-JEPA&nbsp;2 (2025) &amp; Mur-Labadia et&nbsp;al., "
  "V-JEPA&nbsp;2.1 (2026).</b> Latent self-supervised video models; V-JEPA&nbsp;2.1 "
  "shows zero-shot robot planning with image goals via latent action-conditioned "
  "rollout. <b>Take-away:</b> a strong counter-argument to pixel WMs. If your "
  "benchmark shows pixel WMs are not action-faithful, JEPA-style models become the "
  "natural alternative; cite as the contrast class.")

H2("3.2 Action-controllable video world models &mdash; your probe subjects")
P("<b>IRASim (Zhu et&nbsp;al. 2025).</b> Trajectory-to-video formulation. Innovation: "
  "<i>frame-level action conditioning</i> &mdash; each transformer block injects the "
  "action of <i>that</i> frame rather than a single global condition. Trained on "
  "Bridge / RT-1; code released. <b>Strength:</b> the cleanest published architecture "
  "for action-faithful video. <b>Limitation:</b> evaluation uses FVD + a small "
  "in-paper visual comparison, not a controlled counterfactual probe. <b>Probe priority: HIGH.</b>")
P("<b>Ctrl-World (Guo et&nbsp;al. 2026, ICLR).</b> Designed <i>for policy evaluation</i>: "
  "multi-view prediction, frame-level action control, memory module for long-horizon "
  "consistency. Shows that action-faithful rollouts can rank policies. "
  "<b>Strength:</b> author claims action faithfulness <i>and</i> measures policy-rank "
  "preservation &mdash; the closest published precedent for what you want to do. "
  "<b>Limitation:</b> author-defined metrics, single benchmark, no perturbation suite. "
  "<b>Probe priority: HIGH; primary baseline.</b>")
P("<b>Vid2World (Huang et&nbsp;al. 2026, ICLR).</b> Recipe for converting any "
  "pretrained video diffusion model into an interactive WM via causal masking + "
  "action-conditioning layers. <b>Strength:</b> plug-and-play, leverages public "
  "video-diffusion checkpoints. <b>Limitation:</b> conversion does not <i>verify</i> "
  "that the resulting model is action-sensitive. <b>Probe priority: MEDIUM.</b>")
P("<b>EnerVerse-AC (Jiang et&nbsp;al. 2025).</b> Multi-view action-conditional "
  "generator that doubles as a data engine + evaluator. <b>Strength:</b> evaluation "
  "utility is a first-class objective. <b>Limitation:</b> reuses task-success metrics. "
  "<b>Probe priority: MEDIUM.</b>")
P("<b>Cosmos Predict 2.5 / Cosmos Policy (NVIDIA, Ali et&nbsp;al. 2025; Kim et&nbsp;al. "
  "2026).</b> Foundation-scale (multi-billion-parameter) video WM with public "
  "checkpoints; Cosmos Policy fine-tunes for visuomotor control. <b>Strength:</b> "
  "size + availability. <b>Limitation:</b> trained primarily from text/task signal "
  "&mdash; the prime suspect for high-l, low-a behavior. <b>Probe priority: HIGH "
  "for foundation-scale comparison.</b>")
P("<b>Genie Envisioner (Liao et&nbsp;al. 2026).</b> Unified world-foundation "
  "platform integrating video WM + action decoding. <b>Strength:</b> conceptual peer "
  "of Cosmos. <b>Limitation:</b> same training-signal bias. <b>Probe priority: LOW</b> "
  "(use as confirmation, not primary subject).")

H2("3.3 Existing evaluation benchmarks &mdash; your prior art")
P("<b>WorldEval (Li et&nbsp;al. 2025).</b> Asks whether a WM preserves the relative "
  "ranking of policies in real vs. imagined rollouts. <b>Strength:</b> directly "
  "measures decision-utility. <b>Limitation:</b> confounds policy quality with WM "
  "quality &mdash; a WM that ignores a entirely could still preserve ranking if all "
  "policies in the test set are similar in dynamics. No isolated action probe.")
P("<b>WorldGym (Quevedo et&nbsp;al. 2025).</b> Treats WM as a Monte-Carlo evaluation "
  "environment; checks whether estimated policy <i>value</i> matches real value. "
  "<b>Limitation:</b> aggregate value over an episode; no per-step or per-action "
  "diagnostic, no controlled perturbation.")
P("<b>WorldSimBench (Qin et&nbsp;al. 2025).</b> Two-track: (i) perceptual fidelity, "
  "(ii) <i>manipulative evaluation</i> via inverse-dynamics recovery (F6) &mdash; "
  "given &ocirc;<sub>t:t+1</sub>, run an IDM, check whether recovered actions match "
  "input. <b>Closest existing precedent</b> for action faithfulness. <b>Limitation:</b> "
  "uses a single fixed IDM; tests whether <i>some</i> action could explain the "
  "rollout, not whether the rollout responded to <i>this specific</i> action.")
P("<b>WorldArena (Shang et&nbsp;al. 2026).</b> Multi-role evaluation: synthetic data "
  "generation, policy evaluation, action planning. <b>Limitation:</b> each role "
  "scored separately; no isolated causal-conditioning probe.")
P("<b>World-in-World (Zhang et&nbsp;al. 2025).</b> Closed-loop benchmark; reports "
  "end-task success when WM is plugged into a planner. <b>Limitation:</b> compounded "
  "failures invisible &mdash; if the WM is action-blind but the policy is robust, "
  "you cannot tell.")
P("<b>WoW-World-Eval / &quot;Wow, wo, val&quot; (Fan et&nbsp;al. 2026).</b> "
  "Comprehensive embodied WM Turing-style test, includes IDM-based executability. "
  "<b>Closest competitor.</b> <b>Limitation:</b> holistic Turing judgment; no "
  "factor-resolved sensitivity curve and no controlled action-perturbation suite.")
P("<b>WM-ABench (Gao et&nbsp;al. 2025, &quot;Do VLMs have internal world "
  "models?&quot;).</b> Decomposes evaluation into <i>atomic</i> capabilities: spatial, "
  "temporal, mechanistic, counterfactual. <b>Methodologically the closest</b> to "
  "what you want, but applied to <b>VLMs, not video WMs.</b> Transferring the "
  "atomic-capability framework to video WMs is a publishable framing on its own.")
P("<b>EWMBench, RBench, DreamGen-Bench, EVA-Bench.</b> Open-loop predictive "
  "benchmarks with embodied flavor (scene/motion/semantic factors; "
  "structure+physics+action-completeness; instruction following; long-horizon OOD). "
  "<b>Common limitation:</b> ultimately appearance-driven; action sensitivity not "
  "isolated as a primary axis.")

H2("3.4 Critique allies &mdash; your motivation")
P("<b>WorldVLA (Cen et&nbsp;al. 2025).</b> Unified VLA + WM. The authors "
  "<i>explicitly diagnose</i> &quot;weak action conditioning&quot; as the bottleneck "
  "and add a unified-training fix on the <i>training</i> side. Your contribution "
  "complements theirs on the <i>evaluation</i> side.")
P("<b>WoVR (Jiang et&nbsp;al. 2026).</b> Treats WM reliability as the central "
  "bottleneck in WM-as-simulator RL. Key claim: hallucination &quot;does not merely "
  "reduce visual quality, but can directly corrupt the assessment signal itself&quot; "
  "(survey lines 1027&ndash;1029). Quote-worthy.")
P("<b>EVA (Wang et&nbsp;al. 2026).</b> Aligns video WMs with executable robot "
  "actions via inverse-dynamics rewards. Frames the issue as the &quot;executability "
  "gap.&quot;")
P("<b>Causal WM (Li et&nbsp;al. 2026, arXiv:2601.21998).</b> Explicitly causal "
  "framing of WM training. Cite for terminology.")
P("<b>Pan et&nbsp;al. 2026, &quot;Much ado about noising.&quot;</b> Skeptical "
  "position paper challenging diffusion-based generative robot control hype. Cite "
  "as the balancing voice; demonstrates you are aware of the controversy.")

# =========================== PAGE 4: GAP MATRIX ============================
H1("4. Gap matrix")
P("Compact strengths / limitations / pointed-to extensions for every paper above. "
  "Patterns repeat &mdash; that's the point of the synthesis below.")

def cellP(text, style=S_CELL):
    return Paragraph(text, style)

gap_header = [cellP("<b>Paper</b>", S_CELLB),
              cellP("<b>Role</b>", S_CELLB),
              cellP("<b>Strength</b>", S_CELLB),
              cellP("<b>Key limitation</b>", S_CELLB),
              cellP("<b>Implied extension</b>", S_CELLB)]

gap_rows = [
 ("Ha &amp; Schmidhuber 18", "Found.",
  "Imagined rollouts can train policies",
  "Latent only; toy domains; no faithfulness audit",
  "Pixel-space + action-faithfulness audit"),
 ("DreamerV3", "Found.",
  "Stable model-based RL with RSSM",
  "Small latent state; no video extension",
  "Video-scale faithfulness benchmark"),
 ("UniSim", "Found.",
  "Frames video gen as interactive simulator",
  "FVD / task-success metrics only",
  "Counterfactual action probes"),
 ("V-JEPA 2.1", "Found.",
  "Latent action-cond. zero-shot planning",
  "No head-to-head with pixel WMs",
  "Cross-modality faithfulness comparison"),
 ("IRASim", "AC-WM",
  "Frame-level action conditioning",
  "Single-view, single benchmark",
  "Standardized counterfactual eval"),
 ("Ctrl-World", "AC-WM",
  "Multi-view + memory + author rank-test",
  "Author-defined metric; no perturb suite",
  "External standardized benchmark"),
 ("Vid2World", "AC-WM",
  "Recipe to convert any video diffusion WM",
  "Conversion not verified action-sensitive",
  "Post-conversion sensitivity audit"),
 ("EnerVerse-AC", "AC-WM",
  "Eval-aware multi-view design",
  "Reuses task-success metrics",
  "Atomic action-cond. metrics"),
 ("Cosmos Predict / Policy", "AC-WM",
  "Foundation scale, public checkpoints",
  "Trained mostly from text/task signal",
  "<i>a</i>-vs-<i>l</i> disentanglement test"),
 ("Genie Envisioner", "AC-WM",
  "Unified WM+action platform",
  "Same training-signal bias",
  "Same a-vs-l probe"),
 ("WorldEval", "Eval",
  "Real-vs-imagined policy ranking",
  "Confounds policy and WM quality",
  "Decoupled WM-only causal probe"),
 ("WorldGym", "Eval",
  "MC value-fidelity test",
  "Aggregate; no per-step diagnostic",
  "Per-action counterfactual"),
 ("WorldSimBench", "Eval",
  "IDM-recovery test (closest precedent)",
  "Single fixed IDM; not a perturbation test",
  "Multi-IDM ensemble + controlled perturb."),
 ("WorldArena", "Eval",
  "Multi-role functional evaluation",
  "No isolated causal probe",
  "Cross-role causal diagnostic"),
 ("World-in-World", "Eval",
  "Closed-loop end-task success",
  "Compound failures hide root cause",
  "Decomposed mid-loop diagnostic"),
 ("WoW-World-Eval", "Eval",
  "Holistic IDM Turing test",
  "Not factor-resolved over horizon",
  "Sensitivity curve S(t) over time"),
 ("WM-ABench", "Eval (VLM)",
  "Atomic capability decomposition",
  "VLMs not video WMs",
  "Port atomic decomposition to video WMs"),
 ("EWMBench / RBench / DG-Bench / EVA-Bench",
  "Eval (open-loop)",
  "Embodied open-loop predictive scoring",
  "Still appearance-driven aggregates",
  "Action-isolated factor in the score"),
 ("WorldVLA", "Critic",
  "Diagnoses weak action-conditioning",
  "Fix is in training, not in eval",
  "Eval-side benchmark of weak a-cond"),
 ("WoVR", "Critic",
  "Reliability is central bottleneck",
  "Reliability not directly measured",
  "Direct reliability probe"),
 ("EVA", "Critic",
  "Frames executability gap explicitly",
  "Aligns via training only",
  "Eval-side executability metric"),
 ("Causal WM (Li 26b)", "Critic",
  "Causal framing of WM training",
  "No empirical eval of non-causal WMs",
  "Empirical causal-deficiency benchmark"),
 ("Pan et&nbsp;al. 26", "Critic",
  "Skeptic position on diffusion control",
  "Not a benchmark",
  "Quantitative test of skeptic claims"),
]
data = [gap_header] + [
    [cellP(r[0]), cellP(r[1]), cellP(r[2]), cellP(r[3]), cellP(r[4])]
    for r in gap_rows
]
col_widths = [1.18*inch, 0.62*inch, 1.55*inch, 1.65*inch, 1.85*inch]
tbl = Table(data, colWidths=col_widths, repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8e8e8")),
    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 3),
    ("RIGHTPADDING", (0,0), (-1,-1), 3),
    ("TOPPADDING", (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
]))
flow.append(tbl)

# =========================== PAGE 5: SYNTHESIS ============================
H1("5. Cross-paper patterns (the recurring story)")

P("<b>Pattern 1 (across all six AC-WMs).</b> Every action-controllable video WM "
  "<i>asserts</i> action-conditioning is essential and adds an architectural "
  "mechanism for it (frame-level injection, causal mask, multi-view, memory). "
  "<b>None verifies the mechanism worked with a controlled action counterfactual.</b> "
  "Verification is left to FVD-like appearance scores or downstream task success.")

P("<b>Pattern 2 (across all eight evaluation benchmarks).</b> Every benchmark "
  "measures an <i>aggregate</i> quantity &mdash; success rate, value, rank "
  "correlation, FVD, IDM-recovery error &mdash; computed over whole episodes or "
  "whole datasets. <b>None holds (o<sub>0</sub>, l) fixed and perturbs a in a "
  "controlled way.</b> None reports sensitivity as a function of horizon t. None "
  "uses an ensemble of IDMs to robustify the recovery test against IDM idiosyncrasy.")

P("<b>Pattern 3 (across all five critic papers).</b> Everyone agrees weak action "
  "conditioning is the central bottleneck. The fixes are uniformly on the "
  "<b>training side</b> (unified objectives, IDM-reward alignment, causal losses). "
  "<b>No one provides a standardized, model-agnostic, training-free benchmark for "
  "measuring how weak the conditioning is on a given checkpoint.</b>")

P("<b>Pattern 4 (across all foundations).</b> The legitimacy of every world-model "
  "use case &mdash; imagination-based RL, policy evaluation, plan ranking &mdash; "
  "rests on the assumption that the WM tracks action consequences. <b>That "
  "assumption has never been directly stress-tested at the level of individual "
  "checkpoints.</b>")

P("<b>The gap, in one sentence.</b> Across foundations, AC-WMs, eval benchmarks, "
  "and critics, the missing artifact is a <i>training-free, controlled, "
  "horizon-resolved counterfactual probe of action sensitivity</i> that any "
  "released checkpoint can be scored on in an afternoon. The community needs "
  "this; nobody has built it.")

H1("6. Two specific, implementable gaps for Option A")

H2("Gap 1 &mdash; Action-Counterfactual Sensitivity Benchmark (ACSB)")
P("<b>What it measures.</b> Hold (o<sub>0</sub>, l) fixed; for each scripted scene "
  "with ground-truth action a, run four perturbations: "
  "(a) <i>reverse</i>, (b) <i>zero</i>, (c) <i>random Gaussian</i> matched to a's "
  "marginal, (d) <i>swapped</i> &mdash; the action sequence from a different scene "
  "with the same instruction. Roll out each in the WM and the simulator. Report:")
P("&bull;&nbsp;<b>S(t)</b> from F4 as a function of horizon t. Curve, not scalar.")
P("&bull;&nbsp;<b>Pearson(WM-divergence, sim-divergence)</b> across perturbations: "
  "does the WM diverge in the directions and magnitudes the simulator does?")
P("&bull;&nbsp;<b>Counterfactual rank fidelity &rho;<sub>CF</sub></b> from F5 "
  "across the four variants.")
P("<b>Why this is the right gap.</b> It directly answers the question every paper "
  "in the gap matrix dances around: <i>does this checkpoint actually use a?</i> "
  "It is training-free (inference only), model-agnostic (works on any WM with an "
  "action input), and produces a single standardized score that can be reported "
  "alongside FVD.")
P("<b>Concrete scope (6 weeks, one A100).</b> Scenes: 30 from LIBERO-Long "
  "(language-conditioned, long horizon, ground-truth simulator). Subjects: "
  "Ctrl-World, IRASim, Cosmos Policy. Perturbations: 4 above. Seeds: 5. "
  "Total rollouts: 30 &times; 3 &times; 4 &times; 5 = <b>1,800</b>. Easily fits.")
P("<b>Strongest version of the result.</b> If foundation-scale WMs (Cosmos) show "
  "low S(t) for t&gt;1s while smaller dedicated AC-WMs (IRASim, Ctrl-World) show "
  "high S(t), you have a publishable finding: <i>scaling the video backbone does "
  "not buy you action faithfulness</i>. That headline alone clears the bar for "
  "the ICLR World Models workshop or CoRL 2026 workshop.")

H2("Gap 2 &mdash; Action-Reliance Ratio (ARR), F7")
P("<b>What it measures.</b> Hold one input fixed and perturb the other, with the "
  "natural perturbation in each case: random a&prime; for a; LLM-paraphrased "
  "l&prime; for l. ARR = &Delta;<sub>a</sub>/&Delta;<sub>l</sub>. <b>ARR&gt;1</b> "
  "means the model uses a more than l (action-faithful); <b>ARR&lt;1</b> means it "
  "uses l more than a (instruction-driven; the survey's &sect;8.1 failure mode).")
P("<b>Why this complements ACSB.</b> ACSB asks <i>how much</i> the model responds "
  "to a. ARR asks <i>which signal it relies on more</i>. Together they distinguish "
  "&quot;action-blind&quot; from &quot;instruction-driven&quot;: a model can be "
  "responsive to action perturbations <i>and</i> still be more responsive to "
  "instruction paraphrases &mdash; that's a subtle failure ACSB alone misses.")
P("<b>Concrete scope.</b> Same scenes/subjects as ACSB. Add 5 LLM paraphrases per "
  "instruction (Claude or GPT-4 API; trivial). Same total rollouts.")

H2("Recommended order")
P("Build ACSB first (Gap 1). It has a cleaner story (one axis of variation, one "
  "headline number). Reserve ARR (Gap 2) for the extended-paper version. Pitch to "
  "ICLR 2027 World Models Workshop (CFP usually opens November/December); CoRL "
  "2026 robot-learning workshops are the fallback. Both accept 4&ndash;8-page "
  "submissions, which is the right size for a single benchmark contribution.")

# ---- BUILD ----
doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    leftMargin=0.7*inch, rightMargin=0.7*inch,
    topMargin=0.6*inch, bottomMargin=0.6*inch,
    title="Action-Faithful Video World Models: Lit Review",
    author="Compiled for Option A")
doc.build(flow)
print("Wrote", OUT)
