# AFP Leaderboard

Action-Faithfulness Probe results. Sorted by action sensitivity at the final horizon (higher = the predicted future tracks the *action* more).

| model | verdict | gain_action | gain_instruction | sens_action (share) | sens_instruction (share) | monotonicity rho | sep. ratio | gates |
|---|---|---|---|---|---|---|---|---|
| `mock-faithful` | action-faithful | 0.0543 | 0.0225 | 0.663 | 0.337 | 0.961 | 61.79 | 4/4 |
| `mock-balanced` | instruction-driven | 0.0485 | 0.0637 | 0.406 | 0.594 | 0.965 | 49.40 | 4/4 |
| `mock-instruction` | instruction-driven | 0.0131 | 0.0647 | 0.185 | 0.815 | 0.951 | 18.43 | 4/4 |

## Legend

- **gain_action / gain_instruction** -- the headline dose-response gains (DESIGN Rev 4, Sec 2.2): the slope of S vs *normalized perturbation dose* for each channel (action dose = gauss sigma ladder; instruction dose = interpolation toward the counterfactual). Both are in the same units (delta-distance per unit normalized dose), so the (gain_action, gain_instruction) plane is the directly comparable readout.
- **sens_action / sens_instruction (share)** -- per-horizon dominance share `gain_c / (gain_action + gain_instruction)` in [0, 1]: which channel drives the predicted future. (These two sum to ~1; use the raw gains for the 2D plane.)
- **verdict** -- dominance on the dose-response plane: *action-faithful* (action gain dominates), *instruction-driven* (instruction gain dominates), *balanced* (comparable), *inert* (both gains near the noise floor -- under-conditioned).
- **monotonicity rho** -- Spearman of S vs perturbation severity (gate >= 0.7).
- **sep. ratio** -- severe-edit S(T) / seed-floor(T); the gate uses the bootstrap-CI lower bound >= 3x (DESIGN Rev 4).
- **gates** -- Stage-2 GO/KILL checks passed (reactivity, null-floor, monotonicity, separation).
