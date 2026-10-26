"""Command-line interface for the Action-Faithfulness Probe.

    afp gen-data --out data/ --n 24
    afp run --models mock --distance pixel_l2 --data data/ --out results/ --seeds 5
    afp backends

Wired to ``pyproject``'s ``afp = "afp.cli:main"`` console entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from afp import available_backends
from afp.data import (
    load_episodes,
    load_paraphrases,
    make_synthetic_dataset,
    save_episodes,
    save_paraphrases,
)
from afp.rollout import WorldModelAdapter

EPISODES_FILE = "episodes.jsonl"
PARAPHRASES_FILE = "paraphrases.jsonl"


# ===========================================================================
# gen-data
# ===========================================================================
def cmd_gen_data(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    episodes, specs = make_synthetic_dataset(
        n_episodes=args.n, horizon=args.horizon, size=args.size, seed=args.seed,
    )
    ep_path = os.path.join(args.out, EPISODES_FILE)
    pp_path = os.path.join(args.out, PARAPHRASES_FILE)
    save_episodes(episodes, ep_path)
    save_paraphrases(specs, pp_path)
    print(f"wrote {len(episodes)} episodes  -> {ep_path}")
    print(f"wrote {len(specs)} paraphrase specs -> {pp_path}")
    return 0


# ===========================================================================
# run
# ===========================================================================
def _real_adapter_registry() -> dict:
    """Map real-adapter ids -> zero-arg constructors (imported lazily/guarded).

    Real adapters import cleanly without torch/weights and report
    ``available() is False`` until their env vars are set (see ``afp/adapters``).
    """
    registry: dict = {}
    try:
        from afp.adapters.cosmos_sac import CosmosSACAdapter
        registry["cosmos-sac-2b"] = CosmosSACAdapter
    except Exception:
        pass
    try:
        from afp.adapters.irasim import IRASimAdapter
        registry["irasim-bridge"] = IRASimAdapter
    except Exception:
        pass
    try:
        from afp.adapters.ctrl_world import CtrlWorldAdapter
        registry["ctrl-world-droid"] = CtrlWorldAdapter
    except Exception:
        pass
    return registry


def _resolve_adapters(model_spec: str) -> List[WorldModelAdapter]:
    """Resolve ``--models`` into adapter instances.

    Accepts ``mock`` (the full mock suite), ``real`` (all registered real
    checkpoints), or a comma-separated list of specific mock/real ids. Unknown
    ids are reported and skipped.
    """
    from afp.adapters.mock import make_mock_suite

    suite = make_mock_suite()
    by_id = {a.model_id: a for a in suite}
    real = _real_adapter_registry()
    aliases = {"cosmos-sac": "cosmos-sac-2b", "irasim": "irasim-bridge", "ctrl-world": "ctrl-world-droid"}

    spec = (model_spec or "mock").strip().lower()
    if spec in ("mock", "mocks"):
        return suite
    if spec == "real":
        return [ctor() for ctor in real.values()]
    if spec == "all":
        return suite + [ctor() for ctor in real.values()]

    out: List[WorldModelAdapter] = []
    for token in spec.split(","):
        token = aliases.get(token.strip(), token.strip())
        if not token:
            continue
        if token in by_id:
            out.append(by_id[token])
        elif token in real:
            out.append(real[token]())
        else:
            known = sorted(list(by_id) + list(real))
            print(f"warning: unknown model id '{token}' (known: {known}); skipping",
                  file=sys.stderr)
    return out


def _load_or_generate_data(data_dir: str, n: int, horizon: int, size: int, seed: int):
    """Load episodes/paraphrases from ``data_dir`` if present; else generate them."""
    ep_path = os.path.join(data_dir, EPISODES_FILE)
    pp_path = os.path.join(data_dir, PARAPHRASES_FILE)
    if os.path.exists(ep_path) and os.path.exists(pp_path):
        episodes = load_episodes(ep_path)
        specs = load_paraphrases(pp_path)
        print(f"loaded {len(episodes)} episodes from {data_dir}")
        return episodes, specs
    episodes, specs = make_synthetic_dataset(n_episodes=n, horizon=horizon, size=size, seed=seed)
    print(f"generated {len(episodes)} synthetic episodes (no data found in {data_dir})")
    return episodes, specs


def cmd_run(args: argparse.Namespace) -> int:
    from afp.report import plot_curves, write_leaderboard, write_plane, write_scorecards_json
    from afp.score import score_suite

    adapters = _resolve_adapters(args.models)
    # Skip adapters that cannot actually run (real checkpoints without weights /
    # env vars). Do NOT silently fall back to the mock suite.
    runnable: List[WorldModelAdapter] = []
    for a in adapters:
        if a.available():
            runnable.append(a)
        else:
            hint = getattr(a, "availability_hint", None)
            msg = hint() if callable(hint) else "set its checkpoint env vars (see afp/adapters)"
            print(f"skipping '{a.model_id}': not available -- {msg}", file=sys.stderr)
    adapters = runnable
    if not adapters:
        print("no runnable adapters (set the real-adapter env vars, or use --models mock); "
              "nothing to do", file=sys.stderr)
        return 2

    episodes, specs = _load_or_generate_data(args.data, args.n, args.horizon, args.size, args.seed)
    seeds = tuple(range(args.seeds))

    print(f"scoring {len(adapters)} model(s) on {len(episodes)} episodes "
          f"with distance='{args.distance}', seeds={args.seeds} ...")
    scorecards = score_suite(
        adapters, episodes, specs, distance_name=args.distance, seeds=seeds,
    )

    os.makedirs(args.out, exist_ok=True)
    lb_path = os.path.join(args.out, "leaderboard.md")
    pl_path = os.path.join(args.out, "plane.md")
    js_path = os.path.join(args.out, "scorecards.json")
    curves_dir = os.path.join(args.out, "curves")
    write_leaderboard(scorecards, lb_path)
    write_plane(scorecards, pl_path)
    write_scorecards_json(scorecards, js_path)
    written = plot_curves(scorecards, curves_dir)

    _print_summary(scorecards)
    print()
    print(f"leaderboard -> {lb_path}")
    print(f"plane       -> {pl_path}")
    print(f"scorecards  -> {js_path}")
    print(f"curves      -> {curves_dir}/ ({len(written)} files)")
    return 0


def _print_summary(scorecards) -> None:
    rows = sorted(scorecards, key=lambda c: c.sens_action_final, reverse=True)
    hdr = f"{'model':<18} {'verdict':<18} {'sens_act':>9} {'sens_instr':>11} {'rho':>6} {'sep':>7} {'gates':>6}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for c in rows:
        gates_pass = sum(bool(v) for v in c.gates.values())
        print(f"{c.model_id:<18} {c.verdict:<18} {c.sens_action_final:>9.3f} "
              f"{c.sens_instruction_final:>11.3f} {c.monotonicity_rho:>6.2f} "
              f"{c.separation_ratio:>7.2f} {gates_pass:>4}/{len(c.gates)}")


# ===========================================================================
# backends
# ===========================================================================
def cmd_backends(args: argparse.Namespace) -> int:
    status = available_backends()
    print("distance backends:")
    for name, ok in status.items():
        print(f"  {name:<10} {'available' if ok else 'unavailable'}")
    return 0


# ===========================================================================
# parser / entrypoint
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="afp",
        description="Action-Faithfulness Probe: score action-conditioned video world models.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-data", help="generate synthetic episodes + paraphrases")
    g.add_argument("--out", default="data/", help="output directory")
    g.add_argument("--n", type=int, default=24, help="number of episodes")
    g.add_argument("--horizon", type=int, default=12, help="action/prediction horizon")
    g.add_argument("--size", type=int, default=64, help="frame size (HxW)")
    g.add_argument("--seed", type=int, default=0, help="generator seed")
    g.set_defaults(func=cmd_gen_data)

    r = sub.add_parser("run", help="score models and write the leaderboard + plane + curves")
    r.add_argument("--models", default="mock",
                   help="'mock', 'real', 'all', or comma-separated ids "
                        "(e.g. cosmos-sac, irasim, ctrl-world)")
    r.add_argument("--distance", default="pixel_l2", help="distance backend name")
    r.add_argument("--data", default="data/", help="data directory (loads if present, else generates)")
    r.add_argument("--out", default="results/", help="output directory")
    r.add_argument("--seeds", type=int, default=5, help="number of sampling seeds")
    r.add_argument("--n", type=int, default=24, help="episodes to generate if no data found")
    r.add_argument("--horizon", type=int, default=12, help="horizon if generating data")
    r.add_argument("--size", type=int, default=64, help="frame size if generating data")
    r.add_argument("--seed", type=int, default=0, help="data-generator seed if generating")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("backends", help="list available distance backends")
    b.set_defaults(func=cmd_backends)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
