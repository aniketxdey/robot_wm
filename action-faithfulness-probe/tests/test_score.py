"""End-to-end tests for the AFP orchestration / reporting layer on the mock suite.

These exercise the full pipeline (rollout -> perturb -> distance -> metrics ->
Scorecard -> report) with the always-available ``pixel_l2`` backend on a small
synthetic dataset, and assert the metric recovers the *known* ground-truth
sensitivity ordering of the three mock models (DESIGN Section 8, Stage 3 spread).
"""

from __future__ import annotations

import os

import pytest

from afp.adapters.mock import make_mock_suite
from afp.data import make_synthetic_dataset
from afp.report import write_leaderboard, write_plane, write_scorecards_json, plot_curves
from afp.score import Scorecard, score_model, score_suite


@pytest.fixture(scope="module")
def scored():
    """Score the full mock suite once on a small dataset (n=6, 3 seeds)."""
    episodes, specs = make_synthetic_dataset(n_episodes=6, horizon=12, size=64, seed=0)
    suite = make_mock_suite()
    cards = score_suite(suite, episodes, specs, distance_name="pixel_l2", seeds=(0, 1, 2))
    return {c.model_id: c for c in cards}, cards


# ---------------------------------------------------------------------------
def test_action_channel_recovers_faithful_over_instruction(scored):
    """The action-faithful mock must out-score the instruction-driven mock on action."""
    cards, _ = scored
    faithful = cards["mock-faithful"]
    instruction = cards["mock-instruction"]
    assert faithful.sens_action_final > instruction.sens_action_final


def test_instruction_channel_recovers_instruction_over_faithful(scored):
    """The instruction-driven mock must out-score the faithful mock on instruction."""
    cards, _ = scored
    faithful = cards["mock-faithful"]
    instruction = cards["mock-instruction"]
    assert instruction.sens_instruction_final > faithful.sens_instruction_final


def test_sensitivities_in_unit_range(scored):
    """Both channels are normalized fractions in [0, 1] (not trivially 1.0)."""
    cards, _ = scored
    for c in cards.values():
        assert 0.0 <= c.sens_action_final <= 1.0
        assert 0.0 <= c.sens_instruction_final <= 1.0
    # The plane must not collapse to a single point: there is real spread.
    sa = [c.sens_action_final for c in cards.values()]
    assert max(sa) - min(sa) > 0.1


def test_verdicts_are_sensible(scored):
    """Verdicts reflect the ground-truth construction of the mock suite."""
    cards, _ = scored
    valid = {"action-faithful", "instruction-driven", "balanced", "inert"}
    for c in cards.values():
        assert c.verdict in valid
    # The faithful mock should not be labelled instruction-driven, and vice versa.
    assert cards["mock-faithful"].verdict != "instruction-driven"
    assert cards["mock-instruction"].verdict != "action-faithful"
    # Neither headline model is inert (both clearly respond to *something*).
    assert cards["mock-faithful"].verdict != "inert"
    assert cards["mock-instruction"].verdict != "inert"


def test_scorecard_has_expected_structure(scored):
    """The Scorecard carries curves, floor, stats, gates and provenance."""
    cards, _ = scored
    c = cards["mock-balanced"]
    assert c.distance_name == "pixel_l2"
    assert c.horizon == 12
    assert len(c.sens_action) == c.horizon
    assert len(c.sens_instruction) == c.horizon
    assert "null_shuffle" in c.action_curves
    assert "mean" in c.seed_floor and "std" in c.seed_floor
    assert set(c.gates) == {"reactivity_ok", "null_floor_ok", "monotonicity_ok", "separation_ok"}
    assert "counterfactual" in c.instruction_curves
    assert c.separation_ratio >= 0.0


def test_reports_written_to_tmpdir(scored, tmp_path):
    """Leaderboard, plane, scorecards JSON, and curves all land in the out dir."""
    _, cards = scored
    lb = tmp_path / "leaderboard.md"
    pl = tmp_path / "plane.md"
    js = tmp_path / "scorecards.json"
    curves = tmp_path / "curves"

    write_leaderboard(cards, str(lb))
    write_plane(cards, str(pl))
    write_scorecards_json(cards, str(js))
    written = plot_curves(cards, str(curves))

    assert lb.exists() and lb.read_text().strip()
    assert pl.exists() and "sensitivity plane" in pl.read_text().lower()
    assert js.exists() and js.read_text().strip().startswith("[")
    # leaderboard mentions every model
    lb_text = lb.read_text()
    for mid in ("mock-faithful", "mock-balanced", "mock-instruction"):
        assert mid in lb_text
    # plot_curves produced files (PNGs if matplotlib present, CSVs otherwise)
    assert written and all(os.path.exists(p) for p in written)


def test_scorecard_save_load_roundtrip(scored, tmp_path):
    """Scorecard.save/load round-trips through JSON without losing the headline fields."""
    cards, _ = scored
    c = cards["mock-faithful"]
    path = tmp_path / "card.json"
    c.save(str(path))
    loaded = Scorecard.load(str(path))

    assert loaded.model_id == c.model_id
    assert loaded.verdict == c.verdict
    assert loaded.distance_name == c.distance_name
    assert loaded.sens_action_final == pytest.approx(c.sens_action_final)
    assert loaded.sens_instruction_final == pytest.approx(c.sens_instruction_final)
    assert loaded.sens_action == pytest.approx(c.sens_action)
    assert loaded.gates == c.gates


def test_single_model_scoring_smoke():
    """score_model works standalone (no specs -> perturb fallbacks) without error."""
    episodes, specs = make_synthetic_dataset(n_episodes=4, horizon=12, size=64, seed=1)
    adapter = make_mock_suite()[0]
    card = score_model(adapter, episodes, specs, distance_name="pixel_l2", seeds=(0, 1))
    assert isinstance(card, Scorecard)
    assert card.model_id == "mock-faithful"
    assert len(card.sens_action) == card.horizon
