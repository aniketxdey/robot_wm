"""Tests for the real-checkpoint world-model adapters.

These run on any machine: no torch, no GPU, no checkpoints, no video backend.
They assert the adapters import cleanly, instantiate, are unavailable by default,
expose the verified ``horizon`` / ``fps`` / ``model_id``, raise an *informative*
``RuntimeError`` (naming the env var to set) when ``rollout`` is called while
unavailable, and that the shared annotation-json round-trips a ``(T, 7)`` action.

Only numpy + stdlib + pytest are used.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from afp.adapters import _action_io
from afp.adapters.cosmos_sac import CosmosSACAdapter, build_real_suite
from afp.adapters.ctrl_world import CtrlWorldAdapter, droid_action_adapter_required
from afp.adapters.irasim import IRASimAdapter
from afp.rollout import ACTION_DIM, GRIPPER_DIM, WorldModelAdapter

# (adapter class, env var that gates it, expected verified attrs)
ADAPTER_CASES = [
    (CosmosSACAdapter, "AFP_COSMOS_REPO", "cosmos-sac-2b", 12, 4),
    (IRASimAdapter, "AFP_IRASIM_REPO", "irasim-bridge", 12, 4),
    (CtrlWorldAdapter, "AFP_CTRLWORLD_REPO", "ctrl-world-droid", 15, 4),
]

# Every AFP_* env var any adapter consults -- cleared before each test so a
# developer's real configuration cannot leak in and flip availability.
_ALL_ENV_VARS = [
    "AFP_COSMOS_REPO", "AFP_COSMOS_CKPT", "AFP_COSMOS_PYTHON",
    "AFP_COSMOS_INPUT_VIDEO", "AFP_COSMOS_EXTRA_ARGS",
    "AFP_IRASIM_REPO", "AFP_IRASIM_CKPT", "AFP_IRASIM_PYTHON",
    "AFP_IRASIM_ENTRY", "AFP_IRASIM_CONFIG", "AFP_IRASIM_INPUT_VIDEO",
    "AFP_IRASIM_EXTRA_ARGS",
    "AFP_CTRLWORLD_REPO", "AFP_CTRLWORLD_CKPT", "AFP_CTRLWORLD_SVD",
    "AFP_CTRLWORLD_PYTHON", "AFP_CTRLWORLD_ENTRY", "AFP_CTRLWORLD_INPUT_VIDEO",
    "AFP_CTRLWORLD_EXTRA_ARGS",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _dummy_inputs(T=12, A=ACTION_DIM, H=8, W=8, E=8):
    o0 = np.zeros((H, W, 3), dtype=np.uint8)
    actions = np.zeros((T, A), dtype=np.float32)
    instruction = "pick up the red block"
    instruction_emb = np.zeros((E,), dtype=np.float32)
    return o0, actions, instruction, instruction_emb


# --------------------------------------------------------------------------- #
# Import / instantiation / metadata                                           #
# --------------------------------------------------------------------------- #
def test_modules_import_without_torch():
    # torch must not be pulled in merely by importing the adapter modules.
    assert "torch" not in sys.modules


@pytest.mark.parametrize("cls, env_var, model_id, horizon, fps", ADAPTER_CASES)
def test_instantiates_and_subclasses(cls, env_var, model_id, horizon, fps):
    adapter = cls()
    assert isinstance(adapter, WorldModelAdapter)


@pytest.mark.parametrize("cls, env_var, model_id, horizon, fps", ADAPTER_CASES)
def test_verified_metadata(cls, env_var, model_id, horizon, fps):
    adapter = cls()
    assert adapter.model_id == model_id
    assert adapter.horizon == horizon
    assert adapter.fps == fps
    # Class-level attrs match too (frozen interface uses class attrs).
    assert cls.model_id == model_id
    assert cls.horizon == horizon
    assert cls.fps == fps


@pytest.mark.parametrize("cls, env_var, model_id, horizon, fps", ADAPTER_CASES)
def test_unavailable_by_default(cls, env_var, model_id, horizon, fps):
    assert cls().available() is False


def test_cosmos_native_resolution():
    a = CosmosSACAdapter()
    assert a.native_size == (480, 640)


def test_ctrlworld_native_resolution_and_multiview_horizon():
    a = CtrlWorldAdapter()
    assert a.native_size == (192, 320)
    assert a.horizon == 15  # 15-step action chunk (multi-view)


# --------------------------------------------------------------------------- #
# Guarded rollout raises informative errors                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls, env_var, model_id, horizon, fps", ADAPTER_CASES)
def test_rollout_raises_when_unavailable(cls, env_var, model_id, horizon, fps):
    adapter = cls()
    o0, actions, instruction, instruction_emb = _dummy_inputs(T=adapter.horizon)
    with pytest.raises(RuntimeError) as exc:
        adapter.rollout(o0, actions, instruction, instruction_emb, seed=0)
    msg = str(exc.value)
    # The message must name the env var the user has to set.
    assert env_var in msg
    # ...and be actionable (mention download / license / export).
    assert any(k in msg.lower() for k in ("download", "license", "export"))


def test_configured_but_weights_missing_still_unavailable(monkeypatch, tmp_path):
    # Simulate "repo set but checkpoint missing" -> still unavailable, and rollout
    # raises naming the missing checkpoint env var.
    monkeypatch.setenv("AFP_COSMOS_REPO", str(tmp_path))
    adapter = CosmosSACAdapter()
    assert adapter.available() is False
    o0, actions, instruction, instruction_emb = _dummy_inputs()
    with pytest.raises(RuntimeError) as exc:
        adapter.rollout(o0, actions, instruction, instruction_emb)
    assert "AFP_COSMOS_CKPT" in str(exc.value)


def test_ctrlworld_requires_droid_action_adapter():
    with pytest.raises(NotImplementedError) as exc:
        droid_action_adapter_required(np.zeros((15, ACTION_DIM), dtype=np.float32))
    assert "DROID" in str(exc.value)


# --------------------------------------------------------------------------- #
# build_real_suite                                                            #
# --------------------------------------------------------------------------- #
def test_build_real_suite_all_unavailable():
    suite = build_real_suite()
    assert len(suite) == 3
    assert {a.model_id for a in suite} == {
        "cosmos-sac-2b", "irasim-bridge", "ctrl-world-droid"
    }
    assert all(isinstance(a, WorldModelAdapter) for a in suite)
    assert all(a.available() is False for a in suite)


# --------------------------------------------------------------------------- #
# _action_io annotation round-trip                                            #
# --------------------------------------------------------------------------- #
def test_action_annotation_roundtrip():
    rng = np.random.default_rng(0)
    T = 12
    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    actions[:, :GRIPPER_DIM] = rng.normal(size=(T, GRIPPER_DIM)).astype(np.float32)
    # binary gripper so the round-trip is exact
    actions[:, GRIPPER_DIM] = (rng.random(T) > 0.5).astype(np.float32)

    annotation = _action_io.action_to_annotation(actions)
    assert isinstance(annotation, list) and len(annotation) == T
    assert all(len(row) == ACTION_DIM for row in annotation)

    recovered = _action_io.annotation_to_action(annotation)
    assert recovered.shape == (T, ACTION_DIM)
    np.testing.assert_allclose(recovered[:, :GRIPPER_DIM], actions[:, :GRIPPER_DIM], rtol=1e-5)
    np.testing.assert_array_equal(recovered[:, GRIPPER_DIM], actions[:, GRIPPER_DIM])


def test_action_annotation_roundtrip_via_dict():
    actions = np.zeros((5, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = np.arange(5)
    obj = _action_io.build_annotation_dict(actions, instruction="grab the blue bowl")
    assert obj["texts"] == ["grab the blue bowl"]
    recovered = _action_io.annotation_to_action(obj)
    np.testing.assert_allclose(recovered[:, 0], np.arange(5), rtol=1e-5)


def test_gripper_binarization():
    actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
    actions[:, GRIPPER_DIM] = np.array([0.2, 0.5, 0.9], dtype=np.float32)
    ann = _action_io.action_to_annotation(actions)
    grip = [row[GRIPPER_DIM] for row in ann]
    assert grip == [0.0, 1.0, 1.0]


def test_write_temp_annotation_and_read_back(tmp_path):
    actions = np.zeros((4, ACTION_DIM), dtype=np.float32)
    actions[:, 1] = 0.3
    path = _action_io.write_temp_annotation(
        actions, instruction="lift the green cup", path=str(tmp_path / "a.json")
    )
    import json

    with open(path) as f:
        obj = json.load(f)
    recovered = _action_io.annotation_to_action(obj)
    np.testing.assert_allclose(recovered[:, 1], np.full(4, 0.3), rtol=1e-5)


def test_action_to_annotation_rejects_bad_shape():
    with pytest.raises(ValueError):
        _action_io.action_to_annotation(np.zeros((4, 3), dtype=np.float32))


def test_resize_frames_numpy_fallback():
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    out = _action_io.resize_frames(frames, 8, 6)
    assert out.shape == (2, 8, 6, 3)
    assert out.dtype == np.uint8
