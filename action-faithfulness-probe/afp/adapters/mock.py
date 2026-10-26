"""Deterministic synthetic world models with KNOWN action/instruction sensitivity.

The mock renders a moving end-effector "blob" on a small canvas. The blob's
per-step velocity is a weighted sum of:

    v_t = action_gain * (the commanded dx,dy at step t)
        + instruction_gain * (a drift direction read from the instruction embedding)
        + seed_jitter (reproducible per `seed`, independent of the inputs)

Consequences (this is *why* the mock is a valid test of the metrics):
- Perturbing the ACTION moves the trajectory iff ``action_gain > 0``.
- Perturbing the INSTRUCTION changes the drift iff ``instruction_gain > 0``.
- With a fixed ``seed`` and identical inputs, frames are bit-identical
  (``S_identity == 0``); across seeds the jitter differs (the "seed floor").

So a model with high ``action_gain`` / low ``instruction_gain`` is ground-truth
"action-faithful"; the reverse is "instruction-driven". AFP should recover this.
"""

from __future__ import annotations

from typing import List

import numpy as np

from afp.rollout import Rollout, WorldModelAdapter, GRIPPER_DIM


def _unit(v: np.ndarray, default=(1.0, 0.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(default, dtype=np.float64)
    return (v / n).astype(np.float64)


class MockWorldModel(WorldModelAdapter):
    """A synthetic, deterministic action-conditioned video world model.

    ``is_synthetic = True``: mock-only diagnostics (e.g. ``null_shuffle``) run
    here but are skipped on real checkpoints.

    Parameters
    ----------
    model_id : str
        Leaderboard identifier.
    action_gain : float
        How strongly the commanded action moves the predicted blob (>=0).
    instruction_gain : float
        How strongly the instruction's drift direction moves the blob (>=0).
    noise : float
        Std (in pixels) of the per-step seed jitter -> sets the seed floor.
    horizon, size, fps : int
        Output length, canvas H=W=size, and nominal fps.
    gpu_jitter : float
        If > 0, adds *seed-independent* per-call jitter to emulate non-bit-exact
        GPU sampling, so ``S_identity`` becomes a small epsilon rather than 0.
    """

    is_synthetic = True
    instruction_input = "embedding"

    def __init__(
        self,
        model_id: str,
        action_gain: float = 1.0,
        instruction_gain: float = 0.2,
        noise: float = 0.04,
        horizon: int = 12,
        size: int = 64,
        fps: int = 4,
        gpu_jitter: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.action_gain = float(action_gain)
        self.instruction_gain = float(instruction_gain)
        self.noise = float(noise)
        self.horizon = int(horizon)
        self.size = int(size)
        self.fps = int(fps)
        self.gpu_jitter = float(gpu_jitter)

        # Fixed pixel scales (chosen so a unit action / instruction produces a
        # clearly visible, sub-canvas trajectory on a 64px grid).
        self._act_scale = 9.0
        self._instr_scale = 2.2

    # -- rendering ---------------------------------------------------------
    def _render(self, positions: np.ndarray, gripper: np.ndarray, bg: np.ndarray) -> np.ndarray:
        T = positions.shape[0]
        S = self.size
        ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
        frames = np.empty((T, S, S, 3), dtype=np.uint8)
        for t in range(T):
            cx, cy = positions[t]
            g = float(gripper[t])
            sigma = 4.0 + 3.0 * g  # open gripper -> bigger blob
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            blob = np.exp(-d2 / (2.0 * sigma ** 2))
            # gripper toggles warm (open) vs cool (closed) color
            warm = np.array([255, 170, 60], dtype=np.float64)
            cool = np.array([60, 140, 255], dtype=np.float64)
            color = g * warm + (1.0 - g) * cool
            frame = bg.astype(np.float64).copy()
            for c in range(3):
                frame[..., c] = np.clip(frame[..., c] * (1 - blob) + color[c] * blob, 0, 255)
            frames[t] = frame.astype(np.uint8)
        return frames

    def _background(self, o0: np.ndarray) -> np.ndarray:
        S = self.size
        if o0 is not None and o0.shape[:2] == (S, S):
            return (o0.astype(np.float64) * 0.25).astype(np.uint8)
        # faint static gradient if o0 size mismatches
        ys, xs = np.mgrid[0:S, 0:S]
        base = np.stack([(xs * 0).astype(np.uint8)] * 3, axis=-1)
        base[..., 2] = (ys / S * 40).astype(np.uint8)
        return base

    # -- interface ---------------------------------------------------------
    def rollout(self, o0, actions, instruction, instruction_emb, seed: int = 0) -> Rollout:
        actions = np.asarray(actions, dtype=np.float64)
        T = min(self.horizon, actions.shape[0]) if actions.shape[0] else self.horizon
        T = self.horizon  # always emit native horizon; pad/trim actions to it
        A = actions.shape[1] if actions.ndim == 2 else 7
        if actions.shape[0] < T:
            pad = np.zeros((T - actions.shape[0], A), dtype=np.float64)
            actions = np.vstack([actions, pad]) if actions.size else np.zeros((T, A))
        actions = actions[:T]

        drift = _unit(np.asarray(instruction_emb, dtype=np.float64)[:2])

        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        jitter = rng.normal(0.0, self.noise, size=(T, 2)) if self.noise > 0 else np.zeros((T, 2))
        if self.gpu_jitter > 0:
            jitter = jitter + np.random.default_rng().normal(0.0, self.gpu_jitter, size=(T, 2))

        vel = (
            self.action_gain * self._act_scale * actions[:, 0:2]
            + self.instruction_gain * self._instr_scale * drift[None, :]
            + jitter
        )
        start = np.array([self.size / 2.0, self.size / 2.0])
        positions = start[None, :] + np.cumsum(vel, axis=0)
        positions = np.clip(positions, 2, self.size - 3)

        gripper = actions[:, GRIPPER_DIM] if A > GRIPPER_DIM else np.zeros(T)
        gripper = np.clip(gripper, 0.0, 1.0)

        bg = self._background(o0)
        frames = self._render(positions, gripper, bg)
        return Rollout(frames=frames, model_id=self.model_id, episode_id="", seed=int(seed))


def make_mock_suite() -> List[MockWorldModel]:
    """Three mock models spanning the sensitivity plane (the headline demo).

    - ``mock-faithful``     : tracks actions, mostly ignores wording (goal region).
    - ``mock-balanced``     : reads both channels.
    - ``mock-instruction``  : largely ignores actions, repaints from the wording.
    """
    return [
        MockWorldModel("mock-faithful", action_gain=1.0, instruction_gain=0.15, noise=0.04),
        MockWorldModel("mock-balanced", action_gain=0.6, instruction_gain=0.6, noise=0.04),
        MockWorldModel("mock-instruction", action_gain=0.12, instruction_gain=1.0, noise=0.04),
    ]
