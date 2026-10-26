"""Core data contracts and the world-model adapter interface.

This module is the *frozen interface* every other AFP module builds against.
Do NOT redefine these types elsewhere -- import them from here.

A world model under test maps:

    (o0, instruction, instruction_emb, actions[1..T])  ->  predicted frames ohat[1..T]

AFP perturbs the ACTION (and, as a calibration channel, the INSTRUCTION) while
holding the start frame fixed, then measures how the predicted frames diverge.

Frames convention
-----------------
All frames are ``numpy.uint8`` arrays of shape ``(T, H, W, 3)`` in RGB order,
values in ``[0, 255]``. A single frame is ``(H, W, 3)``.

Actions convention
-------------------
An action chunk is a ``float32`` array of shape ``(T, A)``. For the Bridge /
Cosmos-Predict2-2B-Sample-Action-Conditioned primary, ``A == 7``:
``(dx, dy, dz, droll, dpitch, dyaw, gripper)`` -- the first 6 are end-effector
displacement in the gripper frame, the 7th is a binary gripper open(1)/close(0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 7-DoF Bridge action layout (indices into the action vector).
ACTION_DIM = 7
MOTION_DIMS = (0, 1, 2, 3, 4, 5)  # dx, dy, dz, droll, dpitch, dyaw
GRIPPER_DIM = 6


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

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    def validate(self) -> "Episode":
        assert self.o0.dtype == np.uint8 and self.o0.ndim == 3 and self.o0.shape[2] == 3, \
            f"o0 must be (H,W,3) uint8, got {self.o0.shape} {self.o0.dtype}"
        assert self.actions.ndim == 2, f"actions must be (T,A), got {self.actions.shape}"
        assert self.instruction_emb.ndim == 1, "instruction_emb must be 1-D"
        if self.true_future is not None:
            assert self.true_future.dtype == np.uint8 and self.true_future.ndim == 4
        return self


@dataclass
class Rollout:
    """A predicted rollout returned by an adapter."""

    frames: np.ndarray                  # (T, H, W, 3) uint8
    model_id: str
    episode_id: str
    seed: int = 0
    perturbation: str = "identity"      # name of the operator that produced the input action/instruction
    channel: str = "action"             # "action" | "instruction" | "none"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        f = np.asarray(self.frames)
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        assert f.ndim == 4 and f.shape[-1] == 3, f"frames must be (T,H,W,3), got {f.shape}"
        self.frames = f

    @property
    def horizon(self) -> int:
        return int(self.frames.shape[0])


class WorldModelAdapter(ABC):
    """Uniform interface every world model must implement to be scored by AFP.

    Implementations should be *deterministic given a seed* as far as the backend
    allows. GPU sampling is not bit-exact, so AFP measures an empirical
    "determinism floor" rather than assuming S_identity == 0 (see ``afp.metrics``).
    """

    #: short, stable identifier used in the leaderboard (e.g. "cosmos-sac-2b").
    model_id: str = "abstract"

    #: native number of predicted frames per rollout (e.g. 12 for Cosmos-SAC).
    horizon: int = 12

    #: native frames-per-second of the prediction (e.g. 4 for Cosmos-SAC).
    fps: int = 4

    #: which instruction input the model actually conditions on:
    #: "embedding" -> ``instruction_emb`` (mock / latent adapters);
    #: "text"      -> the ``instruction`` string (real checkpoints: the text is
    #: written into the annotation / prompt and the embedding is ignored).
    #: The scorer uses this to route instruction perturbations through the
    #: channel the model can actually see -- perturbing only the embedding of a
    #: text-conditioned model would silently probe nothing (and waste GPU).
    instruction_input: str = "embedding"

    #: True for synthetic/mock adapters. Mock-only diagnostics (e.g. the
    #: ``null_shuffle`` illustrative placebo) are skipped on real checkpoints so
    #: no GPU rollout is spent on an operator that is invalid off-mock.
    is_synthetic: bool = False

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

    # -- convenience -------------------------------------------------------
    def rollout_episode(self, episode: Episode, seed: int = 0) -> Rollout:
        r = self.rollout(
            o0=episode.o0,
            actions=episode.actions,
            instruction=episode.instruction,
            instruction_emb=episode.instruction_emb,
            seed=seed,
        )
        r.episode_id = episode.episode_id
        return r

    def available(self) -> bool:
        """Whether this adapter can actually run (weights present, deps importable).

        Real checkpoint adapters override this to return False with a helpful
        message until the user downloads weights / accepts a license.
        """
        return True
