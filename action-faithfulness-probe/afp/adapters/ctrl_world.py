"""AFP adapter for **Ctrl-World** (ICLR 2026) -- frame-level action-conditioned WM (spread model).

Ctrl-World is a controllable, action-conditioned video world model in the **DROID**
manipulation domain. Unlike the Bridge primary, its conditioning is **frame-level**
and its action space follows the **DROID** convention, so it exercises a *different*
action-layout path through AFP -- useful as a cross-domain spread model.

Verified facts (baked into ``available()`` and the docstrings)
--------------------------------------------------------------
* Code: ``github.com/Robert-gyj/Ctrl-World`` (ICLR 2026).
* Weights: ``yjguo/Ctrl-World`` on Hugging Face (~8 GB) **plus** a Stable Video
  Diffusion base (an SVD checkpoint, ~8 GB) it builds on. Both must be present.
* Conditioning: **frame-level** action conditioning (an action per generated
  frame), DROID domain. Native rendering **192x320**, **multi-view**, processed in
  **15-step action chunks** -> ``horizon = 15``.
* Action layout differs from AFP's 7-DoF Bridge ``(dx,dy,dz,droll,dpitch,dyaw,
  gripper)``. DROID actions are not the Bridge gripper-frame displacement layout,
  so an **action-layout adapter is required** before feeding AFP actions to
  Ctrl-World (see :func:`droid_action_adapter_required` and the note below). This
  adapter therefore does NOT silently reuse the Bridge annotation schema; it flags
  the conversion as a setup responsibility.

Multi-view note
---------------
Ctrl-World renders multiple camera views. AFP frames are a single ``(T, H, W, 3)``
stream, so a real run must select / arrange views into one stream (e.g. pick the
primary external camera, or tile views). This adapter exposes ``view`` to make that
choice explicit; the default selects the primary view.

How AFP would drive it
----------------------
The public Ctrl-World inference path is a script / module invocation that loads the
SVD base + the Ctrl-World weights and rolls out a 15-step action chunk over a start
frame. This adapter encodes that **subprocess** call path at the fidelity public
docs allow (entry script + base/weights paths are constructor / env configurable),
and reads the produced video back to ``(T, H, W, 3)`` uint8 at 192x320. Nothing
torch-related is imported in this process.

Activation
----------
Set (constructor args take precedence over env vars):
  * ``AFP_CTRLWORLD_REPO`` -> a checkout of ``Robert-gyj/Ctrl-World``.
  * ``AFP_CTRLWORLD_CKPT`` -> the downloaded ``yjguo/Ctrl-World`` weights (~8 GB).
Optional but typically required for a real run:
  * ``AFP_CTRLWORLD_SVD``        -> the SVD base checkpoint (~8 GB).
  * ``AFP_CTRLWORLD_PYTHON``     -> python executable for the GPU env.
  * ``AFP_CTRLWORLD_ENTRY``      -> inference entry script (relative to the repo).
  * ``AFP_CTRLWORLD_INPUT_VIDEO``-> default start-frame / context clip.
  * ``AFP_CTRLWORLD_EXTRA_ARGS`` -> extra CLI args (shell-split).
``available()`` returns ``False`` with the full :data:`SETUP` text until both the
repo and the Ctrl-World weights exist on disk.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from afp.adapters import _action_io
from afp.rollout import Rollout, WorldModelAdapter

MODEL_ID = "ctrl-world-droid"
REPO_URL = "https://github.com/Robert-gyj/Ctrl-World"
HF_WEIGHTS = "yjguo/Ctrl-World"

NATIVE_HEIGHT = 192
NATIVE_WIDTH = 320
NATIVE_HORIZON = 15  # 15-step action chunk (multi-view)
NATIVE_FPS = 4

DEFAULT_ENTRY = os.path.join("inference.py")

SETUP = f"""\
Ctrl-World ({MODEL_ID}) is not configured.

Ctrl-World (ICLR 2026) is a frame-level action-conditioned video world model in
the DROID domain: native 192x320, multi-view, 15-step action chunks. To activate:

1. Clone the repo and install its environment:
     git clone {REPO_URL}
   and follow its README.

2. Download BOTH sets of weights:
     - Ctrl-World weights: hf download {HF_WEIGHTS}        (~8 GB)
     - the Stable Video Diffusion base checkpoint it builds on (~8 GB), as named
       in the repo README (accept the SVD license on Hugging Face).

3. Point AFP at the install:
     export AFP_CTRLWORLD_REPO=/path/to/Ctrl-World
     export AFP_CTRLWORLD_CKPT=/path/to/ctrl-world-weights
     export AFP_CTRLWORLD_SVD=/path/to/svd-base        # required for a real run
   Optional, depending on your revision:
     export AFP_CTRLWORLD_ENTRY=inference.py
     export AFP_CTRLWORLD_PYTHON=...  AFP_CTRLWORLD_INPUT_VIDEO=...  AFP_CTRLWORLD_EXTRA_ARGS=...

ACTION LAYOUT: Ctrl-World uses the DROID action convention, NOT AFP's 7-DoF Bridge
layout (dx,dy,dz,droll,dpitch,dyaw,gripper). You MUST supply a DROID action-layout
adapter (see droid_action_adapter_required) before scoring; the Bridge annotation
schema is intentionally not reused here.
"""


def droid_action_adapter_required(actions: np.ndarray) -> np.ndarray:
    """Placeholder for the DROID action-layout conversion (must be provided per setup).

    Ctrl-World consumes **DROID**-convention, frame-level actions, which are not
    AFP's gripper-frame 7-DoF Bridge displacement layout. A correct run requires a
    domain-specific mapping (frame-level alignment + DROID action encoding). Because
    that mapping depends on the exact DROID action spec the checkpoint was trained
    on, it is deliberately not guessed here: calling this without configuring a real
    mapping raises, so a wrong-but-silent conversion can never reach the model.
    """
    raise NotImplementedError(
        "Ctrl-World needs a DROID action-layout adapter: AFP's 7-DoF Bridge actions "
        "(dx,dy,dz,droll,dpitch,dyaw,gripper) must be mapped to the DROID frame-level "
        "action convention the checkpoint expects. Provide this mapping in your run "
        "configuration before scoring Ctrl-World."
    )


class CtrlWorldAdapter(WorldModelAdapter):
    """Run Ctrl-World (DROID) behind the AFP interface via its inference entry script.

    Parameters
    ----------
    repo_dir, ckpt_path:
        Ctrl-World checkout and ``yjguo/Ctrl-World`` weights. Fall back to
        ``AFP_CTRLWORLD_REPO`` / ``AFP_CTRLWORLD_CKPT``.
    svd_path:
        SVD base checkpoint. Falls back to ``AFP_CTRLWORLD_SVD``.
    python_exe, entry, input_video, extra_args:
        GPU-env interpreter, inference entry script (relative to ``repo_dir``),
        default start clip, and extra CLI tokens. Fall back to the matching
        ``AFP_CTRLWORLD_*`` env vars.
    view:
        Which camera view to keep as the single AFP stream (default ``"primary"``).
    horizon, fps, native_size:
        Output geometry (default 15 frames @ 192x320, multi-view).
    env, timeout:
        Subprocess environment overrides and timeout.
    """

    model_id = MODEL_ID
    horizon = NATIVE_HORIZON
    fps = NATIVE_FPS
    #: Ctrl-World conditions on text prompts (if any), not AFP's embedding.
    instruction_input = "text"

    #: Deterministic-CUDA defaults for the inference subprocess (user env wins).
    DETERMINISM_ENV = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        svd_path: Optional[str] = None,
        python_exe: Optional[str] = None,
        entry: Optional[str] = None,
        input_video: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        view: str = "primary",
        horizon: int = NATIVE_HORIZON,
        fps: int = NATIVE_FPS,
        native_size: tuple = (NATIVE_HEIGHT, NATIVE_WIDTH),
        env: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.repo_dir = repo_dir or os.environ.get("AFP_CTRLWORLD_REPO")
        self.ckpt_path = ckpt_path or os.environ.get("AFP_CTRLWORLD_CKPT")
        self.svd_path = svd_path or os.environ.get("AFP_CTRLWORLD_SVD")
        self.python_exe = python_exe or os.environ.get("AFP_CTRLWORLD_PYTHON")
        self.entry = entry or os.environ.get("AFP_CTRLWORLD_ENTRY", DEFAULT_ENTRY)
        self.input_video = input_video or os.environ.get("AFP_CTRLWORLD_INPUT_VIDEO")
        if extra_args is None:
            extra_args = shlex.split(os.environ.get("AFP_CTRLWORLD_EXTRA_ARGS", ""))
        self.extra_args = list(extra_args)
        self.view = view
        self.horizon = int(horizon)
        self.fps = int(fps)
        self.native_size = tuple(native_size)
        self.env = dict(env or {})
        self.timeout = timeout

    # -- availability ------------------------------------------------------
    def available(self) -> bool:
        return bool(
            self.repo_dir
            and self.ckpt_path
            and os.path.isdir(self.repo_dir)
            and os.path.exists(self.ckpt_path)
            and os.path.exists(os.path.join(self.repo_dir, self.entry))
        )

    def _unavailable_message(self) -> str:
        reasons = []
        if not self.repo_dir:
            reasons.append("AFP_CTRLWORLD_REPO is not set")
        elif not os.path.isdir(self.repo_dir):
            reasons.append(f"repo_dir does not exist: {self.repo_dir}")
        elif not os.path.exists(os.path.join(self.repo_dir, self.entry)):
            reasons.append(
                f"entry script {self.entry!r} not found under repo_dir {self.repo_dir} "
                f"(set AFP_CTRLWORLD_ENTRY for your revision)"
            )
        if not self.ckpt_path:
            reasons.append("AFP_CTRLWORLD_CKPT is not set")
        elif not os.path.exists(self.ckpt_path):
            reasons.append(f"ckpt_path does not exist: {self.ckpt_path}")
        if not self.svd_path:
            reasons.append("AFP_CTRLWORLD_SVD is not set (SVD base required for a real run)")
        detail = "; ".join(reasons) if reasons else "missing setup"
        return f"{SETUP}\n[unavailable: {detail}]"

    # -- rollout -----------------------------------------------------------
    def rollout(self, o0, actions, instruction, instruction_emb, seed: int = 0) -> Rollout:
        if not self.available():
            raise RuntimeError(self._unavailable_message())

        # DROID action layout differs from AFP's Bridge layout: a mapping must be
        # configured by the user. Without it we refuse rather than feed wrong actions.
        droid_actions = droid_action_adapter_required(np.asarray(actions, dtype=np.float64))

        workdir = tempfile.mkdtemp(prefix="afp_ctrlworld_")
        action_path = os.path.join(workdir, "droid_actions.json")
        save_path = os.path.join(workdir, "out.mp4")
        with open(action_path, "w") as f:
            import json

            json.dump(np.asarray(droid_actions, dtype=np.float64).tolist(), f)

        cmd = self._build_command(action_path, save_path, seed)
        run_env = {**os.environ, **self.DETERMINISM_ENV, **self.env}
        try:
            subprocess.run(
                cmd, cwd=self.repo_dir, env=run_env, check=True, timeout=self.timeout
            )
            if not os.path.exists(save_path):
                raise RuntimeError(
                    f"Ctrl-World entry script finished but no output at {save_path}"
                )
            # Keep the LAST `horizon` frames: the clip may prepend context frames.
            frames = _action_io.load_video_frames(save_path)
            n_raw = int(frames.shape[0])
            frames = _action_io.take_prediction_frames(frames, self.horizon)
            frames = _action_io.resize_frames(frames, self.native_size[0], self.native_size[1])
        finally:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

        return Rollout(
            frames=frames,
            model_id=self.model_id,
            episode_id="",
            seed=int(seed),
            meta={"backend": "ctrl-world", "repo": REPO_URL, "view": self.view,
                  "n_raw_frames": n_raw},
        )

    def _build_command(self, action_path: str, save_path: str, seed: int) -> List[str]:
        import sys

        python_exe = self.python_exe or sys.executable or "python"
        cmd = [
            python_exe, self.entry,
            "--ctrl_world_ckpt", str(self.ckpt_path),
        ]
        if self.svd_path:
            cmd += ["--svd_ckpt", str(self.svd_path)]
        cmd += [
            "--actions", str(action_path),
            "--save_path", str(save_path),
            "--view", str(self.view),
            "--seed", str(int(seed)),
        ]
        if self.input_video:
            cmd += ["--input_video", str(self.input_video)]
        cmd += self.extra_args
        return cmd
