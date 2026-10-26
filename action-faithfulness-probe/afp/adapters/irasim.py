"""AFP adapter for **IRASim** (ByteDance) -- trajectory-conditioned video (spread model).

IRASim is an action-conditioned ("interactive real-robot action simulator") video
world model: given a start frame and a **7-DoF end-effector trajectory** it renders
the future video. It is the BEST spread-model fit for AFP's Bridge route because it
shares the Bridge domain *and* the same 7-DoF action layout as the primary
(:mod:`afp.adapters.cosmos_sac`), so the very same perturbed ``(T, 7)`` action and
``--input_annotation`` JSON drive both models.

Verified facts (baked into ``available()`` and the run path)
------------------------------------------------------------
* Repo: ``bytedance/IRASim`` (public code + Bridge checkpoints). Built on the
  ``Latte`` video-DiT codebase.
* Conditioning: a per-timestep 7-DoF trajectory ``(dx, dy, dz, droll, dpitch,
  dyaw, gripper)`` in the gripper frame -- AFP's Bridge layout. Reused from
  :mod:`afp.adapters._action_io`.
* Datasets: Bridge / RT-1 / Language-Table. Bridge overlaps the primary.
* Downloads are **large** (~95-129 GB across code, the IRASim Bridge checkpoints,
  and the Bridge data tarball), which is why this is guarded rather than run here.

How AFP drives it
-----------------
The public IRASim inference path is a script invocation (the repo ships
``inference/`` / ``sample`` style entry scripts driven by a config + checkpoint
and an action/trajectory annotation). This adapter encodes that **subprocess**
call path at the fidelity the public docs allow: it writes the perturbed
trajectory to a temp annotation JSON (shared schema with Cosmos), runs the
configured inference entry script in the IRASim repo, and reads the produced mp4
back to ``(T, H, W, 3)`` uint8. The exact entry-script name / config flags vary by
repo revision, so they are **constructor / env configurable** rather than
hard-coded; nothing torch-related is imported in this process.

Native geometry
---------------
IRASim renders Bridge at the dataset's native resolution and the configured
trajectory length. Defaults below mirror the Bridge setup AFP shares with the
primary (``horizon = 12``); override per checkpoint/config via the constructor.

Activation
----------
Set (constructor args take precedence over env vars):
  * ``AFP_IRASIM_REPO`` -> a checkout of ``bytedance/IRASim``.
  * ``AFP_IRASIM_CKPT`` -> the downloaded IRASim Bridge checkpoint.
Optional:
  * ``AFP_IRASIM_PYTHON``     -> python executable for the GPU env.
  * ``AFP_IRASIM_ENTRY``      -> inference entry script (relative to the repo).
  * ``AFP_IRASIM_CONFIG``     -> config file passed to the entry script.
  * ``AFP_IRASIM_INPUT_VIDEO``-> default start-frame / context clip.
  * ``AFP_IRASIM_EXTRA_ARGS`` -> extra CLI args (shell-split).
``available()`` returns ``False`` with the full :data:`SETUP` text until both the
repo and the checkpoint exist on disk.
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

MODEL_ID = "irasim-bridge"
REPO_URL = "https://github.com/bytedance/IRASim"

NATIVE_HEIGHT = 480
NATIVE_WIDTH = 640
NATIVE_HORIZON = 12
NATIVE_FPS = 4

#: Default inference entry script (override via AFP_IRASIM_ENTRY for your revision).
DEFAULT_ENTRY = os.path.join("inference", "sample_autoregressive.py")

SETUP = f"""\
IRASim ({MODEL_ID}) is not configured.

IRASim is a 7-DoF trajectory-conditioned video world model (ByteDance) sharing
the Bridge domain and action layout with the primary, so the same perturbed
(T,7) action drives both. Downloads are large (~95-129 GB). To activate:

1. Clone the repo and install its environment (built on the Latte video-DiT codebase):
     git clone {REPO_URL}
   and follow its README for the conda/pip environment.

2. Download the IRASim Bridge checkpoint(s) as documented in the repo's model
   card / README (Hugging Face / the project's bucket), plus the Bridge data
   (bridge_train_data.tar.gz, ~31 GB) if you don't already have it from the
   Cosmos setup.

3. Point AFP at the install:
     export AFP_IRASIM_REPO=/path/to/IRASim
     export AFP_IRASIM_CKPT=/path/to/irasim_bridge_checkpoint
   Optional, depending on your repo revision:
     export AFP_IRASIM_ENTRY=inference/sample_autoregressive.py
     export AFP_IRASIM_CONFIG=configs/bridge.yaml
     export AFP_IRASIM_PYTHON=...   AFP_IRASIM_INPUT_VIDEO=...   AFP_IRASIM_EXTRA_ARGS=...

The adapter then writes the perturbed 7-DoF trajectory to a temp annotation JSON
(shared schema with Cosmos), runs the IRASim inference entry script as a
subprocess with --seed <seed>, and reads the produced mp4 back.
"""


class IRASimAdapter(WorldModelAdapter):
    """Run IRASim (Bridge) behind the AFP interface via its inference entry script.

    Parameters
    ----------
    repo_dir, ckpt_path:
        IRASim checkout and Bridge checkpoint. Fall back to ``AFP_IRASIM_REPO`` /
        ``AFP_IRASIM_CKPT``.
    python_exe:
        GPU-env interpreter. Falls back to ``AFP_IRASIM_PYTHON`` then current.
    entry:
        Inference entry script, relative to ``repo_dir``. Falls back to
        ``AFP_IRASIM_ENTRY`` then :data:`DEFAULT_ENTRY`.
    config:
        Optional config file passed via ``--config``. Falls back to ``AFP_IRASIM_CONFIG``.
    input_video:
        Default start/context clip. Falls back to ``AFP_IRASIM_INPUT_VIDEO``.
    extra_args:
        Extra CLI tokens appended verbatim. Falls back to ``AFP_IRASIM_EXTRA_ARGS``.
    horizon, fps, native_size:
        Output geometry; override per checkpoint/config.
    env, timeout:
        Subprocess environment overrides and timeout.
    """

    model_id = MODEL_ID
    horizon = NATIVE_HORIZON
    fps = NATIVE_FPS
    #: IRASim conditions on the trajectory + annotation text, not AFP's embedding.
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
        python_exe: Optional[str] = None,
        entry: Optional[str] = None,
        config: Optional[str] = None,
        input_video: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        horizon: int = NATIVE_HORIZON,
        fps: int = NATIVE_FPS,
        native_size: tuple = (NATIVE_HEIGHT, NATIVE_WIDTH),
        env: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.repo_dir = repo_dir or os.environ.get("AFP_IRASIM_REPO")
        self.ckpt_path = ckpt_path or os.environ.get("AFP_IRASIM_CKPT")
        self.python_exe = python_exe or os.environ.get("AFP_IRASIM_PYTHON")
        self.entry = entry or os.environ.get("AFP_IRASIM_ENTRY", DEFAULT_ENTRY)
        self.config = config or os.environ.get("AFP_IRASIM_CONFIG")
        self.input_video = input_video or os.environ.get("AFP_IRASIM_INPUT_VIDEO")
        if extra_args is None:
            extra_args = shlex.split(os.environ.get("AFP_IRASIM_EXTRA_ARGS", ""))
        self.extra_args = list(extra_args)
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
            reasons.append("AFP_IRASIM_REPO is not set")
        elif not os.path.isdir(self.repo_dir):
            reasons.append(f"repo_dir does not exist: {self.repo_dir}")
        elif not os.path.exists(os.path.join(self.repo_dir, self.entry)):
            reasons.append(
                f"entry script {self.entry!r} not found under repo_dir {self.repo_dir} "
                f"(set AFP_IRASIM_ENTRY for your revision)"
            )
        if not self.ckpt_path:
            reasons.append("AFP_IRASIM_CKPT is not set")
        elif not os.path.exists(self.ckpt_path):
            reasons.append(f"ckpt_path does not exist: {self.ckpt_path}")
        detail = "; ".join(reasons) if reasons else "missing setup"
        return f"{SETUP}\n[unavailable: {detail}]"

    # -- rollout -----------------------------------------------------------
    def rollout(self, o0, actions, instruction, instruction_emb, seed: int = 0) -> Rollout:
        if not self.available():
            raise RuntimeError(self._unavailable_message())

        actions = np.asarray(actions, dtype=np.float64)
        workdir = tempfile.mkdtemp(prefix="afp_irasim_")
        annotation_path = os.path.join(workdir, "trajectory.json")
        save_path = os.path.join(workdir, "out.mp4")
        _action_io.write_temp_annotation(
            actions, instruction=instruction, path=annotation_path, as_dict=True
        )

        cmd = self._build_command(annotation_path, save_path, seed)
        run_env = {**os.environ, **self.DETERMINISM_ENV, **self.env}
        try:
            subprocess.run(
                cmd, cwd=self.repo_dir, env=run_env, check=True, timeout=self.timeout
            )
            if not os.path.exists(save_path):
                raise RuntimeError(
                    f"IRASim entry script finished but no output at {save_path}"
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
            meta={"backend": "irasim", "repo": REPO_URL, "n_raw_frames": n_raw},
        )

    def _build_command(self, annotation_path: str, save_path: str, seed: int) -> List[str]:
        import sys

        python_exe = self.python_exe or sys.executable or "python"
        cmd = [python_exe, self.entry]
        if self.config:
            cmd += ["--config", str(self.config)]
        cmd += [
            "--checkpoint", str(self.ckpt_path),
            "--input_annotation", str(annotation_path),
            "--save_path", str(save_path),
            "--seed", str(int(seed)),
        ]
        if self.input_video:
            cmd += ["--input_video", str(self.input_video)]
        cmd += self.extra_args
        return cmd
