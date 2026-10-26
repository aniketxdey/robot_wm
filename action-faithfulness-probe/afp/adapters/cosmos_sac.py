"""AFP adapter for NVIDIA **Cosmos-Predict2-2B-Sample-Action-Conditioned** (PRIMARY).

This is a *genuine* action-conditioned video world model: it maps a start frame
(plus an optional short context) **and a per-timestep 7-DoF action chunk** to a
predicted future video. The action is a true INPUT (fed through
``--input_annotation``), which is exactly what AFP needs to perturb.

Verified model facts (baked into ``available()`` and the run path)
------------------------------------------------------------------
* HF model id: ``nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned``.
* Output: **12 frames at 640x480, 4 fps** (~3 s). Hence ``horizon = 12``,
  ``fps = 4`` and native ``(H, W) = (480, 640)``.
* Conditioning: video + a 7-dim per-step action ``(dx, dy, dz, droll, dpitch,
  dyaw, gripper)`` in the gripper frame, gripper binary open(1)/close(0). Supplied
  as a JSON annotation file via ``--input_annotation`` -- see ``afp.adapters._action_io``.
* Precision: **bf16 ONLY** (fp16/fp32 are unsupported by this checkpoint).
* VRAM: ~**32.54 GB**. OOMs on <= 32 GB without offload. An **A100 40 GB** runs it
  with offload; **80 GB** is comfortable.
* Offload flags are *version-dependent*:
    - ``cosmos-predict2``   : ``--offload_guardrail`` / ``--offload_prompt_refiner``
    - ``cosmos-predict2.5`` : ``--offload-text-encoder`` / ``--offload-tokenizer``
  This adapter passes the ``cosmos-predict2`` (underscore) flags by default and
  lets you override the whole flag list via the constructor / ``AFP_COSMOS_EXTRA_ARGS``.
* Entry point (repo ``nvidia-cosmos/cosmos-predict2``):

      python examples/video2world_action.py \
          --model_size 2B \
          --dit_path <ckpt> \
          --input_video   datasets/bridge/videos/test/13/rgb.mp4 \
          --input_annotation datasets/bridge/annotation/test/13.json \
          --num_conditional_frames 1 \
          --save_path out.mp4 \
          --guidance 0 --seed 0 \
          --disable_guardrail --disable_prompt_refiner

Data caveat
-----------
Bridge is **not bundled** with the repo. It is a documented **~31 GB external
download**, ``bridge_train_data.tar.gz`` from IRASim's public bucket, extracted
under ``datasets/``. Mind a known path mismatch: the example invocation uses
``datasets/bridge/videos/test/13/rgb.mp4`` and
``datasets/bridge/annotation/test/13.json`` (singular ``annotation``, a ``test/``
split) while the tarball lays the data out as
``datasets/bridge/{annotations,videos}/``. You will likely need to symlink or
rename so the entrypoint's expected paths resolve.

How AFP drives it
-----------------
``rollout(o0, actions, ...)`` does **not** re-implement the model. It:
  1. writes the (already perturbed) ``actions`` to a temp ``--input_annotation``
     JSON via :mod:`afp.adapters._action_io`,
  2. invokes ``examples/video2world_action.py`` as a **subprocess** in the
     configured repo, with ``--guidance 0 --seed <seed>`` and the bf16 default,
  3. reads the saved mp4 back to ``(T, H, W, 3)`` uint8 frames.

Everything that touches torch / the checkpoint happens in that subprocess, so
*this* Python process never imports torch and the adapter imports cleanly on a
laptop.

Activation
----------
Set (constructor args take precedence over env vars):
  * ``AFP_COSMOS_REPO`` -> a checkout of ``nvidia-cosmos/cosmos-predict2``.
  * ``AFP_COSMOS_CKPT`` -> the downloaded DiT checkpoint passed as ``--dit_path``.
Optional:
  * ``AFP_COSMOS_PYTHON``     -> python executable for the GPU env (default ``sys.executable``).
  * ``AFP_COSMOS_INPUT_VIDEO``-> default ``--input_video`` if an episode provides none.
  * ``AFP_COSMOS_EXTRA_ARGS`` -> extra CLI args (shell-split) appended to the call.
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

MODEL_ID = "cosmos-sac-2b"
HF_REPO = "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned"

#: Native prediction geometry (do not change without re-verifying the checkpoint).
NATIVE_HEIGHT = 480
NATIVE_WIDTH = 640
NATIVE_HORIZON = 12
NATIVE_FPS = 4

SETUP = f"""\
Cosmos-Predict2-2B-Sample-Action-Conditioned ({MODEL_ID}) is not configured.

This is the PRIMARY action-conditioned world model. To activate it you need an
A100-class GPU (40 GB with offload, 80 GB comfortable; OOMs <=32 GB without
offload) and the following on disk:

1. Clone the inference repo:
     git clone https://github.com/nvidia-cosmos/cosmos-predict2
   and install its environment (Conda or the documented Docker image).

2. Accept the model license on Hugging Face and download the checkpoint:
     hf download {HF_REPO}
   (gated; you must accept NVIDIA's license terms first). This is the bf16 DiT
   passed to --dit_path; it is bf16 ONLY (fp16/fp32 unsupported), ~32.54 GB VRAM.

3. Download the Bridge data (NOT bundled): fetch bridge_train_data.tar.gz (~31 GB)
   from IRASim's public bucket and extract under <repo>/datasets/. Note the path
   mismatch: examples use datasets/bridge/videos/test/13/rgb.mp4 and
   datasets/bridge/annotation/test/13.json (singular 'annotation', a 'test/'
   split) while the tarball lays out datasets/bridge/{{annotations,videos}}/ --
   symlink/rename so the entrypoint paths resolve.

4. Point AFP at the install:
     export AFP_COSMOS_REPO=/path/to/cosmos-predict2
     export AFP_COSMOS_CKPT=/path/to/Cosmos-Predict2-2B-Sample-Action-Conditioned
   Optional: AFP_COSMOS_PYTHON, AFP_COSMOS_INPUT_VIDEO, AFP_COSMOS_EXTRA_ARGS.

The adapter then writes the perturbed 7-DoF action to a temp --input_annotation
JSON, runs examples/video2world_action.py (--model_size 2B, bf16, --guidance 0,
--seed <seed>) as a subprocess, and reads the 12x640x480 @4fps mp4 back.
"""


class CosmosSACAdapter(WorldModelAdapter):
    """Run Cosmos-Predict2-2B-Sample-Action-Conditioned behind the AFP interface.

    Parameters
    ----------
    repo_dir:
        Path to a ``nvidia-cosmos/cosmos-predict2`` checkout. Falls back to
        ``$AFP_COSMOS_REPO``.
    ckpt_path:
        Path to the downloaded DiT checkpoint (``--dit_path``). Falls back to
        ``$AFP_COSMOS_CKPT``.
    python_exe:
        Python interpreter for the GPU environment. Falls back to
        ``$AFP_COSMOS_PYTHON`` then the current interpreter.
    input_video:
        Default ``--input_video`` to use when an episode does not carry one in
        ``meta``. Falls back to ``$AFP_COSMOS_INPUT_VIDEO``.
    num_conditional_frames:
        ``--num_conditional_frames`` (default 1: condition on the start frame).
    extra_args:
        Extra CLI tokens appended verbatim (e.g. version-specific offload flags).
        Falls back to shell-split ``$AFP_COSMOS_EXTRA_ARGS``.
    env:
        Extra environment variables for the subprocess (merged over ``os.environ``).
    timeout:
        Subprocess timeout in seconds (``None`` = no limit).
    """

    model_id = MODEL_ID
    horizon = NATIVE_HORIZON
    fps = NATIVE_FPS
    #: The checkpoint conditions on the instruction TEXT (written into the
    #: annotation), not on AFP's embedding -- the scorer routes instruction
    #: perturbations through the text accordingly.
    instruction_input = "text"

    #: Default subprocess env for reproducibility: deterministic cuBLAS
    #: workspaces (required for torch deterministic mode on CUDA >= 10.2) and a
    #: pinned hash seed. User-supplied ``env`` entries override these.
    DETERMINISM_ENV = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        python_exe: Optional[str] = None,
        input_video: Optional[str] = None,
        num_conditional_frames: int = 1,
        extra_args: Optional[List[str]] = None,
        env: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.repo_dir = repo_dir or os.environ.get("AFP_COSMOS_REPO")
        self.ckpt_path = ckpt_path or os.environ.get("AFP_COSMOS_CKPT")
        self.python_exe = python_exe or os.environ.get("AFP_COSMOS_PYTHON")
        self.input_video = input_video or os.environ.get("AFP_COSMOS_INPUT_VIDEO")
        self.num_conditional_frames = int(num_conditional_frames)
        if extra_args is None:
            extra_args = shlex.split(os.environ.get("AFP_COSMOS_EXTRA_ARGS", ""))
        self.extra_args = list(extra_args)
        self.env = dict(env or {})
        self.timeout = timeout
        self.native_size = (NATIVE_HEIGHT, NATIVE_WIDTH)

    # -- availability ------------------------------------------------------
    def available(self) -> bool:
        return bool(
            self.repo_dir
            and self.ckpt_path
            and os.path.isdir(self.repo_dir)
            and os.path.exists(self.ckpt_path)
            and os.path.exists(
                os.path.join(self.repo_dir, "examples", "video2world_action.py")
            )
        )

    def _unavailable_message(self) -> str:
        reasons = []
        if not self.repo_dir:
            reasons.append("AFP_COSMOS_REPO is not set")
        elif not os.path.isdir(self.repo_dir):
            reasons.append(f"repo_dir does not exist: {self.repo_dir}")
        elif not os.path.exists(
            os.path.join(self.repo_dir, "examples", "video2world_action.py")
        ):
            reasons.append(
                f"examples/video2world_action.py not found under repo_dir {self.repo_dir}"
            )
        if not self.ckpt_path:
            reasons.append("AFP_COSMOS_CKPT is not set")
        elif not os.path.exists(self.ckpt_path):
            reasons.append(f"ckpt_path does not exist: {self.ckpt_path}")
        detail = "; ".join(reasons) if reasons else "missing setup"
        return f"{SETUP}\n[unavailable: {detail}]"

    # -- rollout -----------------------------------------------------------
    def rollout(self, o0, actions, instruction, instruction_emb, seed: int = 0) -> Rollout:
        if not self.available():
            raise RuntimeError(self._unavailable_message())

        actions = np.asarray(actions, dtype=np.float64)
        input_video = self.input_video
        # An episode may carry its own Bridge clip path via meta-injection.
        if input_video is None:
            raise RuntimeError(
                "No --input_video available. Set AFP_COSMOS_INPUT_VIDEO (or pass "
                "input_video=) to a Bridge clip such as "
                "datasets/bridge/videos/test/13/rgb.mp4.\n" + SETUP
            )

        workdir = tempfile.mkdtemp(prefix="afp_cosmos_")
        annotation_path = os.path.join(workdir, "annotation.json")
        save_path = os.path.join(workdir, "out.mp4")
        _action_io.write_temp_annotation(
            actions, instruction=instruction, path=annotation_path, as_dict=True
        )

        cmd = self._build_command(input_video, annotation_path, save_path, seed)
        run_env = {**os.environ, **self.DETERMINISM_ENV, **self.env}
        try:
            subprocess.run(
                cmd,
                cwd=self.repo_dir,
                env=run_env,
                check=True,
                timeout=self.timeout,
            )
            if not os.path.exists(save_path):
                raise RuntimeError(
                    f"cosmos entrypoint finished but no output at {save_path}"
                )
            # Load ALL saved frames, then keep the LAST `horizon`: video2world
            # entrypoints may prepend the conditioning frame(s) to the clip, and
            # slicing from the front would include the (rollout-invariant) input
            # frame while dropping the final prediction.
            frames = _action_io.load_video_frames(save_path)
            n_raw = int(frames.shape[0])
            frames = _action_io.take_prediction_frames(frames, self.horizon)
            frames = _action_io.resize_frames(frames, NATIVE_HEIGHT, NATIVE_WIDTH)
        finally:
            self._cleanup(workdir)

        return Rollout(
            frames=frames,
            model_id=self.model_id,
            episode_id="",
            seed=int(seed),
            meta={
                "backend": "cosmos-predict2",
                "hf_repo": HF_REPO,
                # Stage-0 frame accounting: raw clip length vs kept horizon.
                "n_raw_frames": n_raw,
            },
        )

    def _build_command(
        self, input_video: str, annotation_path: str, save_path: str, seed: int
    ) -> List[str]:
        python_exe = self.python_exe or _default_python()
        cmd = [
            python_exe,
            os.path.join("examples", "video2world_action.py"),
            "--model_size", "2B",
            "--dit_path", str(self.ckpt_path),
            "--input_video", str(input_video),
            "--input_annotation", str(annotation_path),
            "--num_conditional_frames", str(self.num_conditional_frames),
            "--save_path", str(save_path),
            "--guidance", "0",
            "--seed", str(int(seed)),
            # bf16 is the only supported precision for this checkpoint.
            "--disable_guardrail",
            "--disable_prompt_refiner",
        ]
        cmd.extend(self.extra_args)
        return cmd

    @staticmethod
    def _cleanup(workdir: str) -> None:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)


def _default_python() -> str:
    import sys

    return sys.executable or "python"


def build_real_suite() -> List[WorldModelAdapter]:
    """Return one instance of each real action-conditioned adapter.

    All three are **unavailable by default** (no env vars / weights), so this is
    safe to call on any machine; AFP filters on ``available()`` before scoring.
    Kept here (rather than in ``afp.adapters.__init__``) so importing a single
    adapter module never forces the others to import.
    """
    from afp.adapters.ctrl_world import CtrlWorldAdapter
    from afp.adapters.irasim import IRASimAdapter

    return [CosmosSACAdapter(), IRASimAdapter(), CtrlWorldAdapter()]
