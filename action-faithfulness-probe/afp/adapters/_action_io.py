"""Shared I/O helpers for the *real* action-conditioned world-model adapters.

This module is the glue between AFP's in-memory action representation and the
on-disk formats the real checkpoints actually consume on a GPU box. It is kept
free of any heavy dependency at import time: ``numpy`` is the only hard import;
video decoders (``imageio`` / ``decord`` / ``opencv``) and resizers (``PIL`` /
``opencv``) are imported *lazily* inside the functions that need them, with a
dependency-free numpy fallback where feasible. That means the adapters can be
imported, instantiated and introspected on a laptop with no torch, no video
backend and no checkpoints present.

The Bridge "annotation JSON" schema
-----------------------------------
The NVIDIA ``cosmos-predict2`` entrypoint ``examples/video2world_action.py``
and ByteDance's ``IRASim`` both condition on a per-timestep action read from an
``--input_annotation`` JSON file. The canonical *on-disk* action is a
**per-timestep list**: one row per predicted frame, each row a 7-vector

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

where the first six dims are the end-effector displacement (translation +
rotation) expressed *in the gripper frame* and the 7th is a **binary** gripper
command, ``1`` = open and ``0`` = close. This is exactly AFP's ``(T, 7)`` Bridge
action layout (see ``afp.rollout.MOTION_DIMS`` / ``GRIPPER_DIM``), so applying an
AFP perturbation operator is literally editing this list before writing it out.

Real annotation files carry a little more than the bare action list (an
instruction string, sometimes proprioceptive ``state``); :func:`build_annotation_dict`
produces that fuller object, while :func:`action_to_annotation` /
:func:`annotation_to_action` round-trip just the action payload. The loaders
accept either shape so a perturbed action can be written and read back
losslessly.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from afp.rollout import ACTION_DIM, GRIPPER_DIM, MOTION_DIMS

#: Binary gripper encoding used by the Bridge annotation schema.
GRIPPER_OPEN = 1
GRIPPER_CLOSED = 0

#: Keys under which the per-timestep action list may live in a full annotation
#: object. The first is what we write; the rest are accepted on read.
_ACTION_KEYS = ("action", "actions", "actions_gripper", "abs_action")

AnnotationList = List[List[float]]
Annotation = Union[AnnotationList, dict]


# --------------------------------------------------------------------------- #
# Action <-> annotation conversion                                            #
# --------------------------------------------------------------------------- #
def _binarize_gripper(value: float) -> int:
    """Map a (possibly soft) gripper command to the schema's binary open/close.

    AFP actions may carry a continuous gripper value; the on-disk Bridge schema
    is strictly binary. We threshold at 0.5 so ``>=0.5`` -> open(1), else close(0).
    """
    return GRIPPER_OPEN if float(value) >= 0.5 else GRIPPER_CLOSED


def action_to_annotation(actions: np.ndarray) -> AnnotationList:
    """Convert an AFP ``(T, 7)`` action array to the per-timestep annotation list.

    Parameters
    ----------
    actions:
        ``(T, 7)`` array ``(dx, dy, dz, droll, dpitch, dyaw, gripper)``. The
        motion dims are kept as Python ``float``; the gripper dim is binarized.

    Returns
    -------
    list of list of float
        ``T`` rows, each a 7-element list. JSON-serializable.
    """
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions must be (T, {ACTION_DIM}) = "
            f"(dx,dy,dz,droll,dpitch,dyaw,gripper); got shape {arr.shape}"
        )
    rows: AnnotationList = []
    for row in arr:
        motion = [float(row[i]) for i in MOTION_DIMS]
        rows.append(motion + [float(_binarize_gripper(row[GRIPPER_DIM]))])
    return rows


def _extract_action_list(annotation: Annotation) -> Sequence[Sequence[float]]:
    """Pull the per-timestep action list out of a bare list or a full dict."""
    if isinstance(annotation, dict):
        for key in _ACTION_KEYS:
            if key in annotation:
                return annotation[key]
        raise KeyError(
            f"annotation dict has no action list under any of {_ACTION_KEYS}; "
            f"keys present: {sorted(annotation)}"
        )
    return annotation  # already a per-timestep list


def annotation_to_action(annotation: Annotation) -> np.ndarray:
    """Inverse of :func:`action_to_annotation`.

    Accepts either the bare per-timestep list or a full annotation dict (with the
    action under one of :data:`_ACTION_KEYS`) and returns a ``(T, 7)`` float32
    array. The gripper column is re-binarized defensively so the value is always
    exactly ``0.0`` or ``1.0``.
    """
    rows = _extract_action_list(annotation)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValueError(
            f"annotation must decode to (T, {ACTION_DIM}); got shape {arr.shape}"
        )
    arr[:, GRIPPER_DIM] = np.where(arr[:, GRIPPER_DIM] >= 0.5, 1.0, 0.0)
    return arr


def build_annotation_dict(
    actions: np.ndarray,
    instruction: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build the full on-disk annotation object written next to a Bridge clip.

    The action list lives under ``"action"`` (the key the entrypoints read);
    the instruction, when given, is stored under ``"texts"`` as a single-element
    list to mirror the Bridge / IRASim annotation files. ``extra`` is shallow-merged
    in for forward compatibility (e.g. ``"state"`` proprioception).
    """
    obj: dict[str, Any] = {"action": action_to_annotation(actions)}
    if instruction is not None:
        obj["texts"] = [instruction]
    if extra:
        obj.update(extra)
    return obj


def write_temp_annotation(
    actions: np.ndarray,
    instruction: Optional[str] = None,
    path: Optional[str] = None,
    as_dict: bool = True,
    extra: Optional[dict] = None,
) -> str:
    """Write a (perturbed) action to a temporary ``--input_annotation`` JSON file.

    Parameters
    ----------
    actions:
        ``(T, 7)`` AFP action array (already perturbed by the caller).
    instruction:
        Optional task string to embed (under ``"texts"``) when ``as_dict``.
    path:
        Destination path. If ``None``, a uniquely-named file is created in the
        system temp dir and its path returned; the caller is responsible for
        deleting it (the adapters do so after reading the model output).
    as_dict:
        If ``True`` write the full :func:`build_annotation_dict` object; if
        ``False`` write the bare per-timestep list.

    Returns
    -------
    str
        The path the annotation was written to.
    """
    if path is None:
        fd, path = tempfile.mkstemp(prefix="afp_annotation_", suffix=".json")
        os.close(fd)
    payload: Annotation = (
        build_annotation_dict(actions, instruction=instruction, extra=extra)
        if as_dict
        else action_to_annotation(actions)
    )
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


# --------------------------------------------------------------------------- #
# Video frame I/O                                                             #
# --------------------------------------------------------------------------- #
def load_video_frames(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """Load an mp4/video file to an ``(T, H, W, 3)`` uint8 RGB array.

    Tries, in order, the video backends that are *importable in the current
    environment*: ``imageio`` (ffmpeg), ``decord``, then ``opencv``. The imports
    are performed lazily here so this module imports cleanly with none installed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If no supported video backend is importable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"video not found: {path}")

    last_err: Optional[Exception] = None

    # 1) imageio (+ imageio-ffmpeg) -- the most common in the Cosmos/IRASim stacks.
    try:
        import imageio.v3 as iio  # type: ignore

        frames = iio.imread(path, plugin="pyav")
        arr = np.asarray(frames)
        if arr.ndim == 3:  # single frame
            arr = arr[None]
        return _finalize_rgb(arr, max_frames)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        last_err = exc

    # 2) decord -- fast, used by several video-LM pipelines.
    try:
        from decord import VideoReader  # type: ignore

        vr = VideoReader(path)
        n = len(vr) if max_frames is None else min(max_frames, len(vr))
        arr = vr.get_batch(list(range(n))).asnumpy()
        return _finalize_rgb(arr, max_frames)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        last_err = exc

    # 3) opencv -- decodes BGR, so flip to RGB.
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        out: List[np.ndarray] = []
        while True:
            if max_frames is not None and len(out) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame[:, :, ::-1])  # BGR -> RGB
        cap.release()
        if not out:
            raise RuntimeError("opencv decoded zero frames")
        return _finalize_rgb(np.stack(out), max_frames)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        last_err = exc

    raise RuntimeError(
        "No usable video backend found to read frames. Install one of "
        "`imageio[ffmpeg]`, `decord`, or `opencv-python` in the GPU environment. "
        f"Last backend error: {last_err!r}"
    )


def _finalize_rgb(arr: np.ndarray, max_frames: Optional[int]) -> np.ndarray:
    """Coerce a decoded array to ``(T, H, W, 3)`` uint8 RGB, optionally trimmed."""
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[-1] < 3:
        raise ValueError(f"decoded video has unexpected shape {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if max_frames is not None:
        arr = arr[:max_frames]
    return arr


def take_prediction_frames(frames: np.ndarray, horizon: int) -> np.ndarray:
    """Return the LAST ``horizon`` frames: the model's predictions.

    video2world-style entrypoints commonly save the conditioning frame(s)
    *followed by* the predicted frames in one clip. Slicing the first ``horizon``
    frames would then (a) include the input frame -- which is identical across
    every rollout of an episode and so deflates ``S(t)`` at ``t=1`` -- and
    (b) drop the final prediction. Taking the last ``horizon`` frames is correct
    both when the clip is ``[context..., predictions...]`` and when it is the
    predictions alone. If the clip has fewer than ``horizon`` frames it is
    returned unchanged (the caller records the raw count for the Stage-0
    frame-accounting check).
    """
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"frames must be (T,H,W,3); got shape {arr.shape}")
    if horizon <= 0 or arr.shape[0] <= horizon:
        return arr
    return arr[-horizon:]


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an ``(T, H, W, 3)`` (or single ``(H, W, 3)``) array to ``height x width``.

    Prefers a high-quality backend (``PIL`` then ``opencv``) when importable, and
    otherwise falls back to a dependency-free numpy nearest-neighbour resample so
    that callers can always normalise a start frame to the model's native size.
    Returns ``(T, height, width, 3)`` uint8.
    """
    arr = np.asarray(frames)
    single = arr.ndim == 3
    if single:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3) or (H,W,3); got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[1] == height and arr.shape[2] == width:
        out = arr
    else:
        out = _resize_impl(arr, height, width)
    return out[0] if single else out


def _resize_impl(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    # 1) PIL -- bilinear, broadly available.
    try:
        from PIL import Image  # type: ignore

        return np.stack(
            [
                np.asarray(
                    Image.fromarray(f).resize((width, height), Image.BILINEAR)
                )
                for f in arr
            ]
        ).astype(np.uint8)
    except Exception:  # pragma: no cover - depends on optional dep
        pass

    # 2) opencv.
    try:
        import cv2  # type: ignore

        return np.stack(
            [cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR) for f in arr]
        ).astype(np.uint8)
    except Exception:  # pragma: no cover - depends on optional dep
        pass

    # 3) numpy nearest-neighbour fallback (no external dep).
    src_h, src_w = arr.shape[1], arr.shape[2]
    row_idx = (np.arange(height) * (src_h / height)).astype(np.int64).clip(0, src_h - 1)
    col_idx = (np.arange(width) * (src_w / width)).astype(np.int64).clip(0, src_w - 1)
    return arr[:, row_idx][:, :, col_idx].astype(np.uint8)
