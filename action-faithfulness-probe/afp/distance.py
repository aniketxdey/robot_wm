"""Perceptual distance backends for comparing predicted frames.

A ``Distance`` maps two frame sequences of shape ``(T, H, W, 3)`` uint8 to a
per-frame distance vector of shape ``(T,)`` (float). AFP's metrics build
``S(t)`` curves on top of this.

Backends
--------
- ``pixel_l2``  : always available, no heavy deps. Mean per-pixel L2 in [0,1].
- ``lpips``     : LPIPS (AlexNet/VGG); needs ``torch`` + ``lpips``. Inputs
                  normalized to [-1,1] per the lpips package contract.
- ``dreamsim``  : DreamSim ensemble; needs ``torch`` + ``dreamsim``.

The default backend is ``pixel_l2`` so the whole harness runs on CPU with no GPU
and no model downloads. Pass ``--distance lpips`` (etc.) to use a real backend.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _to_float01(frames: np.ndarray) -> np.ndarray:
    f = np.asarray(frames).astype(np.float32)
    if f.max() > 1.5:  # uint8-scale
        f = f / 255.0
    return f


class Distance:
    """Base class: callable mapping (frames_a, frames_b) -> per-frame distances (T,)."""

    name: str = "abstract"
    higher_is_more_different: bool = True

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PixelL2Distance(Distance):
    """Mean per-pixel Euclidean distance in [0,1] space, per frame."""

    name = "pixel_l2"

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        fa, fb = _to_float01(a), _to_float01(b)
        assert fa.shape == fb.shape, f"shape mismatch {fa.shape} vs {fb.shape}"
        diff = fa - fb
        # per-frame RMS over (H, W, 3)
        return np.sqrt(np.mean(diff ** 2, axis=(1, 2, 3))).astype(np.float64)


class LPIPSDistance(Distance):
    """LPIPS perceptual distance. Lazy-imports torch + lpips.

    Frames are normalized to [-1, 1] and laid out as (N, 3, H, W) per the lpips
    package's documented contract ("image should be RGB, normalized to [-1,1]").
    """

    name = "lpips"

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        try:
            import torch  # noqa: F401
            import lpips as _lpips
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "LPIPS backend requires `pip install afp[lpips]` (torch + lpips)."
            ) from e
        import torch
        self._torch = torch
        self._fn = _lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self._device = device

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        torch = self._torch
        fa = _to_float01(a) * 2.0 - 1.0
        fb = _to_float01(b) * 2.0 - 1.0
        # (T,H,W,3) -> (T,3,H,W)
        ta = torch.from_numpy(fa.transpose(0, 3, 1, 2)).float().to(self._device)
        tb = torch.from_numpy(fb.transpose(0, 3, 1, 2)).float().to(self._device)
        with torch.no_grad():
            d = self._fn(ta, tb)  # (T,1,1,1)
        return d.squeeze().cpu().numpy().reshape(-1).astype(np.float64)


class DreamSimDistance(Distance):
    """DreamSim perceptual distance. Lazy-imports torch + dreamsim."""

    name = "dreamsim"

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch  # noqa: F401
            from dreamsim import dreamsim as _dreamsim
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "DreamSim backend requires `pip install afp[dreamsim]` (torch + dreamsim)."
            ) from e
        import torch
        self._torch = torch
        self._model, self._preprocess = _dreamsim(pretrained=True, device=device)
        self._device = device

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        torch = self._torch
        from PIL import Image
        out = []
        for fa, fb in zip(a, b):
            ia = self._preprocess(Image.fromarray(fa.astype(np.uint8))).to(self._device)
            ib = self._preprocess(Image.fromarray(fb.astype(np.uint8))).to(self._device)
            with torch.no_grad():
                out.append(float(self._model(ia, ib).item()))
        return np.asarray(out, dtype=np.float64)


_BACKENDS: Dict[str, Callable[..., Distance]] = {
    "pixel_l2": PixelL2Distance,
    "lpips": LPIPSDistance,
    "dreamsim": DreamSimDistance,
}


def get_distance(name: str = "pixel_l2", **kwargs) -> Distance:
    """Construct a distance backend by name.

    Falls back to ``pixel_l2`` only if explicitly requested; otherwise raises a
    helpful error so silent metric degradation cannot happen unnoticed.
    """
    if name not in _BACKENDS:
        raise KeyError(f"unknown distance '{name}'. options: {sorted(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)


def available_backends() -> Dict[str, bool]:
    """Report which backends can actually be constructed in this environment."""
    status = {}
    for n, cls in _BACKENDS.items():
        try:
            cls()  # type: ignore[call-arg]
            status[n] = True
        except Exception:
            status[n] = False
    return status
