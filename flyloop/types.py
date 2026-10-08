"""Shared types for detectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Detection", "lowpass"]


@dataclass
class Detection:
    """What a detector concluded about one clip.

    trace is the detector's internal decision variable per frame, normalised to 0..1 by
    its own firing threshold, so 1.0 means "committed". Putting both detectors on the
    same scale is what lets the UI plot them on one axis.
    """

    name: str
    fired: bool
    fire_frame: int | None
    trace: np.ndarray
    extra: dict = field(default_factory=dict)

    def fire_time_s(self, fps: float) -> float | None:
        return None if self.fire_frame is None else self.fire_frame / fps


def lowpass(x: np.ndarray, tau_s: float, fps: float) -> np.ndarray:
    """First-order exponential lowpass along axis 0.

    Used for photoreceptor adaptation and for the delay line in the motion detectors.
    A plain IIR filter is the honest model here: real neurons low-pass with a membrane
    time constant, they do not convolve with a windowed kernel.
    """
    if tau_s <= 0:
        return x.astype(np.float64, copy=True)

    alpha = float(np.exp(-1.0 / (tau_s * fps)))
    out = np.empty_like(x, dtype=np.float64)
    acc = np.asarray(x[0], dtype=np.float64)
    out[0] = acc
    for t in range(1, len(x)):
        acc = alpha * acc + (1.0 - alpha) * x[t]
        out[t] = acc
    return out
