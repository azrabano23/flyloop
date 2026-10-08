"""The honest baseline: find the object, measure how big it looks, fire at a threshold.

This is the detector a reviewer will reach for immediately, and they are right to. If the
connectome adds nothing over three lines of trigonometry, we need to know that and say so.

The baseline works like this:

    threshold the frame   ->  pixel area of the dark blob
    area                  ->  equivalent disc radius in pixels
    radius                ->  angular size, theta = 2 * atan(r / f)
    theta >= 20 deg       ->  fire

Note what it needs that the fly does not: it has to **segment the object**. It assumes the
scene contains one dark thing on a light background, and it assumes the camera's focal
length so it can convert pixels to degrees. Both assumptions encode the projective
geometry we are trying to test, which makes the baseline partly circular. That is the
point of the comparison, not a flaw to be fixed.
"""

from __future__ import annotations

import numpy as np

from .types import Detection

__all__ = ["angular_size_deg", "tau_ttc_s", "AnalyticBaseline"]

# Angular size at which a fly's escape commits, from the behavioural literature and
# reproduced by fly-C from unfitted connectome wiring at 19.9 +/- 2.7 deg. We hold the
# baseline to the same number so the two detectors are asked the same question.
ESCAPE_THRESHOLD_DEG = 20.0


def angular_size_deg(
    frames: np.ndarray, fov_deg: float = 90.0, level: float = 0.5
) -> np.ndarray:
    """Estimate the full angular size of the dark object in each frame.

    Measures area rather than width because area degrades gracefully: a few stray dark
    pixels move sqrt(area) far less than they move a bounding box, and a partially
    occluded object still reports a sensible size.
    """
    if frames.ndim == 4:  # colour -> luma
        frames = frames[..., :3].mean(axis=-1)

    size = frames.shape[1]
    f_px = (size / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)

    lum = frames.astype(np.float64) / 255.0
    area = (lum < level).sum(axis=(1, 2)).astype(np.float64)
    r_px = np.sqrt(area / np.pi)
    return np.rad2deg(2.0 * np.arctan(r_px / f_px))


def tau_ttc_s(theta_deg: np.ndarray, fps: float) -> np.ndarray:
    """Optical time-to-contact, the classic tau = theta / (d theta / dt).

    For a real constant-velocity approach this equals the true time remaining, without
    knowing the object's size or distance -- which is why it shows up everywhere in
    animal behaviour. On a clip whose expansion is not physical it returns nonsense, and
    that nonsense is a measurement.
    """
    theta = np.deg2rad(theta_deg)
    dtheta = np.gradient(theta) * fps
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(dtheta > 1e-9, theta / dtheta, np.inf)
    return tau


class AnalyticBaseline:
    """Threshold on measured angular size."""

    name = "analytic expansion"

    def __init__(self, threshold_deg: float = ESCAPE_THRESHOLD_DEG, fov_deg: float = 90.0):
        self.threshold_deg = threshold_deg
        self.fov_deg = fov_deg

    def run(self, frames: np.ndarray, fps: float) -> Detection:
        theta = angular_size_deg(frames, fov_deg=self.fov_deg)

        above = np.flatnonzero(theta >= self.threshold_deg)
        fire_frame = int(above[0]) if above.size else None

        return Detection(
            name=self.name,
            fired=fire_frame is not None,
            fire_frame=fire_frame,
            trace=np.clip(theta / self.threshold_deg, 0.0, 1.0),
            extra={
                "theta_deg": theta,
                "tau_s": tau_ttc_s(theta, fps),
                "threshold_deg": self.threshold_deg,
            },
        )
