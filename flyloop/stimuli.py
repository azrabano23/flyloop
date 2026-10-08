"""Looming stimuli with exact ground truth.

A sphere of radius R approaches the camera head-on at constant speed v. Everything the
detectors are asked to find is known in closed form here, which is the whole point: this
is the ruler we check the instruments against before pointing them at a world model.

Geometry. With the sphere at distance d, the full angular size subtended is

    theta(d) = 2 * atan(R / d)

and a pinhole camera with focal length f (in pixels) projects it to a disc of radius

    r_px = f * tan(theta / 2) = f * R / d

Real looming is hyperbolic: theta blows up as d -> 0. The broken variants below keep the
same start and end size but get the shape of theta(t) wrong, which is exactly the failure
mode we want a detector to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Clip", "looming", "PROFILES", "render"]


@dataclass
class Clip:
    """A rendered approach, with the ground truth that generated it.

    frames: (T, H, W) uint8, light background, dark object.
    theta_deg: (T,) true full angular size of the object in each frame.
    t_contact_s: time of collision measured from the first frame. May be past the end of
        the clip. None when the profile is not a physical approach (the broken controls),
        because "time to contact" is not defined for a trajectory that never happened.
    """

    frames: np.ndarray
    theta_deg: np.ndarray
    fps: float
    label: str
    physical: bool
    t_contact_s: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps


def _theta_from_distance(radius_m: float, distance_m: np.ndarray) -> np.ndarray:
    """Full angular size in radians. Distances are clamped off zero to stay finite."""
    d = np.maximum(distance_m, 1e-6)
    return 2.0 * np.arctan(radius_m / d)


def _profile_looming(t: np.ndarray, radius_m: float, d0_m: float, speed_mps: float) -> np.ndarray:
    """The real thing: constant-velocity approach, hyperbolic expansion."""
    return _theta_from_distance(radius_m, d0_m - speed_mps * t)


def _resample_to_endpoints(shape: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """Stretch an arbitrary 0..1 shape onto the true start and end angular size.

    Keeping the endpoints pinned is what makes the broken controls honest. They start and
    finish at the same apparent size as the physical clip, so a detector cannot tell them
    apart by looking at any single frame -- only the *shape* of the expansion differs.
    """
    lo, hi = float(theta_true[0]), float(theta_true[-1])
    s = (shape - shape.min()) / max(float(np.ptp(shape)), 1e-12)
    return lo + s * (hi - lo)


def _profile_linear(t: np.ndarray, theta_true: np.ndarray, **_) -> np.ndarray:
    """Expands at a constant angular rate. Looks plausible, is geometrically impossible."""
    return _resample_to_endpoints(t, theta_true)


def _profile_reversed(t: np.ndarray, theta_true: np.ndarray, **_) -> np.ndarray:
    """Decelerating expansion -- the time-reverse of a real approach."""
    shape = theta_true[0] + theta_true[-1] - theta_true[::-1]
    return _resample_to_endpoints(shape, theta_true)


def _profile_jump(t: np.ndarray, theta_true: np.ndarray, **_) -> np.ndarray:
    """Correct physics with a discontinuity: the object teleports closer at the midpoint.

    Models the consistency break world models actually make -- scale popping between
    chunks -- rather than a smooth-but-wrong trajectory.
    """
    theta = theta_true.copy()
    mid = len(theta) // 2
    theta[mid:] = np.minimum(theta[mid:] * 1.6, np.pi * 0.98)
    return theta


_PROFILE_FNS = {
    "looming": None,  # the physical case; handled directly
    "linear": _profile_linear,
    "reversed": _profile_reversed,
    "jump": _profile_jump,
}

PROFILES = tuple(_PROFILE_FNS)


def render(theta_rad: np.ndarray, size: int = 192, fov_deg: float = 90.0) -> np.ndarray:
    """Draw a centred dark disc of the given angular size onto a light background.

    Uses the true pinhole projection r = f * tan(theta/2) rather than a small-angle
    shortcut, so the physical clip really is physical out to large angles. The edge is
    antialiased -- a hard-edged disc quantises angular size to whole pixels, which shows
    up as a staircase in anything that differentiates the signal.
    """
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    r_px = f * np.tan(np.clip(theta_rad, 0.0, np.pi * 0.99) / 2.0)

    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)

    # alpha = 1 inside, 0 outside, one pixel of ramp between.
    alpha = np.clip(r_px[:, None, None] + 0.5 - dist[None], 0.0, 1.0)
    frames = (255.0 * (1.0 - alpha)).astype(np.uint8)
    return frames


def looming(
    profile: str = "looming",
    radius_m: float = 0.05,
    d0_m: float = 1.0,
    speed_mps: float = 0.5,
    fps: float = 60.0,
    stop_at_deg: float = 120.0,
    size: int = 192,
    fov_deg: float = 90.0,
) -> Clip:
    """Render one approach.

    Defaults are a 10 cm ball closing from 1 m at 0.5 m/s, which puts contact at 2 s and
    sweeps the angular size through the range where a fly's escape circuit commits.

    stop_at_deg truncates the clip once the object fills that much of the view, so the
    disc never grows past the frame and starts clipping (which would corrupt the very
    expansion signal we are measuring).
    """
    if profile not in _PROFILE_FNS:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")

    t_contact = d0_m / speed_mps
    t = np.arange(0.0, t_contact, 1.0 / fps)
    theta_true = _profile_looming(t, radius_m, d0_m, speed_mps)

    keep = theta_true <= np.deg2rad(stop_at_deg)
    if not keep.any():
        raise ValueError("object already exceeds stop_at_deg in the first frame")
    t, theta_true = t[keep], theta_true[keep]

    fn = _PROFILE_FNS[profile]
    theta = theta_true if fn is None else fn(t, theta_true=theta_true)

    # A profile may push past the cap even though the physical trajectory did not (the
    # jump control does, by construction). Truncate again so every clip stays inside the
    # frame -- a disc that overflows the border stops expanding in the image and would
    # silently flatten the signal we measure.
    keep = theta <= np.deg2rad(stop_at_deg)
    if not keep.all():
        cut = int(np.argmin(keep))
        if cut == 0:
            raise ValueError(f"profile {profile!r} exceeds stop_at_deg in the first frame")
        t, theta = t[:cut], theta[:cut]

    return Clip(
        frames=render(theta, size=size, fov_deg=fov_deg),
        theta_deg=np.rad2deg(theta),
        fps=fps,
        label=profile,
        physical=(profile == "looming"),
        t_contact_s=t_contact if profile == "looming" else None,
        meta={
            "radius_m": radius_m,
            "d0_m": d0_m,
            "speed_mps": speed_mps,
            "fov_deg": fov_deg,
            "size": size,
        },
    )
