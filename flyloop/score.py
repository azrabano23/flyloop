"""Turn two detections into the numbers we actually report."""

from __future__ import annotations

import numpy as np

from .types import Detection

__all__ = ["score_clip", "summarise"]

# How far apart two detectors can commit and still count as agreeing. 4 frames at 60 fps
# is ~67 ms, which is the scale of the escape decision itself, so anything inside it is
# the same call made at slightly different moments rather than a real disagreement.
AGREEMENT_TOLERANCE_FRAMES = 4


def score_clip(
    fly: Detection,
    baseline: Detection,
    fps: float,
    theta_true_deg: np.ndarray | None,
    t_contact_s: float | None,
) -> dict:
    """Score one clip.

    theta_true_deg is None for anything a world model generated -- we do not know the real
    geometry there, which is the entire reason this project exists. In that case the
    angular size at firing falls back to the baseline's *estimate* and is flagged, so a
    reader can never confuse a measured threshold with an inferred one.
    """
    est_theta = baseline.extra.get("theta_deg")

    def theta_at(d: Detection) -> float | None:
        if d.fire_frame is None:
            return None
        src = theta_true_deg if theta_true_deg is not None else est_theta
        if src is None or d.fire_frame >= len(src):
            return None
        return round(float(src[d.fire_frame]), 2)

    gap = (
        abs(fly.fire_frame - baseline.fire_frame)
        if fly.fire_frame is not None and baseline.fire_frame is not None
        else None
    )

    # Time-to-contact error, only meaningful when we know when contact really was.
    ttc_error = None
    if t_contact_s is not None and fly.fire_frame is not None:
        tau = baseline.extra.get("tau_s")
        if tau is not None and fly.fire_frame < len(tau):
            predicted = fly.fire_frame / fps + float(tau[fly.fire_frame])
            if np.isfinite(predicted):
                ttc_error = round(abs(predicted - t_contact_s), 4)

    return {
        "theta_threshold_deg": theta_at(fly),
        "theta_threshold_is_estimate": theta_true_deg is None,
        "baseline_threshold_deg": theta_at(baseline),
        "ttc_error_s": ttc_error,
        "detectors_agree": (gap is not None and gap <= AGREEMENT_TOLERANCE_FRAMES),
        "disagreement_frames": gap,
        # Negative means the fly committed first. On every broken control so far it does,
        # because it reads how fast the image is expanding and the baseline only reads
        # how big it has become.
        "fly_lead_frames": (
            None
            if gap is None
            else int(fly.fire_frame - baseline.fire_frame)  # type: ignore[operator]
        ),
    }


def summarise(clips: list[dict]) -> dict:
    """Aggregate across clips.

    The threshold statistics are computed over *physically correct* clips only. Averaging
    in the broken controls would be measuring our own sabotage, and the spread across
    correct clips is the number that says whether the instrument is stable.
    """
    real = [
        c for c in clips
        if c.get("physical") is True and c["scores"]["theta_threshold_deg"] is not None
    ]
    thetas = np.array([c["scores"]["theta_threshold_deg"] for c in real], dtype=float)

    scored = [c for c in clips if c["scores"]["disagreement_frames"] is not None]
    agree = [c for c in scored if c["scores"]["detectors_agree"]]

    return {
        "n_clips": len(clips),
        "n_physical": len(real),
        "threshold_mean_deg": round(float(thetas.mean()), 2) if len(thetas) else None,
        "threshold_sd_deg": round(float(thetas.std(ddof=1)), 2) if len(thetas) > 1 else None,
        "baseline_vs_fly_agreement": (
            round(len(agree) / len(scored), 3) if scored else None
        ),
        "agreement_tolerance_frames": AGREEMENT_TOLERANCE_FRAMES,
    }
