"""The giant-fibre model: two channels, because one is provably not enough.

We measured a frontier with the single-channel `EMDProbe`: normalise it and the threshold
is stable across approach speed (1.24 deg SD) but it stops noticing clips whose expansion
profile is wrong; leave it raw and it catches them (5-frame lead over trig) but the
threshold slides with speed (4.95 deg SD). One knob, two properties, no setting that gets
both.

The animal does not have this problem, and the published anatomy says why. Two visual
projection populations converge on the giant fibre and they are tuned to different things:
LC4 responds to angular VELOCITY, LPLC2 to angular SIZE (Ache et al. 2019, on looming size
and velocity encoding in the Drosophila giant fibre escape pathway). It is that asymmetry
that makes the pair a size-and-speed detector when neither is one alone.

Two populations, in parallel, summed onto one descending neuron. Our two `EMDProbe`
configurations turn out to be crude versions of exactly those channels:

    normalised pooled signal  ~  what fraction of the field is expanding  ~  SIZE   ~ LPLC2
    raw pooled signal         ~  how fast the image is moving             ~  RATE   ~ LC4

So we run both and combine them with the published model, instead of picking a point on a
frontier that the biology simply does not sit on.

WHAT IS OURS AND WHAT IS NOT. The equations and the parameter values in `GF_PARAMS` are
the published giant-fibre model, not our invention. What is ours is the front end. That
model is normally driven by an analytic looming stimulus that already knows its own
angular size, which is no use at all on a world model's output, because knowing the true
angular size is the thing we do not have. We drive it from the facet array instead, so it
runs on arbitrary video and never segments an object.
"""

from __future__ import annotations

import numpy as np

from .probe import EMDProbe
from .retina import Retina
from .types import Detection

__all__ = ["GF_PARAMS", "GiantFiberProbe"]

# Published giant-fibre model parameters (Ache et al. 2019).
GF_PARAMS = dict(
    w_LC4=1.62, w_LPLC2=1.45, w_i1=2.27, w_i2=1.0,
    lc4_gain_mv_per_deg_per_s=2.567e-4, lc4_delay_ms=19.0,
    lplc2_peak_mv=1.7, lplc2_mu_deg=42.0, lplc2_sigma_log=0.52, lplc2_delay_ms=19.0,
    i1_offset=-0.53, i1_amp=0.59, i1_mid_deg=66.0, i1_slope=-11.0, i1_delay_ms=37.5,
    i2_amp=-0.52, i2_mu_deg=26.0, i2_sigma_deg=7.8, i2_delay_ms=11.0,
)


def _delay(x: np.ndarray, ms: float, fps: float) -> np.ndarray:
    """Shift a signal later in time, holding the first value. Conduction delay."""
    k = int(round(ms / 1000.0 * fps))
    if k <= 0:
        return x
    return np.concatenate([np.full(k, x[0]), x[:-k]])


class GiantFiberProbe:
    """Size and rate channels summed onto one escape command.

    The size channel is calibrated once: a monotonic map from "fraction of the facet array
    showing outward motion" to degrees, fitted against the analytic clip whose angular size
    is known in closed form. That map is the only thing fitted here; the GF weights and
    tuning curves are the published values.
    """

    name = "giant fibre (LC4 + LPLC2)"

    def __init__(self, retina: Retina, params: dict | None = None):
        self.retina = retina
        self.params = dict(GF_PARAMS, **(params or {}))
        # The two channels, as the two EMDProbe configurations we already characterised.
        self._size = EMDProbe(retina, normalize=True, sigma=0.003, tau_adapt_s=0.200)
        self._rate = EMDProbe(retina, normalize=False, tau_adapt_s=0.080)
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self.threshold_mv: float | None = None

    # ------------------------------------------------------------------- channels

    def _channels(self, frames: np.ndarray, fps: float, fov_deg: float):
        f = self.retina.sample(frames, fov_deg=fov_deg)
        _, size_raw = self._size.expansion_signal(f, fps)
        per_facet, rate_raw = self._rate.expansion_signal(f, fps)
        return per_facet, size_raw, rate_raw

    def _size_deg(self, size_raw: np.ndarray) -> np.ndarray:
        """Map the normalised expansion fraction onto degrees."""
        if self._map_x is None:
            raise RuntimeError("call calibrate() before run()")
        return np.interp(size_raw, self._map_x, self._map_y)

    # ---------------------------------------------------------------------- model

    def _gf(self, size_deg: np.ndarray, fps: float) -> dict:
        """The published GF equations, driven by our estimated size trace."""
        p = self.params
        rate_deg_s = np.gradient(size_deg) * fps

        v_lc4 = p["lc4_gain_mv_per_deg_per_s"] * _delay(rate_deg_s, p["lc4_delay_ms"], fps)

        s = np.clip(_delay(size_deg, p["lplc2_delay_ms"], fps), 1e-3, None)
        v_lplc2 = p["lplc2_peak_mv"] * np.exp(
            -((np.log(s) - np.log(p["lplc2_mu_deg"])) ** 2) / (2 * p["lplc2_sigma_log"] ** 2)
        )

        s1 = _delay(size_deg, p["i1_delay_ms"], fps)
        v_i1 = p["i1_offset"] + p["i1_amp"] / (
            1 + np.exp(-(s1 - p["i1_mid_deg"]) / p["i1_slope"])
        )

        s2 = _delay(size_deg, p["i2_delay_ms"], fps)
        v_i2 = p["i2_amp"] * np.exp(
            -((s2 - p["i2_mu_deg"]) ** 2) / (2 * p["i2_sigma_deg"] ** 2)
        )

        v_gf = (p["w_LC4"] * v_lc4 + p["w_LPLC2"] * v_lplc2
                + p["w_i1"] * v_i1 + p["w_i2"] * v_i2)
        return {"v_gf": v_gf, "v_lc4": v_lc4, "v_lplc2": v_lplc2,
                "size_deg": size_deg, "rate_deg_s": rate_deg_s}

    # ----------------------------------------------------------------- calibration

    def calibrate(
        self,
        frames: np.ndarray,
        fps: float,
        theta_true_deg: np.ndarray,
        target_deg: float = 20.0,
        fov_deg: float = 90.0,
    ) -> float:
        """Fit the size map and set the firing threshold, on known physics only."""
        _, size_raw, _ = self._channels(frames, fps, fov_deg)

        # Monotonic map from expansion fraction -> degrees. np.interp needs ascending x,
        # and the raw signal is noisy early, so sort and take a running maximum.
        order = np.argsort(size_raw)
        x, y = size_raw[order], np.maximum.accumulate(theta_true_deg[order])
        keep = np.concatenate([[True], np.diff(x) > 1e-9])
        self._map_x, self._map_y = x[keep], y[keep]

        out = self._gf(self._size_deg(size_raw), fps)
        idx = np.flatnonzero(theta_true_deg >= target_deg)
        if not idx.size:
            raise ValueError(f"calibration clip never reaches {target_deg} deg")

        # Fire where the escape command peaks within the window up to the target size.
        self.threshold_mv = float(out["v_gf"][: int(idx[0]) + 1].max())
        return self.threshold_mv

    # ------------------------------------------------------------------------- run

    def run(self, frames: np.ndarray, fps: float, fov_deg: float = 90.0) -> Detection:
        if self.threshold_mv is None:
            raise RuntimeError("call calibrate() before run()")

        per_facet, size_raw, _ = self._channels(frames, fps, fov_deg)
        out = self._gf(self._size_deg(size_raw), fps)

        # Same warmup guard as the single-channel probe: the size channel inherits the
        # adaptation filter, so an early crossing is settling, not an escape.
        warmup = min(int(3 * self._size.tau_adapt_s * fps), len(out["v_gf"]) // 2)
        above = np.flatnonzero(out["v_gf"][warmup:] >= self.threshold_mv)
        fire_frame = int(above[0] + warmup) if above.size else None

        return Detection(
            name=self.name,
            fired=fire_frame is not None,
            fire_frame=fire_frame,
            trace=np.clip(out["v_gf"] / self.threshold_mv, 0.0, 1.5),
            extra={
                "facet_activity": per_facet,
                "v_lc4": out["v_lc4"],
                "v_lplc2": out["v_lplc2"],
                "size_deg_est": out["size_deg"],
                "threshold_mv": self.threshold_mv,
            },
        )
