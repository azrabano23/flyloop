"""A looming detector built the way a fly builds one.

The pipeline mirrors the known anatomy of the *Drosophila* escape pathway:

    photoreceptors   adapt away the mean, keep the change
    lamina / medulla  pairwise correlations between neighbouring facets -> local motion
    LPLC2             pool motion that points *outward* from the centre of the field
    giant fibre       integrate that pool; one spike commits the escape

What matters for this project is the input it is allowed to see. It gets a vector of
facet brightnesses and the hex topology. It never segments an object, never estimates a
distance, never learns a focal length. So unlike the analytic baseline it cannot be
accidentally measuring the geometry it was handed.

HONESTY NOTE. `EMDProbe` below is a *model of* the circuit, not the circuit. Connectivity
and sign come from the published architecture, but the time constants and gains are ours.
The real article is Ryan's `fly-C`, which compiles the circuit straight out of the
male-CNS connectome; `ConnectomeProbe` is the seam where it plugs in. Until that is wired,
every result must be labelled "emd + lplc2 (fallback)" and never "connectome".
"""

from __future__ import annotations

import numpy as np

from .retina import Retina
from .types import Detection, lowpass

__all__ = ["EMDProbe", "ConnectomeProbe"]


class EMDProbe:
    """Elementary motion detectors feeding a leaky integrator.

    One gain parameter is fitted, once, against a clip whose physics is known exactly
    (see `calibrate`). Everything after that is held fixed. Calibrating an instrument on
    a known standard and then not touching it again is the difference between measuring
    and curve-fitting.
    """

    name = "emd + lplc2 (fallback)"

    def __init__(
        self,
        retina: Retina,
        tau_adapt_s: float = 0.080,
        tau_emd_s: float = 0.035,
        tau_membrane_s: float = 0.050,
        gain: float = 1.0,
    ):
        self.retina = retina
        self.tau_adapt_s = tau_adapt_s
        self.tau_emd_s = tau_emd_s
        self.tau_membrane_s = tau_membrane_s
        self.gain = gain
        self._edges, self._radial_w = self._build_edges(retina)

    # ------------------------------------------------------------------ topology

    @staticmethod
    def _build_edges(retina: Retina) -> tuple[np.ndarray, np.ndarray]:
        """Directed facet pairs, each weighted by how outward-pointing it is.

        A pair contributes to the looming signal in proportion to how closely it aligns
        with the radial direction. Pairs tangential to the centre weigh ~0, so a scene
        panning sideways produces little expansion signal -- which is the selectivity the
        real circuit has, and the reason it is not just a brightness-change alarm.
        """
        radial = retina.radial_unit()
        edges, weights = [], []

        for i in range(retina.n_facets):
            for j in retina.neighbors[i]:
                if j < 0:
                    continue
                d = retina.directions[j] - retina.directions[i]
                n = np.linalg.norm(d)
                if n < 1e-9:
                    continue
                edges.append((i, int(j)))
                weights.append(float(np.dot(d / n, radial[i])))

        return np.asarray(edges, dtype=np.int64), np.asarray(weights, dtype=np.float64)

    # ------------------------------------------------------------------- the model

    def expansion_signal(self, facets: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
        """Run the front end. Returns (per-facet outward motion (T, F), pooled (T,))."""
        # Photoreceptor adaptation: keep what changed, discard the standing level. This is
        # why a uniformly dim scene does not look like an approaching object.
        hp = facets - lowpass(facets, self.tau_adapt_s, fps)

        # Reichardt correlator: facet i delayed against facet j undelayed, minus the
        # mirror image. Positive means motion ran from i to j.
        delayed = lowpass(hp, self.tau_emd_s, fps)
        i, j = self._edges[:, 0], self._edges[:, 1]
        corr = delayed[:, i] * hp[:, j] - hp[:, i] * delayed[:, j]

        # Weight by outwardness and accumulate onto the inner facet of each pair.
        contrib = corr * self._radial_w[None, :]
        per_facet = np.zeros((len(facets), self.retina.n_facets), dtype=np.float64)
        np.add.at(per_facet, (slice(None), i), contrib)

        # Only outward motion excites the escape pathway; contraction is not a threat.
        per_facet = np.maximum(per_facet, 0.0)
        return per_facet, per_facet.mean(axis=1)

    def membrane(self, pooled: np.ndarray, fps: float) -> np.ndarray:
        """Giant-fibre membrane potential: leaky integration of the pooled drive."""
        return self.gain * lowpass(pooled, self.tau_membrane_s, fps)

    def run(self, frames: np.ndarray, fps: float, fov_deg: float = 90.0) -> Detection:
        facets = self.retina.sample(frames, fov_deg=fov_deg)
        per_facet, pooled = self.expansion_signal(facets, fps)
        v = self.membrane(pooled, fps)

        above = np.flatnonzero(v >= 1.0)
        fire_frame = int(above[0]) if above.size else None

        return Detection(
            name=self.name,
            fired=fire_frame is not None,
            fire_frame=fire_frame,
            trace=np.clip(v, 0.0, 1.5),
            extra={"facet_activity": per_facet, "pooled": pooled, "gain": self.gain},
        )

    # ----------------------------------------------------------------- calibration

    def calibrate(
        self,
        frames: np.ndarray,
        fps: float,
        theta_true_deg: np.ndarray,
        target_deg: float = 20.0,
        fov_deg: float = 90.0,
    ) -> float:
        """Set the one free gain so the circuit commits at `target_deg` on known physics.

        Calibrated against a clip whose angular size is known in closed form, not against
        anything a world model produced. After this returns, the gain is frozen and every
        other clip is scored with it -- including the broken controls, which is what makes
        their results meaningful.
        """
        idx = np.flatnonzero(theta_true_deg >= target_deg)
        if not idx.size:
            raise ValueError(
                f"calibration clip never reaches {target_deg} deg "
                f"(max {theta_true_deg.max():.1f}); use a closer approach"
            )
        target_frame = int(idx[0])

        facets = self.retina.sample(frames, fov_deg=fov_deg)
        _, pooled = self.expansion_signal(facets, fps)
        v_unit = lowpass(pooled, self.tau_membrane_s, fps)

        peak = float(v_unit[: target_frame + 1].max())
        if peak <= 1e-12:
            raise ValueError("no expansion signal on the calibration clip; check the retina fov")

        self.gain = 1.0 / peak
        return self.gain


class ConnectomeProbe:
    """Adapter for Ryan's fly-C, which compiles the real circuit from the connectome.

    fly-C (`RyanRana/fly-C`, "Neural Transistor") extracts a named circuit from the
    male-CNS connectome as a signed sparse graph, prunes it, quantises to int8 and emits
    C. Its `collision` circuit already reproduces the behaviour we care about: it fires
    about 56 ms before impact, onset scaling r = -0.9999, and the angular threshold holds
    at 19.9 +/- 2.7 deg across an 8x speed range -- from unfitted wiring.

    The open question, and the first thing to settle with Ryan: does fly-C accept a
    stream of photoreceptor values, or only its own predefined synthetic stimuli? If the
    former this is a thin shim over `nt.circuit("collision")` plus `evaluate`. If the
    latter, the visual front end has to be built, and `EMDProbe` carries the demo.
    """

    name = "connectome (fly-C)"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "fly-C not wired yet -- use EMDProbe. See the module docstring for the one "
            "question that decides how much work this is."
        )
