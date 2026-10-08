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
A circuit compiled directly from the connectome graph would be the real article;
`ConnectomeProbe` is the seam where one plugs in. Until that is wired, every result must
be labelled "emd + lplc2 (fallback)" and never "connectome".
"""

from __future__ import annotations

import numpy as np

from .retina import Retina
from .types import Detection, lowpass

__all__ = ["EMDProbe", "ConnectomeProbe"]


class EMDProbe:
    """Elementary motion detectors feeding a leaky integrator.

    CALIBRATION, STATED PLAINLY. Three numbers in here were fitted by us: the output
    `gain` (see `calibrate`), the adaptation time constant `tau_adapt_s`, and the
    normalisation constant `sigma`. All three were set against *analytic* looming, whose
    geometry is known in closed form, by sweeping them to minimise the drift of the firing
    threshold across an 8x approach-speed range. They were never tuned against anything a
    world model produced, and they are frozen before any generated clip is scored.

    That sweep took the threshold from 19.0 +/- 4.95 deg to 20.97 +/- 1.24 deg. Worth
    being blunt about what that means: a real fly needs none of this. Its threshold holds
    across approach speed because of how the circuit is built, not because anyone tuned
    it. Having to hand-fit three knobs to approximate that is the honest reason to prefer
    a circuit taken from the connectome over a model of one.
    """

    name = "emd + lplc2 (fallback)"

    def __init__(
        self,
        retina: Retina,
        tau_adapt_s: float = 0.200,
        tau_emd_s: float = 0.035,
        tau_membrane_s: float = 0.050,
        gain: float = 1.0,
        normalize: bool = True,
        sigma: float = 0.003,
        contrast_norm: bool = True,
        sigma_c: float = 0.02,
    ):
        self.retina = retina
        self.tau_adapt_s = tau_adapt_s
        self.tau_emd_s = tau_emd_s
        self.tau_membrane_s = tau_membrane_s
        self.gain = gain
        self.normalize = normalize
        self.sigma = sigma
        self.contrast_norm = contrast_norm
        self.sigma_c = sigma_c
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

        if self.contrast_norm:
            # Contrast gain control, and this one was forced on us by real footage.
            #
            # Calibrated on a black disc against white, the circuit is tuned to enormous
            # contrast. Pointed at a generated hallway of flat yellow wallpaper it never
            # came close to firing (peak 0.35 of threshold) even though the wall really was
            # closing on the camera. The expansion was there; the absolute contrast was not.
            #
            # Dividing by the instantaneous spread of activity across the array makes the
            # response depend on the *structure* of the contrast rather than its depth, so
            # a faint scene and a stark one are read on the same scale. Flies do this in
            # the lamina for the same reason: real scenes vary in contrast by orders of
            # magnitude and a fixed-gain detector would be useless outdoors.
            # The normaliser has to be SLOW. Dividing by the instantaneous spread looks
            # equivalent and is not: in the first frames the adaptation filter has not
            # settled, the spread is near zero, and the division turns sensor noise into a
            # spike that fires the circuit at frame 5 of every clip. Real contrast gain
            # control adapts over hundreds of milliseconds, so a lagging estimate is both
            # the correct model and the stable one.
            rms = np.sqrt((hp**2).mean(axis=1, keepdims=True))
            hp = hp / (self.sigma_c + lowpass(rms, 0.3, fps))

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

        if not self.normalize:
            return per_facet, per_facet.mean(axis=1)

        # Per-facet divisive normalisation.
        #
        # Raw, this probe is not speed invariant, and it is not a small effect: across an
        # 8x approach-speed sweep the threshold slides from 25.5 deg down to 14.3 deg. A
        # bare Reichardt correlator is tuned to temporal frequency, not velocity, so a
        # faster approach trips the integrator at a smaller angular size. A real fly does
        # not do that; it commits at a roughly fixed angular size however fast the object
        # is coming, which is the property that makes it useful as a ruler.
        #
        # Dividing each facet's outward motion by the total motion magnitude at that same
        # facet replaces "how fast is this patch moving" with "is this patch moving
        # outward", which carries no speed term. Pooling those gives the *fraction of the
        # visual field that is expanding* -- and that fraction is set by the object's
        # angular size, which is exactly the quantity we want the circuit keyed to.
        #
        # Normalisation pools of this shape are ubiquitous in the fly visual system, so
        # this moves the model toward the biology rather than toward a nicer number.
        mag = np.zeros_like(per_facet)
        np.add.at(mag, (slice(None), i), np.abs(contrib))

        per_facet = per_facet / (self.sigma + mag)
        return per_facet, per_facet.mean(axis=1)

    def membrane(self, pooled: np.ndarray, fps: float) -> np.ndarray:
        """Giant-fibre membrane potential: leaky integration of the pooled drive."""
        return self.gain * lowpass(pooled, self.tau_membrane_s, fps)

    def run(self, frames: np.ndarray, fps: float, fov_deg: float = 90.0) -> Detection:
        facets = self.retina.sample(frames, fov_deg=fov_deg)
        per_facet, pooled = self.expansion_signal(facets, fps)
        v = self.membrane(pooled, fps)

        # Ignore crossings before the filters have settled. The longest time constant in
        # the chain is the adaptation stage, and three of those is the usual rule of thumb
        # for an exponential filter reaching steady state.
        warmup = min(int(3 * self.tau_adapt_s * fps), len(v) // 2)
        above = np.flatnonzero(v[warmup:] >= 1.0)
        fire_frame = int(above[0] + warmup) if above.size else None

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
    """Seam for a circuit taken straight from the connectome graph.

    The models in this package are built from the *published architecture* of the escape
    pathway: which cell types feed the giant fibre, what each is tuned to, and roughly how
    strongly. A connectome-derived probe would instead read the measured wiring itself --
    the male-CNS volume gives a signed, weighted graph over identified neurons, so the
    connectivity would be data rather than a reading of the literature.

    That is the version worth building next, because it removes the hand-fitting that
    `EMDProbe` needs. The work it requires is a mapping from the ommatidial array onto the
    circuit's input neurons and an integrate-and-fire pass over the graph; the retina here
    already produces the former's input in the right shape.
    """

    name = "connectome"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "not wired yet -- use EMDProbe or GiantFiberProbe. See the class docstring "
            "for what building this actually involves."
        )
