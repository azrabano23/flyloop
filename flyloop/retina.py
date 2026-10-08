"""Resample a frame into what a fly's eye actually receives.

A *Drosophila* eye is not a camera. It is roughly 750 ommatidia per eye on a hexagonal
lattice, each one a separate photoreceptor bundle looking in a slightly different
direction. Two numbers matter:

    interommatidial angle  dphi ~ 5.1 deg   how far apart the samples are
    acceptance angle       drho ~ 5.7 deg   how blurry each sample is (Gaussian FWHM)

Because drho is slightly larger than dphi the sampling is mildly oversmoothed, which is
what keeps the fly from aliasing on high-frequency texture. That blur is a real part of
the computation, so we model it rather than nearest-neighbour sampling a pixel grid.

Everything downstream sees only this: a vector of per-facet brightness. No pixels, no
object, no depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Retina"]

# Hex lattice basis in units of the interommatidial angle.
_ROW_STEP = np.sqrt(3.0) / 2.0


@dataclass
class Retina:
    """A hexagonal array of viewing directions with Gaussian acceptance.

    directions: (F, 2) azimuth and elevation per facet, in degrees.
    neighbors:  (F, 6) facet indices of the six hex neighbours, -1 where the facet is on
                the rim and has fewer than six.
    """

    directions: np.ndarray
    neighbors: np.ndarray
    dphi_deg: float
    drho_deg: float

    @property
    def n_facets(self) -> int:
        return int(self.directions.shape[0])

    # ---------------------------------------------------------------- construction

    @classmethod
    def build(
        cls,
        half_fov_deg: float = 40.0,
        dphi_deg: float = 5.1,
        drho_deg: float = 5.7,
    ) -> "Retina":
        """Lay a hex lattice over a circular frontal field of view.

        Defaults give ~200 facets across the frontal field, which is the right order for
        the part of the eye that feeds the looming-sensitive pathway. We do not model the
        whole 750-facet eye because the rear hemisphere never sees the approach.
        """
        n = int(np.ceil(half_fov_deg / dphi_deg)) + 1
        pts = []
        for row in range(-n, n + 1):
            # Every other row is offset by half a step: that is what makes it hexagonal
            # rather than square, and it is why each interior facet has six neighbours.
            offset = 0.5 * (row % 2)
            for col in range(-n, n + 1):
                az = (col + offset) * dphi_deg
                el = row * _ROW_STEP * dphi_deg
                if np.hypot(az, el) <= half_fov_deg:
                    pts.append((az, el))

        directions = np.asarray(pts, dtype=np.float64)
        return cls(
            directions=directions,
            neighbors=cls._hex_neighbors(directions, dphi_deg),
            dphi_deg=dphi_deg,
            drho_deg=drho_deg,
        )

    @staticmethod
    def _hex_neighbors(directions: np.ndarray, dphi_deg: float) -> np.ndarray:
        """Find the six nearest facets of each facet, by angular distance.

        Done geometrically rather than by lattice bookkeeping so it still works if the
        lattice is ever cropped or perturbed. A neighbour must sit within 1.3 * dphi,
        which admits the six true neighbours and excludes the next shell out.
        """
        d = np.linalg.norm(directions[:, None, :] - directions[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        cutoff = 1.3 * dphi_deg

        out = np.full((len(directions), 6), -1, dtype=np.int64)
        order = np.argsort(d, axis=1)[:, :6]
        for i in range(len(directions)):
            cand = order[i]
            out[i, : np.sum(d[i, cand] <= cutoff)] = cand[d[i, cand] <= cutoff]
        return out

    # ------------------------------------------------------------------- sampling

    def _kernels(self, size: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Precompute, per facet, which pixels it sees and how strongly.

        Returns (F, K) pixel indices into a flattened frame and (F, K) weights summing to
        one. K is the largest patch; shorter patches are zero-padded, so a single gather
        handles every facet at once.
        """
        f_px = (size / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        c = (size - 1) / 2.0

        # Facet centres under the same pinhole projection the renderer uses.
        az, el = np.deg2rad(self.directions.T)
        cx = c + f_px * np.tan(az)
        cy = c - f_px * np.tan(el)

        # Gaussian acceptance: convert FWHM in degrees to a pixel sigma at the axis.
        sigma_px = (np.deg2rad(self.drho_deg) * f_px) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        reach = int(np.ceil(2.5 * sigma_px))

        yy, xx = np.mgrid[-reach : reach + 1, -reach : reach + 1]
        k = (2 * reach + 1) ** 2
        idx = np.zeros((self.n_facets, k), dtype=np.int64)
        wts = np.zeros((self.n_facets, k), dtype=np.float64)

        for i in range(self.n_facets):
            px = np.rint(cx[i]).astype(int) + xx
            py = np.rint(cy[i]).astype(int) + yy
            inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)

            w = np.exp(-((px - cx[i]) ** 2 + (py - cy[i]) ** 2) / (2.0 * sigma_px**2))
            w = np.where(inside, w, 0.0)

            idx[i] = np.clip(py, 0, size - 1).ravel() * size + np.clip(px, 0, size - 1).ravel()
            total = w.sum()
            # A rim facet can fall entirely outside the frame; leave it at zero rather
            # than dividing by ~0 and amplifying a single corner pixel into a spike.
            wts[i] = (w / total).ravel() if total > 1e-9 else 0.0

        return idx, wts

    def sample(self, frames: np.ndarray, fov_deg: float = 90.0) -> np.ndarray:
        """Project (T, H, W) frames onto the facet array.

        Returns (T, F) float luminance in 0..1. Frames must be square.
        """
        if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
            raise ValueError(f"expected square (T, H, W) frames, got {frames.shape}")

        size = int(frames.shape[1])
        idx, wts = self._kernels(size, fov_deg)

        flat = frames.reshape(len(frames), -1).astype(np.float64) / 255.0
        return np.einsum("tfk,fk->tf", flat[:, idx], wts)

    # -------------------------------------------------------------------- helpers

    def radial_unit(self) -> np.ndarray:
        """Unit vector per facet pointing away from the centre of the field.

        Looming is radial outward flow, so this is the template the motion detectors get
        projected onto. The centre facet has no defined direction and is left at zero.
        """
        v = self.directions.copy()
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)
