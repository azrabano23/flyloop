#!/usr/bin/env python3
"""Run both detectors over any mp4, generated or real.

The synthetic ladder proves the instrument is calibrated. This proves it survives contact
with actual footage, which is a different and harder claim: real frames have texture,
lighting, compression artefacts and a camera that does not move exactly as asked.

Two things are deliberately different here from `run_synthetic.py`:

  - **There is no ground truth.** We did not build the world, so we do not know the true
    angular size of anything in it. Every number is an estimate and is labelled as one.
  - **The scene is not a dark disc on white.** The analytic baseline has to threshold the
    image to find an object, so it needs to be told which way round the contrast runs. The
    fly circuit does not care, because it reads motion rather than brightness. That
    asymmetry is the point of the comparison, so `--invert` only affects the baseline.

    python scripts/score_clip.py artifacts/clips/veo-yellow-hallway.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyloop.baseline import AnalyticBaseline  # noqa: E402
from flyloop.gf import GiantFiberProbe  # noqa: E402
from flyloop.probe import EMDProbe  # noqa: E402
from flyloop.retina import Retina  # noqa: E402
from flyloop.stimuli import looming  # noqa: E402


def load_square(path: Path, size: int = 192) -> tuple[np.ndarray, float]:
    """Read an mp4 as (T, size, size) uint8 luma, centre-cropped to square.

    Centre crop rather than squash: the retina maps facet directions through a pinhole
    model, so anisotropically scaling the frame would quietly corrupt every angle it
    computes.
    """
    import imageio.v2 as iio2

    # The ffmpeg plugin rather than pyav: ffmpeg is already on this box, and one fewer
    # binary dependency is one fewer thing to install on a laptop at a hackathon.
    reader = iio2.get_reader(str(path), "ffmpeg")
    fps = float(reader.get_meta_data().get("fps") or 24.0)
    frames = np.stack([f for f in reader])
    reader.close()

    if frames.ndim == 4:
        frames = frames[..., :3].mean(axis=-1)

    t, h, w = frames.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = frames[:, top:top + side, left:left + side]

    idx = (np.linspace(0, side - 1, size)).astype(int)
    frames = frames[:, idx][:, :, idx]
    return frames.astype(np.uint8), fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--invert", action="store_true",
                    help="baseline only: treat bright regions as the object")
    args = ap.parse_args()

    frames, fps = load_square(args.clip)
    print(f"{args.clip.name}: {len(frames)} frames @ {fps:.1f} fps, "
          f"{frames.shape[1]}x{frames.shape[2]}, "
          f"luma {frames.min()}..{frames.max()} mean {frames.mean():.0f}")

    retina = Retina.build()
    cal = looming("looming", speed_mps=0.5)

    gf = GiantFiberProbe(retina)
    gf.calibrate(cal.frames, cal.fps, cal.theta_deg)
    emd = EMDProbe(retina)
    emd.calibrate(cal.frames, cal.fps, cal.theta_deg)

    base_frames = (255 - frames) if args.invert else frames
    base = AnalyticBaseline(fov_deg=args.fov).run(base_frames, fps)

    print()
    for name, det in (("giant fibre", gf.run(frames, fps, fov_deg=args.fov)),
                      ("emd 1-channel", emd.run(frames, fps, fov_deg=args.fov)),
                      ("trig baseline", base)):
        if det.fired:
            print(f"  {name:<14s} FIRED at frame {det.fire_frame:>4d} "
                  f"({det.fire_frame / fps:.2f}s), peak trace {det.trace.max():.2f}")
        else:
            print(f"  {name:<14s} did not fire, peak trace {det.trace.max():.2f}")

    theta = base.extra["theta_deg"]
    print(f"\n  baseline's estimated angular size: "
          f"{theta[0]:.1f} -> {theta[-1]:.1f} deg "
          f"(estimate only, no ground truth for generated video)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
