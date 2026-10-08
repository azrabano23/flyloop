#!/usr/bin/env python3
"""Score generated clips and append them to artifacts/results.json.

Separate from `run_synthetic.py` on purpose. That script builds its own stimuli and knows
their geometry exactly; this one is handed footage somebody else's model produced and knows
nothing about it. The difference shows up in the output: every clip added here carries
`ground_truth: null` and `physical: null`, because we did not build the world and cannot
say what the true angular size of anything in it was.

    python scripts/add_generated.py artifacts/clips/veo-*.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyloop.baseline import AnalyticBaseline  # noqa: E402
from flyloop.gf import GiantFiberProbe  # noqa: E402
from flyloop.retina import Retina  # noqa: E402
from flyloop.score import score_clip  # noqa: E402
from flyloop.stimuli import looming  # noqa: E402
from scripts.score_clip import load_square  # noqa: E402

NOTES = {
    "veo-entity-lunge": "something comes out of the wall",
    "veo-yellow-hallway": "forward flight down a backrooms hallway",
    "veo-pool-rooms": "forward flight, poolrooms",
    "veo-parking-column": "forward flight toward a column",
    "veo-office-pillar": "forward flight toward a pillar",
}


def _round(a, nd=4):
    return [round(float(x), nd) for x in np.asarray(a, dtype=float)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--results", type=Path, default=Path("artifacts/results.json"))
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    results = json.loads(args.results.read_text())
    retina = Retina.build()

    probe = GiantFiberProbe(retina)
    cal = looming("looming", speed_mps=0.5)
    probe.calibrate(cal.frames, cal.fps, cal.theta_deg)
    baseline = AnalyticBaseline()

    existing = {c["id"] for c in results["clips"]}
    added = 0

    for path in args.clips:
        cid = path.stem
        if cid in existing:
            print(f"  = {cid} already present, skipping")
            continue

        frames, fps = load_square(path)
        fly = probe.run(frames, fps)
        base = baseline.run(frames, fps)
        # No ground truth, so score_clip falls back to the baseline's *estimate* and flags
        # it. That flag is the whole point: a reader must never mistake an inferred angle
        # for a measured one.
        scores = score_clip(fly, base, fps, None, None)

        act = fly.extra["facet_activity"][:: args.stride]
        peak = float(act.max()) or 1.0

        results["clips"].append({
            "id": cid,
            "source": "veo",
            "model": "veo-3.1-fast-generate-preview",
            "label": "generated",
            "physical": None,
            "note": NOTES.get(cid, "generated approach"),
            "fps": fps,
            "n_frames": int(len(frames)),
            "video_url": f"clips/{path.name}",
            "poster_url": None,
            "ground_truth": None,
            "detectors": {
                "baseline": {
                    "name": base.name, "fired": base.fired,
                    "fire_frame": base.fire_frame, "fire_t_s": base.fire_time_s(fps),
                    "theta_at_fire_deg": scores["baseline_threshold_deg"],
                    "theta_deg": _round(base.extra["theta_deg"], 3),
                    "trace": _round(base.trace, 4),
                },
                "fly": {
                    "name": fly.name, "fired": fly.fired,
                    "fire_frame": fly.fire_frame, "fire_t_s": fly.fire_time_s(fps),
                    "theta_at_fire_deg": scores["theta_threshold_deg"],
                    "trace": _round(fly.trace, 4),
                    "facet_activity": [_round(r / peak, 3) for r in act],
                    "facet_stride": args.stride,
                },
            },
            "scores": scores,
        })
        added += 1
        print(f"  + {cid}: fly {'fired f%d' % fly.fire_frame if fly.fired else 'quiet'}, "
              f"{len(frames)} frames @ {fps:.0f} fps")

    results["summary"]["n_clips"] = len(results["clips"])
    results["summary"]["n_generated"] = sum(
        1 for c in results["clips"] if c["source"] != "synthetic")
    args.results.write_text(json.dumps(results, indent=1))
    print(f"\nadded {added}; results.json now has {len(results['clips'])} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
