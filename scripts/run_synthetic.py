#!/usr/bin/env python3
"""Run both detectors over the synthetic ladder and write artifacts/results.json.

This is the offline half of flyloop: no network, no API keys, no world model. It builds
clips whose geometry is known in closed form, calibrates the fly circuit once against the
correct one, then scores everything -- including the deliberately broken controls and a
sweep of approach speeds.

The speed sweep is the load-bearing experiment. A fly's escape commits at a roughly
constant angular size no matter how fast the thing is coming; the behavioural literature
puts that threshold near 20 deg and reports it holding across a wide range of approach
speeds. If our circuit holds its threshold across an 8x sweep here, the instrument behaves
like the animal and we can point it at a world model. If it does not, nothing downstream
is worth reporting.

    python scripts/run_synthetic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyloop.baseline import AnalyticBaseline  # noqa: E402
from flyloop.gf import GiantFiberProbe  # noqa: E402
from flyloop.probe import EMDProbe  # noqa: E402
from flyloop.retina import Retina  # noqa: E402
from flyloop.score import score_clip, summarise  # noqa: E402
from flyloop.stimuli import looming  # noqa: E402

# Calibration standard: the approach the gain is set against. Every other clip is scored
# with the gain this one produces.
CAL_SPEED_MPS = 0.5

# 8x range, matching the span over which the behavioural threshold is reported to hold.
SPEED_SWEEP_MPS = (0.25, 0.5, 1.0, 2.0)

BROKEN = ("linear", "reversed", "jump")

# Two honest settings of the same circuit, reported side by side because neither wins.
#
# Normalisation buys speed invariance and pays for it in discrimination. Pushed hard, the
# pooled drive becomes "what fraction of the field is expanding", which is just angular
# size -- so the circuit converges on the analytic baseline and stops catching clips whose
# expansion *profile* is wrong. Backed off, it stays sensitive to the shape of the
# expansion but its threshold slides with approach speed.
#
# A real fly is reported to have both at once. We cannot get both out of a single
# channel, which is what sent us to the two-channel model.
PROBE_CONFIGS = {
    "invariant": dict(normalize=True, sigma=0.003, tau_adapt_s=0.200),
    "sensitive": dict(normalize=False, tau_adapt_s=0.080),
}


def _round(a, nd=4):
    return [round(float(x), nd) for x in np.asarray(a, dtype=float)]


def write_video(frames: np.ndarray, path: Path, fps: float) -> bool:
    """Write an mp4 plus a poster frame. Returns False if encoding is unavailable."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(frames[..., None], 3, axis=-1)
    try:
        iio.imwrite(path, rgb, fps=int(round(fps)), codec="libx264")
        iio.imwrite(path.with_suffix(".jpg"), rgb[len(rgb) // 2])
    except Exception as exc:  # noqa: BLE001 - encoding is optional, never fatal
        print(f"  ! video encode failed ({exc}); continuing without it")
        return False
    return True


def build_clips() -> list[tuple[str, str, object]]:
    """(clip id, condition label, Clip). Correct physics first so calibration can use it."""
    out = [
        (f"looming-{s}mps".replace(".", "_"), "looming", looming("looming", speed_mps=s))
        for s in SPEED_SWEEP_MPS
    ]
    out += [(f"broken-{p}", p, looming(p, speed_mps=CAL_SPEED_MPS)) for p in BROKEN]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--facet-stride", type=int, default=2,
                    help="temporal decimation for facet activity sent to the browser")
    args = ap.parse_args()

    retina = Retina.build()
    baseline = AnalyticBaseline()
    cal = looming("looming", speed_mps=CAL_SPEED_MPS)

    probes, gains = {}, {}
    for key, kwargs in PROBE_CONFIGS.items():
        p = EMDProbe(retina, **kwargs)
        gains[key] = p.calibrate(cal.frames, cal.fps, cal.theta_deg)
        probes[key] = p
        print(f"calibrated '{key}' on {CAL_SPEED_MPS} m/s approach -> gain {gains[key]:.4g}")

    # The primary detector: two channels, as the animal has them.
    probe = GiantFiberProbe(retina)
    gains["giant_fibre_threshold_mv"] = probe.calibrate(cal.frames, cal.fps, cal.theta_deg)
    print(f"calibrated 'giant fibre' -> threshold {gains['giant_fibre_threshold_mv']:.4g} mV")
    print(f"retina: {retina.n_facets} facets\n")

    clips_json = []
    for clip_id, label, clip in build_clips():
        fov = clip.meta["fov_deg"]
        fly = probe.run(clip.frames, clip.fps, fov_deg=fov)
        base = baseline.run(clip.frames, clip.fps)
        scores = score_clip(fly, base, clip.fps, clip.theta_deg, clip.t_contact_s)

        # The two single-channel configurations are kept alongside, because the headline
        # claim is that two channels beat either one and that only means something if all
        # three are measured on the same clips.
        for key, p in probes.items():
            d = p.run(clip.frames, clip.fps, fov_deg=fov)
            scores[key] = score_clip(d, base, clip.fps, clip.theta_deg, clip.t_contact_s)

        rel = f"clips/{clip_id}.mp4"
        has_video = (not args.no_video) and write_video(
            clip.frames, args.out / rel, clip.fps
        )

        act = fly.extra["facet_activity"][:: args.facet_stride]
        peak = float(act.max()) or 1.0

        clips_json.append({
            "id": clip_id,
            "source": "synthetic",
            "model": "analytic",
            "label": label,
            "physical": clip.physical,
            "note": (
                f"constant-velocity approach at {clip.meta['speed_mps']} m/s"
                if clip.physical
                else f"deliberately broken control: {label}"
            ),
            "fps": clip.fps,
            "n_frames": clip.n_frames,
            "video_url": rel if has_video else None,
            "poster_url": rel.replace(".mp4", ".jpg") if has_video else None,
            "ground_truth": {
                "theta_deg": _round(clip.theta_deg, 3),
                "t_contact_s": clip.t_contact_s,
            },
            "detectors": {
                "baseline": {
                    "name": base.name,
                    "fired": base.fired,
                    "fire_frame": base.fire_frame,
                    "fire_t_s": base.fire_time_s(clip.fps),
                    "theta_at_fire_deg": scores["baseline_threshold_deg"],
                    "theta_deg": _round(base.extra["theta_deg"], 3),
                    "trace": _round(base.trace, 4),
                },
                "fly": {
                    "name": fly.name,
                    "fired": fly.fired,
                    "fire_frame": fly.fire_frame,
                    "fire_t_s": fly.fire_time_s(clip.fps),
                    "theta_at_fire_deg": scores["theta_threshold_deg"],
                    "trace": _round(fly.trace, 4),
                    "facet_activity": [_round(row / peak, 3) for row in act],
                    "facet_stride": args.facet_stride,
                },
            },
            "scores": scores,
        })

        lead = scores["fly_lead_frames"]
        slead = scores["sensitive"]["fly_lead_frames"]
        print(
            f"{clip_id:22s} theta={str(scores['theta_threshold_deg']):>6s} deg   "
            f"lead invariant={'n/a' if lead is None else f'{lead:+d}':>4s}   "
            f"sensitive={'n/a' if slead is None else f'{slead:+d}':>4s}"
        )

    # The tradeoff, measured rather than asserted. For each configuration: how stable is
    # the threshold across the speed sweep (lower is better), and how far ahead of the
    # trig baseline does it commit on the broken controls (higher means it is catching
    # something the baseline cannot see). The two pull against each other.
    def _tradeoff(key: str) -> dict:
        get = (lambda c: c["scores"]) if key == "giant fibre" else (lambda c, k=key: c["scores"][k])
        th = [
            get(c)["theta_threshold_deg"]
            for c in clips_json
            if c["physical"] is True and get(c)["theta_threshold_deg"] is not None
        ]
        leads = [
            abs(get(c)["fly_lead_frames"])
            for c in clips_json
            if c["physical"] is False and get(c)["fly_lead_frames"] is not None
        ]
        return {
            "channels": 2 if key == "giant fibre" else 1,
            "config": PROBE_CONFIGS.get(key, {"model": "LC4 rate + LPLC2 size, published weights"}),
            "threshold_sd_deg": round(float(np.std(th, ddof=1)), 2) if len(th) > 1 else None,
            "mean_abs_lead_on_broken_frames": round(float(np.mean(leads)), 2) if leads else None,
        }

    tradeoff = {k: _tradeoff(k) for k in (*PROBE_CONFIGS, "giant fibre")}

    results = {
        "version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mock": False,
        "calibration": {
            "standard": "analytic looming, constant velocity",
            "speed_mps": CAL_SPEED_MPS,
            "target_deg": 20.0,
            "gains": {k: round(v, 6) for k, v in gains.items()},
        },
        "retina": {
            "n_facets": retina.n_facets,
            "dphi_deg": retina.dphi_deg,
            "drho_deg": retina.drho_deg,
            "directions": [[round(a, 3), round(e, 3)] for a, e in retina.directions],
        },
        "clips": clips_json,
        "tradeoff": tradeoff,
        "summary": summarise(clips_json),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))

    s = results["summary"]
    print(
        f"\nthreshold {s['threshold_mean_deg']} +/- {s['threshold_sd_deg']} deg "
        f"over {s['n_physical']} correct clips ({SPEED_SWEEP_MPS[0]}-{SPEED_SWEEP_MPS[-1]} m/s)"
    )
    print(f"detectors agree on {s['baseline_vs_fly_agreement']:.0%} of clips")

    print("\nthe tradeoff (neither config wins both):")
    print(f"  {'config':<12s} {'threshold sd':>13s}  {'lead on broken clips':>21s}")
    for key, t in tradeoff.items():
        print(
            f"  {key:<14s} {str(t['threshold_sd_deg']) + ' deg':>13s}  "
            f"{str(t['mean_abs_lead_on_broken_frames']) + ' frames':>21s}"
        )
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
