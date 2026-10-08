#!/usr/bin/env python3
"""Fly a Reactor world with the escape circuit in the loop.

    export REACTOR_API_KEY=rk_...
    python scripts/fly_reactor.py --seed-image artifacts/seeds/yellow-hallway.png

    python scripts/fly_reactor.py --dry-run      # no key, no network, same state machine

The dry run is not a stub. It pushes real synthetic looming frames through the same
`push()` and the same forward/retreat logic the live session uses, so the part that is
easy to get wrong is exercised offline. What it cannot test is Reactor itself: the
session handshake, the frame format coming off the track, and how long a command really
takes to show up in the generated video.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyloop.reactor_loop import MODELS, FlyLoop, LoopConfig  # noqa: E402
from flyloop.stimuli import looming  # noqa: E402


def dry_run(cfg: LoopConfig, out: Path) -> int:
    """Run the loop's logic against synthetic frames, no network involved."""
    out.mkdir(parents=True, exist_ok=True)
    loop = FlyLoop(cfg)
    loop.fps = 60.0

    # A real approach, then a gap, then another. Two events means we also exercise the
    # return to forward flight, which a single approach would never reach.
    a = looming("looming", speed_mps=0.5).frames
    gap = np.repeat(a[:1], 40, axis=0)
    script = np.concatenate([a, gap, a])

    print(f"dry run: {len(script)} synthetic frames @ {loop.fps:.0f} fps "
          f"(would be {MODELS.get(cfg.model, cfg.model)})")

    for i, frame in enumerate(script, 1):
        now = i / loop.fps
        if loop.state == "retreat":
            if now >= loop.retreat_until:
                loop.state = "forward"
                loop._log("cmd:forward", i, {"since_escape_ms": None})
            continue
        if loop.push(frame):
            loop.state = "retreat"
            loop._fired_at = now
            loop.retreat_until = now + cfg.retreat_chunks * (12.0 / loop.fps)
            loop._log("escape", i, {"wall_s": round(now, 3)})
            loop._log("cmd:back", i, {"sent_s": round(now, 3),
                                      "since_escape_ms": 0.0})

    rep = loop.report(out, len(script), live=False)
    print(f"\nescapes: {rep['escapes']}  (expected 2, one per approach)")
    print(f"wrote {out / 'loop_report.json'}")
    return 0 if rep["escapes"] >= 1 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lingbot-world-2", choices=list(MODELS))
    ap.add_argument("--seed-image", type=Path, default=None)
    ap.add_argument("--prompt", default=LoopConfig.prompt)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=60.0,
                    help="hard cap; Reactor bills per session-second from ready")
    ap.add_argument("--invert", action="store_true",
                    help="treat bright regions as the object (dark scenes)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/loop"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = LoopConfig(model=args.model, prompt=args.prompt, seed=args.seed,
                     seed_image=args.seed_image, max_seconds=args.max_seconds,
                     invert=args.invert)

    if args.dry_run:
        return dry_run(cfg, args.out)

    key = os.environ.get("REACTOR_API_KEY")
    if not key:
        print("REACTOR_API_KEY is not set.\n"
              "  1. redeem the promo code in Billing at reactor.inc\n"
              "  2. create a key (rk_...) and: export REACTOR_API_KEY=rk_...\n"
              "  docs: https://docs.reactor.inc/authentication", file=sys.stderr)
        return 1
    if args.seed_image and not args.seed_image.exists():
        print(f"seed image missing: {args.seed_image}\n"
              f"  run: python scripts/seed_images.py", file=sys.stderr)
        return 1
    if args.model == "lingbot-world-2" and not args.seed_image:
        print("lingbot-world-2 requires a reference image before it will start.\n"
              "  pass --seed-image artifacts/seeds/yellow-hallway.png", file=sys.stderr)
        return 1

    t = time.time()
    rep = asyncio.run(FlyLoop(cfg).run(key, args.out))
    print(f"\n{rep['escapes']} escape(s) over {rep['frames']} frames "
          f"in {time.time() - t:.1f}s")
    if rep["mean_escape_to_command_ms"] is not None:
        print(f"mean escape to command: {rep['mean_escape_to_command_ms']} ms "
              f"(floor is Reactor's chunk boundary, not the circuit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
