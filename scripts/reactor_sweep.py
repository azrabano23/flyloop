#!/usr/bin/env python3
"""Fly the same fly through several Reactor world models and compare them.

    export REACTOR_API_KEY=rk_...
    python scripts/reactor_sweep.py --budget-usd 40

This is the point of the project pointed at Reactor itself. One probe, one seed plate, one
prompt, one fixed seed, and the same scripted threat, run against each model in turn. What
comes back is not "which video looked best", it is how often each model produced an approach
geometrically real enough to trip a 400-million-year-old reflex, and how far apart the
commits were.

Reactor's catalog does not share one action space: an image-to-video world model, a driving
model and a text world engine take different commands. So every session asks `request_schema`
first and records which commands the model accepted, and a model that refuses half of them
still produces a row rather than an exception.

Budgeting is wall-clock, because Reactor's meter is. The cap converts to a total number of
session-seconds at a deliberately pessimistic rate, and the sweep stops opening sessions once
it is spent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyloop.reactor_loop import FlyLoop, LoopConfig  # noqa: E402

# Every interactive, publicly priced world model in the catalog that takes a starting frame
# or a prompt and streams video back. Slugs are the ones /models returns, never guessed.
FLEET: tuple[tuple[str, str], ...] = (
    ("reactor/lingbot-world-2", "LingBot-World V2, 4xB200 image-to-video"),
    ("reactor/lingbot", "LingBot-World V1. the same model one version back"),
    ("reactor/helios", "Helios, single-GPU autoregressive text-to-video"),
    ("reactor/longlive-v2", "LongLive 2.0, Wan2.2-TI2V-5B"),
    ("reactor/visko-orbis-dynamic", "Orbis 14.3B, 1.8s chunks with prompt morphing"),
)

# Pessimistic on purpose. LingBot World 2 bills at $0.42/min and the multi-GPU models are
# the expensive end of the catalog, so budgeting above the worst case means the cap is a
# cap rather than a suggestion.
RATE_USD_PER_MIN = 0.70


async def one(slug: str, note: str, args, key: str, out: Path) -> dict:
    cfg = LoopConfig(model=slug, seed=args.seed, seed_image=args.seed_image,
                     max_seconds=args.seconds, lunge_at_s=args.lunge_at)
    loop = FlyLoop(cfg)
    t = time.time()
    try:
        rep = await loop.run(key, out / slug.split("/")[-1])
    except Exception as exc:  # noqa: BLE001 - a model that will not fly is still a result
        return {"model": slug, "note": note, "ok": False, "error": str(exc)[:200],
                "session_s": round(time.time() - t, 1)}
    rep.update({"model": slug, "note": note, "ok": True,
                "session_s": round(time.time() - t, 1),
                "accepted": loop.accepted,
                "escapes_per_min": round(rep["escapes"] / max(rep["frames"] / 24 / 60, 1e-6), 2)})
    rep.pop("events", None)
    return rep


async def sweep(args, key: str) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_s = args.budget_usd / RATE_USD_PER_MIN * 60
    spent = 0.0
    rows: list[dict] = []
    board = out / "leaderboard.json"

    want = [m for m in FLEET if not args.models or m[0].split("/")[-1] in args.models]
    print(f"sweeping {len(want)} models, budget ${args.budget_usd:.2f} "
          f"(~{budget_s / 60:.0f} session-minutes at a pessimistic ${RATE_USD_PER_MIN}/min)\n")

    for slug, note in want:
        if spent + args.seconds > budget_s:
            print(f"budget spent, stopping before {slug}")
            break
        print(f"--- {slug}  ({note})")
        row = await one(slug, note, args, key, out)
        spent += row.get("session_s", args.seconds)
        rows.append(row)
        # Write after every model. A sweep that dies on model four should not lose one
        # through three, and at a hackathon it will die on model four.
        board.write_text(json.dumps({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "live": True,
            "rate_usd_per_min": RATE_USD_PER_MIN,
            "estimated_usd": round(spent / 60 * RATE_USD_PER_MIN, 2),
            "rows": rows,
        }, indent=1))
        if row["ok"]:
            print(f"    {row['escapes']} escape(s), {row['frames']} frames, "
                  f"{row['session_s']}s\n")
        else:
            print(f"    did not fly: {row['error'][:110]}\n")

    print(f"\nwrote {board}")
    print(f"estimated spend ${spent / 60 * RATE_USD_PER_MIN:.2f} over {spent:.0f} session-seconds")
    flew = [r for r in rows if r["ok"]]
    if flew:
        print("\nmodel                           escapes  frames  session")
        for r in sorted(flew, key=lambda r: -r["escapes"]):
            print(f"  {r['model']:<30s} {r['escapes']:>5d}  {r['frames']:>6d}  {r['session_s']:>6.1f}s")
    return 0 if flew else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=40.0)
    ap.add_argument("--seconds", type=float, default=70.0, help="per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-image", type=Path,
                    default=Path("artifacts/seeds/yellow-hallway.png"))
    ap.add_argument("--lunge-at", type=float, default=10.0)
    ap.add_argument("--models", nargs="*", default=None,
                    help="short names, e.g. lingbot-world-2 helios")
    ap.add_argument("--out", type=Path, default=Path("web/artifacts/sweep"))
    args = ap.parse_args()

    key = os.environ.get("REACTOR_API_KEY")
    if not key:
        print("REACTOR_API_KEY is not set.\n"
              "  export REACTOR_API_KEY=rk_...", file=sys.stderr)
        return 1
    if not args.seed_image.exists():
        print(f"seed image missing: {args.seed_image}\n"
              f"  run: python scripts/seed_images.py", file=sys.stderr)
        return 1
    return asyncio.run(sweep(args, key))


if __name__ == "__main__":
    raise SystemExit(main())
