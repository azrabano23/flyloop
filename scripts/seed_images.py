#!/usr/bin/env python3
"""Generate seed images for LingBot World 2 with Nano Banana 2.1.

LingBot World 2 will not start without a reference image: `set_image` is required before
`start`. That image anchors the whole generated world, so it is not a formality. It is the
single biggest lever we have on whether the resulting clip is measurable at all.

What makes a seed good for our purposes is narrow and a bit unusual:

  - **one** obstacle, large, centred, and unmistakably solid
  - a clean floor plane and strong perspective lines, so forward motion is unambiguous
  - no other moving subjects, no people, no animals
  - even, flat lighting: our detector keys on brightness change, and a moving shadow or a
    lens flare reads as motion it should not be seeing

The escape circuit never segments an object. It only sees brightness across a hex array.
So a scene that is visually busy is not merely harder, it is actively noisy for us.

    python scripts/seed_images.py --out artifacts/seeds
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "gemini-nano-banana-2.1"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Liminal-space horror, and not only because it looks good projected.
#
# The Backrooms premise is a space whose geometry is lying to you: corridors that do not
# connect, rooms that repeat, distances that do not add up. That is precisely the failure
# mode we are measuring in a world model, so the aesthetic is an honest label rather than
# a costume.
#
# It also happens to want exactly the conditions the detector wants. Flat buzzing
# fluorescent light means no moving shadows. Empty rooms mean no distractor motion. Long
# hallways mean hard vanishing points and unambiguous forward travel. The style and the
# measurement ask for the same picture.
_STYLE = (
    "Liminal space photography, the Backrooms, A24 horror, Kane Parsons. "
    "Found footage shot on a 1990s consumer camcorder: slightly overexposed, visible "
    "video grain, faded washed-out colour, no author behind the lens. "
    "First-person point of view at eye level, camera perfectly level and centred. "
    "Strong one-point perspective, vanishing point dead ahead. "
    "Flat humming fluorescent ceiling light, evenly lit everywhere, no harsh shadows, "
    "no lens flare, no motion blur, no vignette. "
    "Utterly empty and still. No people, no animals, no furniture, no signage. "
    "Uncanny and quietly wrong, like nobody would have built it this way. "
    "Photorealistic, sharp focus throughout."
)

PROMPTS = {
    "yellow-hallway": (
        "The Backrooms. An endless hallway of damp yellow patterned wallpaper and soggy "
        "beige carpet, lit by buzzing yellow fluorescent panels. A flat blank yellow wall "
        "dead-ends the hallway about twelve metres ahead. No doors, no furniture, no signs. "
        + _STYLE
    ),
    "office-pillar": (
        "An abandoned open-plan office at night, all desks removed, bare grey carpet tiles "
        "running to the horizon. One thick square concrete pillar stands directly ahead "
        "about ten metres away under a flickering ceiling light. Nothing else in the room. "
        + _STYLE
    ),
    "pool-rooms": (
        "The Poolrooms. An empty tiled indoor swimming complex, shallow still water across "
        "the floor, pale aquamarine tiles, no windows. A flat tiled wall blocks the way "
        "about twelve metres ahead. No ladders, no lane ropes, no signage. " + _STYLE
    ),
    "parking-column": (
        "A deserted underground parking garage, no cars anywhere, painted concrete floor "
        "with faded white bay lines running away from the camera. One broad concrete support "
        "column stands squarely ahead about ten metres away under sodium strip lighting. "
        + _STYLE
    ),
}


def generate(prompt: str, api_key: str, timeout: int = 120) -> bytes:
    """One image back as raw bytes. Raises with the API's own message on failure."""
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode()

    req = urllib.request.Request(
        ENDPOINT.format(model=MODEL),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{MODEL} returned {e.code}: {e.read().decode()[:400]}") from None

    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                return base64.b64decode(blob["data"])

    raise RuntimeError(f"no image in response: {json.dumps(payload)[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/seeds"))
    ap.add_argument("--only", nargs="*", help="subset of prompt names")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("set GEMINI_API_KEY (get one at ai.dev, key icon, bottom left)", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or list(PROMPTS)

    failures = 0
    for name in names:
        if name not in PROMPTS:
            print(f"  ? unknown prompt {name!r}, skipping")
            continue
        try:
            img = generate(PROMPTS[name], api_key)
        except RuntimeError as exc:
            print(f"  x {name}: {exc}")
            failures += 1
            continue
        path = args.out / f"{name}.png"
        path.write_bytes(img)
        print(f"  + {name}: {len(img) / 1024:.0f} KB -> {path}")

    (args.out / "prompts.json").write_text(json.dumps(PROMPTS, indent=1))
    return 1 if failures == len(names) else 0


if __name__ == "__main__":
    raise SystemExit(main())
