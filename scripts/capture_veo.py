#!/usr/bin/env python3
"""Animate a seed image into an approach with Veo 3.1, then save it for scoring.

The chain is two Google models end to end: Nano Banana 2.1 draws the room, Veo 3.1 moves
the camera through it. Image-to-video matters here rather than text-to-video, because the
measurement needs the *same* room approached at different rates. Starting every clip from
a fixed plate is the only control we have over that.

Veo is asynchronous: you POST to :predictLongRunning, get an operation name back, and poll
it. Generation takes a couple of minutes, so this script is slow by nature, not stuck.

    python scripts/capture_veo.py --seed-image artifacts/seeds/yellow-hallway.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "veo-3.1-fast-generate-preview"

# Veo has to be told, firmly and more than once, that the camera moves and the world does
# not. Left looser it tends to drift sideways, cut, or animate the room instead, any of
# which destroys a looming measurement. "Dolly" is the word it responds to most reliably.
APPROACH = (
    "The camera dollies slowly and steadily straight forward down the hallway toward the "
    "far wall, moving at a constant speed in a perfectly straight line. The camera stays "
    "level and does not turn, tilt, pan or shake. The room itself is completely static: "
    "nothing in the scene moves, no people, no animals, no flickering. One continuous "
    "unbroken shot, no cuts, no edits, no camera changes. "
    "Found footage from a 1990s camcorder, liminal space, quiet dread."
)


def _post(url: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url.rsplit('/', 1)[-1]} -> {e.code}: {e.read().decode()[:500]}") from None


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def generate(seed_image: Path | None, prompt: str, model: str, key: str,
             poll_s: int = 10, timeout_s: int = 600) -> bytes:
    """Kick off a generation, wait it out, return the mp4 bytes."""
    instance: dict = {"prompt": prompt}
    if seed_image is not None:
        instance["image"] = {
            "bytesBase64Encoded": base64.b64encode(seed_image.read_bytes()).decode(),
            "mimeType": "image/png",
        }

    # Deliberately minimal parameters. The fast variant rejects `generateAudio` outright,
    # and every extra knob here is another way for a request to 400 mid-hackathon.
    op = _post(f"{BASE}/models/{model}:predictLongRunning",
               {"instances": [instance], "parameters": {"aspectRatio": "16:9"}}, key)

    name = op.get("name")
    if not name:
        raise RuntimeError(f"no operation name returned: {json.dumps(op)[:300]}")
    print(f"  operation {name.rsplit('/', 1)[-1]}, polling every {poll_s}s")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(poll_s)
        st = _get(f"{BASE}/{name}", key)
        if not st.get("done"):
            continue
        if "error" in st:
            raise RuntimeError(f"generation failed: {json.dumps(st['error'])[:400]}")

        resp = st.get("response", {})
        samples = (resp.get("generateVideoResponse", {}).get("generatedSamples")
                   or resp.get("generatedSamples") or resp.get("videos") or [])
        if not samples:
            raise RuntimeError(f"done but no video in response: {json.dumps(resp)[:500]}")

        vid = samples[0].get("video", samples[0])
        if vid.get("bytesBase64Encoded"):
            return base64.b64decode(vid["bytesBase64Encoded"])
        uri = vid.get("uri") or vid.get("url")
        if not uri:
            raise RuntimeError(f"no uri or bytes on sample: {json.dumps(vid)[:300]}")
        req = urllib.request.Request(uri, headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    raise RuntimeError(f"timed out after {timeout_s}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-image", type=Path, default=Path("artifacts/seeds/yellow-hallway.png"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("artifacts/clips"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt", default=APPROACH)
    ap.add_argument("--text-only", action="store_true", help="skip the seed image")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("set GEMINI_API_KEY (ai.dev, key icon, bottom left)", file=sys.stderr)
        return 1

    seed = None if args.text_only else args.seed_image
    if seed is not None and not seed.exists():
        print(f"seed image not found: {seed}. run scripts/seed_images.py first", file=sys.stderr)
        return 1

    name = args.name or f"veo-{(seed.stem if seed else 'text')}"
    print(f"{args.model} <- {seed if seed else 'text only'}")

    try:
        mp4 = generate(seed, args.prompt, args.model, key)
    except RuntimeError as exc:
        print(f"  x {exc}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{name}.mp4"
    path.write_bytes(mp4)
    print(f"  + {len(mp4) / 1024:.0f} KB -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
