#!/usr/bin/env python3
"""Generate the site's score with Lyria 3.5, then cut the cues we need from it.

Three cues, because the page needs three different jobs done:

  `descent`  the bed that runs under the fullscreen demo. Long, slow, no rhythm to fight
             the video, and tuned so the fluorescent hum in the clip audio sits on top of
             it rather than against it.
  `stinger`  the hit on the frame the giant fibre commits. Short, loud, and over before the
             next frame is drawn.
  `dread`    the slow swell for the descent section, so scrolling down has a floor.

Lyria returns audio inline on generateContent. We ask for it as WAV where the API will give
it, and transcode otherwise, so the page never has to care which came back.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}"

CUES = {
    # Written clinically rather than luridly. The first pass at this prompt was refused by
    # the content filter for the adjectives, not the music, and the sound we actually want
    # is a tuning description anyway.
    "descent": (
        "Very slow dark ambient drone, continuous, no percussion and no melody. Sustained "
        "low cello and contrabass holding two pitches one semitone apart so they beat "
        "slowly against each other. Deep sine sub bass underneath. Occasional distant "
        "struck metal resonating in a very long reverb. Faint high string harmonic "
        "shimmer. Static and airless, like a room tone."
    ),
    "stinger": (
        "A single violent orchestral horror stinger. Hard transient attack, dissonant "
        "cluster of strings in the upper register, deep sub drop underneath, metallic "
        "crash, then a short reverb tail. No build-up at all, the hit lands immediately."
    ),
    "dread": (
        "Slow rising dread. A low drone that swells gradually in volume and tension, with "
        "a dissonant string cluster creeping upward in pitch and a faint arrhythmic pulse "
        "far away. No percussion, no resolution, it simply keeps tightening. Horror film "
        "score under a descent into a basement."
    ),
}


def generate(model: str, prompt: str, key: str) -> bytes:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["AUDIO"]},
    }).encode()
    req = urllib.request.Request(API.format(m=model, k=key), data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    for cand in out.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            data = part.get("inlineData") or part.get("inline_data")
            if data and data.get("data"):
                return base64.b64decode(data["data"])
    raise RuntimeError(f"no audio in response: {json.dumps(out)[:400]}")


def to_wav(raw: bytes, dst: Path, seconds: float | None, fade: bool) -> None:
    """Normalise whatever came back into a loopable 44.1k WAV at a sane level."""
    src = dst.with_suffix(".raw")
    src.write_bytes(raw)
    filt = "loudnorm=I=-17:TP=-1.5,highpass=f=22"
    if fade:
        filt += ",afade=t=in:st=0:d=1.6"
        if seconds:
            filt += f",afade=t=out:st={max(0.1, seconds - 1.6):.2f}:d=1.6"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    # Lyria hands back raw 48k PCM on this endpoint; the header-sniff keeps us honest if
    # that ever changes to a container.
    if not raw[:4] in (b"RIFF", b"OggS", b"fLaC") and raw[:3] != b"ID3":
        cmd += ["-f", "s16le", "-ar", "48000", "-ac", "2"]
    cmd += ["-i", str(src)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-af", filt, "-ar", "44100", "-ac", "2", str(dst)]
    subprocess.run(cmd, check=True)
    src.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lyria-3.5")
    ap.add_argument("--out", type=Path, default=Path("web/artifacts/audio"))
    ap.add_argument("--only", default=None, help="one cue name, for a quick retry")
    a = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set. Get one at https://ai.dev", file=sys.stderr)
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    plan = {"descent": (None, True), "stinger": (1.6, False), "dread": (None, True)}
    rc = 0
    for name, (secs, fade) in plan.items():
        if a.only and name != a.only:
            continue
        try:
            raw = generate(a.model, CUES[name], key)
            dst = a.out / f"{name}.wav"
            to_wav(raw, dst, secs, fade)
            print(f"  + {name}: {dst.stat().st_size // 1024} KB")
        except Exception as exc:  # noqa: BLE001 - one failed cue must not lose the others
            print(f"  ! {name} failed: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
