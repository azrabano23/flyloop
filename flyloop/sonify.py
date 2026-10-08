"""Turn the escape circuit into something you can hear.

This is not decoration. Electrophysiologists run a speaker off the electrode because the
ear is better than the eye at catching the moment a cell starts to go: you hear the rate
climb before you would see it on a trace. Pointing that same speaker at our giant fibre is
the honest version of an audio layer, not a soundtrack bolted on afterwards.

Three things get mixed:

  **the carrier** rises in pitch and loudness with the membrane potential, so the approach
  sounds like dread building rather than like a countdown,
  **the commit** is a single hard transient at the moment the circuit fires,
  **the room** is mains hum from failing fluorescent tubes, which is not atmosphere either:
  the original Backrooms text specifies "the endless background noise of fluorescent lights
  at maximum hum-buzz", and a dead-silent liminal space reads as wrong.

Stdlib `wave` plus numpy. No external audio dependency, nothing to install on a laptop at
a hackathon.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["sonify", "write_wav", "fluorescent_hum"]

RATE = 44100


def write_wav(path: Path, x: np.ndarray, rate: int = RATE) -> None:
    """16-bit mono. Peak-normalised with headroom so nothing clips on a laptop speaker."""
    peak = float(np.abs(x).max())
    if peak > 0:
        x = x / peak * 0.89
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def fluorescent_hum(n: int, rate: int = RATE, seed: int = 0) -> np.ndarray:
    """Mains hum from a tube that is on its way out.

    A pure 120 Hz sine sounds like a test tone. Real fluorescent noise is the even
    harmonics of mains, each drifting slightly out of phase with the others, plus a thin
    band of hiss from the ballast. The drift is what makes it sit under a scene instead of
    sitting on top of it.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    out = np.zeros(n)
    for k, amp in enumerate((1.0, 0.45, 0.22, 0.12), start=1):
        f = 120.0 * k
        drift = 0.9 * np.sin(2 * np.pi * (0.07 + 0.013 * k) * t + rng.uniform(0, 6.28))
        out += amp * np.sin(2 * np.pi * f * t + drift)

    hiss = rng.normal(0, 1, n)
    # one-pole lowpass, so the hiss sits behind the hum rather than on top of it
    a = 0.06
    for i in range(1, n):
        hiss[i] = a * hiss[i] + (1 - a) * hiss[i - 1]

    flicker = 1.0 + 0.1 * np.sin(2 * np.pi * 0.31 * t) * (rng.random(n) > 0.9995).cumsum() % 2
    return (0.16 * out + 0.5 * hiss) * flicker * 0.3


def sonify(trace: np.ndarray, fps: float, fire_frame: int | None,
           rate: int = RATE, seed: int = 0) -> np.ndarray:
    """Render one clip's membrane trace as audio of exactly the clip's duration."""
    trace = np.asarray(trace, dtype=float)
    n = int(round(len(trace) / fps * rate))
    t = np.arange(n) / rate

    # Upsample the per-frame trace to audio rate.
    v = np.interp(np.linspace(0, len(trace) - 1, n), np.arange(len(trace)), trace)
    v = np.clip(v, 0, 1.5)

    # Carrier: 70 Hz at rest up to about 320 Hz at commit. Integrating the frequency before
    # taking the sine is what keeps the pitch sweep continuous; modulating the argument
    # directly would produce clicks every time the rate changed.
    f = 70.0 + 250.0 * (v / 1.5)
    phase = 2 * np.pi * np.cumsum(f) / rate
    carrier = np.sin(phase) + 0.3 * np.sin(2 * phase)
    carrier *= 0.08 + 0.55 * (v / 1.5) ** 1.6

    out = carrier + fluorescent_hum(n, rate, seed)

    if fire_frame is not None:
        i = int(fire_frame / fps * rate)
        if 0 <= i < n:
            # The commit: a short noise burst through a fast decay. A click, not a beep,
            # because the event is a single spike and should not sound musical.
            k = min(int(0.09 * rate), n - i)
            env = np.exp(-np.arange(k) / (0.012 * rate))
            rng = np.random.default_rng(seed + 1)
            burst = rng.normal(0, 1, k) * env
            burst += np.sin(2 * np.pi * 180 * np.arange(k) / rate) * env * 1.4
            out[i:i + k] += burst * 0.85

            # Everything after the commit drops away: the decision is made.
            tail = np.ones(n)
            tail[i:] = np.exp(-(np.arange(n - i)) / (0.5 * rate)) * 0.6 + 0.12
            out *= tail

    # Short fades so the file can loop without a pop at the seam.
    e = min(int(0.02 * rate), n // 4)
    if e > 0:
        out[:e] *= np.linspace(0, 1, e)
        out[-e:] *= np.linspace(1, 0, e)
    return out
