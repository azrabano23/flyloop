"""Turn the escape circuit into something you can hear.

This is not decoration. Electrophysiologists run a speaker off the electrode because the
ear is better than the eye at catching the moment a cell starts to go: you hear the spike
rate climb a beat before you would see it on a trace. Pointing that same speaker at our
giant fibre is the honest version of an audio layer, not a soundtrack bolted on afterwards.

Five layers, in the order you notice them:

  **the spikes** are the signal. Clicks fired at a rate that follows the membrane, sparse at
  rest and a dense crackle at commit. This is what a recording rig actually sounds like, and
  it is the layer doing the storytelling,
  **the room** is mains hum off failing fluorescent tubes. The original Backrooms text
  specifies "the endless background noise of fluorescent lights at maximum hum-buzz", and a
  dead-silent liminal space reads as wrong,
  **the bed** is a low fifth that beats slowly against itself, which is the one piece of
  score here,
  **the commit** is a hard transient plus a sub drop, because the decision should land in
  your chest and not sound musical,
  **the wings** are what happens next. Drosophila beats its wings near 210 Hz, so the escape
  is a real wingbeat buzz that pans across as the fly leaves, or, on a level it does not get
  out of, an impact and then nothing but the tubes.

Stereo, stdlib `wave` plus numpy. No external audio dependency, nothing to install on a
laptop at a hackathon.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["sonify", "write_wav", "fluorescent_hum", "spike_train", "RATE"]

RATE = 44100


def write_wav(path: Path, x: np.ndarray, rate: int = RATE) -> None:
    """16-bit WAV, mono for (n,) and stereo for (n, 2).

    Peak-normalised with headroom so nothing clips on a laptop speaker, and normalised on
    the pair rather than per channel, which would wander the image around.
    """
    x = np.asarray(x, dtype=float)
    peak = float(np.abs(x).max())
    if peak > 0:
        x = x / peak * 0.89
    ch = 1 if x.ndim == 1 else x.shape[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x.reshape(-1) * 32767).astype("<i2").tobytes())


def _lp(x: np.ndarray, a: float) -> np.ndarray:
    """One-pole lowpass. Vectorised enough for a few seconds of audio."""
    # lfilter without scipy: a geometric decay convolution is close enough for noise shaping
    # and far faster than a Python loop over 44100 samples a second.
    n = len(x)
    k = int(min(n, max(8, 4 / a)))
    h = (1 - a) ** np.arange(k)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def fluorescent_hum(n: int, rate: int = RATE, seed: int = 0) -> np.ndarray:
    """Mains hum from a tube that is on its way out.

    A pure 120 Hz sine sounds like a test tone. Real fluorescent noise is the even harmonics
    of mains, each drifting slightly out of phase with the others, plus a thin band of hiss
    off the ballast. The drift is what makes it sit under a scene instead of on top of it.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    out = np.zeros(n)
    for k, amp in enumerate((1.0, 0.45, 0.22, 0.12), start=1):
        f = 120.0 * k
        drift = 0.9 * np.sin(2 * np.pi * (0.07 + 0.013 * k) * t + rng.uniform(0, 6.28))
        out += amp * np.sin(2 * np.pi * f * t + drift)

    hiss = _lp(rng.normal(0, 1, n), 0.06)

    # A tube on its way out does not hum evenly. Every so often it stutters, and the hum
    # ducks for a few tens of milliseconds.
    gate = np.ones(n)
    for i in rng.integers(0, n, size=max(1, n // (rate * 3))):
        k = int(rate * rng.uniform(0.02, 0.09))
        gate[i:i + k] *= rng.uniform(0.25, 0.6)
    return (0.16 * out + 0.5 * hiss) * gate * 0.3


def _bed(n: int, rate: int, seed: int) -> np.ndarray:
    """A low fifth that beats against itself. The only part of this that is score."""
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed + 7)
    root = 41.2                                  # low E, below where a laptop speaker lies
    x = np.sin(2 * np.pi * root * t)
    x += 0.7 * np.sin(2 * np.pi * root * 1.4983 * t + 0.4)   # a fifth, slightly flat
    x += 0.3 * np.sin(2 * np.pi * root * 2 * t + rng.uniform(0, 6.28))
    swell = 0.55 + 0.45 * np.sin(2 * np.pi * 0.055 * t + rng.uniform(0, 6.28))
    return x * swell * 0.19


def spike_train(v: np.ndarray, rate: int, seed: int = 0) -> np.ndarray:
    """Clicks at a rate that follows the membrane. The layer that carries the tension.

    Rate runs from a few per second at rest to a couple of hundred at commit, which is the
    range a giant fibre recording actually covers, and it is why the approach sounds like
    dread building rather than like a countdown.
    """
    n = len(v)
    rng = np.random.default_rng(seed + 3)
    hz = 6.0 + 210.0 * np.clip(v / 1.5, 0, 1) ** 1.7
    # Thin a Bernoulli draw per sample by the instantaneous rate: an inhomogeneous Poisson
    # process without having to integrate the rate by hand.
    hits = np.flatnonzero(rng.random(n) < hz / rate)
    out = np.zeros(n + 400)
    k = 160
    env = np.exp(-np.arange(k) / (rate * 0.0009))
    click = env * (np.sin(2 * np.pi * 1400 * np.arange(k) / rate)
                   + 0.6 * rng.normal(0, 1, k))
    for i in hits:
        out[i:i + k] += click * rng.uniform(0.7, 1.0)
    return out[:n] * 0.30


def _wings(n: int, rate: int, seed: int) -> np.ndarray:
    """A Drosophila wingbeat. Near 210 Hz, which is a measured figure, not a guess."""
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed + 11)
    # The fly bolts, so the beat rises as it goes.
    f = 210.0 * (1 + 0.22 * t / max(t[-1], 1e-6))
    ph = 2 * np.pi * np.cumsum(f) / rate
    x = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.22 * np.sin(3 * ph)
    x *= 1 + 0.35 * np.sin(2 * np.pi * 13 * t + rng.uniform(0, 6.28))   # flutter
    return x * np.exp(-t / 0.22) * 0.5


def sonify(trace: np.ndarray, fps: float, fire_frame: int | None,
           rate: int = RATE, seed: int = 0) -> np.ndarray:
    """Render one clip's membrane trace as stereo audio of exactly the clip's duration.

    Returns (n, 2). The left-right image is used for one thing only: the fly leaving.
    """
    trace = np.asarray(trace, dtype=float)
    n = int(round(len(trace) / fps * rate))
    t = np.arange(n) / rate

    # Upsample the per-frame trace to audio rate.
    v = np.clip(np.interp(np.linspace(0, len(trace) - 1, n), np.arange(len(trace)), trace),
                0, 1.5)

    # Carrier: 70 Hz at rest up to about 320 Hz at commit, detuned against itself so it
    # sounds like a speaker on an electrode rather than a synth. Integrating the frequency
    # before taking the sine is what keeps the sweep continuous; modulating the argument
    # directly would click every time the rate changed.
    f = 70.0 + 250.0 * (v / 1.5)
    ph = 2 * np.pi * np.cumsum(f) / rate
    carrier = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.12 * np.sin(ph * 1.004)
    carrier *= 0.06 + 0.42 * (v / 1.5) ** 1.6

    mono = carrier + fluorescent_hum(n, rate, seed) + _bed(n, rate, seed) \
        + spike_train(v, rate, seed)

    L = mono.copy()
    Rc = mono.copy()

    if fire_frame is not None:
        i = int(fire_frame / fps * rate)
        if 0 <= i < n:
            # The commit. A noise click for the spike, and a sub drop under it so the
            # decision lands in your chest.
            k = min(int(0.10 * rate), n - i)
            env = np.exp(-np.arange(k) / (0.013 * rate))
            rng = np.random.default_rng(seed + 1)
            burst = rng.normal(0, 1, k) * env
            burst += np.sin(2 * np.pi * 180 * np.arange(k) / rate) * env * 1.4
            ks = min(int(0.38 * rate), n - i)
            sf = np.linspace(95, 27, ks)
            drop = np.sin(2 * np.pi * np.cumsum(sf) / rate) * np.exp(-np.arange(ks) / (0.10 * rate))
            L[i:i + k] += burst * 0.85
            Rc[i:i + k] += burst * 0.85
            L[i:i + ks] += drop * 0.65
            Rc[i:i + ks] += drop * 0.65

            # The room drops away behind the decision, then comes back.
            tail = np.ones(n)
            tail[i:] = np.exp(-np.arange(n - i) / (0.5 * rate)) * 0.55 + 0.16
            L *= tail
            Rc *= tail

            # The wings. It leaves to one side, so the buzz walks across the image.
            kw = min(int(0.6 * rate), n - i)
            if kw > 64:
                w = _wings(kw, rate, seed)
                side = np.linspace(0.5, 0.04, kw)    # starts centred, ends hard left
                L[i:i + kw] += w * side
                Rc[i:i + kw] += w * (1 - side) * 0.35
    else:
        # It never committed, so the level closes on it. One dull impact at the end and then
        # nothing but the tubes, which is the sound we want on a level that failed.
        i = max(0, n - int(0.5 * rate))
        k = n - i
        if k > 64:
            rng = np.random.default_rng(seed + 5)
            sf = np.linspace(70, 20, k)
            thud = np.sin(2 * np.pi * np.cumsum(sf) / rate) * np.exp(-np.arange(k) / (0.07 * rate))
            splat = _lp(rng.normal(0, 1, k), 0.25) * np.exp(-np.arange(k) / (0.02 * rate))
            L[i:] += thud * 0.8 + splat * 0.5
            Rc[i:] += thud * 0.8 + splat * 0.5

    out = np.stack([L, Rc], axis=1)

    # Short fades so the file can loop without a pop at the seam.
    e = min(int(0.02 * rate), n // 4)
    if e > 0:
        out[:e] *= np.linspace(0, 1, e)[:, None]
        out[-e:] *= np.linspace(1, 0, e)[:, None]
    return out
