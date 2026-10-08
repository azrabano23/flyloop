#!/usr/bin/env python3
"""Render one WAV per clip from its membrane trace. No network, no API keys."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyloop.sonify import sonify, write_wav  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=Path("artifacts/results.json"))
ap.add_argument("--out", type=Path, default=Path("artifacts/audio"))
a = ap.parse_args()

res = json.loads(a.results.read_text())
for i, c in enumerate(res["clips"]):
    f = c["detectors"]["fly"]
    x = sonify(f["trace"], c["fps"], f.get("fire_frame"), seed=i)
    p = a.out / f"{c['id']}.wav"
    write_wav(p, x)
    print(f"  + {c['id']}: {len(x)/44100:.2f}s "
          f"{'commit at f%s' % f['fire_frame'] if f['fired'] else 'no commit'}")
print(f"\nwrote {len(res['clips'])} files to {a.out}")
