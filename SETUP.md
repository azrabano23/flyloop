# setup

## offline pipeline (no keys, no network)

```bash
pip install numpy imageio imageio-ffmpeg
python scripts/run_synthetic.py
```
Writes `artifacts/results.json` and the clips the site reads.

## live Reactor loop

Needs **Python 3.10+**. `reactor-sdk` has no wheel for 3.9, and on a Mac the default
`python3` is usually Apple's 3.9. Check with `python3 --version`.

If it is older than 3.10:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then:

```bash
export REACTOR_API_KEY=rk_...        # Billing at reactor.inc, after redeeming the promo code
python scripts/fly_reactor.py --seed-image artifacts/seeds/yellow-hallway.png
```

Caps the session at 60 s and disconnects non-recoverably. LingBot World 2 bills at
$0.42/min, so a capped run is about $0.42.

**This will not work from a sandboxed container.** The SDK carries media over WebRTC, which
needs UDP; allowing `api.reactor.inc` over HTTPS gets you a valid JWT and then a
`transport readiness did not complete` timeout. Run it on a real machine.

## keys

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.
