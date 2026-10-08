"""Close the loop: a fly reflex flying a Reactor world, and dodging inside it.

The offline half of this project scores recorded clips. This is the live half. We open a
session on a Reactor world model, drive the camera forward, run the escape circuit on the
frames as they arrive, and when it commits we send a command back into the same session.
The world the fly is reacting to is being generated in response to the fly.

Two things about Reactor shape the whole design:

**Commands land on chunk boundaries.** LingBot World 2 generates in chunks of about three
latent frames, and a `set_move_longitudinal` applies at the next boundary. So the escape is
not millisecond accurate and we do not pretend otherwise: every run logs the real delay
between the circuit committing and the world visibly responding, and that number goes in
the report rather than a claim about biological reaction time.

**The meter runs on wall-clock, not on work.** Billing starts the moment a session reaches
`ready` and continues whether or not commands are flowing, so every session here is capped,
and the disconnect is non-recoverable. A recoverable disconnect holds the GPU, and the
meter, waiting for a reconnection that a hackathon demo is never going to make.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gf import GiantFiberProbe
from .retina import Retina
from .stimuli import looming

__all__ = ["LoopConfig", "FlyLoop", "Event"]

# Reactor's own slugs. Taken from the model pages, never guessed: some models expose one
# slug per experience and a wrong one fails at connect time with a confusing error.
MODELS = {
    "lingbot-world-2": "reactor/lingbot-world-2",
    "helios": "reactor/helios",
    "fast-h3": "reactor/fast-h3",
}


@dataclass
class Event:
    """One thing that happened, with enough context to reconstruct the run."""

    t: float
    kind: str
    frame: int
    detail: dict = field(default_factory=dict)


@dataclass
class LoopConfig:
    model: str = "lingbot-world-2"
    prompt: str = (
        "An endless Backrooms hallway of damp yellow wallpaper and beige carpet under "
        "buzzing fluorescent lights. The camera moves forward down the hallway."
    )
    seed_image: Path | None = None
    seed: int = 42
    window_s: float = 1.5
    # How long to retreat once the circuit commits. Three chunks is long enough to be
    # visible in the generated video; shorter and the world has barely begun to respond
    # before we are driving forward again.
    retreat_chunks: int = 3
    max_seconds: float = 60.0
    fov_deg: float = 90.0
    invert: bool = False
    # LingBot World 2 generates *infinite* worlds. Drive forward down a hallway and it
    # simply makes more hallway, so nothing ever closes on the camera and a correctly
    # working escape circuit sits quiet for the whole session. That is not a detector
    # failure, it is the model doing exactly what it advertises.
    #
    # To get an approach we hot-swap the prompt mid-generation and bring something to us.
    # set_prompt is valid during generation and lands on the next chunk boundary, so the
    # world turns threatening without restarting the session.
    lunge_at_s: float = 12.0
    lunge_every_s: float = 18.0
    lunge_prompt: str = (
        "A huge dark shape bursts through the wall directly ahead and rushes straight at "
        "the camera, filling the view. Debris in the air."
    )


class FlyLoop:
    """Drives a Reactor session from the escape circuit's output."""

    def __init__(self, cfg: LoopConfig, probe: GiantFiberProbe | None = None):
        self.cfg = cfg
        self.retina = Retina.build()
        self.probe = probe or GiantFiberProbe(self.retina)
        if probe is None:
            # Calibrate against known physics before the session opens. Doing it live would
            # burn metered GPU time on arithmetic that has nothing to do with the world.
            cal = looming("looming", speed_mps=0.5)
            self.probe.calibrate(cal.frames, cal.fps, cal.theta_deg)

        self.frames: list[np.ndarray] = []
        self.events: list[Event] = []
        self.state = "forward"
        self.retreat_until = 0.0
        self.fps = 24.0
        self._t0 = time.time()
        self._fired_at: float | None = None
        self.kept: list[np.ndarray] = []

    # ------------------------------------------------------------------ perception

    @staticmethod
    def to_square_luma(frame: np.ndarray, size: int = 192, invert: bool = False) -> np.ndarray:
        """Reactor frames are 1664x960 RGB. The retina wants square single-channel.

        Centre crop rather than squash: the facet directions are projected through a
        pinhole model, so scaling the axes unevenly silently corrupts every angle.
        """
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        h, w = frame.shape
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        frame = frame[top:top + side, left:left + side]
        idx = np.linspace(0, side - 1, size).astype(int)
        frame = frame[idx][:, idx]
        return (255 - frame if invert else frame).astype(np.uint8)

    def push(self, frame: np.ndarray) -> bool:
        """Add a frame, run the circuit over the rolling window, return True if it commits.

        The window is re-run from scratch each time rather than updated incrementally. The
        filters are IIR and making them resumable correctly is fiddly; at 223 facets over a
        second and a half of frames this is cheap, and cheap and right beats clever and
        subtly wrong when the thing is on stage in two hours.
        """
        self.frames.append(self.to_square_luma(frame, invert=self.cfg.invert))
        keep = int(self.cfg.window_s * self.fps)
        if len(self.frames) > keep:
            self.frames = self.frames[-keep:]

        # Below about half a second there is not enough history for the adaptation and
        # contrast filters to have settled, and anything they report is startup transient.
        if len(self.frames) < max(16, int(0.5 * self.fps)):
            return False

        det = self.probe.run(np.stack(self.frames), self.fps, fov_deg=self.cfg.fov_deg)
        return bool(det.fired and det.fire_frame is not None
                    and det.fire_frame >= len(self.frames) - 3)

    # ------------------------------------------------------------------- the session

    async def run(self, api_key: str, out_dir: Path) -> dict:
        try:
            from reactor_sdk import Reactor
        except ImportError:
            raise SystemExit(
                "reactor-sdk is not installed. Run: pip install reactor-sdk\n"
                "Docs: https://docs.reactor.inc/sdk-reference/python/reactor.md"
            ) from None

        model_name = MODELS.get(self.cfg.model, self.cfg.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_i = 0

        async with Reactor(model_name=model_name, api_key=api_key) as reactor:
            await reactor.connect()
            self._log("connected", 0, {"model": model_name})

            if self.cfg.seed_image is not None:
                ref = await reactor.upload_file(str(self.cfg.seed_image))
                await reactor.send_command("set_image", {"image": ref})
                self._log("set_image", 0, {"file": self.cfg.seed_image.name})

            await reactor.send_command("set_prompt", {"prompt": self.cfg.prompt})
            await reactor.send_command("set_seed", {"seed": self.cfg.seed})
            await reactor.send_command("start", {})
            await reactor.send_command("set_move_longitudinal", {"move_longitudinal": "forward"})
            self._log("flying", 0, {})

            async def provoke():
                """Bring a threat to the fly on a schedule, by rewriting the world."""
                await asyncio.sleep(self.cfg.lunge_at_s)
                while not done.is_set():
                    await reactor.send_command("set_prompt", {"prompt": self.cfg.lunge_prompt})
                    self._log("lunge", frame_i, {"prompt": self.cfg.lunge_prompt[:48]})
                    await asyncio.sleep(6.0)
                    if done.is_set():
                        break
                    await reactor.send_command("set_prompt", {"prompt": self.cfg.prompt})
                    self._log("calm", frame_i, {})
                    await asyncio.sleep(max(1.0, self.cfg.lunge_every_s - 6.0))

            output = reactor.tracks.with_direction("recvonly").with_kind("video").one()
            done = asyncio.Event()

            # The SDK delivers frames on its own thread, not on this event loop, so
            # asyncio.create_task() inside the callback raises "no running event loop" and
            # the dodge is never sent. Capture the loop here and hand work back to it
            # thread-safely instead.
            loop = asyncio.get_running_loop()

            def dispatch(coro):
                asyncio.run_coroutine_threadsafe(coro, loop)

            @output.on_frame
            def on_frame(frame):  # noqa: ANN001 - SDK supplies its own frame type
                nonlocal frame_i
                if done.is_set():
                    return
                arr = getattr(frame, "data", None)
                arr = np.asarray(arr if arr is not None else frame)
                frame_i += 1
                # Keep every third frame. Full rate for a minute of 48fps video is more
                # memory than this needs to cost, and a third is plenty to watch back.
                if frame_i % 3 == 0:
                    self.kept.append(self.to_square_luma(arr, invert=self.cfg.invert))

                now = time.time() - self._t0
                if now > self.cfg.max_seconds:
                    loop.call_soon_threadsafe(done.set)
                    return

                if self.state == "retreat":
                    if now >= self.retreat_until:
                        self.state = "forward"
                        dispatch(self._cmd(reactor, "forward", frame_i))
                    return

                if self.push(arr):
                    self.state = "retreat"
                    self._fired_at = now
                    chunk_s = 12.0 / self.fps  # ~3 latent frames, ~12 pixel frames
                    self.retreat_until = now + self.cfg.retreat_chunks * chunk_s
                    self._log("escape", frame_i, {"wall_s": round(now, 3)})
                    dispatch(self._cmd(reactor, "back", frame_i))

            provoker = asyncio.create_task(provoke())
            try:
                await asyncio.wait_for(done.wait(), timeout=self.cfg.max_seconds)
            except asyncio.TimeoutError:
                pass
            finally:
                provoker.cancel()
                # Non-recoverable on purpose: a recoverable disconnect keeps the GPU, and
                # the meter, reserved for a reconnection that is not coming.
                await reactor.disconnect()
                self._log("disconnected", frame_i, {})

        self.save_clip(out_dir)
        return self.report(out_dir, frame_i)

    def save_clip(self, out_dir: Path) -> Path | None:
        """Write what the circuit actually saw, so the run can be played back."""
        if not self.kept:
            print("  ! no frames captured, nothing to save")
            return None
        try:
            import imageio.v3 as iio
        except ImportError:
            return None
        path = out_dir / "reactor-live.mp4"
        rgb = np.repeat(np.stack(self.kept)[..., None], 3, axis=-1)
        try:
            iio.imwrite(path, rgb, fps=16, codec="libx264")
        except Exception as exc:  # noqa: BLE001 - a missing encoder must not lose the run
            print(f"  ! could not encode clip ({exc})")
            return None
        print(f"  saved {len(self.kept)} frames -> {path}")
        return path

    async def _cmd(self, reactor, direction: str, frame: int) -> None:
        """Send a movement command and record when the world acknowledged it.

        The gap between this and the escape event is the real closed-loop latency, which is
        bounded below by Reactor's chunk boundary. We measure it rather than claim it.
        """
        sent = time.time() - self._t0
        await reactor.send_command("set_move_longitudinal", {"move_longitudinal": direction})
        acked = time.time() - self._t0
        self._log(f"cmd:{direction}", frame, {
            "sent_s": round(sent, 3), "acked_s": round(acked, 3),
            "latency_ms": round((acked - sent) * 1000, 1),
            "since_escape_ms": (None if self._fired_at is None
                                else round((sent - self._fired_at) * 1000, 1)),
        })

    def _log(self, kind: str, frame: int, detail: dict) -> None:
        self.events.append(Event(round(time.time() - self._t0, 3), kind, frame, detail))
        print(f"  [{self.events[-1].t:6.2f}s] {kind:<16s} frame {frame}")

    def report(self, out_dir: Path, n_frames: int, live: bool = True) -> dict:
        escapes = [e for e in self.events if e.kind == "escape"]
        cmds = [e for e in self.events if e.kind.startswith("cmd:")]
        lat = [c.detail["since_escape_ms"] for c in cmds
               if c.detail.get("since_escape_ms") is not None]

        rep = {
            "live": live,
            "model": MODELS.get(self.cfg.model, self.cfg.model),
            "frames": n_frames,
            "escapes": len(escapes),
            "mean_escape_to_command_ms": round(float(np.mean(lat)), 1) if lat else None,
            "events": [{"t": e.t, "kind": e.kind, "frame": e.frame, **e.detail}
                       for e in self.events],
        }
        (out_dir / "loop_report.json").write_text(json.dumps(rep, indent=1))
        return rep
