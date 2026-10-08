# Fright or Fly

**A fly brain, inside AI-generated worlds.** Play as the fly in A24's Backrooms, and the one
room you cannot survive is the one whose physics we faked.

Reactor x Google DeepMind World Models Hackathon. Solo entry by Azra Bano.

- **Play it:** `web/play.html`
- **The write-up:** `web/index.html`

---

## The problem

AI world models are about to be where robots get trained.

Right now the only way anyone checks one is to watch the video and decide it looks fine. But
looking fine and behaving right are different things. If approaches arrive late in the
simulator, a robot trained in it learns to brake late in the street, and nothing in the
footage looks wrong. No vendor can tell you whether an update to their world model changed
how an agent behaves inside it.

## The solution

A fruit fly solved a version of this 400 million years ago.

Two cell types in its brain, LC4 and LPLC2, watch for something expanding across the eye and
feed a single giant fibre neuron. When it crosses threshold, the fly is gone. That circuit
never identifies an object, never separates it from the background and never works out how
far away it is. It only measures how fast an edge spreads.

Which makes it a physics test that cannot be gamed by a world that merely looks right. Put it
inside a generated world and it fires on time if the motion obeys geometry, and sits silent if
the motion is nonsense.

**Measured result:** across an eight-fold approach speed sweep it commits at
**20.7 +/- 0.63 degrees**. The published behavioural figure for a real fly is near 20. On the
seven levels where we know the true geometry it flags 2 of the 3 we corrupted on purpose, with
0 false alarms on the 4 we know are correct.

## The game

You are the fly. You walk down through the Backrooms one AI-generated room at a time.

1. **You see 223 facets and no detail.** The whole game is rendered through the retina itself,
   at the measured 5.1 degree spacing, with no red because a fly has no receptor for it. You
   cannot tell what is in the room with you, because the fly cannot either.
2. **Then one neuron fires and you get six seconds.** Dodge with the arrow keys, or turn the
   camera on and physically swing your hand. A voice tells you to ignore the alarm, because
   staying is the only way to learn anything about the room.
3. **One room kills you, and the reflex never fires.** That room's approach decelerates as it
   arrives, which no real object does, so there is nothing for a reflex tuned to real physics
   to warn you about. Why it stays quiet is the whole point of the project.

## How we used each sponsor

| | | |
|---|---|---|
| **Reactor** | LingBot World 2, LingBot V1, Helios, LongLive 2.0, Orbis | Five live world models, one fly. Sessions, action control, mid-generation prompt rewriting, and the escape sent back into the running world. |
| **Reactor SDK** | `reactor-sdk` 1.9.0 | `request_schema`, `upload_file` + `set_image`, `set_seed`, `set_move_longitudinal`, `set_move_lateral`, recvonly video tracks, `request_recording`, `disconnect`. |
| **Google AI Studio** | Nano Banana Pro | The film stills and the hero plate. |
| **Google AI Studio** | Nano Banana 2.1 | Draws the room every level starts from, which Veo and Reactor then animate. |
| **Google DeepMind** | Veo 3.1 | Animates the eleven generated rooms and the entities walking down them. |
| **Google DeepMind** | Lyria 3.5 | The score. Four cues that follow the game state: a bed, the empty room, the approach, and the hit. |
| **Google** | MediaPipe Hands | Hand landmarks off the webcam, so you dodge with your body instead of a key. |
| **Google Research** | Flood-filling networks | Segmented the *Drosophila* EM volume behind the whole-brain connectome that maps this pathway. We implemented the published equations ourselves; we took no model from anyone. |

## Live on Reactor

One probe, one seed plate, one prompt, one fixed seed, pointed at each world model in turn.
**$14.12 of GPU time across 1210 session-seconds.** Footage from each is on the site.

| model | escapes | frames judged |
|---|---|---|
| `reactor/lingbot-world-2` | 82 | 7964 |
| `reactor/lingbot` (V1) | 47 | 7134 |
| `reactor/visko-orbis-dynamic` | 28 | 3727 |
| `reactor/helios` | 22 | 4328 |
| `reactor/longlive-v2` | 0 | 0 |

Three things this run found:

- **V1 scores 47 and V2 scores 82** on the same probe, same seed, same prompts. That is a
  vendor shipping an update and a number for what it changed about approach geometry.
- **LingBot World 2 generates infinite worlds.** Drive the camera forward down a hallway and
  it makes more hallway. Nothing ever closes on you, so a correctly working escape circuit
  sits silent for a whole session. That is the model doing what it advertises, and it is why
  the loop rewrites the prompt mid-generation to bring a threat in rather than waiting.
- **The catalog does not share one action space.** `longlive-v2` connected and then refused
  `set_image`, `set_prompt` and `set_move_longitudinal`, so there was nothing to fly. Three
  others refused `set_move_longitudinal`. That is why every session asks for the schema before
  it sends anything, and it is recorded rather than hidden.

## What this is for

- Robots trained on fake physics fail on real physics.
- This gives a world model a pass or a fail. Two builds, two scores, and the regression has a name.
- Self-driving sims get checked on time-to-contact, which is the quantity a car actually brakes on.
- It is cheap enough to run on every commit. No judge model, no per-frame bill, just numpy on a laptop.
- The number comes from an animal, not from us, which is why anyone can check it.
- Evolution already paid the training bill. There are more instruments like this in the
  connectome that nobody has picked up.

## Not built yet

Stated plainly so nobody has to guess what is real here:

- **Avatars as the entities.** The voice called Control, the one telling you to override your
  own reflex, is static text. Binding it to `reactor/vidu-s2-avatar` as Captain Clark, the
  human explorer from the film canon, is the obvious next build and the one we would do first.
  We did not get to it, so no avatar model is used in this entry.
- **Telling a broken detector from a broken world.** Right now a silent circuit could mean an
  honest world or a deaf instrument. Two probes that fail in different ways would settle it.
- **Generated rooms have no ground truth**, so the log reports what the fly did rather than
  grading the world. Only the seven control levels are graded.

## Run it

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python scripts/run_synthetic.py                 # score the 18 clips, write artifacts/results.json
python scripts/fly_reactor.py --dry-run         # the live loop's state machine, offline

export REACTOR_API_KEY=rk_...
python scripts/fly_reactor.py --seed-image artifacts/seeds/yellow-hallway.png
python scripts/reactor_sweep.py --parallel 5 --seconds 240 --budget-usd 60
```

The site is static. Serve `web/` with anything.

## Layout

```
flyloop/
  retina.py        223 ommatidia on a hex lattice, 5.1 deg apart, 5.7 deg acceptance
  probe.py         elementary motion detectors, adaptation, contrast gain, wide-field suppression
  gf.py            LC4 rate + LPLC2 size converging on the giant fibre, published weights
  stimuli.py       analytic looming with exact ground truth, plus three broken controls
  reactor_loop.py  the live closed loop on Reactor
  sonify.py        the circuit as audio: spike train, room tone, the commit
scripts/           synthetic sweep, Reactor sweep, seed images, Veo capture, Lyria score
web/               index.html, play.html, and every artifact they read
```
