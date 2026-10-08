# demo

How to walk a judge through this in three minutes, and the paste-ready submission answers.

## the one-liner

> We put a fruit fly's escape reflex inside AI-generated worlds to find out whether their
> physics is real. It fires at 20.7 degrees. A real fly fires at about 20.

## the three minute walkthrough

**Open on the hero. Do not scroll yet.** The thing moving on screen is the algorithm, not a
screensaver. Point at it:

> "That hex grid is a fly's eye, 223 facets at the real 5.1 degree spacing. The blue arrows
> are per-facet optic flow, one Reichardt correlator each. The yellow box is where the
> expansion is coming from, which is what tells it which way to dodge. The bar is one neuron
> charging toward threshold. This is running live on generated footage."

**Then the problem, in one sentence.**

> "World models are about to be where robots get trained. Right now the only way anyone
> checks one is to watch the video and decide it looks fine. Looking fine and behaving right
> are different things."

**Hit [ Play: you are the fly ]. Let it run.** Do not talk over the waiting. The six to ten
seconds of nothing is the demo. When it fires and Control tells them to override, hand them
the laptop:

> "Your call."

Whatever they pick, say what it cost:

> "Retreat and you live and you learn nothing. Override and you bank a verified world. That
> is the actual tradeoff in evaluation, and it is why nobody has good numbers on these
> models."

**If they override three times, let them reach the last room.** It is the one we corrupted on
purpose: the approach decelerates as it arrives, which nothing real does. The reflex stays
quiet and the room kills them. Then:

> "It did not fire because there was nothing real to fire about. That room's geometry was
> impossible. A world model that renders a beautiful hallway with the wrong looming rate
> does not look wrong to you, and it does not look wrong to a vision model grading the
> footage. It looks wrong to the fly."

**Back to the site, scroll to the descent.** Eighteen levels, four strata.

> "Four where we know the exact geometry, eleven generated, three we broke on purpose. The
> stamp is the level's verdict, not the fly's. A level can be escaped and still fail, and one
> is: it got out of that one at 54.6 degrees instead of 20, so the world is flagged."

**Land on the number.**

> "20.7 plus or minus 0.63 degrees across an eight-fold speed sweep. Published behavioural
> figure for a real fly is about 20. That is the whole argument: the number comes from an
> animal, not from our repo, so a vendor and a customer can both check it."

**Close on the real world.** Warning time is the stat that travels:

> "At a quarter metre a second it gives you 1.12 seconds of warning. At one metre a second,
> 0.27. That is what a braking system actually needs, and it falls out of a fixed-angle
> trigger for free."

## if they ask about Reactor

Everything live runs on Reactor and the loop closes inside one session.

- `reactor/lingbot-world-2`, seeded from a Nano Banana plate via `upload_file` + `set_image`
- fixed seed before `start`, so a run is repeatable and two models get the same level
- frames off `tracks(recvonly, video).one()`, each one cropped to 223 facets and judged
- `set_move_longitudinal` forward, then `back` on the frame the circuit commits
- `set_move_lateral` for the banked half, aimed by the threat bearing the eye gives us
- `set_prompt` mid-generation to bring a threat in without restarting the session
- `request_schema` first, so the same probe runs across models that do not share an action
  space, and `request_recording` at the end for Reactor's own copy of the session

Two things worth telling them because they are real findings:

1. **LingBot World 2 generates infinite worlds.** Drive forward down a hallway and it makes
   more hallway. Nothing ever closes on you, so a correct escape circuit sits silent for a
   whole session. That is the model doing what it advertises, and it is why we rewrite the
   prompt mid-generation instead of waiting.
2. **Frames arrive on the SDK's thread.** `create_task` inside the frame callback raises
   "no running event loop", the exception vanishes into the callback, and the dodge never
   sends while the log still prints an escape. Capture the loop and use
   `run_coroutine_threadsafe`. It cost us a session.

## the cross-model sweep

If there is time before you present, kick this off. It spends real credits on the thing the
project is actually for, and the table appears on the site by itself when it finishes.

```
export REACTOR_API_KEY=rk_...
python scripts/reactor_sweep.py --budget-usd 40
```

Same fly, same seed plate, same prompt, same seed, against LingBot World 2, LingBot V1,
Helios, LongLive 2.0 and Orbis. The V1 against V2 row is the pitch in one line: a vendor
shipped an update, and this says whether the update changed how an agent behaves inside it.

## submission answers

**Team.** Solo. Azra Bano, ab2895@scarletmail.rutgers.edu

**Project name.** flyloop

**One line.**

> A playable descent through the Backrooms where you are a fly, your escape reflex is real,
> and the only room that kills you is the one whose physics we broke.

**Longer.**

> Behavioural regression testing for AI-generated worlds. We implement the Drosophila LC4 and
> LPLC2 to giant fibre escape circuit, run it on frames coming out of Reactor and Veo, and
> measure the angle at which it commits. Across an eight-fold approach speed sweep it fires
> at 20.7 plus or minus 0.63 degrees against a published behavioural figure near 20, flags
> two of three deliberately corrupted worlds, and raises no false alarms on the four we know
> are correct.

**Track.** Interactive Narrative, on LingBot World 2.

**Links.**

- Live: the Vercel URL
- Play it: `<vercel-url>/play.html`
- Code: https://github.com/azrabano23/flyloop

**Reactor models used.** `reactor/lingbot-world-2` for the live closed loop, with
`reactor/lingbot`, `reactor/helios`, `reactor/longlive-v2` and `reactor/visko-orbis-dynamic`
in the cross-model sweep.

**Google used.** Nano Banana Pro and Nano Banana 2.1 for the seed plates and film stills,
Veo 3.1 for the eleven generated levels, Lyria 3.5 for the score. The connectome the circuit
is read off was segmented with Google Research's flood-filling networks.

## before you submit

- [ ] Vercel deployed, link opens on a phone
- [ ] `play.html` loads and the first room reaches a decision
- [ ] Sound on for the walkthrough, headphones if the room is loud
- [ ] Form submitted
- [ ] Social post up, link pasted in for the extra credit
