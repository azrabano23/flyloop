# submission

Paste-ready answers for the Reactor x Google DeepMind form. Due 3:15 PM ET.

## team name

Solo entrant. Three options, in order of preference:

1. **Bug Detector** (recommended). A fly is a bug. It finds bugs in world models. It also
   reads like a real dev tool, which is the register we want for the Lightspeed judge.
2. **400 Million Years of QA**
3. **flyloop**

## teammate names and emails

Azra Bano, ab2895@scarletmail.rutgers.edu

## one line description

Recommended:

> **We put a real fruit fly's escape reflex inside AI-generated worlds. Play as the fly in
> A24's Backrooms, and the one room you cannot survive is the one whose physics we faked.**

Alternatives:

> We put a fly's collision-avoidance reflex inside a generated world and measured when it
> panicked, as a way to test whether the world's motion is geometrically consistent.

> Behavioural regression testing for AI-generated worlds, using a 400-million-year-old
> visual reflex as the probe.

## demo link

Live site: `https://<name>.vercel.app` (deploy from `web/`, see below)

What the viewer sees: a Backrooms hallway generated from a Nano Banana seed plate, with
three live panels. The world on the left. The fly's eye in the middle, 223 ommatidia lighting
up with local outward motion. The escape command on the right, charging until it crosses
threshold, at which point the chip flips to ESCAPE in red. Switch between clips whose
geometry we know exactly and generated clips where we do not, and the readout tells you
which is which. Underneath, the measured result: three detector configurations, and what the
second channel buys.

## repo

https://github.com/azrabano23/flyloop

## social post, X

> Spent today at the Reactor x Google DeepMind world models hackathon putting a fruit fly's
> escape reflex inside a generated world.
>
> It fires at 20.7 ± 0.63° across an 8x approach speed sweep. The published behavioural
> figure is near 20°.
>
> The point: nobody can tell you whether a world model update changed how your agent
> behaves, only whether the video looks better. A fly has been solving a version of that
> problem for 400 million years with about 1400 neurons and no training data.
>
> Built on @reactorworld with Nano Banana 2.1, Veo 3.1 and Lyria. Code and honest limits:
> github.com/azrabano23/flyloop

## social post, LinkedIn

> At the Reactor x Google DeepMind World Models Hackathon in NYC today, I built flyloop:
> behavioural regression testing for AI-generated worlds.
>
> The problem is narrow and real. World models are starting to be sold as places to test
> robots. When one ships an update, the video gets better and nobody can tell you whether
> your agent still behaves the same. Reactor's own head of product has said publicly that
> evaluating these models is unsolved, including at DeepMind.
>
> So instead of asking a bigger model whether the video looks right, I used a fruit fly's
> escape circuit as the instrument. It does not segment objects or estimate depth. It reads
> raw optic flow across a hex array of ommatidia, and when something is about to hit it, one
> neuron fires.
>
> Built with Nano Banana 2.1 for the seed worlds, Veo 3.1 and Lyria 3.5 from Google
> DeepMind, and a live closed loop on Reactor's LingBot World 2, where the escape command is
> sent straight back into the session that generated the frame.
>
> Result: the two-channel model commits at 20.7 ± 0.63 degrees across an 8x speed sweep,
> tighter than either single channel. What it does not do yet is separate a broken detector
> from a broken world, and that limitation is written on the site next to the numbers.
>
> Code: github.com/azrabano23/flyloop

## vercel

```
vercel.com/new  ->  import azrabano23/flyloop
Root directory: web
Framework preset: Other
```

Static, no build step. Deploys on every push. Suggested project names:
`bug-detector`, `400-million-years-of-qa`, `unit-tested-by-insects`, `nobody-would-build-this`.

## checklist before 3:15

- [ ] Vercel deployed and the link opens on a phone
- [ ] `python scripts/fly_reactor.py --seed-image artifacts/seeds/yellow-hallway.png` run
      once on the laptop with a real `REACTOR_API_KEY`, so the live Reactor loop is not
      only code
- [ ] Deck open in a tab at `deck/index.html`
- [ ] Form submitted with the links above
- [ ] Social post up, link pasted into the form for the extra credit
