# The two minute walkthrough

Say it like you are telling someone a story, not reading a spec. The bracketed lines are what
to click, not what to say.

---

**[ Start on the hero. Do not scroll yet. ]**

> "So a fruit fly's entire brain has been mapped, neuron by neuron. Google Research built the
> segmentation behind that map.
>
> And somewhere in it there's this one tiny circuit that does a single job. Two cells watch
> for something growing in the fly's eye, and when it gets big enough, one neuron fires and
> the fly is gone. That's it. That's the whole thing.
>
> Here's what's interesting about it. It never figures out *what* the thing is. It just
> measures how fast the edge is spreading. Which means you cannot fool it with something that
> only looks right."

**[ Scroll to "What this is for", point at the first line. ]**

> "And that matters, because AI worlds are about to be where robots get trained. Right now the
> only way anyone checks one is to watch the video and go, yeah, looks good. But if things
> arrive too late in the simulation, the robot learns to brake too late in real life, and the
> video still looks perfect. Nobody can tell you that."

**[ Click PLAY. Let it load, then hand them the laptop. ]**

> "So we made you the fly.
>
> Everything you're seeing is through the fly's eye. 223 facets, no detail, no red, because
> flies can't see red. You genuinely cannot tell what's in the room with you. Neither can the
> fly.
>
> Wait for it. When the alarm goes off you get six seconds. Arrow keys to dodge.
>
> But a voice is going to tell you to ignore it and keep walking, because staying is the only
> way you learn anything about the room."

**[ Let them play one room. Then scroll up to the rooms. ]**

> "There are eleven rooms. Backrooms levels, all AI generated, with things walking down them.
> Smilers, hounds, the guy in the hat from the A24 film.
>
> And then down here there are seven more we built ourselves, where we know exactly what the
> physics should be. Four are correct. Three we broke on purpose.
>
> The fly catches two of the three. And the one room that actually kills you in the game is
> one of the broken ones, because the reflex stays completely silent. There was nothing real
> to warn you about."

**[ Scroll to the Reactor section with the five videos. ]**

> "All of this runs on Reactor. We pointed the same fly at five of their world models.
>
> That footage is real, that's the world the fly was actually flying through. And look at
> this. LingBot version one scores 47, version two scores 82. Same test, same settings. That's
> a company shipping an update, and this is a number for what the update changed.
>
> The rooms are Nano Banana, the monsters are Veo, the music is Lyria, and if you turn the
> camera on in the game you dodge with your hand, which is MediaPipe reading your hand off the
> webcam."

**[ Scroll to the bottom. ]**

> "The point of all of it is that this gives a world model a pass or a fail. Cheap enough to
> run on every single build, no AI judge grading it, and the number comes from an actual
> animal, so anyone can check it against the literature.
>
> Evolution already paid for this test. We just plugged it in."

---

## If they ask things

**"Did you actually use Reactor live?"**
Yes. Five models, about twenty minutes of live GPU time. The escape the fly makes gets sent
straight back into the session that generated the frame, so the world reacts to the fly.

**"What surprised you?"**
Two things. LingBot World 2 makes infinite worlds, so if you just fly forward nothing ever
comes at you and a correct detector sits silent forever. We had to rewrite the world
mid-generation to bring threats in. And the models don't share an action space at all. One of
them refused almost every command we sent, so we ask each one for its schema before we send
anything.

**"Is it a real fly brain?"**
No, and we say that on the site. We implemented the published equations for that circuit
ourselves. The connectome that maps it was segmented using Google Research's work. We didn't
take anyone's model.

**"What's missing?"**
Right now a silent circuit could mean an honest world or a broken detector. Two probes that
fail differently would tell them apart. That's the next thing.

**"Avatars?"**
Not built. The voice telling you to override your reflex should be a Vidu S2 avatar of Captain
Clark from the film. That's the first thing we'd add.
