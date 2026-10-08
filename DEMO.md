# The two minute walkthrough

Bracketed lines are what to click, not what to say.

---

**[ Start on the hero. Do not scroll. ]**

> "A fruit fly's whole brain has been mapped, neuron by neuron. Google Research built the
> segmentation behind that map.
>
> And in there is one tiny circuit that does a single job. Two cells watch for something
> growing in the eye, one neuron fires, the fly is gone.
>
> The clever part is it never works out *what* the thing is. It just measures how fast the
> edge is spreading. So you cannot fool it with something that only looks right."

**[ Point at "What this is for". ]**

> "Which matters, because AI worlds are about to be where robots get trained. And right now
> the only check anyone runs is watching the video and going, yeah, looks good. If things
> arrive too late in the sim, the robot brakes too late in real life, and the footage still
> looks perfect."

**[ Click PLAY, then hand them the laptop. ]**

> "So we made you the fly. Everything here is through its eye. 223 facets, no detail, no red,
> because flies can't see red. You can't tell what's in the room with you. Neither can it.
>
> When the alarm goes off you get six seconds, arrow keys to dodge. But a voice tells you to
> ignore it, because staying is the only way you learn anything."

**[ Let them play one room. Scroll up to the rooms. ]**

> "Eleven Backrooms levels, all AI generated. Then seven we built ourselves where we know the
> real physics. Four correct, three broken on purpose. It catches two of the three.
>
> And the room that kills you in the game is a broken one. The reflex stays totally silent.
> There was nothing real to warn you about."

**[ Scroll to the Reactor videos. ]**

> "This all runs on Reactor. Same fly, five of their world models, and that's real footage of
> what it was flying through.
>
> LingBot version one scores 47. Version two scores 82. Same test. That's a company shipping
> an update, and this is a number for what it changed.
>
> Rooms are Nano Banana, monsters are Veo, music is Lyria, and the hand tracking is MediaPipe."

**[ Scroll to the bottom. ]**

> "So you get a pass or a fail on a world model. Cheap enough to run on every build, no AI
> judging it, and the number comes from a real animal so anyone can check it.
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
