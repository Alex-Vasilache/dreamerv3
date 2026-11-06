# Expressive robot faces: research report for a phone-screen face on a balancing robot

**Sources.** The Anki/Vector findings come mostly from the **leaked or open Vector source**, mirrored at https://github.com/kercre123/victor (branch `snowboy`). I read the code files directly, so those numbers are real. Anything I could not confirm is marked **[unverified]**.

---

## 1. Anki Cozmo and Vector

### Face rig ("ProceduralFace")
Source: `robot/clad/src/clad/types/proceduralFaceTypes.clad` and `docs/architecture/proceduralFace.md` in the kercre123/victor repo.

**Whole-face parameters (6):** `faceAngle_deg`, `faceCenX`, `faceCenY`, `faceScaleX`, `faceScaleY`, `scanlineOpacity`.

**Per-eye parameters on Vector (25 for each eye):**
- **Position and shape:**
  - `EyeCenterX` and `EyeCenterY` are in pixels.
  - `EyeScaleX` and `EyeScaleY`: 1.0 means nominal size.
  - `EyeAngle` rotates the eye.
- **Corner rounding:** 8 corner radii — `Lower/Upper × Inner/Outer × RadiusX/Y`. Each is a fraction of eye width or height. These make the eye a rounded rectangle that can morph into an ellipse or a sharp, angry shape.
- **Lids, each with three values:**
  - `UpperLidY` / `LowerLidY`: 0 = open, 1 = fully closed.
  - `UpperLidAngle` / `LowerLidAngle`, in degrees.
  - `UpperLidBend` / `LowerLidBend`: 0 = straight, >0 bends into the eye, <0 bends away.
  - The lids are filled black polygons drawn over the eye.
- **Colour and glow (Vector only):** `Saturation`, `Lightness`, `GlowSize`, `GlowLightness`, plus `HotSpotCenterX/Y`, a bright "pupil-ish" hotspot that shifts in the direction of gaze.

**Cozmo's version** is reimplemented in pycozmo (https://raw.githubusercontent.com/zayfod/pycozmo/master/pycozmo/procedural_face.py):
- It uses the same rig without the colour and glow fields: 5 face values + 19 per eye = 43 floats.
- Canvas is 128×64. Nominal eye is 28×40 px, and each eye sits ±width/5 from centre.
- The image is resampled nearest-neighbour, and every other row is dropped (`np_im[::2]`), giving 128×32 lit rows. **That row-skipping is the source of Cozmo's visible scanline look.**
- pycozmo ships 25 named presets (Amazement, Anger, Boredom, Confusion, Fear, Happiness, Sadness, Suspicion, Tiredness, and others), based on Catherine Chambers' "Expressive Eyes" (https://pycozmo.readthedocs.io/en/latest/generated/pycozmo.expressions.expressions.html).

**Displays:**
- Cozmo: single-colour OLED (https://en.wikipedia.org/wiki/Cozmo). The 128×64 resolution comes from pycozmo; Anki never documented it **[unverified officially]**.
- Vector 1.0: 184×96 colour IPS, RGB565 over SPI. Vector 2.0: 160×80 (https://os-vector.github.io/vector-docs/5.-Hardware/index.html).
- On Vector the face is drawn in HSV. **The hue is a user setting and is never animated**; only saturation and lightness animate (`proceduralFace.md`).

**Renderer details** (`cannedAnimLib/proceduralFace/proceduralFaceDrawer.cpp`):
- Anti-aliasing uses a box filter.
- Glow is rendered with a hotspot falloff of 0.48.
- Scanlines are optional (`kProcFace_DefaultScanlineOpacity`).
- A per-frame **noise "shimmer"** is applied: lightness randomly multiplied between 0.92 and 1.14, cycling through 5 noise frames.
- A `ScanlineDistorter` shifts rows horizontally and adds "off-noise" pixels for glitch effects.

### Keep-alive layer (blinks and eye darts)
Source: `animProcess/.../faceLayerManager.cpp`.

**Blinks:**
- Randomly spaced every **3–10 s**, and forced at least every 30 s to protect the screen.
- Each blink is 7 frames of about 33 ms. Eye height goes .85 → .6 → .1 → .05 (closed) → .15 → .7 → .9, while width *grows* to 5× at closure.
- **The blink is a squash into a horizontal line, not a lid closing.** It ends with a 100 ms settle.

**Eye darts (saccades):**
- Every **1–2.25 s**, up to **15 px** from centre on a 184 px wide screen, or 1 px when "focused".
- Looking up scales the eyes up to 1.05×; looking down scales them to 0.9×. The outer eye grows 3% when looking sideways.
- The hotspot moves 1.5× as far as the eye.
- Medium darts (>5 px) and long darts (>10 px) get 1–2 in-between frames with a vertical squash of 0.7–0.85. In the code's own words, this is "a sort of mini-blink to the dart (per animators)".
- `LookAt()` shifts the whole face toward the gaze target, adding a bit of perspective by scaling the eyes.

### Emotion engine
Sources: `clad/src/clad/types/emotionTypes.clad`, `engine/moodSystem/*`, `resources/config/engine/*`.

**The five dimensions in Vector's code are Happy, Confident, Social, Stimulated and Trust.** "Calm" is not among them, so your list is not confirmed for Vector. Press coverage describes Cozmo as becoming "less brave, less calm, less happy", so Cozmo may have used a different set **[Cozmo's exact list unverified]**.

**How the mood values behave:**
- Values run from -1 to 1. Stimulated and Trust run from 0 to 1.
- Each event adds fixed deltas, scaled down by a *repetition penalty* when the same event repeats.
- Values decay over time:
  - Default: held for 10 s, down to 60% by 75 s, zero at 150 s.
  - Social: zero by 100 s.
  - Confident: zero by 70 s.
  - Stimulated: decays at a fixed rate per minute.

**Example events** (from `resources/config/engine/emotionevents/*.json`):
| Event | Effect |
|---|---|
| CliffReact | Happy −0.12, Stimulated +0.04 |
| ReactToUnexpectedMovement (picked up or pushed) | Confident −0.2, Happy −0.1, Stimulated +0.05 |
| ReactToObstacle | Confident −0.1, Stimulated +0.1 |
| ReactToDark | Stimulated −1 |
| PlacedOnCharger | Stimulated −0.5 |
| DriveOffCharger | Confident +0.3 |
| EnrolledNewFace | Social +1, Stimulated +1, Happy +.5, Confident +.5 |
| EyeContactReaction | Social +.8, Stimulated +.55 |

**The mood is collapsed into a "SimpleMood"** (`moodManager.cpp`) that selects which group of canned animations plays:
- Stimulated ≤ 0.2 → **LowStim**. This takes precedence "because we don't want to annoy people".
- Otherwise Confident < −0.29 → **Frustrated** (the code notes this is the value used on Cozmo).
- Otherwise Stimulated ≤ 0.8 → **MedStim**.
- Otherwise → **HighStim**.

**Two more mood-to-output links:**
- `stimulationFaceDisplay.cpp` maps Stimulated linearly to **face saturation** (0 → 0.25, 1 → 1.0), smoothed with an EMA (N = 20). Bored means a washed-out face. The feature flag is off by default in that code.
- Stimulated and Trust also feed live **audio parameters**, so the procedural sound tracks the mood.

**Design lesson:** the mood is continuous, but what you *see* is mostly (a) discrete canned reactions to events, chosen by mood bucket, layered over (b) a procedural idle layer of blinks, darts and shimmer.

### Pixar/animator philosophy (Cozmo)
Sources: https://www.therobotreport.com/cozmo-ai-robot-brings-wall-e-to-life/ and https://techcrunch.com/2016/06/27/cozmo/
- Animation was led by ex-Pixar animator **Carlos Baena**. Anki built custom Maya tools so animators could animate the physical robot (head, lift, treads, face and sound together) and test each sequence on a prototype at their desk.
- **Pupils were removed**: "He didn't feel robotic enough, so the animators changed the size and ratio of the eyes" (Tappeiner).
- Frequent eye contact made him read as a dog or toddler; eye contact only at game start felt detached.
- Triggers described in coverage:
  - Pushed or flipped → sad noises and a squint.
  - Bored → plays games on his own screen.
  - Stacking cubes → confidence goes up.
  - Cliffs → bravery goes down.
  - Losing games → frustration.
  - Strangers → "standoffish and suspicious"; familiar people → happy.
- Quote: "feel alive and not like a big if-then-else statement."
- Core77 links Cozmo's eyes to EVE's look (https://www.core77.com/posts/54328/Whats-the-One-Design-Element-That-Can-Make-a-Robot-Lovable).

---

## 2. WALL-E (and EVE)
- Stanton took the eyes from **binoculars** at a ball game. He noticed they look sad or happy depending on their angle, and dropped the nose and mouth (with Buster Keaton in mind).
- He added a zoom lens to make the robot more sympathetic and childlike (Ralph Eggleston).
- **Simplicity was deliberate: too many movements would make him read as human.**
- He was designed as a "machine with function"; arms, treads and head all carry the expression. The whole robot can retract into a box ("boxing") to sleep or hide, a full-body closed posture. (https://en.wikipedia.org/wiki/WALL-E_(character))
- His eyebrow movements are paired with a **Nikon camera-shutter sound** (https://www.mentalfloss.com/entertainment/movies/10-space-age-facts-about-walle).
- **Ben Burtt's sound design** (https://news.symbolicsound.com/?p=4035):
  - He processed his own voice in Kyma and kept its artifacts on purpose. Electronic noise around words "made it so it wasn't human."
  - He treated the voice as "little phrases of music which indicated the feeling."
  - Motor sounds from arms, head and treads are part of the expression.
  - About 2,400 sounds in total.
- **EVE:** she has only her eyes (glowing shapes with horizontal CRT-like striations) plus arm poses. Animators "came up with just the right poses to express emotion" (Core77; the search snippet was second-hand **[partly unverified]**).
- Real replicas: Mike McMaster and Mike Senna each built one (Wikipedia). I found no details on how they handle expression.

---

## 3. Pollen Robotics / Hugging Face Reachy Mini (2025)
**Hardware** (https://huggingface.co/blog/reachy-mini): no screen. It has a **6-DoF head** (Stewart-platform style), **full body yaw**, **2 animated antennas**, a 5 W speaker, 4 microphones and a wide-angle camera. It is 28 cm tall and weighs 1.5 kg.

**Emotion library** (https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library):
- 81 recorded moves. Each is a 50 Hz JSON trajectory of head pose, antennas and body yaw, paired with an Ogg/Opus sound. They are loaded with `RecordedMoves` and played with `play_move`.
- Names include: amazed, anxiety, attentive, boredom, calming, cheerful, confused, contempt, curious, dance1–3, disgusted, displeased, downcast, dying, enthusiastic, exhausted, fear, frustrated, furious, go_away, grateful, impatient, inquiring, irritated, laughing, lonely, lost, loving, no / no_sad / no_excited, oops, proud, rage, relief, reprimand, resigned, sad, scared, serenity, shy, sleep, success, surprised, thoughtful, tired, uncertain, uncomfortable, understanding, welcoming, yes / yes_sad.
- Third-party reporting says the moves were captured by teleoperation with a Vive tracker **[unverified]** (https://www.hardware.com.br/tecnologia/reachy-mini-robo-emocional-open-source/amp/).

**Conversation app** (https://github.com/pollen-robotics/reachy_mini_conversation_app):
- The LLM calls `play_emotion` and `dance` as tools. Other tools: head tracking of the user's face, `move_head`, `sweep_look`, and `go_to_sleep`.
- A **layered motion system** queues primary moves (emotions, dances, breathing idle) and blends a **speech-reactive head wobble** on top.
- **Antennas freeze while the robot is listening** (triggered by voice-activity detection), an attentive "stillness" cue.
- The SDK convention is antennas up = happy, antennas down = sad.

**Legibility study:** Rogel, Yadollahi & Laban (2026), arXiv:2605.12786 (https://arxiv.org/abs/2605.12786).
- 100 participants, 10 clips.
- Exact emotion labels were recognised only modestly. Anger, sadness and interest were recognised best.
- **Valence and arousal were recovered better than discrete labels.** Positive clips were rated warmer.
- **Lesson:** a low-DoF robot reliably communicates valence and arousal, not fine emotion categories. That fits your continuous signals.

---

## 4. "Micro Duck" = Open Duck Mini, and Disney's BDX droids
**Open Duck Mini** (https://github.com/apirrone/Open_Duck_Mini):
- Open-source miniature of Disney's BDX droid by Antoine Pirrone (GitHub user apirrone).
- About 42 cm tall, BOM under $400, Raspberry Pi Zero 2W, reinforcement-learning walking policy trained in simulation. Sponsored by Hugging Face and Pollen.
- Expressive extras listed as add-ons: eye LEDs, speaker, camera.
- The runtime (https://github.com/apirrone/Open_Duck_Mini_Runtime) has gamepad mappings:
  - Triggers → left and right **antennas**.
  - X → **projector** (the head lamp) on/off.
  - B → play a **random sound**.
  - Y → experimental **head control**.
- It is puppeteered; there is no emotion model.

**Disney BDX, "Design and Control of a Bipedal Robotic Character"** — RSS 2024, https://www.roboticsproceedings.org/rss20/p103.html; arXiv:2501.05204 (https://arxiv.org/html/2501.05204v1). The DOI is presumably 10.15607/RSS.2024.XX.103 **[DOI unverified]**.
- **Body:** 5 DoF per leg plus a 4-DoF neck and head. 0.66 m tall, 15.4 kg.
- **Show functions:** actuated antennas, **LED-array eyes**, a head lamp, and two pairs of speakers. All are open-loop.
- **Effort cue:** at high walking speed, **the antennas drop and the eye radii narrow** to convey exertion. This maps directly onto your pain/effort signal.
- **Motion types:**
  - *Perpetual* — standing, with posture and gaze driven by joysticks.
  - *Periodic* — walk style; artists supply reference walks at several speeds, blended by velocity.
  - *Episodic* — clips authored in Maya, such as "happy dance", "excited", "jump" and "tantrum".
- While walking, head commands are offsets added to an animator-defined nominal head motion.
- The operator uses a Steam Deck. A short button press triggers a happy clip, a long press an angry clip.
- **Blend times: 0.1 s for facial show functions, 0.35 s for body motion.** The face reacts faster than the body.

---

## 5. Other robots and the Kalegina et al. study
- **Kismet (MIT):**
  - Its affect space is **Arousal × Valence × Stance**, interpolated between 9 prototype postures (https://groups.csail.mit.edu/lbr/sociable/facial-expression.html).
  - Valence mainly drives lip curvature but also moves the ears, lids, brows and jaw. Arousal spans weary → calm → interested/excited. Expressions intensify toward the extremes.
  - **Stance is essentially your dominance axis** (approach/open vs withdraw/closed). That page doesn't define it, so the detail needs Breazeal's own papers **[unverified]**.
- **Baxter (Rethink):**
  - Screen eyes and brows show state: standby (eyes closed), concentration, confusion when a task fails.
  - **The eyes glance where the arm is about to move**, so gaze announces intent (Rodney Brooks; 2012-era coverage found through search; I could not open qz.com, which returned 403).
- **Jibo:** "very little overt facial detailing is needed". It uses abstract forms plus a rotating body so the motion carries a "line of action"; it has a single eye graphic (https://robohub.org/designing-jibo-subtracting-the-obvious-and-add-the-meaningful/). Kalegina's raters disliked Jibo's face most.
- **Eilik:** EVE-inspired, single-colour OLED face, touch sensors on head, back and belly, and a shock sensor. It has four base states — normal, happy, angry, sad — and its personality shifts with how it is treated (https://hackster.io/news/energize-lab-s-eilik-aims-to-be-your-emotional-playful-eve-inspired-desktop-companion-bot-7cf253953bee).
- **Loona (KEYi):** 2.4-inch LCD with cartoon eyes, plus glowing ear-antennas that **rise when happy and droop when disappointed** (https://newatlas.com/robotics/loona-pet-robot-kickstarter/).
- **EMO (Living AI):** LED-matrix-style face on a small head with "1000+ faces" (marketing claim; the panel specs are **[unverified]**).
- **Kalegina et al., HRI 2018:**
  - The authors are Kalegina, Schroeder, Allchin, Berlin and Cakmak — not Thilo/Sanders/Gerber. DOI **10.1145/3171221.3171286** (https://hcrlab.cs.washington.edu/publications/kalegina2018hri). The PDF would not extract here; the findings below come from the authors' interview (https://spectrum.ieee.org/what-people-see-in-157-robot-faces).
  - 157 rendered faces were coded on 76 properties; 50 participants per survey; still images only.
  - **Removing the mouth hurt friendliness, trust, intelligence and likability most.** Raters called the no-mouth face "creepy" and said it "looks like it is watching my every move."
  - Faces without pupils, and faces *with* eyelids, also scored lower. Low eyebrows raised perceived intelligence.
  - Small eyes, close-set eyes, cheeks and a white face lowered ratings.
  - Circular eyes are the most common; over a third of faces had black backgrounds; eyes-only faces form a common cluster.
  - Kalegina: "blinking helps soften that impact." She also notes that results from motion (video) may differ from stills.

---

## 6. Recommendation for your robot (phone, portrait, dark background)

### Face type: eyes only, no mouth
- Every successful low-DoF character above has no mouth: Cozmo, Vector, EVE, WALL-E, BDX, Loona, Eilik.
- Kalegina's penalty for having no mouth was measured on **static stills**. The fixes are what Anki did:
  - constant life (blinks every 3–10 s, darts every 1–2 s, shimmer);
  - soft, rounded shapes;
  - a glow;
  - letting the body move.
- A mouth adds a second feature that reads as human and pushes the face toward the uncanny.
- Follow Cozmo and use **no pupils**. Use a gaze **hotspot** (a brighter patch of LEDs that shifts about 1.5× the eye offset) so gaze direction still reads.

### Pixel and LED look
- Render a Vector-style vector rig: per eye, centre, scale X/Y, angle, 4 corner radii, upper and lower lid Y/angle/bend.
- Rasterise it into a **coarse virtual LED grid**. About **32 × 56 cells** fits portrait well (cells about 34 px on a 1080-wide screen); each eye spans roughly 10–12 × 14–18 cells.
  - Much coarser than that and lid angles and bends stop reading.
  - Cozmo's effective 128×32 still worked because its eyes are large.
- Draw each cell as a rounded square or circle with a 15–25% gap. Add 1–2 rows of dim "off" LEDs at 3–5% brightness so the matrix stays visible, add a Gaussian bloom, and optionally dim alternate rows (Cozmo-style scanlines).
- Snap shape transitions to the grid with nearest-neighbour sampling, as Anki did. Smooth interpolation of the parameters plus quantised display reads clearly mechanical.

### Colour
- **One fixed hue** (cyan/teal, as Cozmo/EVE did). Animate only brightness and saturation, as Vector does.
- Reserve a **second hue (amber → red) only for pain**, so colour carries a single meaning.

### Mapping your affect signals
Smooth all inputs with an EMA. Vector used N = 20, roughly 0.3–0.7 s; use a faster time constant for pain. Blend face changes in about 0.1 s and body motion in about 0.35 s (Disney's values).

| Signal | Face degrees of freedom | Grounding |
|---|---|---|
| **Valence +** | Lower lids rise with positive bend (crescent, "smiling eyes"); eyes slightly up; rounder corners; small "bounce" on positive TD spikes | Kismet: valence spreads across all features; Cozmo's happy squint |
| **Valence −** | Upper lids droop with the outer ends lowered (sad slope); eyes sit lower on screen and slightly closer; saturation drops slightly | Binoculars' angle reads as sad (Stanton) |
| **Arousal** | Eye size (scale 0.85–1.15); dart rate (0.5–3 s) and amplitude; blink spacing; glow strength; saturation (0.25 → 1, Vector's mapping); shimmer amplitude. Low arousal → upper lids half-lowered, slow drifting gaze, dim | Vector StimFace and LowStim; Kismet's weary ↔ excited axis |
| **Dominance** (empowerment; Kismet's "stance") | High: steady gaze at the person (few darts, the "focused" 1 px mode); eyes level and higher on screen ("chin up"); flat upper lids angled slightly down at the inner corners (determined); wider spacing. Low: gaze averted down or sideways, more frequent darts, smaller eyes set lower and closer, inner corners of the upper lids raised (worried) | Vector's Frustrated bucket (Confident < −0.29); Reachy's shy/uncertain moves |
| **Pain** (closeness to falling) | Overrides the other channels the way Vector's LowStim takes precedence: vertical squash (scaleY ↓, scaleX ↑, like the blink's squash), lids tighten toward ">  <" squeezed eyes, hue shifts toward amber/red, scanline jitter and dropped-out LEDs (Vector's ScanlineDistorter), fast micro-shake. At impact: a hard "flinch blink". Afterwards: a recovery look (blink plus a dart back to centre) | Disney's exertion cue (eye radii narrow); Cozmo's squint when pushed |

### Event-driven reactions on top of the continuous layer
Copy Vector's architecture: continuous base mapping + keep-alive layer + short **episodic** clips.
- Large |TD-error| → surprise widen (scale ×1.2 for about 150 ms) or a disappointed droop.
- Regaining balance after high pain → a relief blink and brightening.
- Give each reaction a **repetition penalty** so it doesn't fire constantly (Vector config).
- **Gaze anticipates motion** (Baxter): look in the direction the robot is about to drive or fall.

### Using the body tilt
- The inverted pendulum already *is* the posture channel: lean, wobble, and stability vs. jitter play the role of WALL-E's "everything was part of his expression" and BDX's gait style.
- Don't duplicate it on screen. Keep the face's global angle about level relative to gravity (counter-rotate any roll), so the eyes read as a head that stabilises itself.
- Use eye position on screen as a small *head offset* added to the body's motion, like Disney's offsets on top of a nominal head motion: forward lean → eyes look down and ahead; backward catch → eyes widen and lift.
- Optionally, bias the balance setpoint or wobble by affect: confident → upright and still; low dominance or low valence → slight hunch and more sway.
- Short procedural chirps whose pitch follows valence and whose rate follows arousal would add the Burtt-style channel. Vector likewise fed mood into its sound parameters.

### Key sources
Code and data:
- Vector source: https://github.com/kercre123/victor
- pycozmo face renderer: https://raw.githubusercontent.com/zayfod/pycozmo/master/pycozmo/procedural_face.py
- Reachy Mini emotion library: https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library

Papers:
- Disney BDX, RSS 2024: arXiv:2501.05204
- Kalegina et al., HRI 2018: DOI 10.1145/3171221.3171286
- Rogel, Yadollahi & Laban, Reachy Mini legibility: arXiv:2605.12786

All other URLs are cited inline above.