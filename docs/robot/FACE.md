# The robot's face: design, references and the page

2026-11-06. The face page is `tools/robot/face/face.html` (served by the phone at
`/face`). The signals behind it are in [EMOTION.md](EMOTION.md); the earlier
literature is in [EMOTION_LITERATURE.md](EMOTION_LITERATURE.md) §7–8. Below:
what other robots do, what biology says, and what we chose.

## Decisions in one place

- **Eyes only, on a visible LED matrix** (44 columns, dim unlit LEDs, bloom).
  No mouth, no pupils, no brows: few degrees of freedom, so the face reads as a
  machine and stays out of the uncanny valley. The per-eye rig follows
  Cozmo/Vector: a rounded rectangle, an upper lid (height, slope), a lower lid
  (height, upward bend), position, size. A gaze hotspot, a brighter patch where
  the eyes look, stands in for pupils.
- **One hue, cyan.** Arousal sets saturation and brightness. Valence and
  pain are carried by shape, not by colour (pain's red was dropped 11-10).
- **The face is separate from the instruments.** The face sits on its own
  black stage with no text. The panel ("Behind the face") holds the numbers.
  `?view=face` or the *Face only* button hides the panel; double-tap the face
  to bring it back.
- **Layers with different timescales:** reflex/phasic (≤ 0.3 s), expression
  (~0.2–0.5 s on, ~1 s off), posture/dominance (~2 s), mood (minutes, already
  in `face_valence` = 0.7 mood + 0.3 TD).

## Mapping (face.html, `drawFace`)

| Signal | Face | Grounding |
|---|---|---|
| valence + | lower lid rises in an arch ("smiling eyes"), rounder corners | AU6 cheek raise marks felt smiles (Frank, Ekman & Friesen 1993); Cozmo's Happiness preset (lower lid y 0.4, bend 0.4) |
| valence − | upper lid lowers, outer ends droop (sad slope); eyes sit lower | corrugator is the bipolar, near-linear valence muscle (Larsen, Norris & Cacioppo 2003); Cozmo Sadness (upper lid 0.6, 20° outward); WALL-E's binocular angle |
| arousal | aperture and size; saturation and brightness; saccade rate; blinks rarer; lids half-closed when low; startle widen on a jump | lid aperture: Susskind et al. 2008 (widening raises sensory intake); pupil/LC: Bradley et al. 2008; speed of movement: Pollick et al. 2001; Vector maps stimulation to saturation |
| dominance + (control) | eyes higher and wider apart, gaze steady and direct, lids level and slightly low | direct gaze with approach emotions (Adams & Kleck 2005); pride is head up and expanded (Tracy & Robins 2004; Tracy & Matsumoto 2008) |
| dominance − (helpless) | eyes lower, closer, inner corners raised (worried), gaze down and away, more darts, half-blinks | averted gaze with avoidance emotions; shame is head down and contracted; half-blinks with fear in cats (Bennett et al. 2017) |
| negative valence × control × arousal | upper lids slant inward (cross) instead of outward (scared) | anger and fear differ mainly in dominance (Knutson 1996; Hess, Adams & Kleck 2005); Cozmo Anger (upper lid 0.6, 30° inward) vs Fear (outward slant + raised lower lid) |
| pain | eyes squeeze into chevrons `> <` (point toward the nose), shake, red; a flinch-blink at onset; lingers ~1 s | pain face = orbital tightening and eye closure (Prkachin 1992; Prkachin & Solomon 2008); Mouse Grimace Scale orbital tightening (Langford et al. 2010); Disney BDX narrows its eyes for effort. **Pain overrides arousal on aperture**, so pain (narrow) never looks like fear (wide). It does not slant the lids: that reads as anger (checked against Cozmo's Anger preset) |
| TD error (phasic) | positive spike: a bounce and a flash; negative spike: a flinch-blink | dopamine RPE bursts last < 200 ms (Schultz 1998) |

Idle life, which keeps an eyes-only face from looking like it is staring:
blinks every 2.5–9 s as a squash to a widening line (Vector's blink), darts
every ~0.4–3 s with a small squash (Vector's animators), a slow bob, and a
slight shimmer. Kalegina et al. (HRI 2018) found no-mouth faces rated
creepier, but on still images. Blinking softens it, and Cozmo, Vector, EVE,
WALL-E, BDX and Loona all have no mouth.

## Sound (face.html, `voice`)

Off until tapped (browsers need a gesture). Three layers like the face:

| Layer | What | Grounding |
|---|---|---|
| utterances | 1–5 babbled syllables (FM sines, a robot timbre), every ~4–16 s (more often when aroused), and a comment when the reading in words changes and holds 1.5 s (at most every 6 s); quiet while in pain | Burtt's WALL-E voice as "little phrases of music which indicated the feeling"; Vector feeds mood into its procedural audio |
| arousal → | tempo (syllables 0.24 → 0.08 s), pitch (±0.45 octave), pitch range, loudness, brightness (FM index) | Juslin & Laukka 2003: high arousal = faster, higher, more variable pitch, louder |
| valence → | scale and contour: major pentatonic rising, each syllable gliding up; minor falling, gliding down; sad and calm ends on a long fall | valence is carried by mode and contour, not pitch height (same review); sadness is best conveyed by sound (Löffler et al. 2018) |
| dominance → | register: −0.4 octave in control, +0.4 helpless | Ohala's frequency code (1984, *Phonetica* 41:1–16, doi:10.1159/000261706): low = large/dominant, high = small/submissive; disputed for human speech (Grawunder & Winter), used here as an animal-call convention |
| events | good surprise (TD+): a rising third-to-fifth "ding"; bad (TD−): a falling "wah"; startle (arousal climbing fast into its upper half): a high blip; pain onset (pain > 0.35): a falling, rough "ouch"; a whimper every ~1 s while pain > 0.5; relief when pain is gone: a soft falling sigh | fired from the same detections as the face's bounce, flinch and widen, since a matching action amplifies an utterance and a clashing one overrides it (Read & Belpaeme 2014, doi:10.1145/2559636.2559680; 2016, *Int. J. Soc. Robot.*); pain is amplitude-modulated at 70 Hz, the roughness band (30–150 Hz) of screams and alarms (Arnal et al. 2015, *Curr. Biol.*, doi:10.1016/j.cub.2015.06.043) |
| repetition | each event has a refractory (0.5–2 s) and repeats are 0.65× quieter, recovering over ~8 s | Vector's emotion-event repetition penalty |
| motors | band-passed noise, centre 250–1650 Hz and level with the mean \|wheel command\| | Burtt: arm, head and tread motors were part of WALL-E's expression |

In 60 s of `demo_trace.json`: ~14 utterances, 5 pain sounds, 6 reliefs, 6
startles (`window.__voice` counts them).

## What other robots do (detail in the research notes below)

- **Cozmo / Vector (Anki).** A procedural face rig: 5 face parameters + 19
  (Cozmo) / 25 (Vector) per eye: centre, scale, angle, 8 corner radii, and
  upper and lower lids each with Y, angle and bend; on Vector also
  saturation, lightness, glow and a hotspot. Cozmo is 128×64 mono with
  every other row dropped (the scanline look). Vector is 184×96 colour with a
  fixed hue and animated saturation and lightness. The mood is 5 dims
  (Vector: Happy, Confident, Social, Stimulated, Trust), each moved by events
  with fixed deltas and decaying over 70–150 s. It is bucketed into
  LowStim / Frustrated / Med / HighStim to pick canned animations, on top of
  a procedural layer of blinks, darts and shimmer. Pixar-trained animators
  (Carlos Baena) removed the pupils: "not robotic enough". Sources: the
  Vector source (github.com/kercre123/victor), pycozmo
  (`procedural_face.py`, `expressions.py`).
- **WALL-E.** Binocular eyes whose angle reads sad or happy, no mouth. The
  whole body expresses (arms, treads, head, boxing up to hide). Ben Burtt's
  voice is processed, phrase-like and musical.
- **Reachy Mini (Pollen / Hugging Face).** No screen. A 6-DoF head, body yaw,
  2 antennas and a speaker. 81 recorded emotion moves (head, antennas, sound)
  played by an LLM tool call. Antennas up = happy, down = sad, frozen while
  listening. A 2026 study (Rogel et al., arXiv:2605.12786) found valence and
  arousal were read better than emotion labels.
- **Open Duck Mini ("micro duck") / Disney BDX.** Open Duck Mini is
  puppeteered (antennas, head lamp, sounds) with no emotion model. BDX (RSS
  2024) has LED-array eyes, antennas, a head lamp and speakers. Its eyes
  narrow and antennas drop at high speed to show effort. Face blends in
  0.1 s, body in 0.35 s.
- **Kismet** blends 9 postures over arousal × valence × stance (stance ≈ our
  dominance). **Baxter**'s eyes look where the arm will move. **Loona**'s
  ears rise when happy and droop when disappointed.

Our states were checked side by side against Cozmo's presets rendered by
pycozmo (`expressions.py`): neutral, glad, excited, sad, scared, cross and
sleepy match Neutral, Happiness, Excitement, Sadness, Fear, Anger and
Tiredness. The first pain face (slanted, heavy lids) matched Anger, hence
the chevrons. Try a state from devtools with `window.__faceSet({...})`.

## Timescales (biology → our constants)

| Process | Biology | Ours |
|---|---|---|
| startle blink, RPE burst | 30–50 ms; < 200 ms | flinch-blink and TD bounce within one frame, decay ~0.25 s |
| facial expression | EMG response 300–400 ms (Dimberg & Thunberg 1998); felt expressions 0.5–4 s | springs ω = 10 rad/s (~0.3 s); input τ 0.5 s, 0.05 s for big jumps |
| pupil / arousal | dilation peaks ~0.9–1 s (Hoeks & Levelt 1993) | arousal through the same 0.5 s filter |
| posture, control | seconds (pride and shame displays) | dominance τ 2 s, position spring ω = 4 |
| pain | grimace persists with the stimulus | attack 0.06 s, release 0.8 s |
| mood | minutes to hours (Beedie et al. 2005; mood as integrated RPEs, Rutledge et al. 2014) | `mood` (~1 min) inside `face_valence` |

## The page

- **Face stage** (top in portrait, left in landscape), no text.
- **Behind the face** panel:
  - *Feeling*: a valence × arousal map (Russell's circumplex, last 10 s as a
    trail), the reading in words ("glad · in control"), and gauges for
    control (dominance, bipolar) and pain (0–1). The 3D box was dropped: on a
    phone it had to be rotated to be read, and two flat views say the same.
  - *Body*: the pendulum from the side (tilt × 4), the bumpers and the pain
    zone before them, and the L/R wheel commands (bar = what the wheels got,
    amber tick = what the policy chose; a gap between them is a reflex
    overriding the policy, which is also pain).
  - *Last 20 s*: lanes for feeling, arousal, control, pain and reward
    (falling back to the TD error, then empty), one scale per lane, shared
    time.

Data: `emotion` in the phone's `/ws` state (`PolicyRunner.emotion_state`).
Since 2026-11-06 it also carries `reward` (mean over the last feel window),
`tilt` (rad from upright), `bumper_back` / `bumper_front` / `pain_zone`,
`act_l` / `act_r` (applied) and `chosen_l` / `chosen_r` (the policy's).
These need the new `policy_runner.py` on the phone
(`tools/robot/dev/push_python.sh dreamerv3/deploy/policy_runner.py`) and the
new page (`adb push tools/robot/face/face.html
/sdcard/Android/data/jp.oist.abcvlib.dreamerBridge/files/face.html`; the app
serves `files/face.html` if present). Without them the panel shows "—".

Replay a sim run: `python tools/robot/face/trace_from_csv.py
LOGDIR/emotion.csv OUT.json`, then `face.html?trace=OUT.json`;
`tools/robot/face/demo_trace.json` is 90 s of e41002.

## Open

- Test legibility with naive viewers (Kalegina; Rogel): show clips, ask for
  valence and arousal ratings, not labels.
- Body as expression: a small affect bias on the balance set-point (upright
  and still when in control, a hunch when helpless) is possible, but it
  touches the controller. Not done.
- Listen with people: whether the phrases read as glad / sad / scared
  without the face, and with it (Read & Belpaeme: context dominates).

## On the phone (2026-11-06)

Pain needs the trainer's emotion settings (`body`: theta_zero, bumpers, pain
zone) passed to `PolicyRunner.configure`. The installed app (5ab754e7) did not
do that, so pain was always 0; fixed in the app's `main.py` `apply_pause`
(smartphone-robot-android 'dreamer-bridge', local commit), currently on the
phone as an override via `tools/robot/dev/push_python.sh` together with
`policy_runner.py`, `numpy_policy.py` and `emotion.py` from this repo (the
app's built-in `emotion.py` lacks the new `step()` arguments). `--clear` the
overrides after the next app build. Emotion on the face during training needs
`--env.robot.weights_emotion True`. The face WebView keeps the page it loaded:
swipe back to the gauges and to the face again after pushing `face.html`.

## Literature

Two reviews behind the design, written 2026-11-06 by the IAS session's
research agents (DOIs checked against Crossref there; each claim marked
[abs]/[sec]/[mem]/[unverified]):

- [FACE_LIT_ROBOTS.md](FACE_LIT_ROBOTS.md): Cozmo/Vector (face rig, blinks,
  darts, mood engine, from the Vector source), WALL-E/EVE, Reachy Mini,
  Open Duck Mini and Disney BDX, Kismet, Baxter, Jibo, Loona, Kalegina et al.
  2018. Recommends eyes only, no pupils, one hue, pain as the only second hue.
- [FACE_LIT_AFFECT.md](FACE_LIT_AFFECT.md): which facial channel carries
  valence, arousal, dominance and pain (facial EMG, pupil, gaze, posture,
  grimace scales), and the timescales of reflexes, expressions, posture and
  mood.

## Changes 2026-11-06 evening (from live use)

- Faster: small affect changes now follow with tau 0.15 s (was 0.5 s), big
  ones at once; springs omega 16 (was 10); the body view draws every frame
  from the raw samples. (Expressions blend in ~0.1 s on Disney's BDX and are
  visible by ~0.3 s in facial EMG.)
- Pain is a squint of the same eyes, not a separate `> <` shape cross-faded
  in: upper lid down, lower lid up (orbit tightening, AU6/7), a slight
  brow-lowering slant (AU4), flatter and wider (Vector's squash). Red replaces
  cyan LED by LED (an ordered dither) from pain 0.15 to 0.5, so the eye never
  passes through a mixed grey or blue. It rises at once and eases over ~0.4 s (was 0.8 s). Pain
  narrows the eyes, fear widens them, which keeps the two apart.
- The phone's empowerment rollout (~100 ms of Python per call) held the
  interpreter and dropped the 65 Hz control loop to ~47-55 Hz. It now runs
  every 3 s (posture-like signal) one rollout step at a time with a control
  period's sleep between steps (`emotion.empowerment_slices`). The app
  publishes the web state at 30 Hz instead of every step (app main.py,
  smartphone-robot-android ea7e25fe, local).

## Changes 2026-11-10 (Alex's notes: snappier, pain without colour, effort)

Maths of the whole chain, with the before/after constants:
[FACE_MATHS.md](FACE_MATHS.md). In short: the phone refreshes pain (and its
share of arousal and valence) every control step instead of every 4th; the
page runs at 60 fps, pain rises unfiltered and squints with a faster spring
(a pain step reaches 90 % in ~95 ms on the page, was ~200 ms); blinks and
darts are faster and more frequent. Pain keeps the squint, stays cyan and
gets a rounded top (the upper lid bends down toward both ends). Effort, the
mean |applied wheel command|, lowers a flat upper lid and narrows and
brightens the eyes, with a fine tremor near full drive. Before/after sheet:
`figs/face/face_1110.png`.

To the phone: `tools/robot/dev/push_python.sh dreamerv3/deploy/policy_runner.py
dreamerv3/deploy/emotion.py` and `adb push tools/robot/face/face.html
/sdcard/Android/data/jp.oist.abcvlib.dreamerBridge/files/face.html`.
