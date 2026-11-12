# Emotion and the face: the maths

How the face's inputs move the eyes. The emotion model itself (pleasure,
arousal, dominance, pain, threat) is in [EMOTION.md](EMOTION.md) since
2026-11-12; the earlier signal definitions are in
[history/FACE_MATHS_2026-11-10.md](history/FACE_MATHS_2026-11-10.md).
Code: `tools/robot/face/face.html` (the face), `dreamerv3/deploy/policy_runner.py`
(cadence and delivery).

## 1-3. Inputs

The face takes three axes: valence (= pleasure), arousal (0-1) and
dominance (-1..1). Since 2026-11-12 pain has no channel of its own: its shape
(squint, then `> <`) is the negative end of pleasure, with
p = max(pain input, clip((−v − 0.6) / 0.4, 0, 1)). A `pain` input, while the
phone still sends one, shows the same way. Effort is no longer shown.

## 4. Delivery and smoothing (before this change)

  phone: P every step → max over K = 4 steps → feel → web state at 30 Hz → page

  page, per frame at 30 fps (dt ≈ 33 ms), for each affect x:

    x̃ ← x̃ + (1 − e^{−dt/0.03}) (x − x̃)                         input average

    x̂ ← x̂ + (1 − e^{−dt/τ_x}) (x̃ − x̂),  τ_P = 0.03 s rising / 0.4 s falling,
         τ_v = 0.05 s, τ_G = 1.5 s, others 0.15 s (0.03 s for jumps > 0.2)

  then each eye feature f follows its goal g(x̂) through a critically damped
  spring: f̈ = ω² (g − f) − 2ω ḟ, ω = 16 rad/s (pain's squint ω = 26).

  A step of pain 0 → 1 reached 90 % of the squint after ~0.3 s:
  ≤ 62 ms (feel) + ≤ 33 ms (web) + ≤ 33 ms (frame) + 2 x 30 ms filters
  + ~150 ms spring (90 % of a critically damped spring at t ≈ 3.9/ω).

## 5. From affect to eye features (face.html, `drawFace`)

With v = V^face, a = A, d = G, p = pain squint (spring on P), v⁻ = max(0, −v),
d⁺ = max(0, d), each eye is a rounded rectangle (half-width w, half-height h,
corner radius r) cut by an upper lid at height `top` (fraction of the eye)
tilted by `tilt` toward the nose, and a lower lid arching up by `bot`:

| feature | goal |
|---|---|
| aperture (open) | (0.62 + 0.5 a)(1 − 0.6 p) x (1 + 0.25 startle) |
| upper lid (top) | 0.3 clip((0.45 − a)/0.45) + 0.2 d⁺ + 0.15 v⁻ + 0.25 v⁻(1−d⁺)(1−a) + 0.3 v⁻ d⁺ a + 0.22 p + half-blink |
| lid slope (tilt) | v⁻(−0.5 (1−d⁺)(1−0.4a) + d⁺ a) − 0.45 max(0, −d) + 0.3 p  (+ inward = cross / hurt, − = sad) |
| lower lid (bot) | 0.5 max(0, v)(1 − p) + 0.25 v⁻ a (1 − d⁺) + 0.3 p |
| roundness | 0.5 + 0.45 max(0, v) − 0.2 p |
| width | 4.4 (1 + 0.12 p) LEDs; size x (1 + 0.08 d + 0.1 (a − 0.5)) |
| height on face | −2.4 d + 0.8 v⁻ (+ a slow bob) |
| gap between eyes | 6.6 + 0.5 d |
| shake | (0.6 p + 2 max(0, a − 0.85)) sin(60 t) |
| colour | cyan, saturation 30 + 60 a, brightness 0.5 + 0.5 (0.3 + 0.7 a); red replaced cyan LED by LED for p from 0.15 to 0.5 |

Phasic events: a TD jump > 0.35 gives a bounce (good) or a blink (bad); pain
crossing 0.35 a flinch-blink; arousal climbing fast into its upper half a
startle (eyes widen). Idle: blinks every 2.5-9 s (0.2 s each), darts every
0.2-3.4 s (rarer when calm or in control).

## 6. Changes (2026-11-10, Alex's face notes)

Snappier, pain in the same colour with a rounder top, effort from the
wheels. The new constants are in the table; section 4's path becomes:

  phone: pain, hurt, arousal's pain term and the body refreshed in the web
  state every control step (the world-model signals stay at 16 Hz)

  page: 60 fps; pain and TD valence skip the input average and pain rises
  without a filter (τ = 5 ms; it still eases out over 0.4 s); small changes
  τ = 0.08 s; springs ω = 26 (expression), pain's squint and shake ω = 45

| | before | after |
|---|---|---|
| pain reaches the web state | ≤ 62 ms (every 4 steps) | ≤ 15 ms (every step) |
| frame | 33 ms | 17 ms |
| pain step to 50 % / 90 % squint (page, measured in Chromium) | 68 / 201 ms | 30 / 95 ms |
| expression spring 90 % | 0.24 s | 0.15 s |
| blink | 0.2 s, every 2.5-9 s | 0.14 s, every 1.8-6 s |
| darts | spring ω 28, every 0.2-3.4 s | ω 45, every 0.15-2 s |
| pain colour | cyan → red | cyan only |
| pain shape | lids squeeze, straight upper lid | squint (to 0.35), then a smooth split into chevrons `> <` (full at 0.6) |

**Pain's chevrons (> <).** (Alex, 11-10: a rounded squint read as happy.)
One continuous change of the same eye, so rising and easing are smooth.
First the eye squints, aperture x (1 − 0.8 s_q) with s_q = clip(p/0.35, 0, 1);
then the slit splits into two strokes pointing at the nose. The split is a
warp of the eye: with u the distance toward the nose, each half is pushed
away from the middle line by

  cl(u) = H clip((w − u) / 2w, 0, 1),   H = 3.4 S clip((p − 0.15)/0.45, 0, 1),

i.e. the LED at (u, y) is lit if the eye contains (u, y − sign(y) cl(u)).
At H = 0 that is the eye itself; as H grows the outer ends open into `> <`
while the strokes keep the slit's thickness. Shake and the flinch-blink
stay; the colour stays cyan. Sequence (steady levels, then the real-time
release from pain 1 to 0): `figs/face/pain_sequence_1110.png`; the face is
back to neutral ~0.6 s after pain ends.

**Effort.** From the wheel commands the robot actually applied,

  e_t = mean_i \|u^applied_{i,t}\| ∈ [0, 1],   ê = low-pass(e, τ = 0.12 s),

  E = clip((ê − 0.2) / 0.5, 0, 1)     (none below a fifth of full drive, full at 0.7)

shown as concentration, not pain: the upper lid lowers flat (top += 0.26 E (1 − p)),
the lower lid rises a little (bot += 0.15 E (1 − p); in pain the squeeze
wins, so the two never close the eye together), the eyes narrow slightly
(width x (1 − 0.06 E)) and brighten (x (1 + 0.15 E)), with a fine strain
tremor at high effort (0.15 max(0, E − 0.6) sin(90 t)). Pain keeps what
effort lacks: the chevrons and the shake, so the two
stay apart (as on Disney's BDX, whose eyes narrow with effort).

Check (2026-11-10): `face.html` rendered in headless Chromium (Playwright)
before and after, states set with `window.__faceSet`, no page errors; sheet
`figs/face/face_1110.png`. Phone side: `test_pain_reaches_the_face_every_step`
(embodied/tests/test_emotion_affect.py). Not yet seen on the phone.

## 7. Changes (2026-11-12): pain on pleasure, stronger dominance

- Pain's shape is driven by p = max(pain, clip((−v − 0.6)/0.4, 0, 1)): mild
  displeasure droops the eyes, below −0.6 they squint, at −1 they are `> <`.
  In the new model a hit drives pleasure down through the outcome term
  (EMOTION.md), so no separate pain channel is needed.
- Dominance follows the face-perception findings: in control the lid sits low
  and flat and the eyes high (Keating et al. 1981; Mignault & Chaudhuri 2003);
  without control the inner corners rise, the eyes sit low and the gaze
  slides down and away (Adams & Kleck 2003). Constants: top +0.2 d⁺ (was 0.1),
  tilt −0.45 max(0, −d) (was 0.15), height −2.4 d (was 1.6), saccade target
  +0.6 max(0, −d) sideways and +1.0 max(0, −d) down (was 0.6 down only).
- Effort is no longer driven by the plan (the face still reads the wheel
  commands if sent).

Renders of the eight PAD corners (±0.8 and ±0.5) and of negative pleasure 0 to
−1: plan page "Robot Emotion Model". The renders step the face 360 frames
before the capture so springs have settled; earlier headless captures had not,
which made dominance look weaker than it is.
