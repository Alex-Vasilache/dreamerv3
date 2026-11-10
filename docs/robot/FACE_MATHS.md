# Emotion, pain and the face: the maths

How the face's inputs are computed now (2026-11-10) and how they move the
eyes. Code: `dreamerv3/deploy/emotion.py` (signals), `policy_runner.py`
(pain, the phone's cadence), `tools/robot/face/face.html` (the face).
Reasons and literature: `EMOTION.md`, `FACE.md`.

## 1. Signals from the world model (every K = 4 control steps, 16 Hz)

At a feel step t, with deterministic state h_t, posterior q_t = q(z_t | h_t, o_t)
and prior p_t = p(z_t | h_t) (32 categoricals x 4 classes, 1 % unimix):

| symbol | definition |
|---|---|
| surprise S_t | KL(q_t ‖ p_t) |
| novelty N_t | H(p_t) |
| unexpected U_t | H(q_t) + KL(q_t ‖ p_t) − H(p_t) |
| value V_t | critic v(h_t, z_t) |
| reward error E_t | \|R_t − r̂(h_t, z_t)\|, R_t = Σ_{k<K} γ^k r_{t−K+k} |
| TD error δ_t | R_t + γ^K V_t − V_{t−K} |
| empowerment C_t | (1/9) Σ_a KL(p^a_{t+8} ‖ p̄_{t+8}), the 9 constant wheel commands a ∈ {−1,0,1}², each rolled 8 prior steps; p̄ their mean. Recomputed every 3 s on its own thread |
| fear F_t | −c(h_t, z_t), the fear critic's predicted pain (if trained) |

Each is z-scored online: z(x_t) = (x_t − m_t)/σ_t with exponential mean and
variance over τ = 650/K ≈ 162 feel steps (10 s); mood uses τ_m = 6τ (~1 min).
The TD error is scaled by its running RMS ρ_t instead of its SD (a steady
disappointment stays negative).

## 2. Pain (every control step, 65 Hz)

With tilt θ_t relative to the balance point, bumpers at θ_lo < 0 < θ_hi and a
zone w = 0.035 rad before each:

  front_t = clip((θ_t − (θ_hi − w)) / w, 0, 1),   back_t = clip(((θ_lo + w) − θ_t) / w, 0, 1)

  P_t = max(front_t, back_t) + min(1, mean_i \|u^applied_i − u^chosen_i\|)

The second term is a reflex (brake, cap, travel limit) overriding the policy.
P is a sensation of the body, not a prediction. Until now the face saw
max P over the last K steps, once per feel step.

## 3. The affect axes

  control c_t = σ(z(C_t))

  helplessness H_t = z(\|δ_t\|) (1 − c_t)

  drive D_t = z(N_t) + z(U_t) + z(E_t) − z(C_t) + H_t

  **arousal**   A_t = σ((D_t + 3 P_t) / 2)                    ∈ (0, 1)

  TD valence    v_t = tanh(δ_t / ρ_t)                        brief joy / distress

  mood          m_t = tanh(z_m(V_t) / 2)                     slow hope / fear

  hurt          h_t = P_t (1 − 0.5 c_t)                      controllable pain hurts less

  **valence**   V^face_t = clip(0.7 m_t + 0.3 v_t − h_t, −1, 1)

  **dominance** G_t = tanh(z(C_t))                             ∈ (−1, 1)

  **pain**      P_t (shown as min(P_t, 1))

The face receives (V^face, A, G, P, v) and the body (tilt, applied and chosen
wheel commands).

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
| upper lid (top) | 0.3 clip((0.45 − a)/0.45) + 0.1 d⁺ + 0.15 v⁻ + 0.25 v⁻(1−d⁺)(1−a) + 0.3 v⁻ d⁺ a + 0.22 p + half-blink |
| lid slope (tilt) | v⁻(−0.5 (1−d⁺)(1−0.4a) + d⁺ a) − 0.15 max(0, −d) + 0.3 p  (+ inward = cross / hurt, − = sad) |
| lower lid (bot) | 0.5 max(0, v)(1 − p) + 0.25 v⁻ a (1 − d⁺) + 0.3 p |
| roundness | 0.5 + 0.45 max(0, v) − 0.2 p |
| width | 4.4 (1 + 0.12 p) LEDs; size x (1 + 0.08 d + 0.1 (a − 0.5)) |
| height on face | −1.6 d + 0.8 v⁻ (+ a slow bob) |
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
