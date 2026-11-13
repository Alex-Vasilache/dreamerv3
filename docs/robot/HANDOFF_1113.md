# Handoff, Fri 11-13 noon CET (overnight 11-12 to 11-13)

All sim; no robot or phone actions. Plan and figures: claude.ai artifact
"Robot Emotion Model"; model text: `EMOTION.md`; study log:
`SAMPLE_EFF_STUDY.md` (sections PF, RF, PF2).

## Built (on the robot branch, all switches; robot defaults unchanged)

- Impact pain (`env.robot.pain_mode impact`, ι0 1.0, ι1 3.0 rad/s per step,
  from the real sessions; per-hit impact tracks MuJoCo contact force,
  Spearman 0.73 table / 0.87 carpet).
- Emotion model v2 (`env.robot.emotion_model v2`): pleasure = outcome + change
  in outlook, faded 0.5 s; arousal switch (total / unexpected / kl, each with
  or without the efference copy); dominance = 32 random commands, 16 steps.
  Every variant is logged per step in sim (`sim_phone_feel_every 1`).
- Catch reflex (`env.robot.catch_reflex`, off), with the RF options (search,
  capped slope, relative limit).
- Face: pain is the negative end of pleasure; dominance as lid, eye height and
  averted gaze.

## Results

**PF** (60 min, n = 8, pushes and walls; mean ± SD, minutes 11-60):

| | a: bumper pain + brake | b: impact pain | c: b + reflex |
|---|---|---|---|
| crossings/min | 4.8 ± 2.0 | 12.0 ± 2.6 | 11.5 ± 4.6 |
| peak contact force (N) | 1.31 ± 0.29 | 3.25 ± 0.17 | 3.13 ± 0.23 |
| falls per push | 0.14 ± 0.05 | 0.47 ± 0.09 | 0.48 ± 0.15 |
| t90 (min) | 3.8 ± 1.4 | 4.6 ± 0.7 | 5.0 ± 1.7 |

a beats b and c (p ≤ 0.004); the reflex fired on 0.07 % of steps and changed
nothing. a differs from b in two things (pain signal and brake): PF2 splits them.

Emotion variants over PF's 24 runs (pooled AUC for bumper hits / 0.1-0.3 s
before): pleasure as designed 0.88 / 0.83 (best; with the outlook level 0.72);
arousal 0.53-0.63 in every version (kl best before a hit, total at it; the
efference copy did nothing because the reflex rarely fired); dominance near
chance because it updates only every ~3.7 s (latent-only 0.57-0.63 on the
update rows; value and threat did not help in sim).

**PF2** splits PF's arm a (Saion V100, n = 8, e44700-e44715; figure
`figs/study/pf2_arms.png`):

| | a: bumper + brake | d: bumper, no brake | e: impact + brake | b: impact, no brake |
|---|---|---|---|---|
| crossings/min | 4.8 ± 2.0 | 4.2 ± 1.2 | 11.1 ± 2.9 | 12.0 ± 2.6 |
| peak contact force (N) | 1.31 ± 0.29 | 1.38 ± 0.31 | 3.18 ± 0.26 | 3.25 ± 0.17 |
| falls per push | 0.14 ± 0.05 | 0.13 ± 0.03 | 0.44 ± 0.12 | 0.47 ± 0.09 |
| t90 (min) | 3.8 ± 1.4 | 3.6 ± 0.7 | 4.9 ± 1.0 | 4.6 ± 0.7 |

- The runaway brake does nothing in sim, with either pain (a vs d, b vs e:
  all p ≥ 0.27). It can go, as planned.
- The pain signal decides it: bumper-closeness pain gives ~2.5x fewer
  crossings, ~2.4x softer hits and ~3x fewer falls per push than impact pain
  (p = 0.001 both ways). Likely reason: closeness pain starts 2° before
  contact and is graded, so the threat critic learns from every near miss;
  impact pain arrives only at contact and, in sim, only on 40 % of hits.
- So "pain = impact only" costs a lot in sim. To keep the no-hand-coding rule
  and still learn early, options: a lower impact threshold (sim hits are
  softer than real ones: 40 % vs 85 % pass ι0 = 1.0), or a learned
  near-miss signal. This is the main open decision.

Emotion signals over all 40 runs (pooled AUC, hit / 0.1-0.3 s before):
pleasure 0.88 / 0.83; arousal total 0.61 / 0.55, kl 0.58 / 0.60; dominance
latent at 16 steps 0.56 / 0.63 on its update rows (value lowers it).

**PF3** (impact pain with the threshold lowered to the sim: ι0 0.5, ι1 1.5;
86 % of sim hits pass; no brake; e44760-e44767, n = 8), against PF2 d
(bumper pain) and PF b (impact pain at 1.0/3.0):

| | f: impact, low threshold | d: bumper pain | b: impact, 1.0/3.0 |
|---|---|---|---|
| crossings/min | 6.1 ± 1.4 | 4.2 ± 1.2 | 12.0 ± 2.6 |
| peak contact force (N) | 1.70 ± 0.37 | 1.38 ± 0.31 | 3.25 ± 0.17 |
| falls per push | 0.20 ± 0.05 | 0.13 ± 0.03 | 0.47 ± 0.09 |
| t90 (min) | 5.6 ± 2.3 | 3.6 ± 0.7 | 4.6 ± 0.7 |

Lowering the threshold closes most of the gap (f vs b: all p ≤ 0.001; force
f vs d n.s., p = 0.11), but bumper pain stays better on crossings (p = 0.015),
falls (p = 0.007) and learning speed (p = 0.02). The threshold is a sim
calibration: on the real robot normal motion reaches ~1.0, so it should be
per platform or relative to the run's own background.

**Catch reflex (RF, RF2):** see SAMPLE_EFF_STUDY.md "RF and RF2 results".
A 4-step search over the drive with a run-relative limit (s4z2) fires on 4 %
of steps and cuts impact pain 42 % and contact force 21 % with no speed cost;
always-on filtering is stronger but rewrites half the commands and slows
learning. Slope-based and 16-step variants do not help.

## Needs you

- The remote is now SSH (`git@github.com:...`); pushes no longer need the editor.
- PF3 runs e44760-e44767 are still in /work (not archived) with snapshot
  /work/DoyaU/vasilache/work/code_pf3; RF IAS runs may still be in ~/robot_runs.
- Decide arousal: none of the versions marks hits well in sim; options are on
  the plan page.
- Nothing changed on the robot; the live face needs `adb push face.html`.
