# Weekend 2026-11-07/09: fastest reliable live learning, balance and commands

Ask (Alex, Fri 11-06 evening): keep the GPUs busy over the weekend with
structured sweeps that find how to make live learning as fast and as reliable
as possible, with little babysitting and little movement, for balancing and
for following commands. Results by Monday morning, German time.

Everything runs in the fitted sim on the Germany floor (sim table schedule +
drive-train randomisation), through the phone's own loop (`sim_phone`), with
the live demo's settings (e40006: publish every sync, drive home capped at
10 s, phone policy age ~1.3 s). Params come from `sbatch/wk/gen.py`.
Experiments e42001 onward.

## Scores

Balance, per run (n seeds per arm, mean ± SD):
- *fast*: first minute with upright (|tilt| < 3°) ≥ 0.9;
- *steady*: bumper crossings a minute, minutes 15-29;
- *reliable*: fraction of seeds up by minute 6 that stay below 10
  crossings a minute after minute 10;
- *movement*: |wheel speed| and roaming after minute 10;
- *compute-invariant*: the same at the slow (P100, ratio ~280) and fast
  (IAS A5000/3090, ~570) learner.

Commands: minutes to forward and turn tracking correlation 0.7, final gain,
falls a minute, and |speed| at a zero stick.

## Stages

| stage | when (German time) | question | runs |
|---|---|---|---|
| IA1/IA2 | Fri → Fri ~22:00 | which drift control keeps the fast start at every learner speed | 108 |
| K1 | Fri 18:00 → Sat ~00:00 | command learning on the Germany floor: base and seven one-factor changes | 32 × 60 min |
| W1 | Fri ~20:00 → Sat ~06:00 | balance speed-up screen on the IA winner, one factor at a time, slow and fast learner | 112 × 30 min |
| W2/K2 | Sat | best factors combined (factorial), n = 6 per speed | ~100 |
| W3/K3 | Sun → Mon 06:00 | confirmation: best two against the live recipe, n = 12, 60 min, three learner speeds, carpet and pushes as holdouts | ~150 |

Each stage is analysed before the next is designed; the stage log below
records what was found and why the next one looks as it does.

## Log

### Fri 11-06 ~19:30: IA1 in, W1 and K1 launched

IA1 (e41001-e41048, n = 4, no ratio schedule): crossings a minute after
minute 15, by learner speed (samples/s).

| arm | P100 17.5k | V100 26k | A5000 uncapped |
|---|---|---|---|
| C control | 16.9 ± 3.1 | 22.4 ± 5.6 | 26.4 ± 9.3 |
| W world model capped at ratio 128 | 16.9 ± 4.5 | 16.5 ± 5.3 | 31.2 ± 15.5 |
| S actor-critic shrink-perturb 0.5 / 4000 updates | 9.3 ± 3.4 | 21.6 ± 5.3 | 22.9 ± 5.3 |
| L actor-critic L2-to-init wd 0.1 | 10.2 ± 0.2 | 21.4 ± 8.0 | 24.8 ± 8.2 |

First minute upright ≥ 0.9: C 4.2 / 3.2 / 3.0, so the faster learner does
learn a little faster; W slows learning 3-8x; S and L lose the fast start on
the A5000 (10.0 and 8.5, seeds stuck). The actor-critic regularisers halve
the drift only at P100 speed, so the drift is not fixed by resetting or
anchoring the actor-critic: it follows the number of updates per sample.
Figure: `figs/weekend/ia1.png` (on the A5000, S and L leave two of four seeds stuck at ~0.5 upright for 10 min).
Decision: keep the ratio schedule (uncapped for 5 min, then 128 — the late
phase is then the same on every GPU), and spend W1 on making the first five
minutes faster. IA2 (world-model regularisers, critic feature norm) still runs.

W1 (e42200-e42327): live recipe e40006 + ratio schedule, one factor at a time,
4 seeds, on Saion P100s (e42200-e42263) and IAS A5000/3090s (e42264-e42327).
Arms: base, batch 16 / 64, batch length 32, both lrs x2, actor-critic lr x2,
small / big model, horizon 1 s / 3 s, entropy 1e-3, fear critic, stronger
smoothing, stronger stillness penalties, switch at 3 min, schedule + L2-to-init.

K1 runs are e42168-e42199 (n = 4 per arm, Saion V100, 60 min).

Lost runs: the first K1/W1 submissions (e42001-e42160) used `weights_emotion
False` like e40006; the sim phone then cannot build its emotion heads, its
actor dies and the job hangs to the wall limit (one wave of 16 runs lost, ~80
min). Fixed in gen.py (True; it only sets what is sent to the phone), smoke
e42161-e42167 for every new flag combination, then resubmitted.

### Fri 11-06 ~22:30: IA2 in

IA2 (e41049-e41108, n = 4, no ratio schedule; control = IA1 C): crossings a
minute after minute 15. Figure `figs/weekend/ia2.png`.

| arm | P100 17.5k | V100 26k | A5000 uncapped |
|---|---|---|---|
| C control | 16.9 ± 3.1 | 22.4 ± 5.6 | 26.4 ± 9.3 |
| MS world-model dynamics shrink-perturb 0.3 / 8000 | 11.2 ± 4.7 | 21.0 ± 3.8 | 25.8 ± 4.0 |
| ML world-model L2-to-init 0.05 | 13.1 ± 6.5 | 19.5 ± 5.9 | 38.0 ± 4.0 |
| BS world model + actor-critic shrink-perturb | 13.3 ± 2.6 | 20.7 ± 1.8 | 24.7 ± 7.3 |
| BL world model + actor-critic L2-to-init | 13.6 ± 3.2 | 22.2 ± 3.2 | 30.4 ± 8.6 |
| FN critic feature normalisation | 12.7 ± 2.4 | 24.9 ± 4.5 | 32.0 ± 3.0 |

No regulariser, on the world model, the actor-critic or both, removes the
drift at V100 or A5000 speed; the L2-to-init arms also leave seeds stuck
half-upright for 10-25 min. At every speed the drift tracks the learner's
updates per sample, so the ratio schedule, which fixes the late ratio at 128
on every GPU, stays the drift control (W1's base, already running).

### Fri 11-06 ~23:30: K1 in, K2 launched

K1 (e42168-e42199, command learning, V100, 60 min, n = 4). Minutes to
tracking correlation ≥ 0.7 for three minutes running; crossings a minute after
minute 10; |wheel speed| after 2 s at a zero stick (this includes the
balancing jiggle; balance runs sit at ~450). Figure `figs/weekend/k1.png`.

| arm | forward 0.7 (min) | turn 0.7 (min) | final forward corr | crossings/min | zero-stick mm/s |
|---|---|---|---|---|---|
| base (robot_command) | 16.8 ± 8.8 | 8.8 ± 1.3 | 0.80 | 15.3 ± 4.4 | 392 |
| actor-critic L2-to-init 0.1 | 11.8 ± 0.4 | 7.8 ± 1.3 | 0.80 | 15.1 ± 2.6 | 400 |
| ratio schedule (5 min -> 128) | 21.0 ± 9.6 | 11.0 ± 3.1 | 0.78 | 37.6 ± 17.9 | 457 |
| narrow kernel 0.15 | 27.5 ± 16.6 | 14.8 ± 9.3 | 0.71 | 19.7 ± 3.9 | 452 |
| p_zero 0.5 | 27.0 ± 0.0 | 18.8 ± 8.9 | 0.78 | 23.9 ± 4.8 | 358 |
| slow commands (1200/1500) | > 60 | 16.0 ± 6.0 | 0.53 | 49.4 ± 26.7 | 323 |
| **smoothing (balance recipe's)** | **11.0 ± 2.3** | 9.0 ± 1.9 | 0.81 | **5.7 ± 0.6** | 421 |
| batch 16 | 11.2 ± 0.4 | 8.5 ± 0.9 | 0.79 | 17.5 ± 12.2 | 350 |

The balance recipe's action smoothing (`action_rate_quad 0.25`,
`imag_loss.smooth 0.15`), which `robot_command` turns off, is the clear win:
as fast as the best and a third of the falls. L2-to-init and batch 16 make
forward learning reliable (SD 0.4 min against 8.8). Unlike balance, commands
need the learner's full ratio: the schedule's 128 after minute 5 slows them
and doubles the falls. Every arm falls 30-60 times a minute for the first 10-20
minutes, while it learns to balance and drive at once.

K2 (e42328-e42359, V100, n = 4): smoothing + {L2-to-init, batch 16, both},
+ command_sigma 0.25, + all three, and smoothing alone and with both at P100
speed (`online_learner_fps 17500`).

New: `env.robot.command_grow_s` (sim only; the phone app would need the same
line) scales random commands by min(1, t / grow), so the robot can learn to
balance before it is asked to drive. Smoke e42360.

### Sat 11-07 ~02:30: W1 in (base broken), W2 launched

W1 (e42200-e42327; figure `figs/weekend/w1.png`). Its base, the live e40006
settings + ratio schedule, learnt far worse than CR2's sim recipe did: on the
P100s first upright minute 16.2 ± 9.0 with two of four seeds lying on a bumper
for 10-20 min, and after minute 15 about 20 crossings a minute on both GPUs
(CR2: 3.5 min, 10). So the one-factor results are measured against a broken
base and only the large effects count:
- **batch 16** is the one change that is fast and tight on both GPUs (P100
  7.5 ± 4.5, IAS 3.5 ± 0.5 min; 17.7 ± 1.4 crossings on IAS, the lowest SD).
- **wheel 1.0 + pos 2.0** never learns: all 8 seeds lie on a bumper for 30
  min (upright 0.05). Stillness penalties that strong make the bumper the
  optimum.
- Fear critic: fast on IAS but triples the wheel speed (950-990 mm/s).
- Both lrs x2, actor-critic lr x2, small/big model, horizon 1 s, entropy 1e-3
  and an earlier switch each leave one or two seeds stuck for the whole run on
  the A5000s; horizon 3 s and batch length 32 are neutral.

Two failure modes run through all of it. (1) A seed that is not up when the
ratio drops to 128 at minute 5 then learns slowly (P100). (2) Lying on a
bumper pays about +0.08 a step, more than falling over and over, so a seed
can settle there for good.

New: `online_ratio_switch_reward` gates the switch on the actor's mean reward
over the last minute (balance: ≥ 0.4 is upright ~0.9; a robot on a bumper sits
at ~0.08), so a seed that has not learnt keeps the fast learner whatever the
GPU.

W2 (e42361-e42440 Saion P100, e42441-e42520 IAS; n = 8, seeds 811-818): CR2's
recipe ("old") against the live settings, each live setting on the old recipe
alone (home drive 1.0/1.2 s/10 s cap, publish every sync, policy age 1.3 s), live
without the schedule, and the reward gate (+ batch 16) on both.

W2 fix: `online_ratio_switch_reward` defaulted to the int -100, so 0.4 was
rejected and the gated arms of the first seed crashed at start (e42367-e42370,
IAS e42447-e42450). Default now -100.0 (snapshots patched, queued runs fine);
the eight reran as e42521-e42524 (P100) and e42525-e42528 (IAS), in arm order
old_gate, live_gate, old_gate_b16, live_gate_b16, seed 811.

Bumper trap found: a stuck seed rests on the front bumper at 8.1° tilt, just
under the penalty zone (bumper_hi 9.2° − margin 1.0° = 8.2°), so lying there
pays +0.08 a step instead of −1. With the balance point randomised by ±1° some
seeds land in this hole. Next balance round: `bumper_margin` 0.035 (2°).

### Sat 11-07 ~06:00: K2 and W2 (IAS half) in; W3, K3 launched

K2 (e42328-e42359, commands, n = 4; figure `figs/weekend/k2.png`):

| arm | forward 0.7 (min) | turn 0.7 (min) | final forward corr | crossings/min after 10 |
|---|---|---|---|---|
| smoothing (K1 best) | 11.0 ± 1.7 | 9.2 ± 3.9 | 0.83 | 5.4 ± 0.8 |
| + L2-to-init | 15.5 ± 7.2 | 10.5 ± 5.0 | 0.83 | 7.5 ± 3.7 |
| + batch 16 | 12.5 ± 4.6 | 10.0 ± 3.5 | 0.83 | 5.2 ± 0.7 |
| **+ L2-to-init + batch 16** | **10.0 ± 3.2** | 8.8 ± 2.0 | 0.81 | **4.9 ± 0.3** |
| + command_sigma 0.25 | 16.5 ± 7.2 | 12.8 ± 4.0 | 0.82 | 6.9 ± 0.7 |
| + all three | 12.0 ± 3.2 | 14.0 ± 2.5 | 0.82 | 6.8 ± 0.4 |
| L2-to-init + batch 16, P100 speed | 11.2 ± 3.3 | 8.8 ± 3.0 | 0.82 | 4.5 ± 0.6 |
| smoothing, P100 speed | 13.2 ± 2.2 | 13.2 ± 5.2 | 0.84 | 8.4 ± 4.2 |

Smoothing does most of the work; L2-to-init + batch 16 on top is the best
cell and holds at P100 speed, but its edge over smoothing alone is within the
seed spread. Command pick so far: `robot_command` + `action_rate_quad 0.25
imag_loss.smooth 0.15` + `ac_opt.wd 0.1 ac_opt.wdinit True` + `batch_size 16`.

W2, IAS half (e42441-e42520 + e42525-e42528, n = 7-8): first upright minute,
seeds stuck past minute 10, crossings after 15.

| arm | first ≥ 0.9 | stuck | crossings 15+ |
|---|---|---|---|
| old (CR2 recipe) | 4.4 ± 2.8 | 1/7 | 15.7 ± 3.1 |
| live (e40006 settings) | 7.6 ± 9.2 | 1/7 | 20.9 ± 2.3 |
| old + live home drive | 13.7 ± 11.5 | 3/7 | 14.5 ± 7.7 |
| old + publish every sync | 15.6 ± 12.6 | 3/7 | 9.1 ± 4.9 |
| old + policy age 1.3 s | 7.1 ± 9.4 | 1/7 | 14.1 ± 5.0 |
| live, no schedule | 5.0 ± 2.0 | 0/7 | 29.2 ± 4.6 |
| old + gate | 8.1 ± 5.4 | 4/8 | 15.3 ± 5.1 |
| live + gate | 10.1 ± 9.5 | 2/8 | 20.3 ± 4.5 |
| old + gate + batch 16 | 8.0 ± 5.2 | 2/8 | 16.9 ± 4.6 |
| live + gate + batch 16 | 5.8 ± 2.1 | 0/8 | 18.6 ± 4.0 |

With 8 new seeds, stuck seeds appear in almost every arm, even on the A5000:
whether a seed falls into the bumper trap dominates the noise, so these arms
cannot be ranked yet. Two things hold: the schedule still cuts late crossings
(live 20.9 against 29.2 without), and the gate alone does not unstick a seed.
The gate itself works (e42522 held the switch until its reward reached 0.40).

W3 (e42529-e42576 IAS, e42605-e42652 Saion P100; n = 8): the bumper margin
0.035 and 0.05 against 0.0175 on the old recipe, with batch 16 or the gate,
and on the live settings.
K3 (e42577-e42604, V100): the command pick, with commands grown over 5 or 10
min (also at P100 speed), relabel 0.9, holds of 2-4 s, and 4 more seeds of
the pick.

### Sat 11-07 ~10:00: W3 and K3 in; W4, K4 launched

W3 (e42529-e42576 IAS, e42605-e42652 P100; n = 8; figure `figs/weekend/w3.png`):

| arm | first ≥ 0.9, P100 | IAS | stuck > 10 min (of 16) | crossings 15+, P100 | IAS |
|---|---|---|---|---|---|
| margin 1.0° (old) | 4.0 ± 1.1 | 13.9 ± 12.7 | 3 | 13.5 ± 3.2 | 10.6 ± 6.5 |
| margin 2.0° | 3.8 ± 0.8 | 3.2 ± 1.1 | 0 | 12.7 ± 4.4 | 13.9 ± 4.4 |
| margin 2.9° | 3.6 ± 0.5 | 3.1 ± 0.9 | 0 | 11.1 ± 4.7 | 11.7 ± 3.5 |
| 2.0° + batch 16 | **2.9 ± 0.6** | **2.8 ± 0.4** | 0 | 13.2 ± 5.1 | 15.3 ± 5.5 |
| 2.0° + reward gate | 3.9 ± 0.6 | 3.4 ± 1.4 | 0 | 12.0 ± 3.4 | 15.2 ± 5.0 |
| 2.0° on the live settings | 5.5 ± 1.0 | 4.0 ± 1.7 | 0 | 22.7 ± 3.3 | 18.1 ± 3.5 |

The wider margin removes the bumper trap (0 of 48 seeds stuck against 3 of
16), and with it the GPU dependence of the start: up in 3-4 min on the P100
and the A5000 alike, 2.8-2.9 min with batch 16. With no stuck seeds the reward
gate changes nothing. The live e40006 settings are worse than CR2's in both
halves (crossings 18-23 against 11-14). Late crossings stay at 11-15 a minute,
so the drift is not solved yet.

K3 (e42577-e42604, figure `figs/weekend/k3.png`): growing the commands over
5 or 10 min does not cut the early falls (minutes 0-10: 29-41 a minute
against 28): they come from learning to balance at all in minutes 0-3.
Relabel 0.9 is the same as 0.5; holds of 2-4 s learn turning sooner (5.0 min)
but end with more falls and lower tracking (0.78). The pick over 8 seeds:
forward 0.7 at 10.5 ± 2.6 and 9.8 ± 1.8 min.

W4 (e42653-e42748 IAS, e42749-e42844 P100; n = 8): on the fixed base (CR2
recipe + schedule + margin 2.9° + batch 16): fear critic, lrs x2, entropy
1e-3, each live setting alone (home drive, publish every sync, policy age
1.3 s), no schedule, late ratio 96 / 192, L2-to-init, batch length 32.
K4 (e42845-e42864, V100, n = 4): the command pick for 90 min, on carpet
(held out), with pushes and walls, with the live phone settings, and with
margin 2.0°.

### Sat 11-07 ~15:30: W4 (IAS half) and K4 in; W5 launched

W4 IAS (e42653-e42748, n = 8, no seed stuck in any arm; figure
`figs/weekend/w4.png`). Base B = CR2 recipe + ratio schedule + bumper margin
2.9° (`bumper_margin 0.05`) + `batch_size 16`.

| arm | first ≥ 0.9 | crossings 3-8 | crossings 15+ | wheel mm/s |
|---|---|---|---|---|
| **B** | 3.2 ± 1.2 | 11.2 ± 3.4 | **9.6 ± 2.4** | 470 |
| + fear critic | 2.8 ± 0.7 | 8.6 ± 3.4 | **5.8 ± 1.7** | 588 |
| + both lrs x2 | 3.1 ± 0.9 | 15.8 ± 5.4 | 15.3 ± 4.0 | 495 |
| + entropy 1e-3 | 2.9 ± 0.9 | 11.1 ± 2.2 | 11.4 ± 3.1 | 471 |
| + live home drive (1.0, 1.2 s, 10 s cap) | 4.1 ± 0.8 | 20.9 ± 7.2 | 13.8 ± 3.0 | 471 |
| + publish every sync | 2.9 ± 0.6 | 13.7 ± 5.7 | 13.5 ± 3.8 | 447 |
| + policy age 1.3 s | 2.8 ± 0.4 | 13.6 ± 3.8 | 12.2 ± 3.2 | 464 |
| no ratio schedule | 3.6 ± 0.9 | 15.9 ± 5.2 | 19.1 ± 7.3 | 473 |
| late ratio 96 | 3.2 ± 0.7 | 11.9 ± 5.3 | 11.2 ± 4.0 | 520 |
| late ratio 192 | 3.2 ± 0.7 | 12.2 ± 4.9 | 11.0 ± 3.0 | 472 |
| + L2-to-init | 3.6 ± 1.3 | 15.3 ± 9.2 | 13.1 ± 2.9 | 433 |
| batch length 32 | 3.1 ± 0.8 | 14.0 ± 3.5 | 14.9 ± 4.0 | 562 |

B is the best balance base so far (CR2 had 10.1 with 4 seeds on a V100). The
schedule halves the late drift (19.1 without). Each live setting alone makes
it worse, the full-speed home drive most. The fear critic is the one addition
that cuts the drift further, at a quarter more wheel movement.

K4 (e42845-e42864, commands, n = 4; figure `figs/weekend/k4.png`): the pick
holds for 90 min (tracking 0.81 at 30-60 min, 0.79 at 60-90; 3.9 falls a
minute after 30), and is unchanged with the live phone settings (forward 9.2
min, 3.3 falls) and slightly better with margin 2.0° (8.5 min, 3.1 falls). It
fails on carpet (forward never 0.7 in 60 min, ~40 falls a minute) and with
pushes and walls (never 0.7, 14-19 falls): on carpet the commands are not
learnt in an hour. The live demo floor is the table-like Germany floor, so
this is a caveat, not a blocker.

W5 (e42865-e42928 IAS, e42929-e42992 V100; n = 8): B, + fear, stillness
(wheel 1.0 + pos 2.0, or wheel 0.75) now that the bumper trap is shut, fear +
stillness, fear + entropy, and the live e40006 recipe with only the two fixes
(margin, batch 16), as it could run today.

### Sat 11-07 ~16:45: W4 (P100 half) in; KF (final, commands) launched

W4 P100 (e42749-e42844, n = 8, no seed stuck) repeats the IAS half: the fear
critic halves late crossings (B 12.1 ± 4.7 → 6.0 ± 2.8 per min after min 15;
first ≥ 0.9 in 3.6 → 3.1 min), the live home drive hurts most (16.7), double
lrs and batch length 32 hurt (15.8, 17.2). At P100 speed the ratio schedule
makes no difference (no schedule 10.9), as expected: the drift is a fast-GPU
problem.

KF (final command round, 60 min, n = 8 per speed; P100 e43001-e43032, V100
e43033-e43064, IAS e43065-e43096): `robot_command` base vs the pick
(smoothing + L2-to-init + batch 16), the pick with bumper margin 2.0° and
2.9° (the balance pick's margin, so both share one live flag).

### Sat 11-07 ~19:30: W5 in; BF (final, balance) launched

W5 (30 min; IAS e42865-e42928 n = 8, V100 e42929-e42992 n = 7 of 8 in; no
seed stuck; figure `figs/weekend/w5.png`). Crossings per min after min 15:

| arm | first ≥ 0.9, IAS | crossings 15+, IAS | crossings 15+, V100 | wheel mm/s, IAS |
|---|---|---|---|---|
| B | 2.9 ± 0.3 | 10.7 ± 4.1 | 12.5 ± 2.5 | 508 |
| + fear | 2.5 ± 0.5 | 4.8 ± 1.3 | 7.5 ± 1.1 | 581 |
| + stillness (wheel 1.0, pos 2.0) | 4.1 ± 0.8 | 10.6 ± 2.1 | 14.9 ± 4.1 | 300 |
| + wheel 0.75 | 3.6 ± 1.0 | 13.1 ± 4.3 | 11.5 ± 4.8 | 416 |
| + fear + wheel 0.75 | 2.5 ± 0.5 | 5.1 ± 1.9 | 6.4 ± 1.5 | 541 |
| **+ fear + stillness** | **2.4 ± 0.5** | **3.2 ± 1.2** | **2.6 ± 1.3** | 433 |
| + fear + entropy 1e-3 | 2.5 ± 0.5 | 6.4 ± 1.9 | 6.4 ± 1.0 | 549 |
| live recipe + margin + batch 16 | 3.6 ± 1.0 | 14.6 ± 4.5 | 17.3 ± 2.4 | 433 |

Fear + stillness is the pick: fastest start, a third of B's late crossings,
and less wheel movement than B. Stillness alone slows learning (4-5 min); the
fear critic makes it safe. The live recipe with only the two fixes stays
worst: the live home drive and phone settings still cost.

BF (final balance round, 60 min, P100 e43101-e43156, V100 e43157-e43212, IAS
e43213-e43268): today's live recipe (e40006: live settings + schedule), B,
B + fear, B + fear + stillness, n = 12 each; holdouts of fear + stillness on
carpet and with pushes + walls, n = 4 each.

### Sun 11-08 ~00:15: KF (Saion half) in

KF on IAS failed at start the first time (the IAS launcher took only balance
lines); `sbatch/ias/run_sim_ias.sbatch` now runs command and still lines too
(wall limit 85 min), resubmitted as e43065-e43096. BF on IAS: 3 of the first
42 runs hit the old 75 min wall limit just short of 60 min (data kept).

KF, 60 min, n = 8 per speed (P100 e43001-e43032, V100 e43033-e43064):

| arm | fwd 0.7 (min), P100 | fwd 0.7 (min), V100 | turn 0.7 (min), V100 | crossings 10+, P100 | crossings 10+, V100 | fwd corr last 20 |
|---|---|---|---|---|---|---|
| `robot_command` base | 15.9 ± 7.6 | 15.0 ± 9.4 | 7.6 ± 4.4 | 37.1 ± 12.8 | 24.2 ± 12.4 | 0.81-0.83 |
| **pick** | **9.4 ± 2.5** | **9.5 ± 2.3** | 6.4 ± 2.8 | 5.0 ± 0.7 | 4.6 ± 0.7 | 0.82 |
| pick + margin 2.0° | 11.2 ± 1.5 | 13.4 ± 2.4 | 11.2 ± 4.3 | 4.0 ± 0.6 | 4.4 ± 0.9 | 0.82 |
| pick + margin 2.9° | 16.8 ± 7.3 | 11.5 ± 2.9 | 12.0 ± 4.5 | 4.6 ± 1.9 | 3.3 ± 1.3 | 0.83-0.84 |

The pick learns forward tracking in ~9.5 min at both speeds with 5-8x fewer
falls than the base. A wider bumper margin slows command learning by 2-7 min
for at most one fall per minute less: keep the default margin for commands
(the balance pick keeps 2.9°).

### Sun 11-08 ~08:30: BF and KF (IAS half) in

BF (final balance, 60 min, n = 12 per speed; holdouts n = 4; figure
`figs/weekend/bf.png`). Crossings per min after min 30; IAS B and B + fear
runs were cut at 43-46 min by the old 75 min wall limit (scored on what ran).

| arm | first ≥ 0.9 P100 / V100 / IAS (min) | crossings 30+ P100 / V100 / IAS | wheel mm/s 30+ (V100) | stuck seeds (of 36) |
|---|---|---|---|---|
| live e40006 (today) | 6.2 / 10.1 / 12.1 | 18.5 / 16.6 / 17.4 | 351 | 8 |
| B | 3.4 / 3.2 / 2.8 | 7.7 / 7.3 / 5.8 | 319 | 0 |
| B + fear | 2.8 / 2.3 / 2.5 | 4.2 / 4.0 / 4.2 | 396 | 0 |
| **B + fear + stillness** | **3.2 / 3.0 / 2.7** | **1.8 / 2.0 / 2.3** | **295** | **0** |
| ... on carpet (n = 4) | 2.0 / 2.0 / 3.0 | 0.0 / 0.1 / 0.0 | 21 | 0 |
| ... pushes + walls (n = 4) | 4.0 / 4.5 / 2.8 | 4.8 / 3.7 / 3.3 | 485 | 0 |

SDs: pick crossings ± 0.9-1.0, live ± 3.2-6.7, first-upright pick ± 0.4-0.7,
live ± 2.8-14.2. The pick is up in ~3 min at every learner speed, never
sticks, crosses a bumper ~2x/min after 30 min (8-9x fewer than today's
recipe) with the least wheel movement, and shows no late drift to 60 min. It
holds on carpet and under pushes.

KF IAS (e43065-e43096, n = 8; figure `figs/weekend/kf.png`): base 8.8 ± 2.5
min to forward 0.7 with 17.6 ± 7.4 falls/min; pick 9.4 ± 2.2 min, 4.3 ± 0.6
falls/min; margins 2.0°/2.9° 15.6 / 10.8 min. The pick's time to tracking is
the same at all three speeds (9.4-9.5 min); the base is fast only on the
fastest GPUs and falls 4-8x more. The common dips at min 24 and 43 are the
command sequence (same seeds in every arm), not learning.

### Sun 11-08 ~13:00: S in; C (carpet diagnosis) and L (120 min) launched

S (60 min, n = 8 per speed; P100 e43333-e43364, V100 e43301-e43332, IAS
e43365-e43396; figure `figs/weekend/s.png`), P100 / V100 / IAS:

| arm | first ≥ 0.9 (min) | crossings 30+ | wheel mm/s 30+ |
|---|---|---|---|
| balance pick + policy age 1.3 s | 3.1 / 2.9 / 2.9 | 1.4 / 1.8 / 2.4 | 292 / 283 / 289 |
| balance pick + all live settings (home drive, publish, age) | 4.4 / 3.9 / 3.5 | 3.7 / 4.7 / 3.9 | 389 / 402 / 394 |

| arm | fwd 0.7 (min) | falls/min 10+ | fwd corr last 20 |
|---|---|---|---|
| command pick + fear critic | 11.6 / 11.9 / 9.8 | 3.9 / 3.6 / 3.8 | 0.82-0.83 |
| ... on carpet | never / never / never | 14.9 / 19.3 / 20.1 | 0.0-0.07 |

The balance pick does not mind the live policy age. With all of today's live
settings it is ~1 min slower and crosses twice as often (still 4x fewer than
today's recipe, no stuck seed): keep the CR2 home drive (half speed, 3 s
ramp, no cap) and publish every second sync. The fear critic gives the
command pick ~1 fall/min less for ~2 min slower tracking on Saion: not worth
it. Commands on CARPET fail completely.

C (Saion, 60 min, n = 8; e43401-e43432): is CARPET too harsh? It carries the
FLOOR's carpet-only scrape (sim_scrape 20, drawn 0-40) on top of the fitted
carpet. Command pick on carpet without scrape, on the plain fitted carpet,
the `robot_command` base on the plain carpet, and the balance pick without
scrape.

L (IAS, 120 min, n = 8; e43441-e43456): both picks for two hours on the
fastest learner, for a late collapse after 60 min.

### Sun 11-08 ~15:00: C in; D launched

C (60 min, n = 8, V100 seeds 811-814 + P100 seeds 815-818):

| arm (carpet) | fwd 0.7 (min) | seeds reaching it | falls/min 10+ | fwd corr last 20 |
|---|---|---|---|---|
| command pick, CARPET without scrape | 55.0 ± 5.4 | 4/8 | 44.9 ± 7.2 | 0.4 ± 0.3 |
| command pick, plain fitted carpet | 25.5 ± 7.2 | 8/8 | 50.6 ± 6.2 | 0.8 ± 0.1 |
| `robot_command` base, plain fitted carpet | 23.9 ± 11.5 | 8/8 | 58.2 ± 8.4 | 0.8 ± 0.0 |
| balance pick, CARPET without scrape | first ≥ 0.9 3.2 ± 0.7 | - | 0.4 ± 0.7 crossings | wheel 21 mm/s |

The carpet holdout of K4/S was not a fitted floor: the table-fitted drive
randomisation (stiction, rolling stiction 5-9x, gear, inertia) on top of the
carpet breaks commands even without the scrape. On the plain fitted carpet
commands are learnt in ~25 min (as e2234-e2263, ~18 min with 5 s holds) but
with ~50 falls/min, and the pick is no faster there than the base. Balance on
carpet is easy (the fitted carpet holds a lean, so the robot stands still).

D (60 min, n = 8): V100 e43501-e43524, the stillness dose of the balance
pick (fear + wheel 1.0/pos 1.0, wheel 1.5/pos 3.0, pos 2.0 alone); P100
e43525-e43548, the command pick on the plain carpet with slower commands
(1200/1500), horizon 3 s, bumper penalty 4.

### Sun 11-08 ~18:30: D and L in (last rounds)

D V100 (balance pick's stillness dose, 60 min, n = 8, all with fear):

| wheel / pos penalty | first ≥ 0.9 (min) | crossings 30+ | wheel mm/s 30+ |
|---|---|---|---|
| 1.0 / 1.0 | 2.9 ± 0.3 | 4.0 ± 1.0 | 364 |
| 0.5 / 2.0 | 3.0 ± 0.7 | 2.0 ± 0.9 | 328 |
| **1.0 / 2.0 (pick, BF V100)** | 3.0 ± 0.7 | 2.0 ± 1.0 | 295 |
| 1.5 / 3.0 | 3.4 ± 1.0 | 1.2 ± 0.6 | 257 |

The dose is smooth: more stillness, fewer crossings and less movement, for at
most ~0.5 min slower learning. 1.5 / 3.0 is a safe step if the live robot
moves too much.

D P100 (command pick on the plain fitted carpet, n = 8): slow commands 49.4
min to forward 0.7 (6/8), horizon 3 s 30.2 (7/8), bumper penalty 4 27.6
(8/8), all with 47-54 falls/min, against the pick's 25.5 in C. Carpet
commands stay slow and fall-heavy; no cheap fix found.

L (IAS, the fastest learner; 7 of 8 balance runs cut at 87-103 min by the
2.5 h wall limit, commands to 103-119 min; figure `figs/weekend/l.png`):
balance pick crossings 1.4 ± 0.5 (min 30-60), 0.9 ± 0.4 (60-90), 0.5 ± 0.4
(90+), wheel speed 175 mm/s after min 90; command pick forward corr 0.8 and
4.0 / 3.8 / 3.5 falls/min over the same windows. No late collapse in either:
both keep improving.

## Result (Mon 11-09)

All in the fitted sim, Germany floor, phone loop; n per speed, three learner
speeds (P100 ratio ~280, V100, IAS A5000/3090 ~570). Figures
`figs/weekend/bf.png`, `kf.png`, `s.png`, `l.png`.

**Balance: fear critic + stillness on the CR2 recipe.** Up in ~3 min at every
learner speed (2.7-3.2 ± 0.4-0.7, n = 12 per speed), no seed stuck on a
bumper (0 of 36; today's live recipe 8 of 36), ~2 bumper crossings a minute
after 30 min (today 17-19), the least wheel movement of any arm, and still
improving at 90-100 min. It holds with the live policy age (1.3 s), on carpet
and under pushes. The live settings that hurt it are the full-speed home
drive and publishing every sync (with them: 4 crossings/min, 1 min slower).

Live flags (e40006's command with these changes):
```
drop:    --online_publish_every 1          (block default 2)
         --env.robot.home_max_s 10
change:  --env.robot.home_speed 0.5 --env.robot.home_ramp 3000
         --env.robot.wheel_penalty 1.0 --env.robot.pos_penalty 2.0
         --env.robot.weights_emotion True
add:     --env.robot.bumper_margin 0.05 --batch_size 16
         --env.robot.cost_obs True --agent.fear.weight 1.0
keep:    --online_ratio_switch 19500 --online_ratio_late 128
```
No app change (the fear critic and the cost signal live on the server). The
bumper margin (2.9°) closes the trap where a policy rests on the front bumper
at 8.1° just outside the penalty zone; on the real robot the bumper sits at
+9.2°, so check the margin against the live tilt log in the first minutes.

**Commands: smoothing + L2-to-init + batch 16 on `robot_command`.** Forward
tracking (corr ≥ 0.7 for 3 min) in 9.4-9.5 ± 2.2-2.5 min at every learner
speed, 4.3-5.0 falls/min (base 17-37), final corr 0.82, steady to 120 min.
```
--configs ... robot_command
--env.robot.action_rate_quad 0.25 --agent.imag_loss.smooth 0.15
--agent.ac_opt.wd 0.1 --agent.ac_opt.wdinit True --batch_size 16
```
Keep the command block's own bumper margin (wider margins slowed learning by
2-7 min). The fear critic is not worth it here (-1 fall/min, +2 min).

**What did not help.** Shrink-perturb / L2-to-init on actor-critic or world
model, critic feature norm (drift control is the ratio schedule); double
learning rates, batch length 32, batch 64, bigger or smaller nets, entropy on
its own; the reward-gated ratio switch; command curriculum
(`command_grow_s`), longer relabelling, shorter holds, narrower tracking
kernel. `command_grow_s` stays in the sim code but is not recommended.

**Caveats.** Carpet commands: on the plain fitted carpet both recipes learn
in ~25 min with ~50 falls/min, and none of slow commands, horizon 3 s or
bumper penalty 4 fixed that; the carpet holdout with the table's drive
randomisation (K4, S) fails outright and is not a fitted floor. All numbers
are sim; the balance pick has not been on the real robot.
