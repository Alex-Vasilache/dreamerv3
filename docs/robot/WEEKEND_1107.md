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

### Sat 11-07 ~00:30: IA2 in

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

### Sat 11-07 ~00:15: K1 in, K2 launched

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
