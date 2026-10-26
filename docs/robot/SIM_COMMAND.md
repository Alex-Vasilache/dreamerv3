# Command following, still balance and floor changes, in the fitted sim

Weekend of 2026-10-24/25. Everything here is the MuJoCo sim
(`embodied/envs/robot_mujoco.py`, `--configs robot [robot_command] robot_sim`),
65 Hz, paced to real time, flat DreamerV3. Experiments e2070–e2424 (main
repo `EXPERIMENTS.md`), launchers `sbatch/sim_*.txt`, figures in
`figs/weekend/`, videos in `/work/DoyaU/vasilache/work/realcmd/videos/`
(copy in the bucket, `results/robot_weekend_20261024/`; runs archived to
`results/dreamerv3/`, with their experience). Tables are mean ± std over seeds; n in each row.

## For Monday

1. **Command following, from scratch on the robot:**
   `CONFIGS="robot robot_command" tools/robot/train.sh cmd`. The block now
   has a 2 s horizon (was 5), no friction compensation (was on) and full
   stick = 2500 mm/s (was 4900). Give it **40–60 min**. In the sim it tracks
   turns after ~10 min, forward on the table after ~10 and on carpet after
   ~20–30 (see 2). Judge it by `epstats/log/cmd_forward` against
   `epstats/log/forward` (or `tools/robot/dev/plot_cmd_tracking.py` on the
   run), not by the reward.
2. **Balance still on the table:** the `robot` block's `wheel_penalty` is now
   1.5 (was 0.3; e2078 used 0.6). No other change needed (see 3).
   The `robot` block also keeps all experience now (`replay.size` 1e6, was
   1e5): with the old 26-min buffer a policy that had stood still for half
   an hour forgot how to get off a bumper (see 5).
3. **Carpet → table:** the sim does not reproduce "never re-learns" with the
   floors as fitted; with a carpet that holds the wheels (as the real one
   does) it re-learns ~1.5x slower than a fresh run, and none of weight
   decay, L2-init or a bigger replay fixes that (see 4). The forgetting
   half (back on carpet it is worse) matches replay eviction: with the old
   buffer the sim falls again during a second carpet visit, with 1e6 not
   (see 5, 6). What would tell us
   most is one real run with the floor written down: carpet ~10 min, table
   ~20 min, carpet ~10 min, `wheel_penalty` 1.5.
4. **Videos** (sim, 32 s each, side by side): `final_cmd_carpet_10_vs_75min`,
   `final_cmd_table_10_vs_75min` (final recipe, refitted floors),
   `final_switch_carpet_to_table`, `still_table_wp03_vs_wp15`; older fit2-floor
   ones: `cmd_carpet_10_vs_60min`, `cmd_table_10_vs_60min`,
   `cmd_carpet_vs_table_60min`, `switch_carpet_policy_on_table`.

## 1. The sim against the robot

**fit2** (the Friday handoff's job) cut the probe error from 0.93 to 0.74. To
check the sim where it matters -- balancing -- `tools/robot/dev/replay_to_sysid.py`
turns a training run's experience into a probe-style recording, and
`sysid_fit.py --phases balance` scores 0.3 s open-loop windows of it.
Held out (e2060 s02 carpet, e2061 s02 table), window error:

| floors | carpet | table |
|---|---|---|
| fit1 | 2.41 | 4.69 |
| fit2 | 2.21 | 3.90 |
| fit3 (motor + floors on probe + balancing) | 2.01 | 3.53 |
| **final** (fit2 motor; carpet fit6, table fit5) | **2.00** | **3.74** |

fit3 fits best but drops the gearbox's stiction (stick 0.38 → 0.004), which
the stand tests contradict (a still wheel does nothing below ~0.3 output), so
the motor stays fit2's. fit5/fit6 add **rolling stiction** per floor (a still
wheel takes `stiction` x the rolling resistance to start, with MuJoCo's
no-slip solver so the static part does not creep). Fitted, the carpet's is
4.5x: the sim carpet now holds a 2–5 deg lean and stops a kicked robot at
1–2 deg, as the real one does, where fit2's tipped over to a bumper
(`figs/weekend/fit2_final_kicks.png`, left column). The table's is
negligible. Known misses: the carpet holds the body on its front bumper a
little too long in PRBS tests, and the table (soft contact, fit5) rests at
+7.7 deg on the front bumper, not the measured +9.2.

Qualitatively the fitted sim also reproduces the real floors: trained on the
table with the old `wheel_penalty` 0.3 it balances while **pivoting on one
wheel** (one at ~3500 mm/s, the other ~40), as e2061 and e2078 did; on
carpet it stands nearly still. It balances on carpet in ~2 min and on the
table in ~5 (real: ~2–3 and ~3–4).

![sysid](figs/weekend/fit2_final_kicks.png)

## 2. Command following

**Why it never trained on the robot.** The Oct 15 runs (the only real
command runs) learned to turn (corr up to 0.8) but barely to drive (forward
gain 0.04–0.32 after 25–31 min) -- `figs/weekend/real_cmd_oct15.png`. They
ran on the stack of that week: 50 Hz, no chatter penalty, a balance that was
itself poor (tilt std ~3 deg against ~0.5 now), and 25–30 min each. In the
sim the same recipe learns both axes, but forward on carpet is slow and only
reaches about half the commanded speed; with rolling stiction (the real
carpet) forward gain after 25–30 min is ~0.3–0.5 -- not far from what the
robot showed. So the robot most likely was on its way and stopped early, on a
carpet that makes forward driving the hard part. What changed:

| change | evidence (sim) |
|---|---|
| horizon 5 → 2 s | carpet: min to forward corr 0.7, 28.5 ± 8.6 (4/5 seeds) → 18.2 ± 1.3 (5/5); table 11.8 ± 2.8 → 10.8 ± 2.7 (n=5) |
| friction_comp on → off | table and carpet as good or better; on the stiction carpet it slowed forward (30.2 ± 8.0 vs 17.8 ± 6.4 min, n=3–4) |
| full stick 4900 → 2500 mm/s | stiction carpet: forward gain 0.58 ± 0.05 → 0.75 ± 0.06 after an hour (n=3–4); table gain unchanged (~0.9) |
| tried, not taken | command_sigma 0.6 (forward no faster, one late collapse), hold 2–4 s, ramp 1.0, product reward (stays on a bumper) |

Final recipe on the refitted floors: see the table in section 5 (filled in
from the final validation).

![command final](figs/weekend/sim_cmd_final.png)

## 3. Balance still on the table

The balance reward's wheel cost is `wheel_penalty * mean_w min(|v_w| * 1e-4, 1)`:
at 0.3 a pivot at ~3000 mm/s costs ~0.05 of a reward near 0.85, so it pays to
pivot. Sim, table, last 3 minutes of 33:

| reward | n | min to balance | median \|wheel\| (mm/s) | tilt std (deg) |
|---|---|---|---|---|
| wheel_penalty 0.3 (old) | 2 | 5.2 ± 1.0 | 2662 ± 596 | 0.60 ± 0.26 |
| wheel_penalty 1.0 | 3 | 5.7 ± 0.4 | 111 ± 60 | 1.44 ± 0.10 |
| **wheel_penalty 1.5** | 5 | 6.3 ± 1.5 | **59 ± 14** | 1.40 ± 0.26 |
| wheel_penalty 2.0 | 3 | 8.2 ± 2.8 | 72 ± 23 | 1.54 ± 0.28 |
| still_speed 2000 (new, multiplicative) | 5 | 7.7 ± 0.8 | 212 ± 334 | 1.82 ± 0.87 |
| still_speed 3000 | 3 | 6.6 ± 1.0 | 47 ± 8 | 1.13 ± 0.81 |

On carpet `wheel_penalty` 1.5 costs ~0.6 min (2.4 ± 0.3 vs 1.8 ± 0.2 to
balance, n=3/2). `still_speed` (a factor exp(-|v|/still_speed) on the upright
score, `robot.py`) is stiller per seed but one in five collapsed back onto a
bumper; 0.6, as in e2078, only halves the pivot in the sim, as it did on
the robot. Video: `still_table_wp03_vs_wp15.mp4`.

![still](figs/weekend/still_r3.png)

## 4. Moving it from carpet to the table

`env.robot.sim_schedule 'carpet:15,table:20,carpet'` moves the sim robot
mid-run (set down on a bumper; `sim_carry_s` also carries it in the air,
which made no difference). Minutes until the 1-min reward holds 0.6 again:

| | n | table after carpet | fresh table | back on carpet |
|---|---|---|---|---|
| fitted carpet (fit2), baseline | 6 | 4.4 ± 0.8 | 5.2 ± 1.0 | 1.2 |
| + weight decay 1e-1 | 6 | 5.3 ± 1.3 | | 1.2 |
| + L2-init 3e-1 (decay toward the initial weights, new `opt.wdinit`) | 6 | 5.2 ± 0.9 | | 1.2 |
| + replay 1e6 (nothing evicted) | 6 | 4.6 ± 0.8 | | 1.2 |
| carpet_hold (holds a lean; too stiff to drive), baseline | 6 | 7.0 ± 2.7 | 4.5 ± 1.6 (n=7) | 1.2–1.8 |
| + weight decay 1e-1 | 5 | 6.9 ± 2.6 (4/5) | | 1.2 |
| + L2-init 3e-1 | 5 | 6.2 ± 0.8 (4/5) | | 1.2 |
| + replay 1e6 | 5 | 7.3 ± 1.6 | | 1.2 |
| carpet_sticky (hand-set stiction 2.5x), baseline | 5 | 6.5 ± 2.2 | | 1.2 |
| fit6 carpet (stiction), baseline | 5 | (section 5) | | |

What the sim shows: right after the move the carpet policy saturates the
wheels for several minutes (`switch_carpet_policy_on_table.mp4`); the
fitted-carpet agent then re-learns as fast as a fresh one, and an agent that
learnt on a carpet that holds it up (carpet_hold, carpet_sticky) takes
~1.5x as long. Nothing is
forgotten: back on carpet it holds within ~1 min, and alternating every 10
min the second table visit takes 1.2 min (n=3). Earlier, smaller rounds (n=3)
had suggested weight decay and a big buffer halve the re-learning time; with
6 fresh seeds that disappears, so I would not change the recipe for this.
Literature notes (plasticity loss, MBRL continual learning) are in
`/work/DoyaU/vasilache/work/realcmd/continual_lit.md`.

What the sim cannot show is why the robot "never" re-learns. Candidates it
leaves out: the user handling the robot while it races off after the move;
a real table policy that pivots and drives off the edge; anything that
degrades over a long session (none seen in the logs, but no logged session
is longer than ~12 min). The cheapest way to find out is one labelled real
run (For Monday, 3).

![floor switch](figs/weekend/floor_switch_r6.png)

## 5. Late collapse, and keeping all experience

On the refitted carpet (with stiction) the balance policy ends up standing
perfectly still -- wheels at the sensor-noise floor, outputs inside the
stiction band -- and then, 25–55 min in, creeps over and falls onto the front
bumper, where it stays for minutes (e2370–e2372: 3/3 seeds). By then the
26-minute FIFO replay (1e5 steps) holds no example of getting off a bumper.
Keeping all experience (`replay.size 1e6`, `online_actor_replay_size 1e6`)
removes it:

| replay | n | seeds that fell after min 10 | bumper-minutes after min 10 |
|---|---|---|---|
| 1e5 FIFO (old) | 8 | 8 | 3.1 ± 2.8 |
| 1e6 (new default) | 8 | 1 (0.3 min) | 0.04 ± 0.1 |
| 1e5 with wheel_penalty 0.3 | 2 | 2 | 4.1 ± 0.5 |
| 1e6 + recency 0.5 | 4 | 1 (0.1 min) | 0.03 ± 0.04 |

80–100 min runs, e2381–e2398, e2421–e2424. This is the forgetting half of "back on
carpet it does not balance as before": any session longer than ~26 min on
one floor evicts the other floor's experience with the old buffer. 1e6
steps is 4.3 h at 65 Hz, ~3.8 GB of RAM per hour of experience. The cost:
command following on carpet may learn turns more slowly with all experience
kept (17.6 ± 3.2 vs 8.5 ± 5.3 min, n=4); forward and the table are within
noise (table below). Drawing half of each batch from recent experience
(`replay.fracs` uniform 0.5, recency 0.5) on top of 1e6 does not bring the
speed back: forward corr 0.7 at 32.2 ± 9.5 min, turn 15.1 ± 2.9 (n=4,
e2417–e2420; `figs/weekend/sim_cmd_recency.png`), so the block keeps uniform
sampling.

![late collapse](figs/weekend/late_collapse.png)

## 6. Final validation (refitted floors, final blocks)

`robot robot_command robot_sim` and `robot robot_sim` as committed, floor
and seed only. Minutes to correlation 0.7 between command and measured
speed, and the last ~2 min:

| floor | replay | n | forward (min) | turn (min) | forward gain (late) |
|---|---|---|---|---|---|
| table (fit5) | 1e5 | 4 | 10.4 ± 2.2 | 8.9 ± 1.4 | 0.94 ± 0.03 |
| table (fit5) | 1e6 | 4 | 11.6 ± 2.3 | 8.6 ± 4.0 | 0.98 ± 0.04 |
| carpet (fit6, stiction) | 1e5 | 4 | 29.4 ± 11.3 | 8.5 ± 5.3 | 0.71 ± 0.17 |
| carpet (fit6, stiction) | 1e6 | 4 | 36.4 ± 15.1 | 17.6 ± 3.2 | 0.66 ± 0.07 |

Balance (`robot`, wheel_penalty 1.5): carpet 2.3 ± 0.2 min, table
5.5 ± 0.2 min to hold 0.6 (n=3).

Carpet 15 min → table 30 min → carpet 55 min on the refitted floors
(e2405–e2412, n=4 each), old and new replay:

| replay | table after carpet (min) | back on carpet (min) | bumper-minutes on the 2nd carpet visit |
|---|---|---|---|
| 1e5 FIFO (old) | 6.6 ± 0.7 | 1.2 | 0.89 ± 0.94 |
| 1e6 (new) | 5.7 ± 1.5 | 1.2 | 0.00 ± 0.00 |

Both hold the carpet again within ~1 min of the move back. With the old
buffer, though, the robot falls again 15–50 min into the second carpet
visit; with 1e6 it does not (`figs/weekend/floor_switch_final.png`). That
is the closest the sim gets to "back on carpet it no longer balances as
before".

![floor switch final](figs/weekend/floor_switch_final.png)

![command final](figs/weekend/sim_cmd_final3.png)

## 7. Faster command learning: relabelling and mirroring (Monday)

`--cmd_relabel F` gives a fraction F of each training batch a fresh command
with its tracking reward recomputed (hindsight relabelling, possible because
the command is exogenous and the reward is a known function of it);
`--cmd_mirror F` swaps left and right in a fraction F (wheels, actions,
turn sign). Both run on the learner (`embodied/envs/robot_relabel.py`).
Minutes to correlation 0.7, final blocks, e2425–e2450/e2463–e2464, 68 min
each; the gain column is the forward gain over the last ~2 min.

| floor, arm | forward | turn | final forward gain |
|---|---|---|---|
| carpet, base | 27.8 ± 5.1 | 9.6 ± 2.8 | 0.52 ± 0.12 |
| carpet, relabel 0.5 | 31.0 ± 7.9 | 11.1 ± 1.7 | 0.68 ± 0.12 |
| carpet, relabel 0.9 | 36.8 ± 3.9 | 7.8 ± 2.0 | 0.72 ± 0.08 |
| carpet, mirror 0.5 | 24.5 ± 7.0 | 11.4 ± 6.9 | 0.51 ± 0.09 |
| carpet, relabel 0.5 + mirror 0.5 | 22.2 ± 4.7 | 12.9 ± 5.7 | 0.72 ± 0.07 |
| table, base | 17.4 ± 3.0 | 10.5 ± 2.9 | 0.96 ± 0.04 |
| table, relabel 0.5 + mirror 0.5 | 12.1 ± 2.5 | 10.1 ± 3.4 | 1.03 ± 0.03 |

n=4 per arm. Relabelling makes the final forward tracking more accurate but
delays it on carpet; mirroring gives the earlier start; together they are
~1.25x (carpet) to ~1.4x (table) faster -- worth using on the robot
(now the `robot_command` default), not the order-of-magnitude gain
asked for. Turning is learnt in ~10 min regardless; forward on carpet is the
bottleneck. P100 seeds (2/3 of the V100's updates) learnt as fast as V100
ones, so more learner compute is not the lever.

![relabel and mirror](figs/weekend/sim_fast_r1.png)

## 8. Sim policies on the robot, and what the sim got wrong (Monday night)

Alex ran two sim balance policies on the robot with training paused:
e2468 (table, final recipe) as e2478 and e2471 (carpet) as e2479 (1.5 and
2.2 min off the bumpers, paused stretches dropped). Both balanced, but far
more restlessly than in the sim (`figs/s2r/closedloop.png`): on the table
the robot drove (median |forward| 387 vs 20 mm/s) and spun one way (mean
turn -246 mm/s); on carpet wheels 52 vs 14 mm/s, tilt rate 4x.

What is **not** wrong: fed the outputs the robot was sent, restarted every
0.3 s from its state, the fitted sim predicts these recordings as well as the
robot-trained runs it was validated on (window error 2.97 table, 1.54 carpet,
against 3.70 / 2.01; `figs/s2r/openloop.png`). Sensor lag (tilt vs tilt rate),
action-to-wheel delay (~19 ms both) and sensor noise also match; the sim's
noise is if anything higher.

What is wrong, found by running the same policy closed loop in the sim with
one thing changed (`tools/robot/dev/sim_closed_loop.py`, `figs/s2r/sweep1.png`):

* **The balance point.** Drawn once per run (+-1 deg), it is learnt by
  heart. Moved 0.25 deg forward, the e2468 policy drives and spins like the
  real table run (|fwd| 390, turn -296 mm/s, tilt -0.95 deg, 4% on a bumper;
  real 387, -246, -0.88, 7.5%); all three seeds go the same way. One wheel
  0.85x, +-0.03 output bias, 2x sensor noise: no effect; +2 ticks delay: small.
* **Gearbox stiction.** Fitted on the stand (wheels in the air) at 0.385 of
  stall. On the floor, under the policy, still wheels break away much more
  often at small outputs:

  | output | 0.15-0.2 | 0.25-0.3 | 0.35-0.4 | 0.4-0.5 |
  |---|---|---|---|---|
  | carpet, robot | 0.09 | 0.14 | 0.38 | 0.47 |
  | carpet, sim | 0.00 | 0.00 | 0.01 | 0.02 |
  | table, robot | 0.37 | 0.49 | 0.58 | 0.58 |
  | table, sim | 0.11 | 0.15 | 0.30 | 0.56 |

  (fraction of still wheels moving within 3 ticks). With 0.385 the sim
  carpet robot stands dead still at any balance point; at 0.2-0.25 it moves
  like the real one. A joint refit including the balancing data gives 0.227.

Fix, now in `robot_sim`: `sim_stick 0.2` and `sim_eq_every 180` (a new balance
point, +-1 deg, every 3 min). In the stick-0.2 sim, with the balance point
moved -1..+1 deg (`figs/s2r/robust_r2.png`, mean over balance points, n=3):

| training | drift (mm/s) | median turn (mm/s) | on a bumper | min to balance (table) |
|---|---|---|---|---|
| stick 0.385, fixed point (old) | 519 ± 48 | 229 ± 16 | 0.10 ± 0.03 | 4.1 ± 0.4 |
| stick 0.2, fixed point | 477 ± 24 | 49 ± 10 | 0.04 ± 0.00 | 4.8 ± 0.6 |
| stick 0.2, redrawn every 60 s | 272 ± 37 | 48 ± 11 | 0.02 ± 0.00 | 9.2 ± 1.5 |
| stick 0.2, redrawn every 180 s | 227 ± 57 | 45 ± 10 | 0.02 ± 0.00 | 6.6 ± 0.8 |

On carpet the redraw costs nothing (2.6 ± 0.1 vs 2.3 ± 0.5 min). None of
the policies is fully still across balance points: the policy has to find
the robot's balance point, and on the table that means some driving.

![closed loop](figs/s2r/closedloop.png)
![sweep](figs/s2r/sweep1.png)
![robustness](figs/s2r/robust_r2.png)

## Bugs found on the way

- `embodied/core/selectors.py`: `Recency` sampled `len(segment)` (undefined
  at the tree's first level) and `Mixture` had no `len`, so any
  `replay.fracs.recency > 0` run under online learning lost its learner at
  startup while the actor kept acting on the initial policy (5eeb3b1).
- The online learner now writes milestone snapshots (`run.save_every_steps`,
  every 5 min in `robot_sim`) for videos; first keyed on the learner's step
  (trained samples), it wrote ~1 GB/min per run before it was caught (fixed
  c01cfc3, the excess deleted).
- `short-a100` preempts and requeues: a few runs restarted from scratch; the
  plots take the last directory of each experiment number.
