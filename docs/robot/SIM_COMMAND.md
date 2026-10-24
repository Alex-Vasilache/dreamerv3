# Command following, still balance and floor changes, in the fitted sim

Weekend of 2026-10-24/25. Everything here is the MuJoCo sim
(`embodied/envs/robot_mujoco.py`, `--configs robot [robot_command] robot_sim`),
65 Hz, paced to real time, flat DreamerV3. Experiments e2070–e2380 (main
repo `EXPERIMENTS.md`), launchers `sbatch/sim_*.txt`, figures in
`figs/weekend/`, videos in `/work/DoyaU/vasilache/work/realcmd/videos/`
(bucket copy after archiving). Tables are mean ± std over seeds; n in each row.

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
3. **Carpet → table:** the sim does not reproduce "never re-learns" with the
   floors as fitted; with a carpet that holds the wheels (as the real one
   does) it re-learns ~1.5x slower than a fresh run, and none of weight
   decay, L2-init or a bigger replay fixes that (see 4). What would tell us
   most is one real run with the floor written down: carpet ~10 min, table
   ~20 min, carpet ~10 min, `wheel_penalty` 1.5.

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

## 5. Final validation (refitted floors, final blocks)

(Filled in when e2362–e2380 finish.)

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
