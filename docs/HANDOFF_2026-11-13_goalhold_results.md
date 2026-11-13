# Handoff 2026-11-13: goal hold until reached — first results are negative, diagnostics running

Start here. Previous handoff: `docs/HANDOFF_2026-11-11_goalhold_surprise.md` (background, house
rules in its section 6 still apply). Full log: `EXPERIMENTS.md`, section "h0374-h0403".

## 0. State at handoff (2026-11-13 18:40 JST)

- **Array 4812978 is live** (`sbatch/goalhold_params.txt`, A100, 8 at a time, time limit 16 h).
  Rows are read when a task starts, so unstarted rows can be edited in place.
- **Autonomy:** the user gave open-ended autonomy (11-13) to implement, test, launch, supervise,
  cancel, fix and reschedule for this plan, until they say otherwise. Log every change, new
  h-numbers, commit and push. The surprise arm (step 5 of the old handoff) is not started.
- **Next free experiment number: h0410.**
- **The session's 40-minute supervision cron dies with the session.** Recreate it if wanted.
  Scripts are now in the repo: `tools/goalhold/supervise.sh [array id]` (read-only status) and
  `tools/goalhold/archive_done.sh` (archives array runs in a terminal sacct state: copy without
  replay, verify, delete run dir + SLURM logs). A run that ended with no metrics fails the verify
  and must be removed by hand.

Running now:

| Run | Row | What | Step at 18:40 |
|---|---|---|---|
| h0383 / h0384 / h0385 | 9-11 | cartpole, cap 32 / 8 / 16, seed 0 | 2.40M / 1.72M / 1.43M |
| h0386 / h0387 / h0388 | 12-14 | cheetah, cap 32 / 8 / 16, seed 0 | 1.12M / 1.15M / 1.00M |
| h0407 | 17 | pin pad four, cap 8, old worker lr (`wkr_lr4e5`) | 0.11M |
| h0408 | 18 | pin pad four, cap 8, reached test off (`goal_hold_noreach`) | 0.11M |

Queued (start as slots free, in this order): h0409 (row 19, pin pad four, cap 8, mean reward
pooling `goal_hold_mean`); h0404-h0406 (rows 20-22, hopper cap 32 / 8 / 16); h0397 (row 23, pin pad
five seed 1 cap 16); h0398-h0400 (rows 24-26, pin pad six seed 1, cap 32 / 8 / 16); h0401-h0403
(rows 27-29, pin pad six seed 2). Consider replacing rows 23-29 once the diagnostics answer.

Finished and archived (bucket, removed from `/work`): h0374-h0382 (pin pads, seed 0, three caps);
h0389 (hopper cap 32, cancelled at 0.48M), h0390 (hopper cap 8, cancelled at 0, empty).

## 1. What was built

Config block `goal_hold_reach` (layer on V: `director director_og wkr_ent_fixed wkr_v3
goal_hold_reach pubmatch`). Worker lr is the new default 1e-4.

- The manager re-decides when the held goal is reached or the cap runs out. No minimum hold.
- Reached = cos_max(goal, deter) >= `goal_reach_frac` (0.8) x ceiling. The ceiling is a running
  mean of cos_max(decode(encode(state)), state) on replay states (`hrl/goals.py: ReachCeil`,
  metric `goal/reach_ceil`), so the threshold is not tuned per task.
- Same rule in the env step, imagination and replay (`hrl/manager.py`).
- Manager: rewards summed over the hold (`mgr_reward_agg: sum`); replay return discounted by hold
  length (it was a flat 0.99 per decision; only changed when `goal_reach` is on).
- Worker: return for a goal ends at the switch, bootstraps from the value under the old goal,
  and the arrival reward is scored against the old goal (dense path, `agent.py`, `hrl/losses.py`).
- Cap = `goal_duration_fixed` (32), `imag_length` 32. Extra blocks, layered after:
  `goal_hold_cap16`, `goal_hold_cap8` (also imag_length 16), `goal_hold_noreach`
  (frac 50, never fires), `goal_hold_mean`.
- Tests: `embodied/tests/test_goal_hold_reach.py`, 7 pass. 12 older tests in
  `test_variable_goals` / `test_block_pooled_*` / `test_hrl_package` fail; 2 were confirmed to fail
  on the unmodified tree, the other 10 show the same stub error from a metrics line not touched
  here but were not rerun on the old code.

Offline calibration (scripts + outputs in `experiments/goal_reach_calibration_20261112/`, data in
`/work/DoyaU/vasilache/work/meetings/tce_investigation/reach/`): ceiling 0.92-0.94 on solved pin pad
six seeds, 0.81 on failed, 0.97 on cartpole. At 0.8 x ceiling on V pin pad six it fires on 34-38%
of steps on the goal cell, 6-7% one cell away, 0.2-1.5% further. With a hand-fixed pad goal about
half of first fires are on the pad, a third one cell before it. cos_max, cosine and MSE separate
reached from not reached equally well, so cos_max stays.

## 2. Results so far (seed 0, one run per cell)

Late window 3.5-4M; in brackets the sustained-50 crossing in M steps (trailing 10 episodes).
Old V = 5 seeds at worker lr 4e-5.

| Task | Old V | Cap 32 | Cap 16 | Cap 8 |
|---|---|---|---|---|
| Pin pad four | 311 (297-335); crossing 0.14-0.30 | 237 (2.86) | 173 (3.63) | 180 (3.95) |
| Pin pad five | 278 (268-287); crossing 1.11-2.59 | 144 (2.96) | 154 (3.77) | 140 (3.85) |
| Pin pad six | 202, 0, 0, 0, 0 | 0 | 0 | 0 |

- The late crossings are dips, not late learning: pin pad four sits at ~160-250 from 0.25M on.
- DMC (running): cartpole about equal to old V at matched steps; cheetah at 1.0-1.25M is 505 at
  cap 8, 393 at cap 32, old V mean 442.
- **Cap 8 is as bad as cap 32 on the pin pads, so long holds are not the cause.**

Open suspects, and the run that tests each (read them at 0.25-0.75M; old V seeds have 238-305
then 271-311 on pin pad four, the cap runs ~200):

| Suspect | Run | Reading |
|---|---|---|
| New worker lr 1e-4 | h0407 | ~300 -> lr is the cause |
| The reached test itself (goal dropped a cell before the pad) | h0408 | ~300 -> reach hurts; ~200 -> the training path |
| Summed manager rewards | h0409 | ~300 -> the sum pooling |

Not covered by these three: the replay hold-length discount and the worker boundary change (both
stay on in all three). If all three stay ~200, test those next (they are tied to `goal_reach`
being on in `agent.py`), or run plain V at lr 1e-4 — the user declined a separate V-base arm on
11-12, so ask or justify. Partial evidence on lr: cancelled V runs at lr 1e-4 (h0344, h0350) had
266 / 245 on pin pad four at 0.25-0.5M.

## 3. Metric caveats (cost me a wrong diagnosis)

- `goal/hold_len_repl` is **not** the real hold length. It runs the current manager's goals over
  replayed trajectories that were not pursuing them, so reach rarely fires. There is no metric
  for the hold length in the real env yet; adding one (log the switch flag from `policy`) is
  worth doing before the next batch.
- `goal/hold_len_imag` = states / decisions per rollout, censored by the horizon. With no reach
  at all it reads 5.7 at cap 8 (17 states, 3 decisions) and 16.5 at cap 32. Use
  `goal/hold_reach_frac` (share of imagined switches triggered by reach) alongside it.
- Pin pad: sustained-50 crossing plus late-window mean, never the last-15 alone.

## 4. Queue notes

- The array waited ~26 h. Cause: fair-share (our user is ~78% of the group's recent usage; the
  job gets 10 of 50,000 fair-share points). `--nice=100` was removed and the time limit cut from
  48 h to 16 h so backfill can place jobs. Runs take ~7-8.7 h on pin pad (130-160 fps) and ~10 h
  on DMC (105-125 fps).
- Robot session jobs (e44xxx simrobot etc.) hold V100/P100: never touch them.

## 5. Next steps

1. Read h0407 / h0408 (and h0409 when it starts) at 0.5M and 0.75M; update `EXPERIMENTS.md`.
2. Fix or drop the guilty part, re-test, relaunch pin pads under new numbers (from h0410); edit
   the unstarted rows 23-29 of the array or submit a new one.
3. Let cartpole / cheetah finish; archive with `tools/goalhold/archive_done.sh`.
4. Add a real-env hold-length metric.
5. Only then: more seeds, and the surprise arm (old handoff, step 5; reward the excess
   max(0, KL - running baseline), not raw surprise).
6. Update the paper (`code/26_04_HRL-paper`) once the goal-hold result is settled; not done yet.
