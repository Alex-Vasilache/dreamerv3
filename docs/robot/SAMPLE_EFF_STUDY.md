# Sample-efficiency study (from 2026-11-09)

Goal (Alex): learn **from scratch on the real robot** as fast as possible.
The fitted sim is the test bed; no sim-pretrained warm starts as a recipe
(runs that load trained weights are diagnostics only). Baseline is the
weekend balance pick (`WEEKEND_1107.md` "Result"): up in ~3 min, ~2
crossings/min after 30 min, with the ratio schedule as the drift fix.

Literature: `LIT_AUGMENT_DV4.md` (modality-agnostic augmentation, DreamerV4,
multi-rollout imagination) and `LIT_ZERO_SHOT.md`, both fact-checked 11-09.

## Plan

| step | what | question | status |
|---|---|---|---|
| 0 | live runs e43457, e43605-e43607 (weekend balance pick) vs e40006 | does the pick hold on the robot? | analysing |
| 1 | Z1, Saion V100, 60 min, n = 8 (e43700-e43731): pick uncapped / with schedule / uncapped + world model frozen at min 5 / uncapped + actor-critic frozen at min 5 | which component drifts late? | queued (Saion jobs 4807464-95) |
| 1 | Z2, IAS, 30 min, n = 8 (e43740-e43771): pick fresh / start with trained world model / actor-critic / both (diagnostic) | which component limits the first minutes? | running (IAS array 200247) |
| 2 | world-model dropout 0.05 / 0.1 / 0.2 (uncapped), 0.1 with the schedule | does it stop the drift and let data be reused harder? | Y, Saion V100, e43800-e43831 (unc d0.05/0.1/0.2, sched d0.1), queued behind Z1 |
| 3 | observational dropout 10 / 30 % (open-loop steps in training) | better imagination from the same data? | X, IAS, e43840-e43879 (unc, sched, unc o0.1/0.3, sched o0.3), array 200255 after Z2 |
| 4 | N rollouts per start (2, 4) | does spare GPU buy speed? | V, IAS, e43880-e43911 (unc/sched x N 2/4), array 200256 after X |
| 5 | winners on commands; best to live | fastest from-scratch recipe | |
| 6 | reanalysis: what worked, what did not, why; combine the parts that help together (factorial on the 2-3 best), confirm n = 12 at three GPU speeds | most improvement | |

If Z1 shows the actor-critic drifts rather than the world model, step 2's
dropout moves to the critic (DroQ-style).

Experiment numbers: the live tool on IAS takes max+1 of IAS `robot_runs`;
sims stay in blocks well above the live numbers (smoke e43600-e43606 share
e43605/e43606 with two live runs; different suffixes).

Code pinning: Saion sims run from the working tree at job start, so it stays
at 49e142f until every Z1/Y job has started (Alex's 2cc011e changes the drive
home, which the sim uses too; mixing it into a round would confound it). IAS
rounds run from snapshots (`~/robot_runs/code_z`, `code_s` = 49e142f). Rounds
after Z/Y/X/V use the current branch, drive-home stall stop included.

## Log

### Mon 11-09 ~12:30: live runs of the weekend balance pick (step 0)

`e43457` (11 live min, stopped by Alex), `e43606` (3 min), `e43607` running;
`e43605` recorded no live steps. Per minute (`tools/robot/dev/demo_stats.py`,
figure `figs/study/live_1109.png`):

| min | e43457 upright | crossings | \|speed\| mm/s | e40006 upright | crossings |
|---|---|---|---|---|---|
| 1 | 0.21 | 147 | 1323 | 0.19 | 126 |
| 2 | 0.67 | 15 | 1397 | 0.33 | 44 |
| 3 | 0.80 | 15 | 711 | 0.73 | 47 |
| 4 | 0.97 | 5 | 616 | 0.90 | 19 |
| 5 | 0.93 | 16 | 1168 | 0.95 | 9 |
| 6 | 0.96 | 8 | 451 | 0.94 | 11 |
| 7-9 | 0.89-0.92 | 8-22 | 648-826 | 1.00 | 0-1 |
| 10 | 1.00 | 1 | 333 | 1.00 | 0 |

e43606: 0.32 / 0.90 / 0.95 upright, 99 / 20 / 14 crossings in minutes 1-3.
The pick learns as fast or faster than e40006 in minutes 1-4 (fewer
crossings at minutes 2-4: 15/15/5 against 44/47/19), but at minutes 5-9 it
moves fast (450-1170 mm/s) and crosses 8-22 times a minute where e40006 had
settled to 0-1 by minute 7. n = 1 each; the sim predicted the opposite late
(~2 against 17-19 crossings/min). Open: is the stillness penalty weaker live
(position scale, home drive moving the robot), or is this run-to-run
variation? Repeat runs are needed before reading more into it.
