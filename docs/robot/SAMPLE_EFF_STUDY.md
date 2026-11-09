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
| 1 | Z1, Saion V100, 60 min, n = 8 (e43700-e43731): pick uncapped / with schedule / uncapped + world model frozen at min 5 / uncapped + actor-critic frozen at min 5 | which component drifts late? | done: no late drift with the full pick on V100; freezing either part is worse (log 11-10) |
| 1 | Z2, IAS, 30 min, n = 8 (e43740-e43771): pick fresh / start with trained world model / actor-critic / both (diagnostic) | which component limits the first minutes? | done: world model is the early bottleneck (log 11-09 23:30) |
| 2 | world-model dropout 0.05 / 0.1 / 0.2 (uncapped), 0.1 with the schedule | does it stop the drift and let data be reused harder? | Y, Saion V100, e43800-e43831 (unc d0.05/0.1/0.2, sched d0.1), queued behind Z1 |
| 3 | observational dropout 10 / 30 % (open-loop steps in training) | better imagination from the same data? | X, IAS, e43840-e43879 (unc, sched, unc o0.1/0.3, sched o0.3), array 200255 after Z2 |
| 4 | N rollouts per start (2, 4) | does spare GPU buy speed? | V, IAS, e43880-e43911 (unc/sched x N 2/4), array 200256 after X |
| 4b | DreamerV4's one rollout per context: imag_last 1 (b16, b64), imag_last 8, vs every step | do fewer, more diverse starts learn as fast for less compute? | done: not better (fewer starts = more restless policy); not adopted (log 11-10) |
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

### Mon 11-09 ~23:30: Z2, which component limits the first minutes (step 1)

Sim, weekend pick with the ratio schedule, 30 min, n = 8 per arm (e43740-e43771,
IAS). Trained parts come from e43159 at step 214500 (diagnostic only).
Mean ± SD; t90 = first minute with upright ≥ 0.9. Figure `figs/study/z2.png`.

| arm | t90 (min) | upright min 1 | upright min 2 | crossings/min, min 1-5 | crossings/min, min 10-30 |
|---|---|---|---|---|---|
| fresh (the pick) | 2.9 ± 0.3 | 0.17 ± 0.05 | 0.79 ± 0.08 | 33 ± 5 | 5.0 ± 1.1 |
| trained world model, fresh actor-critic | 3.3 ± 1.1 | 0.66 ± 0.02 | 0.83 ± 0.07 | 22 ± 3 | 3.9 ± 1.7 |
| trained actor-critic, fresh world model | 10.0 ± 2.4 | 0.02 ± 0.01 | 0.05 ± 0.02 | 60 ± 44 | 13 ± 7 |
| both trained | 3.5 ± 1.4 | 0.76 ± 0.08 | 0.66 ± 0.19 | 18 ± 5 | 4.8 ± 1.2 |

Reading. (1) The world model is the early bottleneck: with a trained world model
a fresh actor-critic is already 66 % upright in minute 1, so the actor-critic
learns in well under a minute once the model exists. (2) A trained actor-critic
on a fresh world model is far worse than starting fresh (10 against 3 min): its
inputs are latents of a different model, and the critic's confident values
steer learning wrongly until the model settles. (3) Starting with both trained
gives the best minute 1 but dips at minute 2 (0.66): early training at the high
ratio on very little new data degrades a good policy. (4) After minute 3 none
of the three starts differs from fresh; late crossings are ~4-5/min for all but
the actor-critic arm. So for from-scratch speed the lever is how fast the world
model forms from the first ~2 minutes of data: the target of Y (world-model
dropout), X (observational dropout) and V/U (more imagination per sample).
Caveat: the actor process also loaded the trained policy for its first ~3 s
until the first learner sync (applies to all loaded arms).

### Tue 11-10: Z1, which component drifts late (step 1)

Sim, Saion V100, 60 min, n = 8 per arm (e43700-e43731). Freezes at actor step
19500 (min 5); the learner log confirms each freeze. Mean ± SD. Figure
`figs/study/z1.png`.

| arm | t90 (min) | crossings/min, min 6-15 | crossings/min, min 30-60 | upright, min 30-60 | \|speed\| mm/s, min 30-60 |
|---|---|---|---|---|---|
| pick, uncapped ratio (~260 on V100) | 3.3 ± 0.4 | 5.1 ± 1.8 | 2.0 ± 1.2 | 0.98 ± 0.01 | 283 ± 36 |
| pick, ratio schedule | 3.0 ± 0.7 | 7.3 ± 2.0 | 1.6 ± 0.7 | 0.98 ± 0.01 | 286 ± 21 |
| uncapped, world model frozen at min 5 | 2.9 ± 0.3 | 14 ± 7 | 22 ± 16 | 0.88 ± 0.07 | 911 ± 259 |
| uncapped, actor-critic frozen at min 5 | 2.9 ± 0.3 | 46 ± 61 | 61 ± 80 | 0.53 ± 0.37 | 1256 ± 833 |

Reading. (1) With the full pick (fear critic, stillness, batch 16) the uncapped
ratio no longer drifts late on V100: 2.0 against 1.6 crossings/min with the
schedule. The late collapse seen before the pick does not reproduce here; it
may still appear at the faster IAS ratio (~570), which the uncapped controls
in X and V test. (2) Freezing either part makes things much worse, so neither
part "is" the drift: the policy depends on the world model's latent space.
Freezing the actor-critic while the model keeps learning moves the latents
under a fixed policy (61 crossings/min, same failure as Z2's trained
actor-critic on a fresh model). Freezing the model at 5 min leaves the
actor-critic training on a model that has seen only 5 min of data, and it
exploits that model's errors (22/min, 3x faster driving). Both parts need to
keep learning together; the speed lever stays the world model (Z2).

### Tue 11-10: U, DreamerV4 one rollout per context (step 4b)

Sim, Saion P100, pick with the ratio schedule, 60 min, n = 8 per arm
(e43920-e43951; three jobs were requeued once, the data-holding attempt is
used). `agent.imag_last k` starts imagination from the last k steps of each
replayed sequence only (the pick starts from all 16 x 64 = 1024). Mean ± SD.
Figure `figs/study/u.png`.

| arm | starts per update | t90 (min) | upright min 1 | crossings/min, min 6-15 | crossings/min, min 30-60 | \|speed\| mm/s, min 30-60 |
|---|---|---|---|---|---|---|
| pick | 1024 | 3.0 ± 0.0 | 0.18 ± 0.03 | 5.4 ± 1.5 | 2.2 ± 0.8 | 296 ± 15 |
| one per sequence (DreamerV4) | 16 | 3.6 ± 1.3 | 0.32 ± 0.08 | 11 ± 2 | 4.8 ± 1.0 | 523 ± 69 |
| one per sequence, batch 64 | 64 | 3.4 ± 1.2 | 0.25 ± 0.04 | 13 ± 6 | 7.0 ± 4.3 | 632 ± 89 |
| last 8 per sequence | 128 | 2.9 ± 0.8 | 0.25 ± 0.02 | 8.9 ± 2.4 | 3.2 ± 1.5 | 379 ± 34 |

Reading. Fewer starts give a slightly better first minute (0.25-0.32 against
0.18 upright) but no faster t90, and the policy ends up roughly twice as
restless (crossings and speed). Updates per minute barely change (~7-8k
train fps in all arms), so in our setting the starts cost little and buy a
lower-variance actor-critic update; DreamerV4's one-start design pays off
where imagination is the expensive part. Not adopted.
