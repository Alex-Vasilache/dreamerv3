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
| 2 | world-model dropout 0.05 / 0.1 / 0.2 (uncapped), 0.1 with the schedule | does it stop the drift and let data be reused harder? | done: no early speed-up, slightly calmer late; candidate for step 6 (log 11-10) |
| 3 | observational dropout 10 / 30 % (open-loop steps in training) | better imagination from the same data? | X, IAS, e43840-e43879 (unc, sched, unc o0.1/0.3, sched o0.3); done: 0.1 neutral, 0.3 worse; not adopted (log 11-10) |
| 4 | N rollouts per start (2, 4) | does spare GPU buy speed? | V, IAS, e43880-e43911 (unc/sched x N 2/4); done: N = 4 slower, N = 2 within noise; weak candidate (log 11-10) |
| 4b | DreamerV4's one rollout per context: imag_last 1 (b16, b64), imag_last 8, vs every step | do fewer, more diverse starts learn as fast for less compute? | done: not better (fewer starts = more restless policy); not adopted (log 11-10) |
| 5 | winners on commands; best to live | fastest from-scratch recipe | |
| 6 | reanalysis: what worked, what did not, why; combine the parts that help together (factorial on the 2-3 best), confirm n = 12 at three GPU speeds | most improvement | reanalysis logged 11-10; round M (dropout 0.05 x 2 rollouts, n = 12, V100/P100/IAS) running |

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

### Tue 11-10: Y, world-model dropout (step 2)

Sim, Saion V100, 60 min, n = 8 per arm (e43800-e43831); the no-dropout
controls are Z1's uncapped and schedule arms (same GPU type and seeds).
`agent.dyn.rssm.drop p` = dropout on the hidden layers of the RSSM core,
posterior and prior, during training only. Mean ± SD. Figure `figs/study/y.png`.

| arm | t90 (min) | upright min 2 | crossings/min, min 6-15 | crossings/min, min 30-60 | \|speed\| mm/s, min 30-60 |
|---|---|---|---|---|---|
| uncapped, no dropout (Z1) | 3.3 ± 0.4 | 0.78 ± 0.07 | 5.1 ± 1.8 | 2.0 ± 1.2 | 283 ± 36 |
| uncapped, dropout 0.05 | 3.0 ± 0.7 | 0.80 ± 0.10 | 4.7 ± 1.8 | 1.1 ± 0.6 | 256 ± 22 |
| uncapped, dropout 0.1 | 3.1 ± 0.3 | 0.78 ± 0.08 | 7.1 ± 2.5 | 1.4 ± 0.9 | 262 ± 22 |
| uncapped, dropout 0.2 | 3.1 ± 1.2 | 0.82 ± 0.07 | 5.2 ± 2.1 | 1.7 ± 0.7 | 292 ± 16 |
| schedule, no dropout (Z1) | 3.0 ± 0.7 | 0.86 ± 0.04 | 7.3 ± 2.0 | 1.6 ± 0.7 | 286 ± 21 |
| schedule, dropout 0.1 | 3.1 ± 0.3 | 0.82 ± 0.06 | 5.4 ± 1.6 | 1.2 ± 0.6 | 281 ± 20 |

Reading. Dropout does not speed up the first minutes (t90 3.0-3.3 in every
arm). Late, every dropout arm sits slightly below its control (uncapped 1.1-1.7
against 2.0; schedule 1.2 against 1.6 crossings/min), smallest at 0.05; the
differences are within one SD at n = 8. Cheap (no speed cost) and harmless, a
small late-stability gain: a candidate for the combination round (step 6), not
a speed-up on its own.

### Tue 11-10: X lost to the IAS time limit

The IAS launcher still had the 1 h 25 min limit sized for 30-min runs; 33 of
the 40 one-hour X runs timed out (one more hit the 5 GB memory cap) and the
final copy back from the node's /tmp did not finish before the kill, so their
data is gone. Fixed in `sbatch/ias/run_sim_ias.sbatch`: 2 h 30 min, 7 GB
(the largest finished run peaked at 4.0 GB), and the logdir is copied back
every 5 min. The 33 lost X runs (array 202005) and all of V (202006, V's two
started runs cancelled) were resubmitted, 4 concurrent each.

### Tue 11-10 evening: X, observational dropout (step 3)

Sim, IAS (RTX 3090 / A5000, uncapped ratio ~330), 60 min, n = 8 per arm
(e43840-e43879; 33 reruns, see above; one control run has 39 min).
`agent.dyn.rssm.obsdrop p` = in training, a fraction p of steps take the
prior instead of the posterior (the model predicts open-loop there).
Mean ± SD. Figure `figs/study/x.png`.

| arm | t90 (min) | upright min 2 | crossings/min, min 6-15 | crossings/min, min 30-60 | \|speed\| mm/s, min 30-60 |
|---|---|---|---|---|---|
| uncapped | 2.6 ± 0.5 | 0.86 ± 0.12 | 6.4 ± 1.9 | 2.4 ± 1.1 | 344 ± 53 |
| uncapped, obs. dropout 0.1 | 3.0 ± 0.5 | 0.82 ± 0.07 | 4.3 ± 1.3 | 2.3 ± 1.2 | 331 ± 24 |
| uncapped, obs. dropout 0.3 | 4.0 ± 1.0 | 0.68 ± 0.10 | 12 ± 3 | 4.8 ± 1.1 | 472 ± 49 |
| schedule | 2.8 ± 0.4 | 0.82 ± 0.10 | 7.2 ± 2.8 | 1.3 ± 0.6 | 272 ± 17 |
| schedule, obs. dropout 0.3 | 3.8 ± 0.8 | 0.64 ± 0.11 | 12 ± 3 | 2.5 ± 1.2 | 414 ± 49 |

Reading. 0.1 is neutral; 0.3 slows learning by ~1 min and leaves a restless
policy. Not adopted. On IAS's faster GPUs the schedule does help late (1.3
against 2.4 crossings/min uncapped), unlike on V100 (Z1): the late drift
grows with the replay ratio, and the schedule stays in the pick.

### Tue 11-10 evening: V, N imagination rollouts per start (step 4)

Same setting, n = 8 per arm (e43880-e43911), controls = X's uncapped and
schedule arms. `agent.imag_repeat N` imagines N rollouts (different sampled
actions) from each start and averages the actor-critic losses over them. It
costs updates: 77 % (N = 2) and 56 % (N = 4) of the control's learner steps.
Figure `figs/study/v.png`.

| arm | t90 (min) | upright min 2 | crossings/min, min 1-5 | crossings/min, min 30-60 | \|speed\| mm/s, min 30-60 |
|---|---|---|---|---|---|
| uncapped, N = 1 | 2.6 ± 0.5 | 0.86 ± 0.12 | 33 ± 7 | 2.4 ± 1.1 | 344 ± 53 |
| uncapped, N = 2 | 2.8 ± 0.4 | 0.81 ± 0.08 | 36 ± 3 | 1.7 ± 0.6 | 267 ± 29 |
| uncapped, N = 4 | 3.5 ± 0.5 | 0.66 ± 0.08 | 47 ± 6 | 1.4 ± 0.9 | 258 ± 27 |
| schedule, N = 1 | 2.8 ± 0.4 | 0.82 ± 0.10 | 34 ± 5 | 1.3 ± 0.6 | 272 ± 17 |
| schedule, N = 2 | 2.4 ± 0.5 | 0.88 ± 0.07 | 35 ± 2 | 1.7 ± 0.7 | 286 ± 24 |
| schedule, N = 4 | 3.0 ± 0.0 | 0.76 ± 0.04 | 41 ± 6 | 1.7 ± 0.6 | 288 ± 25 |

Reading. N = 4 slows the start (fewer world-model updates, which Z2 says is
the bottleneck). N = 2 is within noise early (2.4 against 2.8 min with the
schedule, 2.8 against 2.6 uncapped) and calmer late when uncapped (1.7
against 2.4), i.e. it acts like a milder ratio schedule. Weak candidate.

### Tue 11-10 evening: reanalysis so far (step 6, first part)

What worked: the weekend pick itself (fear critic, stillness, batch 16, ratio
schedule) is up in ~3 min in every round and settles to ~1-2 crossings/min.
The schedule matters on fast GPUs (IAS: 1.3 against 2.4 late), not on V100.

What did not: nothing tested shortened the first minutes. Z2 says why: the
world model is the bottleneck, and a fresh actor-critic learns in under a
minute once a model exists. Dropout in the world model (Y), open-loop steps
(X), fewer imagination starts (U) and more rollouts per start (V) all leave
t90 at 2.4-3.3 min; the ones that cost world-model updates (X 0.3, V N = 4)
make it slower. Freezing parts (Z1) breaks the coupling between latents and
policy and is much worse.

Small late gains: world-model dropout 0.05 (Y) and 2 rollouts per start (V),
each within one SD at n = 8. Round M tests them together on the pick, 2 x 2,
n = 12, at three GPU speeds: Saion V100 (e44000-e44047), P100 (e44050-e44097),
IAS (e44100-e44147, array 204440, snapshot `code_m` = 4023756). Saion runs use
the current tree, which includes Alex's drive-home stall stop (2cc011e), so
compare M arms only with M's own `pick` arm.

Open for after M: if the factorial shows nothing beyond noise, the pick stays
as is, and the next lever is the world model's data efficiency in the first
2 min directly (e.g. extra world-model-only updates early, which `wm_ratio`
already allows, or a smaller/faster-fitting model). Step 5 (commands, live)
then uses whatever M picks.

### Tue 11-10 night: M, dropout x rollouts on the pick (step 6)

Sim, 60 min, n = 12 per arm at each of Saion V100 (e44000-e44047), P100
(e44050-e44097) and IAS (e44100-e44147), all with the ratio schedule (figure
`figs/study/m.png`). Late = crossings/min over minutes 30-60.

| arm | t90 V100 / P100 / IAS (min) | late V100 / P100 / IAS | t90, all (n = 36) | late, all (n = 36) |
|---|---|---|---|---|
| pick | 2.9 / 3.1 / 2.8 | 2.0 / 2.0 / 1.9 | 2.9 ± 0.5 | 2.0 ± 1.0 |
| + world-model dropout 0.05 | 2.8 / 3.1 / 2.8 | 1.7 / 1.5 / 1.4 | 2.9 ± 0.5 | 1.6 ± 0.6 |
| + 2 rollouts per start | 3.2 / 3.9 / 2.9 | 2.1 / 1.4 / 2.1 | 3.4 ± 0.6 | 1.9 ± 0.9 |
| + both | 3.5 / 3.7 / 2.5 | 2.1 / 1.6 / 1.4 | 3.2 ± 0.8 | 1.7 ± 0.8 |

Reading. Dropout 0.05 is the only consistent change: calmer late at all
three GPU speeds (about 20 % fewer crossings, and less spread) at no cost in
speed. Two rollouts per start costs half a minute to a minute on the slower
GPUs and adds nothing late, alone or with dropout. Step 6b recipe: the pick
plus `--agent.dyn.rssm.drop 0.05`. All M arms keep the hand-set ratio
schedule; whether the drift can be controlled without it is round N and the
learning-progress control below.

### Tue 11-10 night: N, the late drift without the ratio schedule

Alex (11-10): the ratio schedule is handcrafted (a switch at a fixed minute
to a fixed ratio); the recipe has to work without it. Two routes, run in
parallel:

1. Regularise instead of throttle (round N, IAS, the fastest learner where
   the drift shows; e44200-e44231, array 204855, snapshot `code_n` = 04a47da,
   n = 8, 60 min): uncapped, scheduled, and uncapped with dropout 0.05 plus 2
   or 4 rollouts per start (V: N = 4 was the calmest uncapped arm, 1.4).
2. Let the learner set its own ratio from learning progress
   (`online_progress`, new). Every 20 s it scores the world model on a batch
   of held-out recent experience (10 % of chunks, kept out of training for
   2 min, then released, `replay.holdout`), trains on, and scores the same
   batch again with the same seed. If the held-out loss fell, the ratio grows
   x1.25, otherwise it shrinks x0.8, within [32, the GPU's pace]. No switch
   time and no target ratio: the learner trains as long as training still
   improves prediction of experience it has not seen, so the ratio follows
   the data rather than the GPU. Smoke e44232 (V100, 8 min).

### Wed 11-11 morning: N and P results

Figure `figs/study/np.png`. Late = crossings/min over minutes 30-60, mean ±
SD. Achieved ratio = `fps/train` / 65 Hz over the second half of the run.

Round N (IAS A5000, n = 8 per arm):

| arm | t90 (min) | late | achieved ratio |
|---|---|---|---|
| uncapped | 2.8 ± 0.7 | 2.6 ± 0.9 | 287 |
| hand-set schedule | 2.9 ± 0.4 | 1.7 ± 0.7 | 114 |
| uncapped + dropout 0.05 + 2 rollouts | 2.6 ± 0.5 | 1.6 ± 0.9 | 248 |
| uncapped + dropout 0.05 + 4 rollouts | 3.8 ± 1.0 | 1.2 ± 0.8 | 167 |

Round P (Saion, e44300-e44371, V100 and P100 n = 8 each, pooled n = 16):

| arm | t90 V100 / P100 | late V100 / P100 | t90, pooled | late, pooled |
|---|---|---|---|---|
| uncapped | 3.0 / 3.1 | 1.8 / 1.7 | 3.1 ± 0.4 | 1.8 ± 0.5 |
| hand-set schedule | 2.8 / 3.1 | 1.7 / 2.0 | 2.9 ± 0.6 | 1.8 ± 0.8 |
| learning-progress control | 3.1 / 3.1 | 3.4 / 3.7 | 3.1 ± 0.6 | 3.6 ± 1.5 |
| learning-progress + dropout 0.05 | 3.2 / 3.1 | 2.8 / 3.5 | 3.2 ± 0.4 | 3.2 ± 1.3 |

Reading.

- The drift needs a fast learner. At Saion's pace (uncapped ratio about
  250-280) uncapped and scheduled are the same late (1.8 vs 1.8); on IAS
  (about 290 late, up to 570 early) uncapped drifts (2.6 vs 1.7).
- Dropout 0.05 plus 2 rollouts removes the IAS drift without any ratio limit:
  1.6 late, t90 2.6, at a ratio of about 250. Rollouts also cost compute per
  update, so part of this is a lower achieved ratio (287 to 248), but not
  most: the schedule needed 114 for the same late value. Four rollouts is
  calmer still (1.2) but costs a minute to t90, through a ratio of 167.
- Learning-progress control as built is worse than both baselines (3.6 late,
  twice uncapped). Held-out world-model loss stops improving after about
  15-20 min, so the controller drives the ratio to its floor of 32-40 and
  holds it there (lower left of the figure); a policy trained that rarely
  drifts more, not less. The probe also costs pace (start ratio 239 / 177 vs
  about 253 / 280 uncapped on V100 / P100). World-model progress is the wrong
  signal for how much the actor-critic should train, so this route is parked.
- Recipe without a hand-set ratio (sim): the pick, uncapped, plus
  `--agent.dyn.rssm.drop 0.05 --agent.imag_repeat 2` (`sbatch/wk/gen.py`
  N_ARMS `unc_d05_r2`). IAS round P (array 204901, same four P
  arms, n = 8) repeats the comparison on the fast learner; results below if
  in before noon.
