# Goal-code / goal-space structure correlation

Answers a question from the paper (`code/26_04_HRL-paper/sections/motivation.tex`,
Sec. "Goal Code Learning", Figure "Goal-code geometry only partly tracks
goal-space geometry"): Director's goal autoencoder maps a continuous RSSM state
`s_t` (`deter`, 1024-d) to a discrete `(L=8, C=8)` categorical code `z`. Does a
small change in `s_t` correspond to a small change in `z`, and vice versa — or
is the code space arbitrary with respect to goal-space geometry?

Covers all **4 project environments** (cartpole swingup, hopper hop, acrobot
swingup, cheetah run) across **however many training seeds have finished** —
as of 2026-08-05 all **5** (seeds 0-4, `e390`-`e409`) are complete.
Checkpoints are discovered automatically by
[`../baselines_common.py`](../baselines_common.py) — re-running the pipeline
later picks up newly-finished seeds with no code changes.

**Result** (pure Director baselines, full-length 4M-step, no
`goal_struct_weight`/`goal_struct_adapt` ever applied during training — this
reads the *unforced* baseline geometry): goal-space and goal-code similarity
are positively but only moderately correlated. Hard-code (argmax, the object
the manager actually samples via REINFORCE) Pearson r, mean ± std **across
seeds** (each seed's own value already averaged over its 10 measurement
batches):

Numbers are printed by `make_figure.py` (which reads directly off the panel
annotations, `r = mean ± std (n=<seeds> seeds)`) and also shown on the figure
itself — regenerate any time more seeds land.

Low within-seed batch-to-batch std relative to the between-env spread means
this is a stable property of each checkpoint, not sampling noise. See the
paper section above for the full write-up.

## Files

- `../baselines_common.py` — shared checkpoint discovery (4 tasks x N
  completed seeds from the `e390`+ baseline series), used by both this
  experiment and `../worker_success/`.
- `diag_goal_struct_corr.py` — the measurement. Loads a trained checkpoint,
  drives the live policy across parallel env workers, collects
  `(deter, soft-code)` pairs at every step via a diagnostic hook in the agent
  (see "Code changes" below), splits the collected pool into disjoint batches,
  computes the pairwise-`cosine_max`-Gram-matrix Pearson correlation (soft and
  hard code) per batch, and additionally computes a **fixed-bin trend curve**
  (goal-code similarity vs. goal-space similarity, on global bin edges shared
  across every seed/task — see `TREND_XMIN`/`TREND_XMAX`/`TREND_NBINS`) so
  different seeds' curves can be averaged point-for-point at the figure stage.
  Same `--seed 0` (state-sampling seed), `--n_batches 10`, `--n_states`
  (defaults to the run's own `batch_size x batch_length`) for every checkpoint
  — identical sampling methodology applied to every seed, so seed-to-seed
  differences in the result reflect the checkpoints, not the measurement.
- `submit_diag_goal_struct_corr.sh` — discovers all currently-complete
  baseline seeds via `baselines_common.discover_baselines()`, writes
  `manifest.tsv` (one row per task x seed), and submits
  `run_diag_goal_struct_corr.sbatch` as a SLURM array sized to match. Re-run
  any time; the array just grows as more seeds finish.
- `run_diag_goal_struct_corr.sbatch` — SLURM array job, `gpu-v100`,
  ~2-3 min/task (`n_batches=10`, `n_envs=8` parallel workers per task) — all
  array tasks run **concurrently** across separate GPU allocations, so total
  wall time stays ~2-3 min regardless of how many (task, seed) pairs are in
  the manifest. `-t 0-00:15:00` gives headroom without over-provisioning.
- `make_figure.py` — hand-rolled SVG plotter (no matplotlib: this cluster's
  `rl_env` has a broken matplotlib install, missing system `libpng15` on both
  login and compute nodes) + `rsvg-convert` to PDF. Reads
  `results/{tag}_{task}_seed{N}.npz`, groups by task (4 panels), pools the
  scatter cloud from all seeds, plots the cross-seed mean trend line ± 1 std
  shaded band, and reports `r = mean ± std (n=<seeds> seeds)`. Writes
  `goal_code_struct_corr.{svg,pdf}` here and overwrites the copy in
  `code/26_04_HRL-paper/figures/introduction/goal_code_struct_corr.pdf`.
- `results/*.npz` — committed outputs, so the figure can be regenerated
  (`python3 make_figure.py`) without re-running on the cluster. Each contains:
  `corr_soft_mean/std`, `corr_hard_mean/std`, `corr_soft_per_batch`,
  `corr_hard_per_batch` (per-batch arrays), `sd_pairs`/`sz_pairs`/`sh_pairs`
  (upper-triangle pairwise similarities, batch 0, for the scatter), and
  `trend_x`/`trend_y`/`trend_counts` (fixed-bin trend curve over a 4096-state
  subsample of the full pool).
- `results/archive_e123_e124_e180_single_seed/` — the old single-seed,
  3-environment (hopper/cheetah/acrobot only) results this experiment used
  before it was extended to all 4 environments x multiple seeds. Kept for
  reference; not read by `make_figure.py` (different filename convention).

## Reproducing from scratch

```bash
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
./submit_diag_goal_struct_corr.sh              # discovers checkpoints, submits the array job
# wait for all array tasks (squeue -u $(whoami) -n diag_structcorr); ~2-3 min total
python3 make_figure.py                         # regenerates the figure + updates the paper's copy
cd ../../../26_04_HRL-paper && ./build.sh
```

To point at different checkpoints, edit `../baselines_common.py` (the
`390 + seed*4 + task_index` numbering, or `TASKS`), or run
`diag_goal_struct_corr.py` directly on any run dir:
```bash
python -u diag_goal_struct_corr.py --run_dir <path_to_run_dir> \
    --n_batches 10 --n_envs 8 --out results/my_run.npz
```

## Checkpoints used

Pure Director baselines (`RECIPE=director` defaults: fixed `K=8`, no
masking/variable-goal-length, `goal_struct_weight=0.0`, `goal_struct_adapt:
false`), full 4M-step training, from the `e390`-`e409` series
(`EXPERIMENTS.md`). A checkpoint counts as "usable" once its `metrics.jsonl`
reaches ≥99% of its configured `run.steps` budget — see
`baselines_common.is_complete`. As of 2026-08-05, all 5 seeds are complete:

| Task | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Cartpole Swingup | e390 ✅ | e394 ✅ | e398 ✅ | e402 ✅ | e406 ✅ |
| Hopper Hop | e391 ✅ | e395 ✅ | e399 ✅ | e403 ✅ | e407 ✅ |
| Acrobot Swingup | e392 ✅ | e396 ✅ | e400 ✅ | e404 ✅ | e408 ✅ |
| Cheetah Run | e393 ✅ | e397 ✅ | e401 ✅ | e405 ✅ | e409 ✅ |

These runs' replay was deleted at archive time (`SKIP_REPLAY=1`, see
`copy_dreamerv3_runs_to_bucket.sh` in the top-level CLAUDE.md) once archived,
so this measurement drives a **live on-policy rollout** with the restored
checkpoint rather than reading states from replay. As of 2026-08-03 the
`e390`-`e409` runs have not yet been archived and are read directly from
`/work` — `baselines_common._find_run_dir` checks `/bucket` first, then
`/work`, so this needs no changes once they are archived.

## Measuring at multiple points in training (not just the final checkpoint)

`elements.Checkpoint` (the framework's checkpoint class) defaults to
`keep=1`: every save deletes the previous one. None of the run loops
(`train.py`, `online.py`, `train_eval.py`) override this, so **the
already-completed baselines (e390-e397) only have their final checkpoint —
their mid-training states are permanently gone.** There was no way to
retroactively recover them.

For baselines that are **still training** (e398-e409), `watch_and_diag_running.py`
works around this by watching each run's `metrics.jsonl` and, the moment it
crosses 1M/2M/3M env steps, copying that run's *current* `ckpt/latest` folder
aside (to `/work/DoyaU/vasilache/work/goal_struct_corr_snapshots/` -- **not**
under the repo: each snapshot is a full ~1GB checkpoint copy, and `/apps` is
a 50GB filesystem that filled to 100% and crashed this watcher once ~20GB of
snapshots had accumulated there, 2026-08-03) before the next periodic save (`run.save_every`,
900s) rotates it out from under us, then submitting a
`run_diag_struct_corr_snapshot.sbatch` job against that frozen snapshot.
Results land as `results/{tag}_{task}_seed{N}_step{1,2,3}M.npz` (with a
`train_step` field recording the step the snapshot was taken at), alongside
the existing final-checkpoint results
(`results/{tag}_{task}_seed{N}.npz`, effectively the ~4M-step point).

```bash
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
nohup python3 -u watch_and_diag_running.py --poll_interval 300 > watch.log 2>&1 &
# state (which milestones already captured, job ids) persists to
# results/milestones_state.json -- safe to re-run after an interruption,
# already-captured milestones are skipped.
```

Once some snapshot results exist:
```bash
python3 plot_struct_corr_vs_step.py    # writes struct_corr_vs_step.{svg,pdf}:
                                        # hard-code r vs. training step, one
                                        # line per (task, seed) -- seeds aren't
                                        # pooled here since only whichever
                                        # seeds happened to be mid-training
                                        # when the watcher ran have snapshots.
```

This only applies going forward — it cannot backfill e390-e397's history.
For *future* baseline sweeps where a dense, cheap training-time trajectory is
wanted from step 0 without the snapshot/rollout machinery at all, the
better-scaling alternative is to log the correlation as an in-training
scalar (reusing the batch already sampled for the goal-VAE loss each train
step, the same way `train/wkr_goal_rew` already works for `../worker_success/`)
rather than driving a live rollout after the fact — not yet implemented.

## Code changes this experiment depends on

Two small, opt-in, default-`False` additions to the main agent code (zero
effect on training/production runs unless explicitly enabled):

1. **`dreamerv3/configs.yaml:392`** — new flag `agent.policy_struct_diag:
   False`.
2. **`dreamerv3/agent.py`, `Agent.policy()`** (~line 618) — when
   `policy_struct_diag=True`, encodes the *current real state* (not a
   manager proposal) through the goal encoder at every policy step and emits
   `out['log/struct_diag_deter']` / `out['log/struct_diag_probs']` (pure
   forward pass, `stop_gradient`d elsewhere in the codebase already covers
   this path — no gradients, no effect on the action actually taken).
3. **`dreamerv3/agent.py`, `Agent.policy_keys`** (~line 472-483) — `goal_enc`
   is normally train/report-only (the live policy step only *decodes* the
   manager's sampled code via `goal_dec`); (2) needs `goal_enc`'s params
   available in the policy-side param group too, so `policy_keys` includes
   `goal_enc` in its regex **only when `policy_struct_diag=True`** — this
   doesn't touch the actor/learner param sync for any real training run.

These are the only codebase changes required; everything else lives in this
folder.
