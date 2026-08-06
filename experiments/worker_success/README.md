# Worker success (worker goal-reward across training)

Answers a `\todo` from the paper (`code/26_04_HRL-paper/sections/motivation.tex`,
Sec. "Worker success"): "Across training, worker success is relatively low
[Check this with actual numbers]." In Director, the worker is rewarded by
`cosine_max(goal, deter)` (`dreamerv3/hrl/tensors.py:42`,
`goal_reward_cosine_max`) between the manager-proposed goal and the worker's
achieved RSSM deterministic state — this **is** the worker-success metric,
logged during training as `train/wkr_goal_rew` (`dreamerv3/hrl/losses.py:109`,
`imag_loss_wkr`). `cosine_max(x, x) = 1`, so `1.0` is the ceiling for a
worker that has exactly matched its goal.

No new rollout is needed: `train/wkr_goal_rew` is already logged at every
training step in each run's `metrics.jsonl`, so this experiment just reads
the existing logs. Covers all **4 project environments** across **however
many training seeds have finished** — as of 2026-08-05, all **5** —
via the same checkpoint discovery ([`../baselines_common.py`](../baselines_common.py))
used by `../goal_code_struct_corr/`.

**Result** (pure Director baselines, full 4M-step runs, mean ± std across
the 5 seeds trained):

| Env | Overall mean | Late 10% mean | Final value |
|---|---|---|---|
| Cartpole Swingup | 0.38 ± 0.03 | 0.49 ± 0.02 | 0.54 ± 0.04 |
| Hopper Hop | 0.33 ± 0.06 | 0.49 ± 0.07 | 0.49 ± 0.07 |
| Acrobot Swingup | 0.28 ± 0.02 | 0.33 ± 0.05 | 0.34 ± 0.04 |
| Cheetah Run | 0.59 ± 0.03 | 0.76 ± 0.03 | 0.78 ± 0.02 |

(see `results/summary.json` for the exact numbers; re-run the two scripts
below any time a further seed lands.)

Worker success rises over training in all four environments (the worker is
learning to reach manager-proposed goals better as training proceeds) but
never gets close to the `1.0` ceiling, and plateaus well below it: cheetah is
the best case; acrobot plateaus lowest. Seed-to-seed variability (the std
column above) is small relative to the gap to the ceiling in every
environment. This confirms the paper's qualitative claim — the worker only
ever partially achieves the goals it is given — and gives it concrete,
multi-seed numbers. See `worker_success.svg`/`.pdf` for the full training
curves (mean line ± shaded std band per environment).

## Files

- `../baselines_common.py` — shared checkpoint discovery (4 tasks x N
  completed seeds from the `e390`+ baseline series).
- `extract_worker_success.py` — reads `train/wkr_goal_rew` from every
  discovered baseline's `metrics.jsonl` (no cluster job required — plain
  stdlib `json`, runs on the login node), computes per-seed overall/early/
  late summary stats plus cross-seed mean/std, writes
  `results/{tag}_{task}_seed{N}.json` (full step/value series per seed) and
  `results/summary.json` (cross-seed stats only, grouped by task).
- `make_figure.py` — hand-rolled SVG line plot (same no-matplotlib
  workaround as `../goal_code_struct_corr/make_figure.py`). Interpolates
  each seed's curve onto a common step grid per task (so seeds with slightly
  different logging cadences/lengths become directly comparable), then plots
  the cross-seed mean line ± 1 std shaded band for each of the 4
  environments in one panel. Reads `results/*.json`, writes
  `worker_success.{svg,pdf}` here and overwrites the copy in
  `code/26_04_HRL-paper/figures/introduction/worker_success.pdf`.
- `results/*.json` — committed outputs, so the figure and table can be
  regenerated without re-reading the bucket/work dirs.
- `results/archive_e123_e124_e180_single_seed/` — the old single-seed,
  3-environment results this experiment used before being extended to all 4
  environments x multiple seeds. Kept for reference only.

## Reproducing from scratch

```bash
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/worker_success
python3 extract_worker_success.py     # reads metrics.jsonl for all discovered
                                       # baselines, writes results/*.json
python3 make_figure.py                # regenerates the figure + updates the paper's copy
cd ../../../26_04_HRL-paper && ./build.sh
```

Both scripts are plain-stdlib-`json`/`numpy` (no `jax`/`dreamerv3_env`
needed) and run fast (a few seconds total) on the login node — no SLURM job.

## Checkpoints used

Same discovery and baseline series as `../goal_code_struct_corr/` — see that
experiment's README for the full per-seed completion table (`e390`-`e409`,
fixed `K=8`, no masking/variable-goal-length, `goal_struct_weight=0.0`).

No code changes required — `train/wkr_goal_rew` is logged unconditionally by
`imag_loss_wkr` in every training run.
