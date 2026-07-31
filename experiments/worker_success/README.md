# Worker success (worker goal-reward across training)

Answers a `\todo` from the paper (`code/26_04_HRL-paper/sections/introduction.tex`,
Sec. "Efficiency: Worker success"): "Across training, worker success is
relatively low [Check this with actual numbers]." In Director, the worker is
rewarded by `cosine_max(goal, deter)` (`dreamerv3/hrl/tensors.py:42`,
`goal_reward_cosine_max`) between the manager-proposed goal and the worker's
achieved RSSM deterministic state — this **is** the worker-success metric,
logged during training as `train/wkr_goal_rew` (`dreamerv3/hrl/losses.py:109`,
`imag_loss_wkr`). `cosine_max(x, x) = 1`, so `1.0` is the ceiling for a
worker that has exactly matched its goal.

No new rollout was needed: `train/wkr_goal_rew` is already logged at every
training step in each run's `metrics.jsonl`, so this experiment just reads
the existing archived logs.

**Result** (pure Director baselines, same three checkpoints as
`../goal_code_struct_corr/`):

| Env | Overall mean | Early 10% mean | Late 10% mean | Final value |
|---|---|---|---|---|
| Hopper Hop (e124) | 0.24 | 0.13 | 0.48 | 0.48 @ 2.14M steps |
| Cheetah Run (e123) | 0.57 | 0.40 | 0.74 | 0.80 @ 2.61M steps |
| Acrobot Swingup (e180) | 0.29 | 0.19 | 0.35 | 0.37 @ 3.72M steps |

Worker success rises over training in all three environments (the worker is
learning to reach manager-proposed goals better as training proceeds) but
never gets close to the `1.0` ceiling, and plateaus well below it: cheetah is
the best case and still averages 0.57 over the whole run (final 0.80);
hopper and acrobot plateau in the 0.35–0.48 range. This confirms the paper's
qualitative claim — the worker only ever partially achieves the goals it is
given — and gives it concrete numbers. See `worker_success.svg`/`.pdf` for
the full training curves (noisy, non-monotonic — cheetah in particular has
large dips around step ~2.2M coinciding with the manager's own instability
in that window, consistent with `goal_struct_corr`'s finding that goal-code
geometry is only moderately structured).

## Files

- `extract_worker_success.py` — reads `train/wkr_goal_rew` from each
  baseline's `metrics.jsonl` (`/bucket/.../<run>/logdir/metrics.jsonl`,
  already present from training — no cluster job required), computes
  overall/early/late summary stats, writes `results/<tag>.json` (full
  step/value series) and `results/summary.json` (stats only).
- `make_figure.py` — hand-rolled SVG line plot (same no-matplotlib
  workaround as `../goal_code_struct_corr/make_figure.py`: this cluster's
  `rl_env` has a broken matplotlib install) + `rsvg-convert` to PDF. Reads
  `results/*.json`, writes `worker_success.{svg,pdf}` here and overwrites
  the copy in `code/26_04_HRL-paper/figures/introduction/worker_success.pdf`.
- `results/*.json` — committed outputs, so the figure and table can be
  regenerated without re-reading the bucket.

## Reproducing from scratch

```bash
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/worker_success
python3 extract_worker_success.py     # reads bucket metrics.jsonl, writes results/*.json
python3 make_figure.py                # regenerates the figure + updates the paper's copy
cd ../../../26_04_HRL-paper && pdflatex main.tex && pdflatex main.tex
```

Both scripts are plain-stdlib-`json`/`numpy` (no `jax`/`dreamerv3_env`
needed) and run fine on the login node.

## Checkpoints used

Same pure-Director baselines as `../goal_code_struct_corr/` (fixed `K=8`, no
masking/variable-goal-length, `goal_struct_weight=0.0`):

| Tag | Run dir (under `/bucket/DoyaU/vasilache/bucket/results/dreamerv3/`) |
|---|---|
| e124_hopper | `e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29` |
| e123_cheetah | `e123_dmc_cheetah_run_director_baseline_20260706_155117_Vk7oa4` |
| e180_acrobot | `e180_dmc_acrobot_swingup_director_baseline_j4660140` |

No code changes required — `train/wkr_goal_rew` is logged unconditionally by
`imag_loss_wkr` in every training run.
