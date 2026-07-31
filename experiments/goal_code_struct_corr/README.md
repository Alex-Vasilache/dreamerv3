# Goal-code / goal-space structure correlation

Answers a question from the paper (`code/26_04_HRL-paper/sections/introduction.tex`,
Sec. "Efficiency: Goal Code Reuse", Figure "Goal-code geometry only partly tracks
goal-space geometry"): Director's goal autoencoder maps a continuous RSSM state
`s_t` (`deter`, 1024-d) to a discrete `(L=8, C=8)` categorical code `z`. Does a
small change in `s_t` correspond to a small change in `z`, and vice versa — or
is the code space arbitrary with respect to goal-space geometry?

**Result** (pure Director baselines, no `goal_struct_weight`/`goal_struct_adapt`
ever applied during training, so this reads the *unforced* baseline geometry):
goal-space and goal-code similarity are positively but only moderately
correlated. Hard-code (argmax, the object the manager actually samples via
REINFORCE) Pearson r, mean ± std over 10 independent batches of N=1024 states
each:

| Env | r (hard code) |
|---|---|
| Hopper Hop (e124) | 0.42 ± 0.01 |
| Cheetah Run (e123) | 0.26 ± 0.02 |
| Acrobot Swingup (e180) | 0.31 ± 0.01 |

Low batch-to-batch std relative to the between-env spread means this is a
stable property of each checkpoint, not sampling noise. See the paper section
above for the full write-up and the open `\todo` on whether this level of
correlation is actually a problem or expected given the geometry of the goal
state space.

## Files

- `diag_goal_struct_corr.py` — the measurement. Loads a trained checkpoint,
  drives the live policy across parallel env workers, collects
  `(deter, soft-code)` pairs at every step via a diagnostic hook in the agent
  (see "Code changes" below), splits the collected pool into disjoint batches,
  and computes the pairwise-`cosine_max`-Gram-matrix Pearson correlation
  (soft and hard code) per batch.
- `run_diag_goal_struct_corr.sbatch` — SLURM array job (`--array=0-2`, one
  task per baseline checkpoint below), `gpu-v100`, ~2-3 min/task. Each task
  additionally runs 8 parallel env workers internally, so getting 10 batches
  x 1024 states (10,240 states) per checkpoint takes about the same wall
  time the single-worker version previously spent on one batch of 1000.
- `make_figure.py` — hand-rolled SVG plotter (no matplotlib: this cluster's
  `rl_env` has a broken matplotlib install, missing system `libpng15` on both
  login and compute nodes) + `rsvg-convert` to PDF. Reads `results/*.npz`,
  writes `goal_code_struct_corr.{svg,pdf}` here and overwrites the copy in
  `code/26_04_HRL-paper/figures/introduction/goal_code_struct_corr.pdf`.
- `results/*.npz` — committed outputs from the run described below, so the
  figure can be regenerated (`python3 make_figure.py`) without re-running on
  the cluster. Each contains: `corr_soft_mean/std`, `corr_hard_mean/std`,
  `corr_soft_per_batch`, `corr_hard_per_batch` (both length-`n_batches`
  arrays), and `sd_pairs`/`sz_pairs`/`sh_pairs` (upper-triangle pairwise
  similarities for batch 0 only, used for the scatter plot).

## Reproducing from scratch

```bash
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
sbatch run_diag_goal_struct_corr.sbatch        # writes into ./results/ by default
# wait for all 3 array tasks (squeue -u $(whoami)); each ~2-3 min
python3 make_figure.py                         # regenerates the figure + updates the paper's copy
cd ../../../26_04_HRL-paper && pdflatex main.tex && pdflatex main.tex
```

To point at different checkpoints or knobs, edit the `RUNS` map in the
sbatch script, or run `diag_goal_struct_corr.py` directly:
```bash
python -u diag_goal_struct_corr.py --run_dir <path_to_run_dir> \
    --n_batches 10 --n_envs 8 --out results/my_run.npz
# --n_states defaults to batch_size x batch_length from the run's own config.yaml
# (1024 = 16 x 64 for all three baselines below)
```

## Checkpoints used

Pure Director baselines only (`RECIPE=director` defaults: fixed `K=8`, no
masking/variable-goal-length, `goal_struct_weight=0.0`, `goal_struct_adapt:
false` — see each run's `config.yaml`):

| Tag | Run dir (under `/bucket/DoyaU/vasilache/bucket/results/dreamerv3/`) |
|---|---|
| e124_hopper | `e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29` |
| e123_cheetah | `e123_dmc_cheetah_run_director_baseline_20260706_155117_Vk7oa4` |
| e180_acrobot | `e180_dmc_acrobot_swingup_director_baseline_j4660140` |

These runs' replay was deleted at archive time (`SKIP_REPLAY=1`, see
`copy_dreamerv3_runs_to_bucket.sh` in the top-level CLAUDE.md), so this
measurement drives a **live on-policy rollout** with the restored checkpoint
rather than reading states from replay.

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
