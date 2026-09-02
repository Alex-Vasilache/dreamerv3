# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This is the MacBook (local dev) copy.** Experiments run on the Saion HPC cluster, which
has its own CLAUDE.md covering SLURM, GPU partitions, `/work`+`/bucket` staging, and the
module-based environments. None of that applies here — see *Not available locally* below.

## Replying to the user

Keep answers short. 1-3 sentences by default, one short paragraph at most.
No headers, bold labels, bullet lists or tables unless asked for a comparison.
After doing work, say what changed in one sentence; add at most one caveat, and
only if it changes what to do next. Long replies are the failure mode here even
when every word is correct and was asked for.

## Repository Overview

This repo is **DreamerV3** (`dreamerv3/`), a JAX-based world-model RL algorithm carrying
our Director-style hierarchical RL work. Note the path shape differs from the server: here
the repo root *is* the project, so paths the server writes as `code/dreamerv3/sbatch/` are
just `sbatch/`.

It is a **single git worktree on `main`** (consolidated 2026-09-01 from five worktrees and
eleven branches). It holds **both** research arms — the Director baseline and the
SOM/Ligoal autoencoder — selected by config block, not by branch. `archive/hrl` is the
only other branch.

Sibling checkouts under `~/work/research/`:

| Path | What it is |
|---|---|
| `../26_04_HRL-paper/` | The working paper — see *Paper* below |
| `../director/` | Original TF Director implementation; the reference when auditing ours |

## Environment

A local venv lives at `.venv/` (Python 3.13, JAX 0.4.34). Use it directly, no activation
script needed:

```bash
.venv/bin/python -u dreamerv3/main.py ...
```

**Runs use the M5 GPU automatically.** `jax-metal` is installed in `.venv`, and
`embodied/jax/internal.py:resolve_platform` maps the configured platform onto what this
machine has: on macOS it returns `METAL` when the plugin is present, so no flag or config
edit is needed. On Linux it is the identity, so Saion is unaffected. Override with the
env var:

```bash
DREAMERV3_PLATFORM=cpu .venv/bin/python -u dreamerv3/main.py ...
```

On macOS that env var is the *only* way to force a platform — the configured
`jax.platform` (including the `cpu` that the `debug` block sets) is superseded. There is
no CUDA here, so `cuda` in a config is never usable locally.

Expect ~1.4x wall-clock over CPU at `size1m` and roughly parity at `debug` scale, where
the network is too small to load the GPU.

Not installed in `.venv`: `pytest`, `wandb`, `tensorboard`. Training logs a
`WandB init failed, skipping WandB output` line and continues — that is expected here, not
a bug. Install what you need ad hoc (`.venv/bin/pip install pytest`).

Do **not** set the Saion-only env vars locally: `MUJOCO_GL=egl` (Linux/EGL; macOS uses the
default glfw path) and `DREAMERV3_CONV_IMPL=reference` (a workaround for the Saion GPU
stack — locally the native conv is correct and faster).

## Running Training Locally

Everything goes through `dreamerv3/main.py` directly; there is no scheduler here. Put
`debug` **last** in `--configs` so its small-network/CPU overrides win:

```bash
# Director baseline, debug scale
.venv/bin/python -u dreamerv3/main.py \
  --logdir ~/logdir/$(date +%Y%m%dT%H%M%S) \
  --configs debug \
  --task dmc_cartpole_swingup

# SOM/LiP + UCB arm, debug scale
.venv/bin/python -u dreamerv3/main.py \
  --logdir ~/logdir/$(date +%Y%m%dT%H%M%S) \
  --configs director_match goal_som_lipvq_line_prod lip_relu mgr_ucb debug \
  --task dmc_cartpole_swingup
```

Both of the above are verified to run end-to-end on this machine, on the GPU. Cap a smoke
test with `--run.steps 600`; anything much shorter finishes before the first metrics write
(`run.log_every` is a 5-second clock under `debug`), so an empty `metrics.jsonl` means the
run was too short, not that training failed. Resume a stopped run by re-running the same
command with the same `--logdir`.

DMC tasks need `dm-control`, which will **not** install on this Python 3.13 venv
(`dm-tree` has no cp313 wheel). `pinpad_three`, `pinpad_six` and `dummy` are pure numpy and
work today; a Python 3.12 venv is needed for the DMC suite.

macOS has no `timeout` command — use a background run or install coreutils (`gtimeout`)
rather than reaching for it.

The two arms take the same config sets the server passes via `CONFIGS=`, e.g.
`director_match director_stable` for the Director arm and
`director_match goal_som_lipvq_line_prod lip_relu mgr_ucb` for the SOM/LiP+UCB arm.

## Not available locally

Present in the repo but **server-only** — do not try to run these here:

- `sbatch/` (132 `run_*.sbatch` scripts + `submit_*.sh` matrix wrappers) — SLURM launchers
  for Saion. Editing them here and pushing is fine; running them is not.
- Anything referencing `/work/DoyaU/...`, `/bucket/DoyaU/...`, `/apps/unit/DoyaU/...`,
  `source_dreamerv3_env.sh`, `source_rl_env.sh`, `module load`, `sbatch`, `squeue`, or the
  bucket-archiving script. Those paths do not exist on this machine.
- The `rl_env` (Python 3.7 + TF2) TensorBoard workflow and `tensorboard_*.sh` helpers.

Scalar metrics are written as JSONL at `$logdir/metrics.jsonl`, which is the practical way
to inspect a local run without TensorBoard.

## Configuration System

Config is hierarchical YAML (`dreamerv3/configs.yaml`). Named blocks layer on top of
`defaults` in the order specified:

```bash
--configs crafter             # single block
--configs size1m debug        # size1m then debug overrides
```

All keys can also be overridden from the CLI as dotted flags:

```bash
--agent.use_single_rollout True
--jax.platform cpu
--batch_size 32
```

Key config blocks:
- `debug` — small network, CPU, fast cycle for development
- `size1m`, `size6m`, `size12m`, `size50m` — model scale presets
- `director_match`, `director_stable` — Director baseline arm
- `goal_som_*`, `goal_lipvq_*`, `lip_relu`, `mgr_*` — SOM/Ligoal autoencoder arm
- `dmc_proprio`, `dmc_vision`, `atari`, `crafter`, … — task-specific settings

## DreamerV4 Feature Toggles

Active research adds DreamerV4-style features as opt-in config flags (all default `False` /
DreamerV3 behavior):

| Flag | Effect |
|---|---|
| `agent.use_single_rollout` | Single rollout (`K=1`) instead of multi-branch |
| `agent.use_rms_loss_norm` | Per-term running RMS loss normalization |
| `agent.imag_loss.use_pmpo_actor` | PMPO actor loss instead of standard actor entropy |

See `docs/dreamerv4_upgrades_plan.md` for implementation details and caveats (e.g.
`repval_loss` still requires ≥2 replay steps even when `use_single_rollout=True`).

## Code Architecture

```
dreamerv3/          # repo root is the project
├── dreamerv3/
│   ├── main.py     # entry point; wires config, run mode, env, replay, agent
│   ├── agent.py    # Agent class: RSSM world model + actor-critic on imagined rollouts
│   ├── rssm.py     # RSSM (GRU-based recurrent state-space model)
│   └── configs.yaml# all hyperparameters
├── embodied/       # framework library
│   ├── core/       # Env/Agent base classes, Replay, Driver, wrappers, streams
│   ├── jax/        # JAX-specific: Agent wrapper, nets (CNN/MLP/GRU), heads, optimizer
│   ├── run/        # run loops: train, train_eval, eval_only, parallel, online
│   └── tests/      # pytest tests
├── sbatch/         # SLURM launchers — server-only (see above)
├── docs/           # standing planning/reference notes
├── experiments/    # per-experiment analysis scripts
├── tools/          # one-off Python/shell utilities (diag scripts, smoke checks)
├── job_logs/       # per-batch job-id tsv logs from server matrix launches
└── paper/          # STALE earlier draft — do not update (see Paper below)
```

`EXPERIMENTS.md`, `EXPERIMENTS_ARCHIVE_*.md`, `README.md`, `scores/`, `baselines.yaml`, and
`plot*.py` stay at the repo root: `plot*.py` load `baselines.yaml`/`scores/` via paths
relative to their own location.

**Data flow:** `main.py` parses config and dispatches to `embodied.run.*` → actor collects
transitions into `Replay` (chunked, disk-backed under `$logdir/replay/`) → learner samples
sequences via `embodied.streams.Consec` (consecutive-episode slices with replay context) →
`Agent.train` runs one RSSM encode+imagine step and updates all heads; `Agent.report`
produces reconstructions/videos for logging.

**Key abstractions:**
- `embodied.core.Env` / `embodied.core.Agent` — protocol classes; all envs and agents
  implement these
- `embodied.jax.Agent` — wraps a model class, handles JAX JIT, device sharding,
  checkpointing, and policy/train/report dispatch
- `ninjax` — stateful module system used for all neural net modules (analogous to Flax
  `nn.Module` but mutable-state style)
- `elements` — utilities: `Config`, `Path`, `Logger`, `Flags`, `Counter`, `Timer`
- `portal` — multiprocessing RPC used for learner/actor in online mode

## Tests

Tests live in `embodied/tests/`. `pytest` is not in `.venv` by default:

```bash
.venv/bin/pip install pytest        # once
.venv/bin/python -m pytest embodied/tests/test_train.py -x -v
.venv/bin/python -m pytest embodied/tests/ -x -v
```

Single test:

```bash
.venv/bin/python -m pytest embodied/tests/test_replay.py::TestReplay::test_sample -xvs
```

This is the main thing the MacBook is good for — the suite is CPU-bound and needs no GPU.

## Paper (`../26_04_HRL-paper/`)

The project maintains a continuously-updated working paper — the core of the eventual
publication for these experiments: `../26_04_HRL-paper/main.tex` → `main.pdf` (NeurIPS
shell, content in `sections/*.tex`, figures built by `figures/motivation/make_*.py`).

**`paper/` inside this repo is a stale earlier draft — do not update it.**

- **Keep it current**: whenever the project state materially changes (new results land, an
  interpretation is corrected, a new mechanism/formula is added), update `main.tex` and
  rebuild the PDF in the same session that updates `EXPERIMENTS.md`.
- **Style**: concise but clear; objective tone, no exaggerated language. Use formulas,
  tables, and figures wherever they explain an idea better than prose. State problems and
  negative results as plainly as benefits.
- **Build**: `./build.sh` from `../26_04_HRL-paper/` (`--clean` forces bibtex to redo).
  `pdflatex` is at `/Library/TeX/texbin/pdflatex` locally, so the paper builds fine on this
  machine. Run `build.sh` rather than a bare `pdflatex`: the acronym package needs the
  multi-pass sequence or it reports spurious "Acronym ... is not defined" warnings.
- **Figures**: built by `figures/motivation/make_*.py` (matplotlib, installed locally). They
  read archived runs from the bucket, which is **not** reachable here — figure regeneration
  needs the server or a local copy of the run data.

## Experiment Log

`EXPERIMENTS.md` is the experiment log; it was reset on 2026-09-01 and is now a slim
four-section file (Conventions · Live board · Dead ends · Config reference), with history in
`EXPERIMENTS_ARCHIVE_20260901.md` (which carries the two earlier archives behind it).

Runs are launched on the server, not here, so **local scratch runs do not get an experiment
number and do not go in the log.** The parts that do apply locally:

- Experiment numbers are global and monotonic — never reuse one, and don't allocate one for
  a local smoke test.
- Keep the live board short — move finished batches into the archive rather than growing the
  log again.
- **Pinpad scores**: never quote the final `last-15` alone; it has misled three times.
  Report the sustained-50 crossing step plus a late-window mean.

## Paths

| Path | Contents |
|---|---|
| `~/work/research/dreamerv3/` | This repo |
| `~/work/research/26_04_HRL-paper/` | The working paper |
| `~/work/research/director/` | Reference TF Director implementation |
| `.venv/` | Local Python 3.13 env (CPU JAX) |
| `~/logdir/` | Suggested local run directories — outside the repo, so runs never dirty it |
