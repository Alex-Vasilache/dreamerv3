# DreamerV3 HRL — Experiment Log

Director-style hierarchical RL on DreamerV3. Companion working paper:
`code/26_04_HRL-paper/main.tex` (the `paper/` directory in this repo is a stale
earlier draft).

This log was reset on **2026-09-01**. Everything before that date is preserved
verbatim in `EXPERIMENTS_ARCHIVE_20260901.md`, which carries the two earlier
archives (`..._20260809.md`, `..._20260713.md`) behind it. Nothing was deleted —
only the sections still worth reading every day are repeated here.

**Structure:** §1 Conventions · §2 Live board · §3 Dead ends · §4 Config flags &
metrics reference.

---

## 1. Conventions

- **Naming.** Runs are `e{N}_{env}…`; `N` is global and monotonic, never reused. A requeue
  with a tweak gets a new `N`; a literal restart keeps it. SLURM job name = run name.
- **Scores.** DMC episode return (max 1000). Reported as last-10/15-episode mean and best
  trailing-10/15 window ("peak"). Single seed unless stated — treat all orderings as
  provisional until replicated.
- **Scales.** `small` = size6m, 32×32, 1×V100 (or pinned P100). `BIG` = `director_match`
  (deter 1024, 512-wide MLPs, CNN depth 64), 64×64, batch 16×64, imag 16, train_ratio 64,
  native conv, 1×A100-80GB. All cross-recipe comparisons are within-scale.
- **Standard metrics** (from `metrics.jsonl`, `train/` prefix):

| Metric | Key | Meaning |
|---|---|---|
| mask_frac | `goal/mask_frac_mean` | fraction of L=8 blocks edited per decision |
| K | `goal/mgr_duration_mean` | mean hold length (steps between decisions) |
| **blk/step** | mask_frac × 8 / K | avg blocks edited per env step (Director = 1.0; lower = sparser) |
| worker reliability | `wkr_goal_rew` | mean per-step cosine to goal (healthy 0.32–0.46; collapse-zone 0.14–0.23) |
| sparse-task life | `epstats/reward_rate` | dead floor ≈1e-4; **alive bar = 1e-2 by 1M steps** (e124 ref) |
| manager signal | `mgr_extr_adv` | healthy ~0.01–0.02 sustained; collapse = spike then ≈0 permanently |

  Caveat: low blk/step from a long hold (large K) is not per-decision sparsity — always
  report mask_frac and K alongside. Post-2026-07-10, mask stats are valid-slot-weighted
  (NOT comparable to earlier runs, which were ~75% weighted to the last imagined decision).
- **Recipe names** (used everywhere below):
  - **combined** = masked edits (`prob_entropy`: rate→0.3 + per-block entropy→0.5) +
    variable durations (Lagrangian on |E[dur]−τ_d|, tol 0.1) + struct 200. Formerly "Group D"/"D-fix".
  - **plain var-K** = variable durations with weak fixed prior (reg 0.01→τ_d), no mask, struct 0.
  - **mask-only** = masked edits at fixed K=8, struct 200.
  - **pure Director** = base hierarchy, defaults only (fixed K=8, whole-code overwrite, no struct).
  - **+countdown** = `worker_timed_goals`: normalized steps-left appended to worker policy+value
    inputs (HiTS-style timed subgoals). **+struct-adapt** = `goal_struct_adapt` multiplier.

### How to update this document

- **At launch** (same session as `sbatch`): add the run to the §2 live board (exp, job id,
  recipe name from the glossary above, task/scale, one-line question) and write its
  pre-registered entry in §6 — hypothesis, quantitative expectation against a named
  reference value, and explicit "→ if X then Y" branches — **before results exist**. New
  mechanisms get a 3k-step smoke first; record the smoke job id.
- **When results land**: fill the ledger row in §5 (last-N score, peak, one-line verdict
  referencing the §6 branch that fired); remove the run from the live board; if it changes
  a conclusion, add/amend a finding in §3 (new findings get the next F-number; supersede,
  don't delete) or an entry in §4 (dead ends). Refresh the §2 prose and best-configs block
  if the state of the art moved. Update `paper/main.tex` in the same session (CLAUDE.md §Paper).
- **Cancellations**: mark in the live board with step count, the metric that justified it,
  and what comparator it lost to; move to the ledger + §4 if it establishes a dead end.
  Archive the run dir to the bucket before deleting from `/work` (CLAUDE.md §Archiving).
- **Corrections**: never silently rewrite history. When a past interpretation turns out
  wrong, keep the correction visible as part of the relevant finding (F14 style: state the
  old reading, the new one, and the evidence that flipped it).
- **Bugs / infra gotchas** → §7 (with fix date and affected run range). **New config flags
  or template knobs** → §8, defaults preserving old behavior.
- **Style**: this file stays condensed — findings and verdicts, not narratives. Long
  analysis lives in the session that produced it and in the paper; if a section outgrows
  its job, compress it here and rely on `EXPERIMENTS_ARCHIVE_20260809.md` + git for the
  full text. Experiment numbers are global and monotonic; never reuse or renumber.

---


---

## 2. Live board

**Nothing is running.** The queue was cleared on 2026-09-01: e718–e721 were
cancelled at ~3.47M/4M and e706–e712 (the UCB tilt batch) were cancelled before
starting. Their run dirs are still under `/work/DoyaU/vasilache/work/` and are
not yet archived to the bucket.

**Repo state.** One worktree (`code/dreamerv3`) on `main`, holding both the
Director and SOM/LiP arms. `archive/hrl` is the only other branch. Local `main`
is ahead of `origin/main` by 179 commits and **has not been pushed**.

**Where the science stands.**

- **The `director_stable` arm is a null.** Its four factors each hit their
  mechanical target and the seed-averaged pinpad-five curve is indistinguishable
  from the plain baseline (peak ~191 vs ~202 at 1.0M, same decay). It fixed the
  *recovery-blocker* — the manager value runaway, which only ever hit 2 of 5
  baseline seeds — not the *trigger*, which is present in every seed. Full
  measurements in the 2026-09-01 archive.
- **`director_stable` is a five-factor change, not four.** `mgr_advnorm` and
  `wkr_goal_advnorm` read one shared `config.advnorm` key, so turning on the
  manager's advantage normalization also raised the worker's update magnitude
  ~17x (0.037–0.043 -> 0.699–0.700). Splitting that key is a prerequisite for
  ablating either half.
- **The collapse trigger is not identified.** The one signal that moves with
  every collapse is `wkr_goal_rew` roughly doubling (0.19–0.25 -> 0.50–0.53)
  while the score falls ~90% — the worker gets better at reaching goals as the
  score dies. Manager entropy is held at target throughout and is ruled out.
- **The goal VAE now matches Director except for the nets.** Audited line by
  line on 2026-09-01 (`docs/VERIFICATION.md`, *Goal autoencoder*). The one active
  difference was `goal_autoencoder_beta` 0.25 vs Director's 1.0, now available as
  the `director_vaebeta` block.

**Next, in order.**

1. `goal_autoencoder_beta: 1.0` (`director_vaebeta`) — the only active goal-VAE
   difference from Director.
2. Split `advnorm` into worker and manager keys, then ablate the 17x worker
   change apart from the manager normalization.
3. Log `success_manager` (Director's "final goal reward > 0.7" fraction) and a
   time-to-reach within the K-block, so the "worker arrives early and idles"
   reading of the collapse can be tested at all.
4. Archive the cancelled e706–e721 run dirs to the bucket and clear `/work`.

**New config blocks (2026-09-01), all composable and off by default:**
`director_optmatch` (lr 1e-4, wd 1e-2 — Director's optimizer; layered after
`director_stable` this holds shrinkage at `wd*lr` = 1e-6 and only raises the lr),
`director_vaebeta` (goal-VAE KL weight 1.0), `v3nets` (all eleven MLPs back to
DreamerV3 base sizes, undoing the size half of `director_match`).
`goal_struct_adapt` and `goal_soft_reuse_adapt` now default to **False**, so a
Director baseline is a clean control; recipes that want them must opt in.

---

## 3. Dead ends (don't retry)

- `mask_sparsity_mode` **`sample`** / **`reinforce`**; fixed-weight sparsity→0 (F1).
- `mask_topk` hard budgets — strangle the policy at every K (e2/e4/e6, ≤100).
- `goal_duration_reg ≥ 0.1` — magnitude domination collapse (§7).
- One-sided priced costs as sparsity/length levers: `goal_edit_cost` any magnitude
  (e70–e77), `goal_edit_cost_ach`/`prob_ach` (e78/e81), switch-cost-only (rails K to max).
- `variable_goal_block_rew` without a duration prior (e62: 216, duration drifts 2.5).
- Duration-entropy bonus or min-hold floor as no-prior collapse fixes (e63/e64: ≤181).
- `mgr_reward_agg=sum` as a sparse-task rescue (e181 dead) or general improvement (e66).
- Combined-recipe sparse-task rescues that keep masks on: τ_d=8 (e182/e184), 10× mgr_expl
  (e183) — all dead at 0.6M while comparators learned.
- Entropy-only masking at BIG scale (e145: railed 0.72–0.80 + goal-KL divergence).
- `hrl_auto` fully-adaptive targets as a transfer fix (e126–e138: interior but dead).

---



---

## 4. Config flags & metrics reference

All flags default to DreamerV3/pre-HRL behavior.

- **Masking:** `use_masked_goals`, `mask_sparsity_mode {prob,sample,reinforce,none,entropy,prob_entropy}`,
  `mask_sparsity_target/_max/_fixed_weight`, `mask_topk`, `mask_actent_*` (entropy
  anti-collapse), `mask_kl_enable` + `mask_kl_*` (KL-to-sparse-prior, off by default),
  `mask_perblock_credit`, `perblock_edit_cost`, `mask_sparsemax`.
- **Durations:** `variable_goal_length`, `goal_duration_{min,max,target,reg,fixed}`,
  `goal_duration_adapt(_max)`, `goal_duration_lagrange(_impl,_min,_max,_vel,_init,_tol)`
  (own loss key `goal_duration_prior`; mutually exclusive with adapt),
  `variable_goal_block_rew`, `goal_duration_relabel_truncated` (default `True`;
  hindsight-relabels a block-pooled hold's duration class when `imag_length` cuts
  it short — see §7), `goal_switch_cost`, `goal_edit_cost(_ach)`,
  `manager_actent_duration_target`.
- **Struct:** `goal_struct_weight`, `goal_struct_target {deter,feat}`,
  `goal_struct_loss {mse,margin}`,
  `goal_struct_adapt(_init,_max,_min,_target,_vel,_one_sided)` — `_target` is a
  `goal/struct_loss` setpoint (default 0.005 as of 2026-07-14, was 0.01 before),
  dual-ascent by default: grows while loss is above target, shrinks while
  below. `_one_sided` (default `False`) drops the shrink branch — scale still
  grows on violation but holds rather than relaxing once cleared. See §7 for
  the three-revision history (a corr-based 0.97 target was tried and reverted
  the same day after e171/e175 showed BIG-scale runs never reach it; one-sided
  was tried and reverted back to dual-sided the same day).
- **Worker:** `worker_timed_goals` (countdown conditioning, 07-13).
- **Manager misc:** `mgr_cond_goalcode`, `mgr_cond_achieve`, `mgr_reward_agg {mean,sum}`,
  `mgr_expl_weight`.
- **Implicit sparsity:** `impl_sparsity_mode {none,reinforce}` + `impl_sparsity_target(_init,_vel,_one_sided,_min,_max)`
  (F18: catastrophic in every cell tested — leave `none`). **`goal_soft_reuse_adapt` +
  `goal_struct_adapt` are now `True` by default (2026-07-22)**, target 0.5 / 0.01
  respectively, ratcheted at the BIG-scale rate (`goal_soft_reuse_target_vel: 3.9e-6`) —
  the e286/e290 recipe (§2), currently the best-performing cell in the project (both
  above their Director baseline late in training). Override to `False`/old targets for a
  clean baseline; override `goal_soft_reuse_target_vel` to `1.0e-6` at small scale. See
  `dreamerv3/configs.yaml` `defaults.agent` for the exact shipped values. `goal_delta_mode` +
  `goal_delta_clip` (07-16, F18 follow-up, mutually exclusive with
  `use_masked_goals`): differentiable reuse via additive sigmoid-vote combination
  with the previous goal code, see §7. Carry field `skill_probs` (delta mode
  only) does double duty: it's the soft pre-sample distribution the NEXT
  decision's votes get added to (so stickiness scales with how confidently a
  block was last chosen, not a flat 1.0), and it also feeds `mgr_cond_goalcode`'s
  next-step conditioning instead of the collapsed one-hot. `goal_reuse_adapt`
  (recommended) / `goal_reuse_weight` (fixed ablation) (07-16, targets the
  DECODED goal directly rather than code identity, see §7): own loss key
  `goal_reuse`, metrics `goal/reuse_sim_mean` + `goal/reuse_adapt_scale_mean`
  (adapt mode); compatible with any goal-generation mechanism (plain, masked,
  joint, delta). Gradient reaches `manager_pol` (via the code's
  straight-through gradient) but NOT `goal_dec`'s own weights — blocked via
  `_decode_goal_no_decoder_grad` (`.values`/`.write()`-based parameter
  freeze, verified in isolation, `test_freeze_grad.py`). `goal_reuse_target`
  (adapt mode, default 0.9) is the similarity setpoint a dual-ascent Lagrange
  multiplier holds, `inverse=True` sense (push up when below target).
  `goal_reuse_target_init`/`_vel` (07-16) ratchet the target itself (`agent.Ratchet`,
  same mechanism as `impl_sparsity_target_init/_vel`) from `_init` up to
  `goal_reuse_target` over training instead of imposing the final target from step 0;
  default `_init == goal_reuse_target` (no-op). Mutually exclusive with
  `goal_reuse_weight` (raises if both set). Forces `mgr_cond_decgoal` on (also
  un-gated from `use_masked_goals` this session — was over-restricted the same
  way `mgr_cond_goalcode` was before the delta-mode fix).
- **Template env knobs** (`run_v3_prior_vargoal_{small,big_a100,short_a100}.sbatch`):
  `RECIPE={vark_masked,mask_fixedk,plain_vark}`, `DUR_TARGET`, `DUR_MODE={fixed,lagrangian}`,
  `MASK_MODE`, `STRUCT_W`, `STRUCT_ADAPT`, `WORKER_TIMED_GOALS`, `MGR_REWARD_AGG`,
  `MGR_EXPL_W`, `MGR_FREQ`, `SEED`, `RUN_STEPS`, `RUN_DIR` (fixed = resumable),
  `GOAL_DELTA_MODE`, `GOAL_DELTA_CLIP`, `GOAL_REUSE_WEIGHT`, `GOAL_REUSE_ADAPT`,
  `GOAL_REUSE_TARGET(_INIT,_VEL)`, `VARIABLE_GOAL_BLOCK_REW` (big_a100 only so far,
  2026-07-23), `GOAL_DURATION_RELABEL_TRUNCATED` (default `True`, big_a100 only),
  `IMAG_LENGTH` (big_a100 only; was hardcoded 16, now overridable — watch for OOM
  above 16 at BIG scale, untested).
- **Key metrics:** `goal/mask_frac_mean`, `goal/mask_prob_mean`, `goal/struct_corr`,
  `goal/rec_mean`, `goal/mgr_duration_mean/std`, `goal/mgr_switch_rate`, `wkr_goal_rew`,
  `wkr_ent/action`, `mgr_extr_rew(_block)`, `mgr_extr_adv`, `epstats/reward_rate`,
  `mgr_duration_lagrange_scale_mean`, `goal/mask_actent_scale_mean`,
  `goal/struct_adapt_scale` (pilot), plus §1's derived blk/step.

### Infra gotcha: WORK is set by the login profile (2026-08-15)

The watchdog was moved between partitions and, on its first pass in the new
job, **resubmitted all 32 finished runs at once**. Cause: the cluster's login
profile exports `WORK=/work`, the sbatch wrapper is `#!/bin/bash -l`, and the
script's `WORK="${WORK:-/work/DoyaU/vasilache/work}"` therefore kept `/work`.
Every glob became `/work/e510_*` instead of
`/work/DoyaU/vasilache/work/e510_*`, every probe returned "no run directory",
and "no run directory" reads as "the run died".

Caught within a minute; the 32 jobs were cancelled before any started, so no
run directory was touched and no data was lost. Two fixes:

1. The variable is now `WD_RUNS_DIR`. This is the **second** collision of the
   batch after `GROUPS` (bash builtin) silently emptied the launcher's loops.
   Generic environment names are not safe in scripts that run under a login
   shell on this cluster.
2. A pass now **aborts without acting** if fewer than half the experiments have
   a run directory under the configured root. A missing mount, a wrong root or
   an unreadable filesystem all look exactly like "every run died", and the
   difference is the scale — no plausible failure kills 32 runs between two
   30-minute passes. Verified by re-running under the exact login-shell
   conditions that broke it: `done=32 resubmitted=0`.

The deadline was also extended to 2026-08-17T06:00 in the same restart: the
original 08-16T12:00 predates the last eight runs' finish, and a late failure
there would have cost a cell its fourth seed for want of a resume that needs
minutes, not a full run.

### Infra gotcha: the score collector averaged unfinished runs (2026-08-13)

`collect_scores.py --at-step N` took, per run, the last 15 episodes recorded at
or before N. For a run that had reached 700k of a 4M target that silently
returns its 700k score and lets it into the cell mean as a finished seed. Read
two hours into round 3 it produced hopper `som_line` = **573.7 ± 436** from two
finished seeds at ~810 and one that had barely started, and every arm's mean
was depressed the same way.

Fixed with `--min-frac` (default 0.99): runs short of that fraction of
`--at-step` are dropped and **listed** rather than averaged in, so the exclusion
is visible in the output instead of silent. Same class of error as the watchdog
one the day before — **an automated read has to check that a run reached the
step it is being read at, not just that a number exists there.**

### Infra gotcha: a run does not stop on `run.steps` exactly (2026-08-12)

The driver advances in chunks and checks the step counter between them, so a
run ends a few thousand steps *short* of its target, not on it. Round 1, all
targeting 4M, finished between **3,995,992 and 3,998,592**; the e502–e509
baselines did the same.

This bit `sbatch/watchdog_e510_e557.sh`, whose "is it done" test was
`step >= 0.999 * RUN_STEPS` = 3,996,000. e542 ended at 3,995,992 — **eight
steps under** — so the watchdog correctly followed its rule and resubmitted a
finished run to collect them, taking a GPU slot to do it. Caught at the next
check-in, the job cancelled, and `DONE_FRAC` lowered to **0.995**, which clears
the whole observed spread while staying far above any genuinely partial run
(one round is 27h, so nothing lands accidentally within 0.5% of target).

Worth stating as a general lesson for anything that automates on a step count:
**compare against what runs actually reach, not against what they were asked
for.**

### Run-cost / logging flags (added 2026-08-09)

None of these change what training computes; they only add or remove bookkeeping, so runs
that differ only in these values stay comparable.

| Flag | Default | Effect |
|---|---|---|
| `run.save_every_steps` | `500000` | Permanent checkpoint every N env steps under `logdir/ckpt_milestones/<step>/` (agent params + step, no replay). Never pruned; copied to the bucket by the archiver. `0` disables. |
| `run.log_video` | `True` | Stack worker 0's frames into an episode video for the logger. `False` in the A100 launch script. |
| `agent.goal_struct_diag` | `False` | `goal/struct_corr*` diagnostics: an O((batch_size·batch_length)²) similarity matrix **per train step** (64 per env step at train_ratio 64). Enters no loss. |
| sbatch `PREALLOC` | `True` | `--jax.prealloc`: take the GPU allocation once instead of growing it. |
| sbatch `LOG_OUTPUTS` | `jsonl,wandb` | `scope` dropped — it wrote ~400MB of episode video per run that nothing reads. |

Resuming from a milestone for analysis:
`--run.from_checkpoint <run>/logdir/ckpt_milestones/000000500000`.

### Where the 4M-step wall time actually goes (measured 2026-08-09)

A 4M-step BIG run takes **~26–27 h** on one A100 (40–42 env fps). Timer shares
are now logged as scalars (`timer/<section>/frac` in `metrics.jsonl`), and they
say the budget is not recoverable from bookkeeping:

| Section | Share of wall time |
|---|---|
| waiting on the jitted train step (`jaxagent_policy` + `jaxagent_train`) | ~0.85–0.90 |
| replay sampling (`replay_sample`) | 0.007 |
| episode logging (`logfn`) | 0.001 |
| logger writes, nvsmi, psutil, checkpointing | ~0.000 |

GPU compute utilisation is 0.90. Everything the run does outside the train step
adds up to under 1.5%, so the four run-cost defaults above are worth **~2%
together** (42.15 vs 41.29 env fps, A/B on one node) --- real, but not a route
to 24 h. Two further things were measured and rejected:

- **One process per env** (`run.debug False`): *slower*, 36.4 vs 40.0 fps,
  GPU 0.79 vs 0.90 --- piping 64x64 observations costs more than stepping DMC
  inline.
- **XLA command buffers / latency-hiding scheduler**
  (`--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL`,
  `--xla_gpu_graph_min_graph_size=1`): 40.2 vs 40.0 fps, i.e. nothing.

Getting under 24 h therefore requires changing something that *is* a parameter
(train_ratio, batch, imag_length, or the step budget), or splitting one run
across 2 GPUs --- which does not help throughput here, since the 8-GPU quota
already runs all 8 seeds concurrently.
