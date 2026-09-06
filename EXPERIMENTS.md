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

**e735–e788 launched 2026-09-04** — the benchmark matrix. 54 runs: three
configs x six tasks x three seeds, one GPU each, 1.1M env steps at replay
ratio 256. Arrays `4706650` (a100, 36), `4706651` (v100, 12), `4706660`
(p100, 6); watchdog `4706677`; log `job_logs/bench_20260904_204028.tsv`.
W&B project **`dreamerv3-bench-2026-09`**.

| config | what it is |
|---|---|
| `dreamerv3` | flat DreamerV3, no hierarchy. The control. |
| `director` | hierarchy + Director's categorical goal VAE |
| `som_lip` | hierarchy + SOM/Lipschitz VQ goal VAE |

Tasks: `dmc_cheetah_run`, `pinpad_six`, `dmc_hopper_hop`, `pinpad_five`,
`dmc_cartpole_swingup`, `pinpad_four`. Submission order is seed-major then
environment then config, so a full seed-0 sweep of all six environments and
all three arms is in flight before any seed-1 run starts, and within a cell the
order is director -> som_lip -> dreamerv3.

**This is the first batch on DreamerV3 hyperparameters throughout** (38f8bf6,
856ffb8), so nothing in the archive is a valid comparator — not even the
2026-09-01 batch. What changed: lr 4e-5 / wd 0.0, `manager_policy.unimix` 0.01,
`mgr_retnorm` percentile, fixed manager entropy instead of the AutoAdapt,
`director_stable` deleted, replay 5e6, and `size6m` in place of
`director_match`. That last one is the big one: **3.94M parameters against
107M**, because `size6m` is DreamerV3's ladder rung at deter 1024 and satisfies
their scaling rules (deter = 8*units, classes = units/16, depth = units/16)
while `director_match` ran deter 1024 against units 1024 — a 107M model with a
6M recurrent state.

**Question:** does the hierarchy beat flat DreamerV3 at all, on a matched and
now-honest hyperparameter set, and does the SOM/Lipschitz goal AE beat
Director's on the sparse pinpad tasks where the collapse lives?

**Pre-registered expectations.** On the three DMC tasks all three arms should
be close; DreamerV3 reaches ~800+ on cheetah at 1M steps and the hierarchy has
never bought anything on dense control, so a large gap in *either* direction
there means something is misconfigured rather than interesting. The claim under
test is pinpad. Director's pinpad curve has peaked near 1M and decayed in every
batch we have run; if the decay is a replay-eviction artifact it should be gone
here, because replay 5e6 against 1.1M steps never evicts. -> If Director still
peaks and decays with a non-evicting buffer, eviction is ruled out and the
trigger is in the algorithm. -> If it holds, the whole 2026-08 collapse
programme was chasing a buffer-size artifact.

**e789–e824 added 2026-09-04 22:50** — the paired exploration-floor arm.
`director` and `som_lip` with `mgr_expl_perc01` layered on (36 runs, seed-major,
arrays `4706728` p100 / `4706729` short-a100). Everything else is identical to
e735–e788, so the pairs differ only in `mgr_expl_retnorm.limit`.

**Why.** Measured across all 15 HRL runs of the first batch, both arms and four
tasks, the manager's exploration return is 0.19–0.55 against a `perc` floor of
1.0. `perc` returns `max(limit, hi−lo)`, so **the exploration stream is never
normalized, in any run, on any task**, while the extrinsic stream (return ~17
on cheetah) is normalized properly. Where extrinsic reward is exactly 0 — every
pinpad task until the first pad sequence is found — that leaves `mgr_adv_mag`
at 4e-4 against 8e-2 on cheetah, and manager entropy pinned at its 16.64-nat
ceiling (measured 16.47 on pinpad_six against 1.76 on cheetah).

The cause is a unit mismatch rather than a broken normalizer: the exploration
reward is a per-dim *mean* of squared reconstruction error over 1024 deter
dims, so it is intrinsically ~1e-3, while the floor is absolute and was chosen
for a task-reward stream. DreamerV3 never met this because it has one stream;
Director avoids it with per-stream `std`.

`perc` is kept rather than switched to `meanstd` because `meanstd` divides by
the running std unconditionally, and would keep handing the manager a
unit-scale advantage built from noise once the goal AE reconstructs everything
equally well. A floor gives the wanted annealing for free: real novelty
structure → divide by the real range → signal; structure gone → range falls
under the floor → advantage decays → entropy returns to maximum.

**0.1 is provisional.** It is ~1/3 of the estimated current 5–95 range, but
propagating the measured spread gives 0.07–0.8 — the uncertainty is entirely in
how correlated reconstruction errors are along a trajectory, which the logs
could not settle. `mgr/rscale_expl` is now logged; set the floor from that
rather than from this arithmetic. **Read it first: if it equals the configured
limit, the stream is still clamped.**

**Branches.** → If the paired arm's manager entropy falls below the ceiling on
pinpad while the baseline's stays pinned, the floor was suppressing the
exploration signal and 0.1 (or lower) becomes the default. → If both stay
pinned, `mgr_expl_weight` is the binding constraint instead, not the normalizer.

**Measured while setting this up** (size6m, one GPU per run):

| GPU | conv | env fps | GPU mem | ETA for 1.1M |
|---|---|---|---|---|
| A100-80GB | native | 24.2 | 2.4 GB (3%) | ~12.6 h |
| V100-16GB | reference | 7.8 | 4.2 GB (26%) | ~39 h |
| P100-16GB | reference | 7.2 | 3.8 GB (24%) | ~42 h |

Two standing answers from that. **One GPU per run is enough everywhere** — the
old multi-V100 scripts used four because the 107M model needed ~59 GB, and at
3.94M a single 16 GB card is 4x oversized. **A GPU cannot be shared between
runs** despite 2.4 GB of an 80 GB A100 being nothing: the cluster exposes
`GresTypes = gpu,mic` with no MPS or shard type, so SLURM hands out whole GPUs.

**Two bugs found by launching it**, both fixed in 856ffb8:

- `som_lip` died on every V100 within two minutes (`agent.py:254 assert all
  finite`) while running fine on A100. `goal_vq_enc/dec` use `norm: none`
  because a norm would undo the Lipschitz bound, and an unnormalized trunk in
  bfloat16 goes non-finite. `goal_vq.dtype: float32` is what `LipMLP.dtype`
  exists for. Applied to A100 too, so the arm is not bf16 on one card and f32
  on another.
- The partition split assigned runs by index mod 9, but algorithm is index
  mod 3, so every P100 task was `dreamerv3` and every V100 task was
  `director`/`som_lip`. The unit of assignment is now a *cell* — the three
  arms for one (environment, seed) — so the arms being compared always ran on
  identical hardware.


### First results (2026-09-05 08:00, ~36% of the matrix)

Two runs finished, both flat `dreamerv3` at seed 0:

| run | task | last-15 | best |
|---|---|---|---|
| e737 | `dmc_cheetah_run` | **910.5** | 923.5 |
| e740 | `pinpad_six` | 0.0 | 0.0 |

**The flat control reproduces published DreamerV3 on cheetah** (~800–900 at 1M)
with a 4.19M-parameter `size6m` model, against the ~200M `size200m` that v3
uses for its own `dmc_vision` benchmark. That is a useful calibration: the
harness, the config and the 30x-smaller model are all sound, and any deficit
elsewhere is not "our DreamerV3 is broken".

**The hierarchy is far behind on dense control.** At comparable steps on
cheetah: flat 910.5, `som_lip` 418.3, `director` 166.7. Manager skill entropy
there is ~1.5 of a possible 16.64 nats, i.e. the manager has collapsed onto
essentially one goal and the worker is chasing a constant target. That is the
predicted consequence of `manager_actent_adapt: False` — DreamerV3's fixed
`actent` 3e-4 is calibrated for a worker-sized action space against
percentile-normalized returns, and nothing holds an 8x8 categorical open.
A `mgr_actent_adapt` block exists to restore the controller, but **e825–e828
were cancelled before running on 2026-09-05 and the probe was not taken**. The
DMC ordering therefore stands unattributed: we know the manager collapses onto
one goal under the fixed coefficient, and we do not know whether restoring the
controller recovers the deficit.

**Nothing has scored on pinpad_six**, in any arm, flat or hierarchical, at
~980k steps. One run has found reward on a pinpad task at all: e781
(`som_lip`, pinpad_five, seed 2) at last-15 192.7 after 104 episodes.

**The exploration floor works directionally.** On pinpad_six the paired arm
(floor 0.1) has manager advantage 0.00844 against the baseline's 0.00096 —
8.8x, exactly the 1.0/0.1 ratio — and manager entropy has come off its ceiling
to 13.06 against 15.13. But `mgr/rscale_expl` still reads exactly 0.1000, so
the stream is *still* clamped and the true 5–95 range is below 0.1. The new
`mgr/range_expl` metric logs the unclamped range so the floor can be set from a
measurement rather than the arithmetic that produced 0.1.

### Standings at 2026-09-06 09:00 (17 of 90 runs finished, 37% of env steps)

Best run per cell, **single seeds, so treat every ordering as provisional**.
`last-15` episode return; `*` marks a run still in flight (~520-590k steps).

| task | dreamerv3 | director | som_lip | director+expl | som_lip+expl |
|---|---|---|---|---|---|
| `dmc_cartpole_swingup` | **858.9** | 528.2 | 744.9 | 704.0 | 704.0 |
| `dmc_cheetah_run` | **910.5** | 338.7 | 282.1 | 361.4* | 381.1* |
| `dmc_hopper_hop` | **336.8** | 0.0 | 65.1 | 65.1 | — |
| `pinpad_four` | **381.3** | 0.0 | 12.0 | 234.0 | 326.7 |
| `pinpad_five` | **297.3** | 0.0 | 2.0 | — | 0.0 |
| `pinpad_six` | 0.0 | 0.0 | 0.0 | 0.0* | 0.0* |

**Flat DreamerV3 leads every task**, including the sparse pinpad ones the
hierarchy exists for. Director's paper claims the opposite there, so either the
hierarchy is misconfigured or something about this setup differs from theirs in
a way we have not identified. Do not read this as "hierarchy does not work"
yet: it is one seed per cell, and the manager is in a state we already know is
degenerate (below).

**The exploration floor is the largest single effect measured.** On
`pinpad_four` it moves the HRL arms from 0.0/12.0 to **234.0/326.7**. That is
direct confirmation of the diagnosis behind e789-e824: with `perc`'s floor at
1.0 the exploration return (0.19-0.55) was never normalized, so on a task whose
extrinsic reward is 0 the manager had almost no gradient. Lowering the floor to
0.1 gave it one. Note `mgr/rscale_expl` still reads exactly 0.1000, so the
stream is *still* clamped and a lower floor should help further --
`mgr/range_expl` now logs the unclamped range to set it from.

**Open, and the most likely reason the HRL arms trail:** manager skill entropy
sits at ~1.5 of a possible 16.64 nats on dense tasks under the fixed
`actent: 3e-4` (`manager_actent_adapt: False`). A manager collapsed onto one
goal makes the worker chase a constant target. The `mgr_actent_adapt` block
restores Director's controller but **was deliberately not run**, so this stays
a hypothesis with no experiment attached.

**Nothing has scored on `pinpad_six` in any arm**, flat or hierarchical, at
~1.09M steps.

### e829–e864: exploration floor 0.02, set from measurement (2026-09-07)

`director` and `som_lip` with `mgr_expl_perc002`, 6 tasks x 3 seeds, arrays
`4707513` (a100, 24) / `4707514` (v100, 8) / `4707517` (p100, 4). Third point on
the floor axis, against the unmodified arm (1.0) and the 0.1 arm.

**The value comes from `mgr/range_expl`, not from arithmetic.** Across 39 runs
past 200k steps the exploration return's unclamped 5–95 range is:

| arm | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| baseline (floor 1.0) | 24 | 0.034 | 0.096 | **0.167** | 0.216 | 0.481 |
| floor 0.1 | 15 | 0.056 | 0.082 | **0.115** | 0.164 | 0.506 |

Sparse tasks (pinpad/hopper) sit lower than dense ones: median 0.090 against
0.137 in the 0.1 arm. **At floor 0.1, four of fifteen cells were still clamped**
— both `pinpad_four` runs, `dmc_hopper_hop`, and one cheetah, with ranges
0.056–0.076. Those are exactly the tasks where lowering the floor produced the
largest gains, so the remaining headroom sits where it matters.

0.02 is ~40% below the lowest range ever observed (0.034), so nothing clamps at
launch, while still guarding a genuine collapse toward zero. Lower would sit
inside the noise floor of a stream whose per-step reward is ~5e-3.

**Correction to the 2026-09-06 entry**, which said the 0.1 arm was "still
clamped": that came from four early runs where the range estimate had not
settled. With the full set it is 4 of 15, not all — 0.1 landed near the median
of the distribution and fixed most cells. The earlier estimate of 0.15–0.5,
derived from the per-state reward spread, was about 2x too high; the lambda-
return damps that spread more than assumed. This is why the metric was added.

**Also on 2026-09-07:** the baseline (non-exploration) `director`/`som_lip`
runs still early or unstarted were dropped, keeping the seven already past 900k
so the control retains its seed coverage. `bench_droplist.txt` stops the
watchdog resurrecting them.

### Standings at 2026-09-07 08:30 (38 of 90 finished, 63 with scores)

Mean last-15 over the seeds started; (n seeds, lowest step reached).

| task | dreamerv3 | director | som_lip | director+expl | som_lip+expl |
|---|---|---|---|---|---|
| `dmc_cartpole_swingup` | **865.6** (3) | 624.2 (3) | 691.8 (3) | 704.0 (1) | 744.5 (1) |
| `dmc_cheetah_run` | **909.2** (2) | 280.8 (3) | 373.9 (3) | 481.3 (2) | 232.0 (2) |
| `dmc_hopper_hop` | **278.7** (2) | 0.3 (2) | 48.7 (2) | 88.6 (2) | 142.2 (1) |
| `pinpad_four` | **382.0** (3) | 23.6 (3) | 128.0 (3) | 326.0 (1) | 250.0 (1) |
| `pinpad_five` | **99.1** (3) | 0.0 (3) | 6.7 (3) | — | 0.0 (1) |
| `pinpad_six` | 0.0 (2) | 0.0 (2) | 0.0 (2) | 0.0 (2) | 0.0 (2) |

**The exploration floor now helps across tasks, not just one.** Comparing each
HRL arm against its `+expl` pair: director gains on pinpad_four (23.6 -> 326.0),
cheetah (280.8 -> 481.3) and hopper (0.3 -> 88.6); som_lip gains on hopper
(48.7 -> 142.2) and pinpad_four (128.0 -> 250.0). Five of six comparable cells
improve. The single regression, som_lip+expl on cheetah (232.0 against 373.9),
is two partial runs at ~870k steps. This is now repeated evidence rather than
one striking cell, and it is the strongest result in the batch.

**Flat DreamerV3 still leads every task.** The gap narrows a lot with the
exploration fix but does not close. Since the manager is known to be collapsed
onto ~1.5 of 16.64 nats under the fixed `actent`, the honest reading is that
this batch does not yet test "hierarchy vs flat" -- it tests one particular
degenerate manager against flat.

**`pinpad_six` is unsolved by everything**, two seeds per arm, all at ~1M steps,
all exactly 0.0. Whatever the six-pad sequence needs, none of these agents has
it, and that includes the flat agent that solves pinpad_four at 382.

### Packing runs onto one GPU: measured and rejected (2026-09-05)

A run uses 2.4 GB of an 80 GB A100, which looks like 97% waste. It is not.
Four runs packed onto one A100 reach **21.6 aggregate env-fps against 24.2 for
a single run** — 0.89x, slightly *slower* in total, with each run crawling at
5.4 fps. Memory was never the constraint; the card is already compute-saturated
by one run, and `nvidia-smi`'s `compute_avg` of 0.97 meant what it said.
SLURM also cannot subdivide a GPU here (`GresTypes = gpu,mic`, no MPS or shard
type), so packing inside one job was the only route and it does not pay.

`sbatch/run_benchmark_packed.sbatch` is kept with the measurement in its header
in case a smaller model or shorter imagination horizon changes the arithmetic.
**One run per GPU stands.**

A 1.26x reading taken at ~1000 steps was warmup, not signal — the runs had not
finished filling replay or settled after JIT. Do not trust a throughput number
from the first minutes of a run.

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
