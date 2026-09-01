# DreamerV3 HRL — Experiment Log

Director-style hierarchical RL on DreamerV3. Companion working paper: `paper/main.pdf`.

This log was restarted on **2026-08-09**. Everything before that date — the masked-edit /
variable-K programme (e1–e501), its findings, dead ends and technical notes — is preserved
verbatim in `EXPERIMENTS_ARCHIVE_20260809.md` (which itself carries
`EXPERIMENTS_ARCHIVE_20260713.md` inside it) and in git. Sections worth carrying forward
are repeated below; anything not repeated is in the archive, not lost.

**Why the restart:** the hopper-variance investigation closed (2026-08-09). Three
independent measurements agree that the implementation is sound and the task was the
problem:

1. `dmc_walker_walk`, Director arm, two seeds: **923.7** and **897.2** at ~850k steps —
   near the DMC maximum, agreeing within 3%.
2. On `hopper_hop` the original TF Director is no better than ours (matched 310k steps:
   ours median 31.9, TF Director 10.3, same 2-of-5 bimodality).
3. Five seeds on `hopper_hop` have **1% power**: a 15x median difference at 1M steps gives
   permutation p = 0.38. ~20–30 seeds would be needed.

**Consequences, carried into this log:**
- `hopper_hop` is not a comparison substrate. Use `cartpole_swingup`, `walker_walk`,
  `hopper_stand` — and report seeds, not single runs.
- The replay-capacity fix (`replay.size` 1e6, so the buffer actually evicts) is correct on
  its own terms but was never validated by a measurement; e495–e499 do not establish it.

**Structure:** §1 Conventions · §2 Live board · §3 Findings · §4 Dead ends (carried over) ·
§5 Experiment ledger · §6 Pre-registered hypotheses · §7 Config flags & metrics reference.

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

## 2. Live board

Launched 2026-08-09 via `sbatch/submit_e502_e509_director_baselines.sh`
(`ARM=director`, BIG/A100, 4M steps, replay 1e6, 8 envs, train_ratio 64).

| exp | job | task / scale | recipe | question | status |
|---|---|---|---|---|---|
| e502–e505 | 4676883–6 | cartpole_swingup / BIG | pure Director | seeds 0–3 | **done 4M** → §5 |
| e506–e509 | 4676887–90 | hopper_stand / BIG | pure Director | seeds 0–3 | **done 4M** → §5 |

These are the reference distributions every later arm is compared against: four
seeds per task on two tasks that are not `hopper_hop`. They are also the first
runs with the 2026-08-09 run-cost defaults and with 500k-step milestone
checkpoints (`logdir/ckpt_milestones/`), so any arm launched from here on can be
compared against them at 0.5M, 1M, ... and not only at the end.

**Finished 2026-08-10 ~20:00 at 4M**, results in §5: cartpole 753.7 ± 83.3
(spread 204), hopper_stand 822.3 ± 3.5 (spread 7.6). Both unimodal, first §6
branch fired.

### e510–e557 — the goal-autoencoder comparison, four seeds a cell

Launched 2026-08-10 via `sbatch/submit_e510_e557_goal_ae_comparison.sh`. Five
goal-AE arms x {cartpole_swingup, hopper_stand} x seeds 0–3, plus eight
lower-priority Director baselines on two further tasks. BIG/A100, 4M steps,
identical to e502–e509 in everything except the goal autoencoder.

| exps | arm | config | isolates |
|---|---|---|---|
| e510–e517 | `som_line` | `goal_som_line` | SOM on a line, straight-through **on** |
| e518–e525 | `som_orig_line` | `goal_som_orig_line` | the same, straight-through **off** (motivation.tex Eq. 12 verbatim) |
| e526–e533 | `lipvq_prod` | `goal_lipvq_prod` | Lipschitz alone, no SOM |
| e534–e541 | `som_lipvq_line_prod` | `goal_som_lipvq_line_prod` | = `som_line` + Lipschitz |
| e542–e549 | `som_orig_lipvq_line_prod` | `goal_som_orig_lipvq_line_prod` | = `som_orig_line` + Lipschitz |
| e550–e553 | `director` | — | cartpole_swingup_**sparse**, seeds 0–3 |
| e554–e557 | `director` | — | cheetah_run, seeds 0–3 |

Within each arm block the first four exps are cartpole_swingup seeds 0–3 and the
next four hopper_stand seeds 0–3. Job ids are in
`job_logs/e510_e557_goal_ae_comparison.tsv`.

**The Lipschitz form.** The three `_prod` blocks are new (commit `b7aae0c`) and
take the penalty as Liu et al. write it — one trainable scalar bound per *layer*
(`lip_per_row: False`) and the *product* over layers (`lip_impl: prod`,
`lip_scale: 1.1e-11`) — rather than the per-unit `logprod` default, under which
`bound()`'s `max` reduction let the penalty reach one unit of each 1024-unit
layer. They keep `silu`, which is the one place they depart from
`goal_som_orig_lipvq_line_paper`: the activation is not part of the mechanism
under test, and changing it would confound every LiP-vs-no-LiP contrast in the
family. Cost of that choice: silu's slope peaks at ~1.0998, so the composed
bound understates the true constant by ~1.33x over three hidden layers.
Calibration verified at production scale before launch — the smoke logs
`lip_penalty` = 6.6e10 against the 6.7e10 the weight was derived from, so the
`_prod` arms differ from their `logprod` siblings in the *form* of the penalty
and not its size (0.73 vs 0.75 absolute).

**α and β are the defaults**, `commit_scale: 1.0` and `som_scale: 0.9`, i.e. the
official SOM-VAE weights. The e485–e494 sweep did not establish that α matters
on a line (perplexity at 100k was 4.96 / 4.03 / 4.75 / 5.14 for α = 0.1 / 0.25 /
0.5 / 1.0, an ordering that reshuffled between readings), and 1.0 was nominally
the healthiest as well as the value in the paper.

**Scheduling.** The account's two A100 lanes have different limits: `gpu-a100`
is 8 concurrent with a 48h wall, `short-a100` is 16 concurrent (256 CPUs at 16
a job) with a **2h** wall, so runs there requeue themselves ~16 times and resume
from checkpoint. Seeds 0–1 and half of seed 2 went to `short-a100`, where they
started immediately; the tail went to `gpu-a100`, where it starts as e502–e509
finish. Two bookkeeping knobs differ by lane and change nothing about what is
computed: `run.save_every` 300s on the sliced lane vs 900s, and a persistent XLA
compilation cache (`JAX_COMPILATION_CACHE_DIR`) so a 2h slice does not spend
10–15 minutes recompiling from cold.

Jobs are **submitted seed-major** — all five arms on both tasks at seed 0, then
seed 1, and so on — so the first wave to finish is a complete comparison at n=2
rather than two finished arms and three that never started.

**Capacity, measured at launch.** All 32 A100s in `saion-gpu[23-26]` were
allocated (`sinfo -O GresUsed` reads `gpu:a100:8(IDX:0-7)` on all four nodes);
most of that is other groups, whose jobs `squeue` hides from us
(`PrivateData`). So the `short-a100` backlog is not a scheduling artifact to be
tuned around — there is nothing free to backfill into. Holding our own pending
`gpu-a100` jobs for a few minutes to test whether they were suppressing tier-1
backfill changed nothing, which confirms it.

What this account is actually guaranteed is the `gpu-a100` association: 8 GPUs
at PriorityTier=10. That lane processes 8 runs per ~27h, so the 40 arm runs need
five rounds ≈ 135h from when e502–e509 release it — landing 2026-08-15/16, ahead
of the user's return, with `short-a100` (4 running at launch) as upside rather
than as the plan. The eight `director` extras are the part that may not finish,
which is the priority order they were given.

**The analysis horizon is chosen at the end, not now.** Every run logs
continuously and snapshots at each 500k, so all 40 can be compared at whatever
step the *slowest* of them reaches, against the e502–e509 baselines read at that
same step. Nothing about the comparison requires every run to see 4M.

### Revision, 2026-08-10 13:00 — the lane split, at 4M

`short-a100` was measured rather than assumed. e510 ran there for **28:43**
before being preempted — of which ~10 minutes was compiling the BIG graph,
leaving ~19 minutes of training (45,952 steps at 41 fps) — and then sat queued
for another 80 minutes. A ~17% duty cycle, because every A100 on the cluster is
allocated and a PriorityTier=1 job gets only what nobody else wants. Preemption
itself worked exactly as designed: same job id, same `RUN_DIR`, progress kept.

Shortening the arms to 3M was proposed on schedule grounds and **rejected: the
arms run to 4M.** The resulting layout:

| lane | runs | wall |
|---|---|---|
| `gpu-a100`, 8 guaranteed GPUs | 32 arm runs, 4 rounds x ~27h | frees ~Aug 10 20:00 → **Aug 15 08:00** |
| `short-a100`, opportunistic | 8 arm runs (seed 3 of arms 2–5), 2h slices | accumulate in parallel; whatever is unfinished moves to `gpu-a100` after Aug 15 → **≤ Aug 16 11:00** |
| `gpu-a100`, after the arms | 8 Director extras | dependency-gated, ~27h → Aug 17, i.e. probably not |

Putting the last eight arm runs on the opportunistic lane costs nothing — they
would otherwise sit in a queue until Aug 15 — and turns five rounds on the
guaranteed lane into four. Even at a 17% duty cycle they accumulate a
substantial fraction of 4M in the five days before the guaranteed lane frees up
for them, and `PIN_DIRS=1` lets them be moved between lanes at any point
without losing a step.

The Director extras (e550–e557) are **not dropped and not merely niced**. A
nice value orders a queue but does not stop a niced job taking the last free
GPU once the arm queue is nearly drained, which would put a late arm restart
behind 27 hours of Director. They carry `--dependency=afterany:` on all forty
arm jobs, so they cannot start until every arm has ended — `afterany` rather
than `afterok` so a single failed arm does not strand them.

Runs that had already started were resubmitted with `RUN_DIR` pinned to their
existing directory (`PIN_DIRS=1`), so they continued from their checkpoints
rather than starting over. The launch TSV carries a per-run step target and the
watchdog reads it, so mixed horizons cannot confuse it.

### Round 1 complete, 2026-08-12 (seed 0, 4M)

The first full seed of four arms. `lipvq_prod` seed 0 had not started; its
seed 1 is running. **last-15 / peak**, against the e502–e509 Director means:

| arm | cartpole | hopper_stand | perplexity | used | rec_q |
|---|---|---|---|---|---|
| som_line | 744.9 / 824.3 | 821.6 / 839.4 | 7.24 / 7.36 | 1.00 | 11.2 / 8.3 |
| som_orig_line | **283.2** / 862.5 | 797.7 / 823.8 | 3.61 / 3.04 | 0.79 / 0.86 | 41.8 / 33.8 |
| som_lipvq_line_prod | **830.0** / 869.0 | 816.3 / **886.5** | 6.94 / 7.35 | 1.00 | 15.6 / 9.3 |
| som_orig_lipvq_line_prod | **843.4** / 858.9 | 755.6 / 826.1 | 3.38 / 4.46 | 0.75 / 0.90 | 21.0 / 20.8 |
| *Director (4 seeds)* | *753.7* | *822.3* | — | — | — |

**The Lipschitz penalty may be what prevents the STE-off collapse.** On
cartpole the two STE-off arms end 283.2 (no LiP, from a peak of 862.5) and
**843.4** (with LiP) — the only difference between them is `lip` — and the
geometry measurement puts their decoder expansion at 25.9 and 16.2. That is a
single seed each and the two hopper cells do not show the same gap (797.7 vs
755.6), so it is a hypothesis for the remaining seeds to test, not a finding.

One seed a cell, so no cell is a result yet — with the baseline's
own cartpole spread at 204 points, a single seed cannot clear anything. What
the round does establish is the shape of the comparison, and four things to
carry forward:

1. **The straight-through split is the dominant effect on codebook health, and
   the Lipschitz penalty does not touch it.** STE-on arms sit at perplexity
   7.0–7.5 with the full codebook in use and reconstruction 5–7; STE-off arms
   at 3.5–4.5, 74–92% used, reconstruction 15–26. Adding `lip` moves neither
   group across the gap. The pre-registered expectation for
   `som_line` vs `som_orig_line` is holding, and holding at both LiP settings,
   which is what makes it a clean single-factor read.
2. **The `_prod` penalty bites.** `goal/lip_bound_max` has fallen 28.7 → 26.2–28.1
   and `lip_penalty` 6.6e10 → 1.9e10–3.4e10 in a third of a run. Under the old
   `logprod` form the bound moved 27.88 → 27.33 over an entire 3.7M-step run.
   The pre-registered failure case — "if `lip_bound_max` is still ≈ 28 at 2M,
   the `_prod` arms did not test anything new" — will not fire.
3. **Do not read early hopper numbers.** e538 was at last-15 = **13.1** (peak
   108) at 11% and is at **833.6** at 78%. Any mid-run intervention on that
   evidence would have killed the arm that currently leads both tasks.
4. **CORRECTED at 4M — the near-collapsed codebook did not get away with it.**
   The 18:47 reading of this entry said e518 (`som_orig_line`, cartpole) was
   "most of the way to collapse and still beating the baseline" at 792.5 with
   perplexity 2.60. It finished at **283.2**, with perplexity 3.61, 79% used
   and reconstruction error **41.8**. The peak (862.5) was real and so was the
   collapse after it; the earlier reading mistook a run on its way down for a
   standing result. This is the same late-collapse signature the archive
   records for the ring version of this arm (e415: perplexity 1.57 → 1.28 with
   reconstruction 5.1 → 35.7) and for e486 (`som_orig_line` on hopper_hop,
   ended 0.3 against a peak of 186.4). **Read `som_orig_*` arms at the end,
   not in the middle, and report peak alongside final for every cell.**

Note also that last-15 is volatile late in a run: e510 read 592.8 at 84% and
744.9 at 100%; e522 read 512.5 and finished 797.7. The e502–e509 baselines did
the same (e502 wandered 615 → 569 → 655). Both statistics are reported for
every cell for that reason, and the final comparison should not rest on
last-15 alone.

### FINAL, 2026-08-16 — all 40 runs complete at 4M, four seeds a cell

**last-15 episode mean**, against the e502–e509 Director baselines. `Δ` is the
difference of means; `p` is the exact two-sided permutation test, floor 0.029.

| arm | cartpole | Δ | p | hopper_stand | Δ | p |
|---|---|---|---|---|---|---|
| *Director* | *753.6 ± 83.3* | — | — | *822.3 ± 3.5* | — | — |
| som_line | 785.9 ± 59.6 | +32.2 | 0.66 | 816.4 ± 18.5 | −5.9 | 0.91 |
| som_orig_line | 625.5 ± 238.0 | −128.1 | 0.43 | 706.4 ± 189.0 | −115.9 | **0.029** |
| lipvq_prod | 784.7 ± 37.3 | +31.1 | 0.54 | 812.6 ± 20.3 | −9.7 | 0.34 |
| som_lipvq_line_prod | 721.9 ± 104.5 | −31.7 | 0.60 | **835.6 ± 21.4** | +13.3 | 0.26 |
| som_orig_lipvq_line_prod | **838.1 ± 7.5** | +84.5 | 0.14 | 767.9 ± 46.8 | −54.4 | **0.029** |

**Geometry at four seeds** (`ratio` = displacement at k=7 over k=1):

| arm | cartpole ratio | hopper ratio | cartpole k=1 | dec Lipschitz (cp / hop) |
|---|---|---|---|---|
| Director | 1.13 ± 0.10 | 1.00 ± 0.10 | 2.31 ± 0.49 | 2.58 / 1.28 |
| som_line | **9.67 ± 0.63** | **5.18 ± 1.02** | 1.26 ± 0.21 | 2.47 / 2.09 |
| som_orig_line | 1.55 ± 0.33 | 1.45 ± 0.51 | 6.70 ± 0.94 | 38.3 / 29.1 |
| lipvq_prod | 1.07 ± 0.17 | 1.10 ± 0.34 | 5.03 ± 0.49 | **1.08 / 0.86** |
| som_lipvq_line_prod | **9.27 ± 1.94** | **4.28 ± 1.07** | 1.33 ± 0.31 | 2.38 / 1.95 |
| som_orig_lipvq_line_prod | 1.46 ± 0.37 | 2.71 ± 0.44 | 6.69 ± 0.62 | 33.3 / 15.9 |

**The second pre-registered branch fires. No arm beats Director on either
task.** On hopper_stand, the task whose baseline resolves to ±3.5, the three
straight-through arms land at −5.9, −9.7 and +13.3 — ties. The two
estimator-free arms are reliably *worse*, both at the 0.029 floor, i.e. every
one of their seeds below every Director seed. On cartpole nothing is
resolvable: the baseline's own seeds span 204 points, so even the largest
effect (+84.5) sits inside it, and the two arms that look best there
(`som_orig_lipvq_line_prod` +84.5) and worst on hopper (−62.1) are the same
arm, which is what task-to-task noise looks like.

**Read against F19 this is the substantive result of the batch.** The same
arms that leave return unchanged turn the code index from a label into a
coordinate — gradation 1.1 → 4.4–10.0, decoder expansion cut where the
Lipschitz penalty is added. Geometry preservation is therefore *achievable and
not what limits return*, which is precisely the §6 branch that reads: "then
geometry preservation is achievable and not what limits return, which
contradicts motivation.tex's argument as stated and is the more interesting
negative result. It must be written up as such, not buried: the paper's Sec.
'Goal Code Learning' would need the claim narrowed from 'this confound costs
performance' to 'this confound exists'."

### Three seeds in, 2026-08-14 (complete runs only, 23 of 40)

| arm | cartpole (n) | p | hopper_stand (n) | p |
|---|---|---|---|---|
| *Director* | *753.6 ± 83.3 (4)* | — | *822.3 ± 3.5 (4)* | — |
| som_line | 805.1 ± 55.7 (3) | 0.46 | **825.5 ± 3.8** (3) | 0.31 |
| som_orig_line | 555.8 ± 236.2 (3) | 0.11 | 800.9 ± 3.3 (3) | **0.029** |
| lipvq_prod | 787.4 ± 51.0 (2) | 0.67 | 799.3 (1) | 0.20 |
| som_lipvq_line_prod | 704.5 ± 177.5 (2) | 0.73 | **823.6 ± 10.3** (2) | 0.87 |
| som_orig_lipvq_line_prod | 844.3 ± 1.2 (2) | 0.33 | 786.0 ± 42.9 (2) | 0.07 |

**The shape is a tie on return, plus one small reliable loss.** On
hopper_stand, where the baseline resolves to ±3.5, both straight-through arms
sit on top of Director (825.5 and 823.6 against 822.3). Cartpole's ±83 cannot
resolve anything at three seeds and probably will not at four.

`som_orig_line` on hopper_stand is the **first cell to reach the floor of this
test**: p = 0.029 is 1/35, the smallest value C(7,4) permutations can return,
meaning all three of its seeds (798–805) fall below all four Director seeds
(818–825) with no overlap. Read it precisely — it says the separation is
consistent, not that it is large. The effect is **−21 points on a 1000 scale**,
about 0.4% of the achievable return, and it is a *loss*, from the arm whose
codebook F19 shows collapsing. Nothing here says the geometry work helps
return; the tight task says the straight-through arms match the baseline and
the estimator-free ones fall slightly behind it.

Meanwhile F19 shows the geometry *is* measurably different: the same arms turn
a flat displacement curve into a ramp (ratio 1.1 → 4.2–9.9).

If that holds at four seeds it fires the **second pre-registered branch**:
geometry preservation is achievable and is *not* what limits return. That is
the more interesting negative result and the §6 entry commits to writing it up
as one — narrowing motivation.tex's claim from "this confound costs
performance" to "this confound exists" — rather than burying it.

**Lane change, 2026-08-12 00:50.** The eight arm runs parked on `short-a100`
(seed 3 of arms 2–5) had waited **34 hours without starting a single time**,
and `sinfo -O GresUsed` still reads all 32 A100s allocated, so the lane never
paid off. They were cancelled and resubmitted to `gpu-a100` (fresh — they had
no progress to preserve), the Director extras were re-gated with an
`afterany` dependency on the new set of forty arm job ids, and the launch TSV
was pruned so the watchdog sees exactly one row per experiment. Verified after:
48 rows, `done=7 alive=41`, no spurious resubmissions. All forty arm runs are
now in the one queue that actually schedules, at ~4.25 rounds of 8 remaining
(→ ~Aug 16), which is the honest projection rather than one that assumed
capacity that was never there.

### Geometry tooling: validated, and the Director reference values

`experiments/goal_geometry/diag_goal_geometry.py` (2m28s on one V100 per run,
so it costs nothing and does not need the A100 lanes). First run: e502
(cartpole, Director) at its 3M milestone, 2048 states.

**It agrees with the published measurement**, which is the check that matters
before trusting anything else it says: hard-code `cosine_max` Pearson **0.586**
against motivation.tex's 0.62 for cartpole (seed range 0.41–0.80), and soft-code
**0.721** against the paper's 0.71. Same quantity, independent code path.

**The new measurements, as Director reference values:**

| quantity | e502 @3M | reads as |
|---|---|---|
| `cosmax/hard` Pearson / Spearman | 0.586 / 0.708 | the published figure |
| `dist/embed` Pearson / Spearman | 0.618 / 0.721 | distance geometry ≈ cosine geometry here |
| `dist/latent` Pearson / Spearman | **0.916 / 0.953** | the *continuous* logits track goal distance well |
| empirical Lipschitz, encoder | p50 2.81, max 4.51 | |
| empirical Lipschitz, decoder | p50 1.89, p99 5.46, max 7.35 | the number the `_prod` penalty should reduce |

`dist/latent` 0.916 against `dist/embed` 0.618 puts a number on where the
geometry is lost: not in the encoder, which maps goal distance to latent
distance almost linearly, but in the discretization after it. That is the same
conclusion motivation.tex draws from the soft-vs-hard `cosine_max` gap, now in
the geometry the Lipschitz condition is actually written in.

**Decoded-goal displacement vs code index distance** — the direct form of the
architecture's claim. For Director, moving one block by *k* classes and
decoding:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| ‖Δgoal‖ | 1.47 | 1.54 | 1.45 | 1.44 | 1.63 | 1.49 | 1.67 |

**Flat.** One class away and seven classes away move the goal by the same
amount, because the entry index is an arbitrary label — exactly what a
categorical head with no topology should do, stated as a measurement rather
than as an argument. If the SOM-line arms return a monotone ramp here, that is
the cleanest evidence in the programme that the topology does what it is for,
and it needs no correlation and no p-value to read. Single seed at one
milestone; all arms and seeds get measured once the batch finishes.

`sbatch/watchdog_e510_e557.sh` (job 4677013, on `intel`) supervises them every
30 minutes until 2026-08-16: it resumes runs that died and requeues runs whose
`metrics.jsonl` has gone stale for 90 minutes, always into the lane the run was
launched in and always with `RUN_DIR` pinned so it continues from its
checkpoint. It never cancels or deletes anything, gives up after five
resubmissions of the same experiment, and stops resubmitting past the deadline.

---

### FIVE benchmarks, 2026-08-20 — the aggregate rests entirely on cheetah

All five benchmarks now have 4 seeds of Director and 4 of SOM-line+LiP.

| benchmark | Director | SOM-line+LiP | Δ | % | p (exact) |
|---|---|---|---|---|---|
| Cartpole Swingup | 753.6 ± 72.2 | 721.9 ± 90.5 | −31.7 | −4.2% | 0.600 |
| Hopper Stand | 822.3 ± 3.0 | 835.6 ± 18.5 | +13.3 | +1.6% | 0.257 |
| Cartpole sparse | 775.6 ± 16.0 | 805.1 ± 19.4 | +29.5 | +3.8% | 0.114 |
| **Cheetah Run** | 442.6 ± 114.5 | **660.9 ± 44.6** | **+218.4** | **+49.3%** | **0.029** |
| Hopper Hop | 220.4 ± 93.6 | 211.3 ± 57.5 | −9.1 | −4.1% | 0.943 |

Aggregate (normalized = return/1000): Director 0.603, SOM-line+LiP 0.647.
**+7.31%**, 95% CI [+0.63%, +14.76%], exact stratified permutation
**p = 0.110** (1.68e9 assignments).

**Leave-one-benchmark-out is the whole story.** Dropping any single benchmark
leaves the gain at 0.048–0.063 — except cheetah, which leaves **+0.0005**:

| dropped | remaining gain |
|---|---|
| Cartpole Swingup | +0.0630 |
| Hopper Stand | +0.0518 |
| Cartpole sparse | +0.0477 |
| **Cheetah Run** | **+0.0005** |
| Hopper Hop | +0.0574 |

So the honest statement is **not** "our method is 7% better across five
benchmarks". It is "our method is decisively better on cheetah (+49%, complete
seed separation, at the 0.029 permutation floor) and indistinguishable on the
other four". Remove cheetah and the aggregate is zero to three decimals.

Adding hopper_hop moved the aggregate the wrong way (4-benchmark: +8.2%,
p=0.056 → 5-benchmark: +7.3%, p=0.110), which is what a tie on a new benchmark
does. It did not corroborate cheetah.

Note the bootstrap CI [+0.63%, +14.76%] excludes zero while the permutation
test does not reject at 0.05. They disagree because the CI resamples seeds
within benchmarks while the permutation test respects the benchmark strata; the
stratified permutation is the more conservative and the more appropriate test
for "does the arm label matter". Read it as not significant.

## 3. Findings

### F19 — the SOM-on-a-line turns the code index into a coordinate, and the straight-through estimator is what makes it happen

Measured on every round-1 checkpoint at 4M with
`experiments/goal_geometry/diag_goal_geometry.py` (job 4677770). Batch-to-batch
std on these quantities is 0.002–0.006, so the differences below are not noise.

**Move one code block *k* classes away, decode, and measure how far the goal
moved** (cartpole, seed 0, raw units):

| arm | k=1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Director | 1.47 | 1.54 | 1.45 | 1.44 | 1.63 | 1.49 | 1.67 |
| `som_line` (STE on) | **1.44** | 3.24 | 5.14 | 7.12 | 8.98 | 11.45 | **13.23** |
| `som_orig_line` (STE off) | **6.25** | 6.97 | 7.50 | 8.12 | 8.84 | 9.30 | 7.10 |

Director is flat: its entry indices are arbitrary labels, one class away and
seven classes away move the goal equally. `som_line` is a near-straight ramp
from 1.44 to 13.23 — the index has become a **coordinate**, which is exactly
the property motivation.tex argues is missing, now delivered and measured
rather than asserted. `som_lipvq_line_prod` is the same or stronger (11.9x at
k=7).

**Without the straight-through estimator the mechanism does not merely fail, it
inverts.** `som_orig_line`'s *nearest* edit already moves the goal 6.25 — four
times Director's nearest edit, and further than Director's farthest. Its
locality is worse than having no topology at all.

**The Lipschitz measurement says why.** Empirical ratios over state pairs,
median (max):

| arm | encoder ‖Δz‖/‖Δs‖ | decoder ‖Δgoal‖/‖Δz_q‖ |
|---|---|---|
| Director | 2.81 (4.51) | 1.89 (7.35) |
| `som_line` | 0.43 (1.43) | 2.53 (8.08) |
| `som_orig_line` | **0.022 (0.048)** | **25.9 (140.1)** |
| `som_orig_lipvq_line_prod` | 0.023 (—) | 16.2 (—) |

The STE-off encoder is collapsed — it barely moves when the state moves
(perturbation slope 0.040 against Director's 3.058) — and its decoder is wildly
expansive to compensate, at a maximum ratio of **140** against Director's 7.4.
That single pair of numbers explains the whole arm: a dead encoder gives a
codebook nothing to separate (hence perplexity 3.0–3.6 and 21–34% of entries
unused), and an expansive decoder makes every code edit a jump, which is why
e518 peaked at 862.5 and finished at 283.2. The `_prod` penalty reduces the
decoder ratio (25.9 → 16.2) but does not rescue it.

**A caution about the metric motivation.tex currently uses.** Ranked by
hard-code `cosine_max` correlation, the *worst* arm here looks like the best:
`som_orig_line` scores r = 0.831 and `som_orig_lipvq_line_prod` 0.912, against
Director's 0.586 and `som_line`'s 0.503. A degenerate codebook inflates that
correlation — with three entries in use, most state pairs share a code and
"agree" trivially. The correlation is not wrong, it is unidentifiable in the
collapsed regime. The displacement sweep and the Lipschitz ratio separate the
two cases cleanly and should carry the argument in the paper; the correlation
should be reported next to codebook occupancy or not at all.

Where the correlation *does* work, the SOM arms improve it in the geometry that
matters: quantized-embedding distance against goal distance is r = 0.94/0.83
(`som_line`) and 0.92/0.82 (`som_lipvq_line_prod`) against Director's 0.62.

**Two seeds (2026-08-13) reproduce it, and sharpen what the claim is.** Mean
over seeds of the k=1 displacement, the k=7 displacement, their ratio, and the
median decoder Lipschitz ratio:

| task | arm | n | k=1 | k=7 | ratio | dec L |
|---|---|---|---|---|---|---|
| cartpole | Director | 1 | 1.47 | 1.67 | **1.1** | 1.88 |
| cartpole | som_line | 2 | 1.25 | 12.29 | **9.9** | 2.32 |
| cartpole | som_lipvq_line_prod | 2 | 1.22 | 12.09 | **9.9** | 2.20 |
| cartpole | som_orig_line | 2 | 7.24 | 11.53 | 1.6 | **39.5** |
| hopper | som_line | 2 | 1.53 | 6.84 | 4.5 | 2.04 |
| hopper | som_lipvq_line_prod | 2 | 1.10 | 4.63 | 4.2 | 1.56 |
| hopper | som_orig_line | 2 | 3.25 | 5.77 | 1.8 | **25.8** |

**Full baseline (4 seeds) + 2–3 seeds an arm, 2026-08-14.** mean ± std over
seeds. `ratio` is k=7 displacement over k=1 — the gradation. `dec L` is the
median decoder Lipschitz ratio.

| task | arm | n | k=1 | ratio | dec L |
|---|---|---|---|---|---|
| cartpole | Director | 4 | 2.31 ± 0.49 | 1.13 ± 0.10 | 2.58 ± 0.76 |
| cartpole | lipvq_prod | 2 | 4.87 ± 0.66 | **1.12 ± 0.04** | **1.03 ± 0.18** |
| cartpole | som_line | 3 | 1.33 ± 0.19 | **9.70 ± 0.72** | 2.52 ± 0.32 |
| cartpole | som_lipvq_line_prod | 2 | 1.22 ± 0.04 | **10.01 ± 1.86** | 2.20 ± 0.17 |
| cartpole | som_orig_line | 3 | 6.74 ± 1.08 | 1.46 ± 0.34 | 36.7 ± 4.0 |
| cartpole | som_orig_lipvq_line_prod | 2 | 6.73 ± 0.51 | 1.33 ± 0.28 | 35.6 ± 3.5 |
| hopper | Director | 4 | 1.71 ± 0.19 | 1.00 ± 0.10 | 1.28 ± 0.14 |
| hopper | lipvq_prod | 2 | 3.14 ± 0.23 | **0.93 ± 0.23** | **0.92 ± 0.10** |
| hopper | som_line | 3 | 1.45 ± 0.28 | **5.27 ± 1.16** | 1.97 ± 0.14 |
| hopper | som_lipvq_line_prod | 2 | 1.10 ± 0.13 | **4.36 ± 1.43** | 1.56 ± 0.12 |
| hopper | som_orig_line | 3 | 3.56 ± 0.44 | 1.41 ± 0.58 | 24.4 ± 2.0 |
| hopper | som_orig_lipvq_line_prod | 2 | 3.21 ± 0.06 | 2.48 ± 0.36 | 16.9 ± 0.7 |

**The five arms separate the mechanism cleanly, which is what they were for:**

1. **Ordering needs the SOM *and* the estimator; neither alone does it.**
   Ratio is 1.0–1.1 for Director, and **1.12 / 0.93 for `lipvq_prod`** — a
   Lipschitz bound with no topology leaves the index exactly as arbitrary as
   Director's. Drop the estimator instead and it is 1.3–2.5. Only the two arms
   with both reach 4.4–10.0.
2. **The Lipschitz penalty does the job it claims, and only that job.** It
   lowers the decoder's expansion in *every* pairing where it is added:
   2.52 → 2.20 and 1.97 → 1.56 on the SOM arms, 24.4 → 16.9 on the STE-off
   hopper arm, and `lipvq_prod` alone reaches **1.03 / 0.92**, below Director's
   2.58 / 1.28. It changes ordering not at all, and (per §2) return not at all.
3. **CORRECTS the 08-13 note.** That note said Director's k=1 (1.47) was as
   local as `som_line`'s, so locality was not part of the claim. That reading
   came from Director's *single* measured seed. With all four, Director is
   2.31 ± 0.49 on cartpole against `som_line`'s 1.33 ± 0.19 — non-overlapping.
   The SOM+STE arms improve locality **and** ordering; hopper's locality gap
   (1.71 vs 1.45) is within noise, cartpole's is not. Two corrections in two
   days on the same sentence, both caused by reading a one-seed baseline —
   which is the argument for having measured all four.

Batch-to-batch noise on these quantities is 0.002–0.006, and the seed spreads
above are small next to the effects.

### F20 — Director's blocks are not independent, which is the real reason per-block credit assignment cannot work

Measured 2026-08-17 on the same 48 checkpoints as F19 (`hamming/mse` in
`experiments/goal_geometry/results/*.npz`, figure
`experiments/goal_geometry/make_figure_blocks.py`). F19 asked which *class* a
block moves to; this asks how many *blocks* move. Change `m` of the L=8 blocks
to uniformly random other classes, decode, record MSE against the reference
goal, for m = 0..8.

Every arm rises with m — so unlike the class index, blocks-changed is a real
distance for all of them. What separates them is curvature, measured as the
**additivity ratio** `α = MSE(8) / (8 × MSE(1))`: 1 if blocks contribute
independently, >1 if they interact, <1 if the shift saturates.

| arm | α cartpole | α hopper | MSE m=1 cartpole | MSE m=1 hopper |
|---|---|---|---|---|
| director | **3.07 ± 1.15** | **2.30 ± 0.58** | 0.0078 | 0.0040 |
| lipvq_prod | 0.80 ± 0.12 | 0.94 ± 0.10 | 0.0403 | 0.0160 |
| som_line | 0.75 ± 0.04 | 0.75 ± 0.15 | 0.0324 | 0.0224 |
| som_lipvq_line_prod | 0.88 ± 0.21 | 0.87 ± 0.15 | 0.0321 | 0.0165 |
| som_orig_line | 0.43 ± 0.02 | 0.88 ± 0.29 | 0.0742 | 0.0328 |
| som_orig_lipvq_line_prod | 0.42 ± 0.07 | 0.80 ± 0.16 | 0.1110 | 0.0444 |

All 8 Director seeds are above 1.72; all 40 other seeds are below 1.25 — the
groups do not overlap (exact two-sided permutation test, p = 9.4e-05 on each
task).

**Do not read Director's small m=1 as good locality.** It has the smallest
single-block edit of any arm, which taken alone looks like exactly the property
we want. It is the opposite. The decoder reads the eight blocks jointly, so one
block moved on its own barely disturbs the goal and the movement only appears
once several move together — and F19 shows that edit is *unsteerable* anyway
(d7/d1 = 1.19 / 0.75, every class the same distance away). Director offers one
edit size, in the middle, with no way to ask for less. The SOM+STE arms have a
larger *average* block edit (their codebook has spread out, so a random class is
typically far) but the smallest *available* one, and it can be chosen: in MSE
their nearest index edit is 0.0019–0.0026 against Director's 0.0078, still
smallest after normalizing by each arm's own full-code range (1.0% vs 4.7%).

The two sweeps have to be read together. Either alone gives the wrong answer.

### F21 — the cosine_max geometry result is not an artifact of a scale-free metric

Re-measured 2026-08-17 on all 20 four-env Director baselines (e390–e409,
restaged from `/bucket`), adding pairwise MSE alongside `cosine_max` in
`diag_goal_struct_corr.py`. The fresh rollout reproduces the published cosine
numbers to within 0.01 (hard 0.61/0.48/0.35/0.37 vs the paper's
0.62/0.48/0.36/0.37), so the two metrics are being compared on like data.

| | cartpole | hopper hop | acrobot | cheetah |
|---|---|---|---|---|
| hard, cosine_max | 0.61 ± 0.15 | 0.48 ± 0.08 | 0.35 ± 0.06 | 0.37 ± 0.06 |
| hard, MSE | 0.62 ± 0.14 | 0.50 ± 0.04 | 0.34 ± 0.06 | 0.43 ± 0.07 |
| soft, cosine_max | 0.70 ± 0.14 | 0.61 ± 0.10 | 0.44 ± 0.06 | 0.52 ± 0.03 |
| soft, MSE | 0.75 ± 0.11 | 0.68 ± 0.05 | 0.41 ± 0.06 | 0.67 ± 0.02 |

Same moderate values, same soft > hard ordering, under both metrics. What the
distance view adds is **saturation**: plotted as MSE-vs-MSE the soft code rises
steeply over the first fifth of the goal-space range and is then flat — past
~0.1 goal MSE on cartpole, moving the goal twice as far again does not move the
code at all. A single r understates the problem at the far end and overstates it
at the near end.

### F22 — at identical code distance, only the straight-through SOM code knows how far it moved

Measured 2026-08-17, same 48 checkpoints (`index_total/*` in the geometry npz,
`make_figure_index_total.py`). Third sweep: move m of the L blocks so the index
shifts sum to exactly `D = Σ|c'−c|`, D = 0..L(C−1) = 56.

**Reachability caveat.** A block at class c can move at most `max(c, C−1−c)`, so
D=56 needs every class at an end of its range. Real codes reach D≈50 at best and
coverage falls below half past D≈43. The sweep records per-D coverage; the
figure draws solid to coverage 0.5 and dashed beyond. **Do not quote a number
from the faded tail** — it averages over a shrinking, increasingly extreme
subset, and it is where hopper Director appears to go *down*.

Gain from D=8 (one class step in every block) to the coverage limit:

| arm | cartpole | hopper |
|---|---|---|
| director | 1.29 ± 0.22 | 1.35 ± 0.65 |
| lipvq_prod | 1.39 ± 0.18 | 1.66 ± 0.77 |
| som_orig_line | 1.28 ± 0.09 | 1.83 ± 0.29 |
| som_orig_lipvq_line_prod | 1.09 ± 0.12 | 2.51 ± 0.99 |
| **som_line** | **5.96 ± 0.72** | **4.90 ± 2.13** |
| **som_lipvq_line_prod** | **6.91 ± 1.82** | **6.39 ± 1.39** |

Director and LiP-only flatten by D≈15: once all 8 blocks have changed there is
nothing left to vary but index distance, which for them means nothing.

**The controlled version is the decisive one.** The decoder's input is the
one-hot code for every arm, so code-space MSE = 2·(blocks differing)/(L·C) — it
counts blocks and cannot see index distance. Take only the draws where **all L
blocks changed** (code MSE identical, 0.25, for every one) and compare the
bottom quartile of D (≈19) against the top (≈40):

| | cartpole | hopper |
|---|---|---|
| STE-SOM arms pooled | **2.13** (1.64–2.42) | **2.41** (1.33–3.30) |
| all other arms pooled | 0.99 (0.79–1.15) | 1.14 (0.75–1.61) |
| exact 2-sided perm. test | p = 1.4e-06 | p = 2.5e-05 |

At a distance the manager's code space reports as identical, the STE-SOM code
still moves the goal 2x further when the index distance is larger; Director,
LiP-only and the estimator-free arms move it not at all.

Note the *raw* spread at fixed code distance is large for **every** arm (p90/p10
of 1.7–11.7), because it also contains reference-state and which-blocks
variation. Only the controlled comparison isolates the index contribution — do
not cite the raw band as an arm difference.

### e495–e499 cannot be resumed — the replay is gone

Asked on 2026-08-17 to resume the five hopper_hop Director baselines (e495–e499,
stopped at ~2.3M). They were archived with `SKIP_REPLAY=1` and deleted from
`/work`, so **only the checkpoints survive**. Restarting from one refills replay
from scratch at 2.3M, which is a different experiment from a fresh 4M run and is
not a valid baseline for one. Ran e566–e569 fresh to 4M instead, matching the
e570–e573 arm runs they exist to be compared against.

This is the second time the archive policy has cost a resume. `SKIP_REPLAY=1`
is right for a *finished* run and wrong for a stopped one; check
`metrics.jsonl` reaches the target step before archiving.

### F23 — Director has no Lipschitz bound; our LiP arms do

Measured 2026-08-19 from the checkpoint weights (`experiments/goal_geometry/
lipschitz_bound_from_weights.py`, CPU-only on `intel`), answering the
motivation.tex todo "actually measure the lipschitz bound for director".

**The two families use different trunks, and this is the whole finding.**
Director's goal AE is configured by `goal_enc`/`goal_dec` (`norm: rms`,
unconstrained). The VQ arms use `goal_vq_enc`/`goal_vq_dec` — **`norm: none`,
`strict_bound: true`** — and in the LiP arms all five layers including the
output projection carry the `c` bound. So:

| arm | trunk norm | layers constrained | product enc | product dec | is it a bound? |
|---|---|---|---|---|---|
| director | rms | no | 1.28–2.85e7 | 0.67–2.05e7 | **NO** — RMSNorm is not Lipschitz |
| som_line | none | no | 2.9–6.4e7 | 3.1e7 | yes, but unconstrained |
| lipvq_prod, som_lipvq_line_prod | none | **yes** | **1.35–1.87e5** | **0.51–1.13e5** | **yes** |

Director's number is a weight statistic, not a bound: RMSNorm's gain is
`scale/rms(x)` → ∞ as x → 0, so no product of weight norms bounds that network.
Among the VQ arms — which all have a valid composed bound — the LiP penalty
reduces it by **~200–400x** against the same trunk without the penalty
(som_line). Caveat: `act: silu` peaks at slope ~1.0998, so the true constant can
exceed the product by that factor per activation.

Spectral norms move the other way (Director enc 308–375 / dec 3.6–9.1e3; LiP enc
1.3–4.8e3 / dec 1.9–7.1e4). A row-wise ∞-norm constraint does not control the
2-norm and the network compensates where the constraint cannot see.

**Three traps, all of which produce confident wrong numbers:**
* `agent.pkl` stores Adam's two moment buffers under `params/opt/state_goal/
  {1,2}/` with the **same leaf names** as the weights — an unfiltered walk finds
  each layer three times (15 "layers" for a 5-layer decoder, product 1.3e18).
* Detecting norm layers from parameter names is unreliable; read the config.
* **Reading the wrong config key.** `goal_enc`/`goal_dec` exist in *every* run's
  config as the Director-side defaults, and the checkpoint *module* is named
  `goal_enc`/`goal_dec` for the VQ arms too. The VQ trunk is configured by
  `goal_vq_enc`/`goal_vq_dec`. I first read the Director keys and concluded no
  arm had a bound, which is wrong. Distinguish the arms by whether the layers
  carry a `c` parameter, which is a fact about the checkpoint and not about
  which config key you happened to read.

---
## 4. Dead ends (don't retry)

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


## 5. Experiment ledger

| exp | job | task | arm | seeds | steps | score (last-15) | verdict |
|---|---|---|---|---|---|---|---|
| e558–e561 | 4693736–43 | cartpole_swingup_sparse / BIG | SOM-line + LiP | 0–3 | 4M | *launched 2026-08-17* | baseline is e550–e553 |
| e562–e565 | 4693737–43 | cheetah_run / BIG | SOM-line + LiP | 0–3 | 4M | *launched 2026-08-17* | baseline is e554–e557 |
| e566–e569 | 4693744–47 | hopper_hop / BIG | pure Director | 0–3 | 4M | **mean 220.4, std 93.6** | complete 2026-08-20. Fresh, **not** a resume of e495–e499 (see below). |
| e570–e573 | 4693748–51 | hopper_hop / BIG | SOM-line + LiP | 0–3 | 4M | **mean 211.3, std 57.5** | complete 2026-08-20. Δ −9.1 (−4.1%), p=0.943 — a **tie**, as the low-power warning predicted. |
| ~~e574–e581~~ | ~~4696162–69~~ | cheetah_run + hopper_hop / BIG | SOM-line | 0–3 | 4M | **cancelled 2026-08-19, never started** | queued 2026-08-18, cancelled while still PENDING to free the lane for e582–e586. Their rows were removed from the watchdog TSV first, or it would have resubmitted them. Nothing was lost — no job ever ran. |
| e582–e586 | 4696814–18 | all 5 benchmarks / BIG | SOM-line + **Poisson manager** | 0 | 4M | **PAUSED 2026-08-20 at ~1.9M** | one seed per task. Cancelled (not archived) to free the lane for e587–e591; checkpoint + 4–7GB replay intact, resume with `sbatch/resume_e582_e586_som_line_poisson.sh`. Rows pulled from the watchdog TSV at pause time and recorded in `job_logs/e582_e586_paused.tsv`. Early-health prediction confirmed, details below. |
| ~~e587–e591~~ | ~~4697638–42~~ | all 5 benchmarks / BIG | SOM-line+LiP + Poisson manager | 0 | 4M | **cancelled 2026-08-20 at ~45min** | superseded by e592–e611 before producing anything usable; the Gaussian head was preferred at this scale (see §6). Watchdog rows removed before the scancel. |
| e592–e599 | 4697649–56 | cartpole_swingup, hopper_stand, cheetah_run, hopper_hop / BIG | SOM-line+LiP (**ReLU**) + **Gaussian manager** | 0–1 | 4M | **done** — last-15 mean at 4M: cartpole **776.4**, hopper stand **830.2**, cheetah **439.1**, hopper hop **224.0**; aggregate normalized **0.567** vs Director 0.560 (p 0.83) and SOM-line+LiP 0.607 (p 0.29) | 4 tasks × 2 seeds. ReLU makes `prod_l softplus(c_l)` an exact Lipschitz bound; straight-through stays on; one scalar bound per layer. Not a single-factor change from e534–e573 — activation and manager head move together. **Level with Director overall, 6.6% under SOM-line+LiP, and the whole gap is cheetah: −221.8 (−33.6%), p at the 1/15 floor with no seed overlap — exactly the benchmark where SOM-line+LiP had beaten Director.** Numbers from `experiments/manager_head/gaussian_scores.py`; in the paper as Table 2 + Figure 13 of §motivation. |
| ~~e600–e611~~ | ~~4697657–68~~ | same | same | 2–4 | 4M | **cancelled 2026-08-21, never ran** | seeds 2–4 dropped to free the lane for the eps comparison. e600 alone had 26 min before being cancelled. |
| e612–e619 | 4697887–94 | cartpole_swingup, hopper_stand, cheetah_run, hopper_hop / BIG | same **+ 10% ε-greedy index jump** | 0–1 | 4M | *launched 2026-08-21* | differs from e592–e599 by `mgr_explore_eps` **alone** (0 → 0.1), so it is a clean single-factor test on the same seeds. Jump is actor-only, so the manager's REINFORCE gradient stays on-policy — see §6. |
| e620–e623 | 4697896–99 | **antmaze XL** / BIG | Director (2) vs SOM-line+LiP/ReLU/Gaussian/**eps** (2) | 0–1 | **10M** | *queued held 2026-08-21* | first exploration-limited benchmark. `EXTRA_CONFIGS=loconav`, `TRAIN_RATIO=256`. ~10× the compute of a 4M DMC run; will requeue across several 48h slices. |
| e624–e627 | 4697904–07 | **pinpad six** / BIG | Director (2) vs SOM-line+LiP/ReLU/Gaussian/**eps** (2) | 0–1 | **4M** | **first 4M done 2026-08-23** — mean score over 3M–4M: Director 1.6 / 0.0, arm **282.8** / 0.2 | no config block, launcher defaults (train_ratio 64). Cheap image env. Both Director seeds are essentially at zero at 4M and one arm seed is too, so 4M is too short to rank the arms here. |
| e620–e623 (paused) | 4699098–4699101 | **antmaze XL** / BIG | Director (2) vs SOM-line+LiP/ReLU/Gaussian/eps (2) | 0–1 | 10M → **6M** | **PAUSED 2026-08-24 at ~1.8M** | target cut 10M→6M on 08-24, then cancelled (**not archived**) to free the lane for e628–e635. Checkpoints + 6.6–7.6GB replay all intact in `/work`; state in `job_logs/e620_e623_paused.tsv`, resume with `sbatch/resume_e620_e623_antmazexl.sh`. No watchdog tracked them, so nothing needed removing. |
| e628–e629 | 4699190–91 | **pinpad six** / BIG | SOM-line+LiP/ReLU + **Student-t (Cauchy ν=1)** manager, no eps | 0–1 | 4M | *launched 2026-08-24* | new `mgr_studentt` head: polynomial tails, so the far half of the ordered codebook is reachable **and still ordered**, which lets the manager walk `μ` there instead of needing the eps jump. Runs at `manager_actent_target` **0.7** — ν=1's tie floor is 0.601, so 0.5 is unreachable. Differs from e626/e627 by head + target + eps (three factors); a Gaussian-at-0.7 arm would separate them. |
| e630–e631 | 4699192–93 | cheetah_run / BIG | same Cauchy arm | 0–1 | 4M | *launched 2026-08-24* | dense-control control for the Cauchy head against e614/e618 (Gaussian+eps, mean at 4M). |
| e632–e633 | 4699180–81 | **pinpad five** / BIG | pure Director | 0–1 | 4M | **done 2026-08-25** — last-15 mean **0.0 (s0) / 158.0 (s1)**, peaks 1270 / 1350 | baseline for e634–e635. **Both seeds find reward** (peaks >1200) but s0 ends at 0.0 while s1 ends at 158.0 — the arm is not failing to explore, it is failing to *hold* the behaviour. That spread across two seeds is larger than any effect we would be trying to measure, so **2-seed comparisons on pinpad five are uninformative**; this is the case for the 5-seed target in `DIRECTOR_BASELINES.md`, and it applies directly to e636–e637 (count-novelty on the same task). Shorter pad sequence than pinpad six, so less exploration-bound. |
| e634–e635 | 4699194–95 | **pinpad five** / BIG | same Cauchy arm | 0–1 | 4M | *launched 2026-08-24* | vs e632–e633, a clean single-factor comparison: both new, both 4M, differing only in the manager head (and its required target). |
| ~~e628–e635 (first try)~~ | ~~4699176–83~~ | as above | as above | 0–1 | 4M | **cancelled 2026-08-24 after ~1 min** | launched with `manager_actent_target` **0.5**: `run_v3_goal_ae_ablation_big_a100.sbatch` passed the target on the CLI, which beats the `mgr_studentt` block that sets 0.7 — the same "a CLI flag beats a config block" trap the script already documented for `TRAIN_RATIO`. At 0.5 a ν=1 Cauchy cannot reach target and the adapter rails (e479). Fixed with an `ARM_ACTENT_TARGET`/`MGR_ACTENT_TARGET` variable; the two Director rows (e632/e633) were correct at 0.5 and were left running. |
> **PINPAD EPISODE LENGTH CORRECTED 2026-08-26.** The Director paper (Sec. 3.1) states *"Episodes last for 2000 steps"*, but both our `pinpad.py` and the released Director repo default the constructor to `length=10000` and neither ships an env override — so **every pinpad run before this date used a 5x-too-long episode** (measured `episode/length` = 10,001). Reward is +10 per completed pad sequence with the position reset after, so score scales roughly linearly with episode length and those numbers are not comparable to the paper's Figure 5. Fixed by adding `env.pinpad: {length: 2000}` to the defaults. **Invalidated as pinpad results:** e624–e625, e626–e629, e632–e635 (finished) and e636–e639, e644–e647 (cancelled mid-flight). Relaunched as e648–e651 (novelty) and e652–e655 (Director). Non-pinpad cells are unaffected.

| e636–e643 | 4700173–80 | **pinpad five, pinpad six, cheetah_run, cartpole_swingup** / BIG | SOM-line+LiP/**ReLU**/categorical + **count-based novelty as a THIRD manager reward and critic** | 0–1 | 4M | *launched 2026-08-25* | New `mgr_novel`: the manager now carries one task reward and TWO exploration rewards, each with its own critic — the existing goal-AE reconstruction error (`mgr_expl_val`) and decaying visit counts over the joint goal code (`mgr_novel_val`). Both enter at `mgr_expl_weight` 0.1, so total exploration pull is **0.2, double** the usual. Counts live on two grids over the JOINT code (4 bins/block fine = 65,536 cells, 2 bins/block coarse = 256); per-block counting was measured to miss real structure (shuffling blocks independently, which preserves every marginal, spreads the codes over 3–4× more regions). Decay `exp(−batch·len/(train_ratio·replay.size))` = 0.999984 and weight 1/`train_ratio`, both derived, so the memory forgets exactly as fast as replay and a count reads as "env steps spent near this code in the replay window" (table settles at `replay.size`). **Hypothesis:** reconstruction error is coupled to autoencoder quality — a better AE explores less by construction and the bonus fades just as the manager becomes able to use it; a visitation count is independent of that and does not fade. **Expected:** clearest gain on the exploration-bound pinpads. **Caveat:** ReLU+categorical has never been run alone (`lip_relu` only ever shipped with `mgr_gaussian`/`mgr_studentt`), so there is no matched baseline and these cells measure the combination, not the exploration change in isolation — control arm `som_lipvq_line_relu` deliberately not run per instruction. | **INTERIM 2026-08-26, step-matched, mid-run:** cheetah @1.56M — novelty **561.7** [478–646] n=2 vs SOM-line+LiP silu 507.0 [403–596] n=4 vs Director 444.2 [235–553] n=4. cartpole @0.93M — novelty **713.4** [643–784] vs silu 632.0 [565–691] vs Director 517.5 [209–743]. Same ordering on both tasks, both novelty seeds above the silu mean in each. **Not conclusive:** n=2 against n=4, ranges overlap, and arms can still cross before 4M. **Not single-factor** either — the novelty arm differs from the silu arm by ReLU *and* the count channel. Note the two arms that previously carried ReLU both underperformed (ReLU+Gaussian aggregate 0.567 vs silu 0.607, losing 33.6% on cheetah specifically), so if ReLU is neutral-to-harmful the count channel is doing more work than the raw gap shows — but that inference needs the `som_lipvq_line_relu` control, which was deliberately not run. Both pinpad cells remain unresolvable: pinpad five novelty 482.7/48.0 against Director 932.7/2.7, one high and one dead seed in each arm.
| e644–e647 | 4700195–96, 4700201–02 | **pinpad five** (s2,s3,s4), **pinpad six** (s2) / BIG | pure Director, seed fill | 2–4 | 4M | *launched 2026-08-25* | Not a new arm — filling toward **5 seeds per environment**, tracked in `DIRECTOR_BASELINES.md` (12 of 65 seeds done at launch). Training-identical to the A100 Director seeds via `run_v3_director_baseline_multigpu.sbatch`, only device placement differs: BIG needs **4 GPUs** on a 16 GB card (batch 16×64 is ~59 GB on A100, ~12 GB/device split 4 ways), so v100 and p100 give 2 concurrent jobs each. Measured **26.3 env fps on V100** (~43 h/seed), **18.2 on P100** (~61 h). e644–e646 complete pinpad five at 5/5. **First attempt (4700184/85, 4700186) died within a minute**: the A100 scripts `unset DREAMERV3_CONV_IMPL` for native cuDNN conv, which fails on V100/P100 in the stock env (`All algorithms tried for __cudnn$convForward failed`). Fixed by branching the env on partition — Route-A clone + `--xla_gpu_strict_conv_algorithm_picker=false` + explicit `WANDB_API_KEY`. Reference conv is **not** a fallback on 16 GB: forcing it (4700192) OOM'd at 13.4 GB/device because it costs more memory than cuDNN. |
| e640–e641 FINAL | 4700177–78 | cheetah_run / BIG | count novelty, equal weight | 0–1 | 4M | **3,999,984 / 3,995,992 — late collapse** | Both finished 2026-08-26 and are archived. **The count arm does not hold what it learns.** Over the final 400k all four Director seeds are flat (−1%, −1%, −2%, −8%) while both count seeds fall: e640 764→548 (**−28%**), e641 430→183 (**−57%**). Peaks are higher than any Director seed (832 / 461), so the arm climbs further and then gives it back. The earlier "count 541 vs Director 454" reading of the 3.5–4.0M bin is measured *during* that collapse and should not be quoted as a win. Consistent with the mechanism: the recon bonus fades to 3% of the manager's advantage so Director consolidates on task reward alone, while counts hold 20–25% to 4M and keep pulling the policy off learned behaviour. With the pinpad start-delay this makes the count weight wrong at **both** ends and motivates a decaying schedule on `mgr_novel_weight`. |
| e642–e643 FINAL | 4700179–80 | cartpole_swingup / BIG | count novelty, equal weight | 0–1 | 4M | **857 / 835 — flat, no collapse** | Finished 2026-08-27. **This retracts the generalisation drawn from cheetah.** Both cartpole count seeds are flat over the final 400k (−0%, +0%) and finish at the top of the Director range (602/749/755/856), so the late collapse is *not* a property of the count arm. Collapse severity tracks fine-grid saturation monotonically across all four equal-weight runs: e641 98.7% occupied / 2.06× spread → −57%; e640 92.0% / 2.87× → −28%; e643 81.6% / 3.83× → +0%; e642 78.9% / 3.58× → −0%. Mechanism: once the grid fills, the bonus stops discriminating (spread → uniform) but keeps its 20–25% share of the manager's advantage, so the policy is pushed by what has become noise. Fixes: decay `mgr_novel_weight`, tie it to measured spread, or raise `fine_bins`. |
| e648–e651 | 4700537–40 | **pinpad five, pinpad six** / BIG | SOM-line+LiP/ReLU/categorical + count novelty | 0–1 | 4M | *launched 2026-08-26* | Relaunch of e636–e639 at the paper's **2000-step episode**. Identical in every other respect. |
| e652–e655 | 4700541–44 | **pinpad five** / BIG | pure Director, seed fill | 0–3 | 4M | **ALL FOUR FINAL** (2026-08-28, archived + removed from `/work`): 3,997,904 / 3,997,632 / 3,997,824 / 3,998,656 steps — final last-15 **35.3 / 32.0 / 20.0 / 107.3** | Relaunch of the pinpad five baseline at **2000-step episodes**; e632/e633 and e644–e646 are void. Paper reports Director solving Pin Pad Five reliably over 5 seeds within ~6M steps (Fig. 5, y-axis to ~240), so a seed at 0.0 would be a genuine failure rather than an early read. **e652/e653 finished 2026-08-28 and are archived** (bucket, `SKIP_REPLAY=1`, deleted from `/work`). **The four final last-15 values are 35.3 / 32.0 / 20.0 / 107.3 — a 5.4× spread across seeds of the SAME algorithm on the SAME task.** e655 alone read 181 an hour before it finished and ended at 107.3. End-of-run snapshots are therefore a poor summary of pinpad five for any arm, and the sustained-50 crossing step remains the metric to compare on; quoting a final score here without the spread would misrepresent every comparison built on it. |
> **PINPAD FIVE LADDER, INTERIM 2026-08-28 — the selection form fixes what the reward form broke.** All at the corrected 2000-step episode. Metric: step at which the last-15 episode mean first clears 50 (first-reward is useless here, it fires on one fluke episode). **Director** (n=5, updated 2026-08-28): 144k / 160k / 176k / 256k / **496k**, scoring on 77–90% of episodes for the first four. **The fifth seed (e676) lands at 496k — 3.4× the fastest seed — and it alone widens the baseline band from 144–256k to 144–496k.** That weakens the "3–12×" delay claim below: read against n=5, the w=0.03 seed at 304k falls *inside* Director's own spread, and the w=0.1 seed at 608k is only 1.2× Director's slowest rather than 2.4×. Only 928k and 1953k remain clearly outside. Caveat in both directions: e676's crossing is marginal (last-15 = 55.3, barely over the bar, 8% of episodes scoring at 511k) and may not hold. **UCB c=20** (n=2, at ~2.55M): **176k / 208k**, scoring 71/86% — inside Director's band. **Count as a REWARD**: w=0.1 → 608k and **1953k** (50% / 19% scoring); w=0.03 → 304k / 928k (60% / 63%). **No-count control** (n=2, at 2.24M): 240k and never (81% / 0%). So counts *as a reward* delay sustained scoring versus Director — but **by how much depends on how many Director seeds you have**, and at n=5 the honest range is ~1.2–3.9× on the slowest seeds rather than 3–12×; the same counts *as a selection-time tilt* do not delay it at all. This is the bootstrapping argument confirmed on the task where it matters: `mgr_novel_val` learns discounted FUTURE novelty and permanently distorts what the manager wants, while a selection bonus never enters a return. **Do not over-read:** n=2, UCB runs unfinished, and pinpad five oscillates violently for every arm including Director (e652 runs 224→226→95→30→154 across its 4M), so final ordering can still change. |

| e668–e675 | 4701772–79 | **cheetah_run, cartpole_swingup, pinpad_five, pinpad_six** / BIG | UCB arm, selection counts from policy **SAMPLES** | 0–1 | 4M | *launched 2026-08-27* | Replaces e660–e667, killed at ~100k. Identical config (`mgr_ucb_c` 20, 8 candidates, `mgr_expl_weight` 0.15); the only change is that `_mgr_select_update` draws samples instead of the mode. |
| ~~e660–e667~~ | 4701727–34 | as below | UCB arm, selection counts from the **mode** — **VOID, killed at ~100k** | 0–1 | 4M | **no annealing: flip rate pinned at 76%** | **The table and the candidates were on different distributions.** UCB candidates are policy *samples*, but `_mgr_select_update` counted the manager's *mode* (`deterministic=True`), and with a 74%-dominant class per block the mode collapses to nearly one code per state. Measured live at 100k: `select_eff_cells` **21–59** of 65,536, i.e. mass in ~30 cells, so **99.95% of the grid sat at bonus 1.0 permanently**. Candidates almost never landed in a counted cell, so their bonus never decayed — simulated candidate bonus 0.73 and flip rate **76.3% at both 12.5k and 500k decisions**, i.e. the manager overridden three times in four for the whole run with zero fade. That is precisely the pathology the cumulative table was introduced to avoid (cf. the cheetah late collapse). Counting samples instead puts the table on the candidates' own distribution: bonus 0.31→0.08 and flip 93%→32% across a 4M run. Caught by the standing instruction to stop and requeue *all* runs together if the constants stop making sense. |
 **cheetah_run, cartpole_swingup, pinpad_five, pinpad_six** / BIG | SOM-line+LiP/ReLU/categorical + **counts as selection-time UCB, no reward, no third critic** (`mgr_ucb_c` 20, 8 candidates), `mgr_expl_weight` **0.15** | 0–1 | 4M | *launched 2026-08-27* | **Counts move from the return to the decision.** A count reward is bootstrapped by its critic into "prefer goals that LEAD to novelty later" — a permanent change to the manager's objective, and the reason an uninformative bonus could still drag cheetah down 28–57% late. Here the bonus is added to the score when picking a goal in the environment rollout only, never to a return, so the learned objective stays pure task reward. **Two tables now:** selections (cumulative, never decays) drive the UCB bonus so it fades as 1/sqrt(n) like textbook UCB; visits (decaying replay window) become coverage metrics plus a measured `reach_ratio` that replaces the saturating coarse-grid frontier test. Counting *proposals* rather than arrivals is what makes it self-limiting — a goal the worker can never reach still gets counted, so the manager cannot fixate on it. **`mgr_ucb_c`=20 is derived, not guessed:** at the 50% entropy target the manager sits at 8.7 of 16.6 nats, so a code's log-prob has std 3.7 nats and among 8 candidates the best beats the second by ~1.6 nats, while the bonus differs by only ~0.05–0.10 between candidates from the same concentrated policy — c must be ~15–30 to change any decision. The initial guess of 3.0 would have shifted scores by 10–20% of the gap and made this arm indistinguishable from the no-count control. **Grid stays at 4 bins** (65,536 cells): 500k decisions per 4M run gives 7.6/cell (22–49 per *effective* cell at the measured 2.9–6.4× concentration), inside the range where 1/sqrt(n+1) actually moves — and matching the visit grid is required for `reach_ratio` to mean anything. **Known approximation:** the selection table counts the manager's untilted mode on replay states, not the tilted choice made during collection, because running the tilt inside the training scan would break the env-rollout-only scoping that keeps REINFORCE unbiased. e656–e659 paused (requeued+held) to free the partition. |
| e656–e657 | 4700585–86 | **pinpad five** / BIG | SOM-line+LiP/ReLU/categorical + count novelty at **`mgr_novel_weight` 0.03** | 0–1 | 4M | *queued 2026-08-26* | Single-factor test against e648/e649: identical arm, only the count bonus weight changes (0.1 → 0.03). **Why:** step-matched at the corrected 2000-step episode, **all 4** Director seeds (e652–e655) find reward by 224k — three by 80k — and hold 50–200 through 200–400k, while the equal-weight count arm managed **1 of 2** by 800k, and that seed spiked to 183 then collapsed to 7. **Mechanism, measured:** before any task reward exists exploration is the manager's *only* signal, and at equal weights the count term takes ~50% of it (`mgr_expl_adv` 0.056 vs `mgr_novel_adv` 0.058) — halving the reconstruction term's share in exactly the phase that has to find the first pad sequence. Director's exploration is 100% reconstruction. At 0.03 the count term takes 23% of early exploration. **Expected:** first reward inside ~250k like Director; if it stays slow, the delay is the goal AE rather than the count channel, which the equal-weight comparison cannot separate (no matched no-count arm exists at length 2000). **Note the opposite sign late:** on cheetah to 2.9M the reconstruction bonus fades 12%→3% of the manager's advantage while counts hold 20–25%, and that run (e640) reaches **800** at 3M against Director's best 589 over 4 seeds — so counts look harmful at the start and useful later, which would favour ramping the weight rather than a flat cut. New knob `agent.mgr_novel_weight`, negative = "same as `mgr_expl_weight`" so all earlier runs stay bit-identical (`hrl/losses.py:resolve_novel_weight`, `TestNovelWeightKnob`). | **INTERIM 2026-08-26, e656 at 0.32M (n=1, e657 still held):** on **sustained scoring** — the step where the last-15 episode mean first clears 50, which is the metric that matters because first-reward fires on a single fluke episode — e656 reaches it at **304k**. Director: 144k / 160k / 176k / 256k. Equal weight: 608k (e648) and **1953k** (e649 — it was at one scoring episode in 912 as late as 1.82M, then took off; the arm is very slow, not dead). So 0.03 is **2× faster than the only equal-weight seed that got there** and within reach of Director's slowest, but still ~2× behind Director's median (~168k). The residual gap is consistent with the goal autoencoder rather than the count weight, which this arm cannot separate — that needs the `som_lipvq_line_relu` no-count control, still unrun. **Do not over-read:** n=1 at 8% of the run, and pinpad five seed variance is large (Director e655 first-scored at 32k yet sits at a last-15 of 74.7). |
| e658–e659 | 4700786–87 | **pinpad five** / BIG | SOM-line+LiP/ReLU/categorical, **no count reward** (`som_lipvq_line_relu`) | 0–1 | 4M | *queued held 2026-08-26* | **The missing control.** Every count-novelty cell so far differs from Director by *two* things — the goal autoencoder (SOM-line+LiP/ReLU/categorical) and the count channel — so no result among e636–e657 can attribute a difference to either. This arm is the count arm with `mgr_novel` off, making e648/e649 (w=0.1), e656/e657 (w=0.03) and e658/e659 (no counts) a single-factor ladder on one task. **Reads:** if e658/e659 sustain at Director's 144–256k, the ~2× residual gap at w=0.03 is the count channel and the weight should go lower still; if they sustain near e656's 304k, the residual is the autoencoder and the count weight is already about right. Submitted **held** — a fresh submission outranks a requeued one here (e657 at priority 10377 vs preempted e648 at 10126), so these release only after e648 and e657 are running. |
| e676–e681 | 4701951–56, **e678/e679 resubmitted as 4702072–73** | **pinpad five** (e676), **pinpad six** (e677–e681) / BIG | pure Director, seed fill to 5/env | 0–4 | 4M | *running / queued 2026-08-28* | Completes the 2000-step-episode baselines. **e676 (pinpad five seed 4) crossed sustained-50 at 496k**, versus 144k/160k/176k/256k for the first four seeds — a 3.4× spread that widens the pinpad-five band to 144–496k and moderates the "counts as a reward delay learning 3–12×" claim in the ladder note above. **e676 FINAL: 3,996,944 steps, last-15 60.0** (finished 2026-08-30, archived + removed from `/work`). **Pinpad five Director baseline is now COMPLETE at 5 seeds** — crossings 144k/160k/176k/256k/496k, final last-15 **35.3 / 32.0 / 20.0 / 107.3 / 60.0**. **Pinpad six Director band, n=4: 640k (e677) / 848k (e679) / 864k (e678) / 1217k (e680)**
— exact sustained-50 crossings, so roughly **4–7× later than pinpad five's 176k median**,
which is the expected cost of the sixth pad. e680 first scored at 1,009k; **e681 has still
never scored at 1.2M**, so the band may widen further. First-nonzero across seeds:
400k / 688k / 704k / 1009k. Caveat: e678/e679 had scored on only **4–5%** of
episodes at the moment they crossed (last-15 of 52.0 and 74.0 carried by a handful of
recent episodes), exactly the marginal pattern e676 showed on pinpad five — treat both as
provisional until they hold it. e677 is unambiguous: 74% of episodes scoring, and it **FINISHED at 3,999,776 steps with a
final last-15 of 154.7** (2026-08-30, archived + removed from `/work`). Director
first-nonzero-score on pinpad six: **400k / 688k / 704k**. **All five pinpad six seeds are
now accounted for**: e677 done, e678–e681 running (e680/e681 started on the v100 slots
e676/e677 freed, with no idle gap).

> **e678 FINISHED — AND ITS "last-15 = 0.0" IS A TRAP (2026-08-31).** COMPLETED at
> 3,996,528 steps, 13 h 16 m on p100. The headline number is **last-15 0.0**, which read
> alone says the seed died. It did not: last-50 is **21.4**, last-100 **58.3**, and the
> final 500k bin averages **89 with 90% of episodes scoring** — the best bin of the run.
> Binned in 25s, the run holds 24–25/25 scoring with means of 40–157 all the way to ~3.95M
> and then collapses in the **final 25 episodes** (1/25 scoring, mean 1.2, ~48k steps). So
> it is a real collapse, but one confined to the last 1% of the run, and quoting 0.0 as this
> seed's result would misrepresent it completely. This is the third time end-of-run last-15
> has misled on a pinpad run (cf. e655 reading 181 an hour before finishing at 107.3, and
> e632/e633) — **for pinpad, report the crossing step plus a late-window mean, never the
> final last-15 alone.** Sustained-50 crossing confirmed at **864k**, matching the band above.
> **This seed is also a third independent case for the `director_stable` hypothesis
> (e718–e721) — and it refines it.** `mgr_extr_val` by 500k bin: 0.000 0.182 0.756 0.635
> **27.802** **−11.852** 1.440 3.238 (final row 0.851), on a task whose reward is 0 or +10
> and can never be negative — the same runaway seen in e654 (−23) and e655 (−210), now
> reproduced on **pinpad six** rather than pinpad five, so the pathology is not
> pinpad-five-specific. `goal/rec_mean` grows **6.68 → 36.9** (5.5×), matching the "the
> manager's exploration reward grows in absolute scale all run" observation (they measured
> 6.4 → 52). **The refinement:** the e718–e721 note says the negative excursion is "what
> blocks recovery, not what starts the fall", with both never-negative seeds recovering.
> e678 went to −11.85 — and **recovered anyway**, from a 0-scoring 2.5–3.0M bin to 90%
> scoring at the end. So going negative does not always block recovery; magnitude plausibly
> matters (−11.9 here against −23 and −210 in the seeds that stayed down). Worth checking
> against e719–e721 when they land.
>
> **e679 FINISHED THE SAME DAY and completes the picture (2026-08-31).** COMPLETED at
> 3,992,032 steps, 13 h 21 m on p100. Crossing **848k** (matches the band). Final last-15
> **33.3**, but again read the window: last-100 **67.5**, final 500k bin **79 with 96% of
> episodes scoring**. Its curve is the classic Director shape this project keeps hitting —
> bins **0 38 77 172 145 73 56 79**, peaking at 1.5–2.0M and settling ~55% below the peak.
> **`mgr_extr_val` never goes negative here** (0.000 0.716 2.836 5.038 4.964 3.618 2.414
> 3.054) and the seed holds. With e678 the four-seed picture across pinpad five and six is
> now: **large negative (−23 e654, −210 e655) → stays down; small negative (−11.9 e678) →
> recovers; never negative (e679, plus the two pinpad-five seeds) → recovers.** Magnitude,
> not sign, is what tracks the outcome — sharper than the original "negative blocks
> recovery" statement. `goal/rec_mean` grows on both seeds regardless (e678 6.7 → 36.9,
> e679 7.9 → 38.5, both ~5×), so the wd=0 parameter-norm diagnosis is independent of whether
> the value diverges. **e679 also left in `/work` unarchived**, same 4M → 6M reasoning as
> e678. Pinpad six Director is now **3 of 5 finished** (e677 154.7, e678, e679); e680/e681
> still running on v100.
> **NOT archived, deliberately:** pinpad six Director runs have precedent for being extended
> 4M → 6M (e624/e625 were, 2026-08-24), and archiving with `SKIP_REPLAY=1` would destroy
> resumability ([[dreamerv3-archiving-destroys-resumability]]). Left in `/work` pending a
> decision on whether the pinpad six baseline goes to 6M. `/work` is at 26%, so there is no
> space pressure forcing it.

> **PINPAD SIX, TILT vs DIRECTOR (interim, 2026-08-29, e710/e711 at ~0.5M of 4M).**
> **e710 crossed sustained-50 at 336k — before *any* Director seed had scored even once**
> (earliest Director first-reward is 400k), and holds 42% of episodes scoring with last-15
> 128.7. Against Director's crossings of 640k / 848k / 864k that is roughly **2× faster
> than the best baseline seed**. **e711 has not scored at all at 512k — and that is
> normal, not a failure:** every Director seed was also at 0% by 512k, and two of the three
> did not score once until ~690–700k. So the pair is one clearly-fast seed and one that is
> simply too early to read, which at n=2 supports no mean and no ordering. Worth flagging
> as the most promising signal so far, and worth *not* over-reading: pinpad six variance is
> exactly what makes n=2 useless here (see the n=2 resolution problem above). **e678/e679 moved v100 → p100 on 2026-08-28**: they were blocked behind `AssocGrpGRES` on v100 until e676/e677 finish (~2 days), while p100 nodes saion-gpu[13-14] sat idle and our p100 quota freed as soon as e654/e655 completed. Same script, same config, only the GPU type differs — the script documents the p100 path (`-p gpu-p100 --gres=gpu:p100:4 --nodelist=saion-gpu[11-14]`, Route-A env for both v100 and p100). **P100 is 1.4× SLOWER, and the move is still right — the justification is queue position, not throughput.** Measured `fps/policy` from the logs: **v100 26.30, p100 18.75** (matching the long-standing 26.3/18.2 note; an earlier claim here that p100 ran at ~35 steps/s was wrong — it came from reading heartbeat step-deltas as 1-hour intervals when the readout fires every 2 h). The arithmetic that actually decides it: on p100 starting now, 4M at 18.75 fps ≈ **59 h**. Waiting for a v100 slot means 2.82M left on e676/e677 at 26.30 fps ≈ 30 h, *then* 4M at 26.30 ≈ 42 h ≈ **72 h**. So starting ~30 h earlier more than pays for the 1.4× slowdown. e680/e681 stay on v100 (p100 is at its 8-GPU cap). |
| e706–e717 | 4702043–54 | **antmaze L, pinpad five, pinpad six, cartpole_swingup, cheetah_run, hopper_hop** / BIG | UCB arm with **tilted sampling**, `mgr_ucb_c` **10**, smeared selection grid `select_bins` 4 / `select_h` 1.0 | 0–1 | 4M | **ALL FIVE DENSE-TASK RUNS COMPLETE at 4M** (2026-08-31, archived + removed from `/work`): cartpole s1 **659.1**, cheetah **795.4 / 584.4** (*both above every Director seed*), hopper **161.1 / 172.5**. **The annealing claim is falsified** — gain rises in all five, 4 of 5 peak in the last bin, all at 7.6 counts/cell. e708–e712 eligible, e706/e707 (antmaze L) still held | **Replaces e668–e705, all void for the sharpening bug.** The old rule picked `argmax(log π + c·bonus)` over 8 policy samples, which double-counts the policy: it is a best-of-K likelihood filter, so it *sharpened* the manager rather than exploring it — measured `E[log π]` **−9.46 → −6.70**, discarding ~25% of the manager's entropy, and a directly sampled goal is the best of its own eight only **12%** of the time. New rule is `P(k) ∝ exp(c·bonus_k)` by Gumbel-max, **with no logp term**; `c=10` derived from the manager's real per-state probabilities (chosen goal 1.3–1.5× rarer than the candidate average for ~1.4 nats of 9.5). Also fixes a half-class offset in the cell centres (`(j+0.5)·C/b` → `(j+0.5)·C/b − 0.5`) that put 3 of 8 classes in the wrong cell. **Pre-launch validation (CPU, 2026-08-28):** `_ucb_diag` runs inside a real train step; `ucb_logp_shift` mean **−0.0021 ± 0.0175** (indistinguishable from 0, i.e. not sharpening); smearing live, `select_eff_cells` 748 → 10,171 where a hard deposit would sit at ~1. **Reading the diagnostics:** an empty table and a saturated one give *identical* readings (gain exactly 1.0, shift noise ±0.05), so gate on mean count = `select_mass`/65,536 — sharpening if shift > 0.25 with mean count > 0.5; saturated if mean count > 2 with >85% of cells carrying mass and gain < 1.02. |
| e718–e721 | 4703164–67 | **pinpad_five** / BIG, a100 | Director baseline + `director_stable`: `manager_slowtar` **True**, `advnorm.impl` **meanstd** (was `none`), `mgr_retnorm.limit` **1e-2** (was 1e-8), `opt/ac_opt/goal_opt.wd` **2.5e-2** (was 0) | 0–3 | 4M | launched 2026-08-31 | **Hypothesis: the four DreamerV3 defaults `director_match` never matched are what make our pinpad Director curve peak early and decay, where the paper's rises late and holds.** The e652–e655/e676 five-seed baseline peaks at mean **214** by 1M and falls to **83** by 3.75M; Hafner et al. Fig. 5 is flat 0 until ~1.8M then climbs steadily to ~130 at 6M. Diagnosis from those runs' metrics: (1) the manager's task critic bootstraps off ITSELF (`slowtar` False → live critic is its own target, `hrl/losses.py:241`), and in 2 of 5 seeds it ran away to `mgr_extr_val` **−23** (e654) and **−210** (e655) on a task whose reward is 0 or +10 and can never be negative — the score collapse came FIRST (e654: 75.9 → 0.1 between 3.0M and 3.25M, value flipped at 3.287M), so this is what blocks recovery, not what starts the fall; both seeds that never went negative recovered. (2) TF Director normalizes TWICE — per stream by running std, then the weighted SUM by running mean+std (`agent.py:329-337`) — and our second stage was `impl: none`, so manager `mgr_adv_std` ran a median 0.3 with peaks 10.3/10.5/14.7, i.e. policy updates 30–50× normal size. (3) our per-stream std floor was 1e-8 against Director's 1e-2 (`retnorm.max: 1e2`). (4) with wd 0 nothing bounds parameter norm: AC param RMS 0.046 → 0.109 (2.4×, still linear at 4M), world model 0.0293 → 0.0386; the goal AE reconstructs `deter` with a summed MSE so `goal/rec_mean` grew 6.4 → 52, and that error IS the manager's exploration reward (`mgr_expl_rew` 0.0069 → 0.0559, the same 8×) — its second objective grew in absolute scale all run instead of converging. wd **2.5e-2** not 1e-2 because both decays are decoupled (shrinkage = wd·lr) and our lr is 4e-5 vs Director's 1e-4: 2.5e-2·4e-5 = 1e-2·1e-4. **Expected:** no negative `mgr_extr_val`, `mgr_adv_std` pinned near 1, `goal/rec_mean` and the parameter RMS flat rather than climbing, and the peak held to 4M instead of decaying. Four-factor change on purpose — if it works, split it. Still unmatched: `mgr_retnorm.rate` 0.01 vs Director's decay 0.999, `mgr_retnorm.debias` False vs their always-on correction, and the manager discount (ours 0.997⁸ ≈ 0.976 per decision ≈ 330 env steps, Director's 0.99 ≈ 800). Baseline to beat: e652–e655/e676. Job log `job_logs/e718_e721_director_stable.tsv`. |
> **e718–e721 INTERIM AT 3.4M/4M: ALL FOUR FIXES LANDED, THE CURVE DID NOT MOVE
> (2026-09-01).** Each of the four `director_stable` factors hit its stated mechanical
> target, and the seed-averaged score curve is indistinguishable from the e652–e655/e676
> baseline. Binned 500k means, ours (4 seeds, live) vs plain Director (5 seeds, done):
> **43/155/191/73/51/88/67** vs **69/144/202/130/109/86/75**. Same peak height, same peak
> location (1.0M), same decay; the stable arm is *worse* through 1.5–2.0M (73/51 vs
> 130/109) and equal by 2.5M. **The four factors, scored individually at 3.35M:**
> (1) `manager_slowtar` **WORKED** — `mgr_extr_val` min is exactly 0.000 in all four seeds
> against the baseline's −22.98 (e654) and −209.64 (e655); the runaway is gone.
> (2) `advnorm meanstd` **WORKED, AND ALSO SILENTLY CHANGED THE WORKER** — `mgr_adv_mag`
> median 0.67–0.70, max ≤1.01 everywhere, against baseline median 0.21–0.27 with maxima
> 1.00/1.41/2.62/1.81/0.77. Note the *typical* manager update is now ~3x larger, not
> smaller: the fix compresses the range from roughly [0.1, 14] into [0.6, 1.0].
> **`director_stable` IS A FIVE-FACTOR CHANGE, NOT FOUR (found 2026-09-01).**
> `agent.py:384-385` builds `mgr_advnorm` and `wkr_goal_advnorm` from the *same*
> `config.advnorm` key (separate instances, separate running stats, but one `impl`), so
> `advnorm: {impl: meanstd}` normalized the **worker's** advantage as well. Measured
> `wkr_goal_adv_mag`: **0.037–0.043 (plain baseline) → 0.699–0.700 (stable)**, a **~17x
> increase in the worker's policy update**, while `wkr_goal_adv_std` is unchanged at
> 0.055–0.061 in both arms — so this is purely the normalization, not a change in the
> underlying advantage. The unlogged fifth factor is **larger than the manager effect the
> arm was designed around** (17x vs ~3x).
> **Direction is Director-matching:** TF Director builds worker and manager both as
> `ImagActorCritic`, whose `__init__` does `self.advnorm = Normalize(**config.advnorm)`,
> and the manager's `mconfig` overrides only `actor_grad_cont` and `actent.target` — so
> Director normalizes *both* actors with `mean_std`. Our plain baseline (`impl: none` on
> both) was the deviation. Parameters are effectively equivalent: our `rate: 0.01` is their
> `decay: 0.99`, and their `max: 1e8` / our `limit: 1e-8` are both effectively inert.
> **What it explains:** not the collapse — the baseline collapses just as hard with no
> worker advnorm — but it is the leading candidate for why the stable arm is *worse than
> baseline* exactly through 1.5–2.0M (73/51 vs 130/109), the window it was meant to help.
> It is also a concrete mechanism for the worker-converges-harder reading of the collapse
> signature below. **To separate the manager and worker arms, `advnorm` has to be split
> into two config keys first; today it cannot be done from config.**
> (3) `mgr_retnorm.limit 1e-2` **WORKED** — `mgr_adv_std` max 0.76–0.95 against baseline
> 0.86/1.68/4.79/6.35/10.48.
> (4) `wd 2.5e-2` **FAILED ON ITS OWN TARGET** — `goal/rec_mean` still grows 5–6x
> (6.5→28–41) against the baseline's 4–5x (8.9–11.9→32–57), and `opt/ac_param_rms` lands at
> 0.094–0.099, *inside* the baseline's 0.088–0.116 range. wd lowered the starting point,
> not the growth rate.
> **Why the curve did not move.** The original note said the value runaway is "what blocks
> recovery, not what starts the fall". That reading was right, and it is the reason the arm
> is a null: only 2 of 5 baseline seeds ever had the runaway, the other 3 decayed just as
> hard and recovered unaided, so removing the recovery-blocker could only ever rescue a
> minority of seeds and the mean curve was never driven by them. The decay is present in
> every seed, including the healthy baseline ones, so its trigger is untouched by all four
> factors. **Consistent with that, the fix did do the one thing it should:** no seed is
> permanently dead. Final 250k bins read 37/4/114/122 (e718/e719/e720/e721) with e720 and
> e721 mid-recovery, against the baseline's e654 at 8.5 and e676 at 32.0. The runs
> oscillate with a ~500k–1M period and >200 amplitude, so any single endpoint is
> uninformative — cf. [[pinpad-final-last15-is-a-trap]].
> **Collapse signature (four seeds, 50k bins through each crash).** `wkr_goal_rew`
> roughly doubles across every collapse and does not come back: e718 0.19→0.52,
> e721 0.25→0.53, e720 0.34→0.53. The worker gets *better* at reaching goals while the
> score falls ~90%, which points at the manager proposing easy, worthless goals rather than
> at any execution failure. Not airtight: e720 recovered to 177 with `wkr_goal_rew` at its
> run maximum. `mgr_extr_val` correlates +0.89 with binned score — a mirror of the
> collapse, not a cause. **The entropy story is ruled out:** normalized manager entropy
> holds **0.503–0.545** against its 0.5 target through every crash, so the raw
> `mgr_ent/skill` moving 8.35→8.88 (pooled corr −0.61 with score) is a ~6% wobble on a
> tightly-held controller, not drift.
>
> **TWO NEW UNMATCHED DIRECTOR DIFFERENCES, FOUND AUDITING THE GOAL VAE AGAINST
> `code/director/` (2026-09-01).** Prompted by the e718–e721 null. Full audit and the
> matching items are in `docs/VERIFICATION.md` (new *Goal autoencoder* section — the
> component had **no row at all**, not even TODO, before today).
>
> **(1) `goal_autoencoder_beta: 0.25` — ours, not Director's, and it is binding.** Our goal
> AE loss is `rec + 0.25 * kl_adapted`. Director's `train_vae_replay`/`train_vae_imag` are
> `loss = (rec + kl).mean()` with no coefficient. Verified exhaustively: `0.25` does not
> appear anywhere in Director's `configs.yaml`; `encdec_kl` occurs exactly once in the
> whole codebase (the `AutoAdapt` construction) and `goal_kl` only as an on/off boolean; no
> task preset overrides either (only `dmc_proprio` touches the goal AE, shrinking the nets
> to 3x64); their `Optimizer.__call__` has one multiplier, the fp16 grad scale, divided
> back out. **It is active, not nominal:** `goal/kl_adapt_scale_mean` is pinned at its 1.0
> ceiling from 1M onward with `goal/kl_raw_mean` at 11.9 against target 10 — the controller
> is saturated asking for more pressure — and we then apply a quarter of what it asks.
> Director in the same state applies 4x ours. **Provenance:** introduced at **1.0** on
> 2026-05-15 (`e1ab1ed`, "Add initial goal autoencoder (rec + KL vs uniform prior)"),
> changed to 0.25 on 2026-06-02 in `713d15b`, whose message says "tune related
> hyperparameters: goal_autoencoder_beta 1.0→0.25". A tuning choice, never a match.
> **Correction to the record:** `EXPERIMENTS_ARCHIVE_20260713.md` line 722 lists
> `goal_autoencoder_beta 0.25` under "Everything else passed in both scripts ... identical
> or an inert default in both". That was e118-vs-e123 — two of *our* runs, checked for
> internal consistency so the masked-goals comparison would be clean. It was never a
> Director comparison, and it reads like one.
>
> **(2) RETRACTED — there is no missing manager discount.** This slot briefly claimed our
> manager was undiscounted because `hrl/losses.py` sets `disc = 1` under `contdisc: True`.
> That is wrong: `agent.py:1289` folds the discount into the **continue head's training
> target** (`con = f32(~is_terminal); if contdisc: con *= 1 - 1/horizon`), so the 0.997 per
> env step is already inside `con` and `cumprod(con)` carries it. `disc = 1` is correct
> *because* the discount lives in `con`; `horizon: 333` is read, just not in the return
> computation. **The `VERIFICATION.md` `discount` row was right as written** ("0.99 vs
> horizon 333 (~0.997), DIFF, 4.5% effect") and has been restored. The real difference
> remains the one already recorded — ours ≈ 0.997^8 = 0.976 per manager decision against
> Director's 0.99, i.e. an effective manager horizon of ~42 decisions against ~100 — a
> genuine 2.4x gap, but a known one and small per decision, **not** a missing contraction
> and not a leading candidate for the oscillation.
>
> **Two corrections to earlier claims in this investigation**, both from reading the live
> run's `logdir/config.yaml` rather than the base defaults: the goal AE architecture is
> **not** a difference (`director_match` overrides the 3x1024 default, so these runs use
> `goal_enc`/`goal_dec` at **4 layers x 512**, exactly Director's shape; only act/norm
> differ, silu+rms vs elu+layer), and **both sides use `deter: 1024`**, so the summed-MSE
> reconstruction is over the same number of dimensions and the rec:KL balance is directly
> comparable — which is what makes the beta a clean 4x difference rather than a
> compensation for a scale mismatch elsewhere.
>
> **Next, in order.** (a) **`goal_autoencoder_beta: 1.0`** — the only *active* VAE
> difference found, run it before anything more elaborate; shipped as the
> **`director_vaebeta`** config block. (b) **Split `advnorm` into worker and manager
> keys** so the 17x worker step-size change above can be ablated apart from the manager
> normalization it was bundled with; needs an `agent.py` edit, which must NOT be made in
> this tree while e718–e721 are live (they run from it and re-read it on requeue).
> (c) **Instrumentation, because we cannot currently test the "worker arrives early and
> idles" hypothesis at all:** Director logs `success_manager` (fraction of rollouts whose
> final goal reward > 0.7) and we log no equivalent; adding that plus a time-to-reach
> within the K-block would show whether the worker is arriving early and idling for the
> remainder, which is the mechanism the collapse signature above points at but does not
> establish.
>
> **Config blocks added 2026-09-01** (composable, so the factors can be split later —
> the e718–e721 lesson): **`director_optmatch`** sets `opt`/`ac_opt`/`goal_opt` to
> Director's **lr 1e-4, wd 1e-2**. Layer it *after* `director_stable`, which set
> wd 2.5e-2 against our lr of 4e-5: both decays are decoupled, so shrinkage = `wd*lr`
> and `2.5e-2*4e-5 == 1e-2*1e-4 == 1e-6`. The block therefore **holds the shrinkage
> `director_stable` already had and raises the lr 2.5x**, rather than moving both — worth
> stating plainly, because "increase the weight decay to Director's 1e-2" reads like more
> decay and is in fact the same decay. Still unmatched in the optimizer: `eps` 1e-20 vs
> 1e-6, and our `agc: 0.3` vs their global-norm `clip: 100`.
> **`director_vaebeta`** sets `goal_autoencoder_beta: 1.0`.
> Composed as `director_match director_stable director_optmatch director_vaebeta`, the
> resolved config verifies as: lr 1e-4 / wd 1e-2 on all three optimizers (shrinkage 1e-6),
> beta 1.0, `manager_slowtar` True, `advnorm` meanstd, `mgr_retnorm.limit` 1e-2, goal
> enc/dec 4x512.
>


> **e706/e707 (antmaze L) HAVE NO BASELINE TO COMPARE AGAINST (2026-08-30).** Every
> archived antmaze Director run is antmaze **M** — `e475–e477` (clean) and `e410–e414`
> (contaminated, see [[director-baseline-contamination]]). The only antmaze **XL** runs,
> e620–e623, were paused at ~1.8M and never finished. **Nothing exists at antmaze L**, so
> these two runs currently have no matched comparator at any budget, and at 4M they may not
> score at all (0.0 at 500k; the Director paper uses ~9M for its maze results and our own
> earlier antmaze target was 6M). Options for resolving it: run antmaze L Director seeds,
> re-point the arm at antmaze **M** where clean baselines already exist, or treat these two
> as a demonstration rather than a comparison. **Combined with hopper_hop below, 4 of our 7
> usable a100 slots are on runs that cannot produce a comparison**, while e708–e711 — which
> do have complete baselines and carry 0.5–1.2M of banked progress — sit queued behind them.
>
> **e716/e717 (hopper_hop) CANNOT SUPPORT A COMPARISON — flagged, deliberately NOT
> cancelled (2026-08-30).** `dmc_hopper_hop` was retired as a comparison task on
> 2026-08-09: its seed distribution is bimodal and heavy-tailed, and bootstrap power to
> detect a difference at p<0.05 is **0.01 at five seeds** (0.24 at ten, 0.60 at twenty) —
> a 15× median difference at 1M gave a permutation p of 0.38. This matrix runs **two**
> seeds, so these two runs have even less power than that and cannot order the arms.
> They were kept because the task list was specified explicitly by the user, and dropping
> a task is a scope decision rather than a fault to be fixed autonomously; the runs also
> still surface a *gross* failure even though they cannot resolve an ordering. The cost is
> real though: 2 of our 8 a100 slots, while e708–e711 (0.5–1.2M of banked progress, and the
> runs that answer the annealing question) sit queued behind them. If slots are needed,
> these are the two to cancel first.

> **e706–e717 ARE BLOCKED, AND IT IS NOT FIXABLE FROM OUR SIDE (2026-08-29).** All 12
> sit `PD` on `gpu-a100` with an estimated start of **2026-09-01** — three days out. Two
> of them (e706/e707) did run for 40 minutes on 08-28 before being preempted, so
> opportunistic capacity does appear and the estimate is a pessimistic bound, not a
> promise. What was checked and ruled out: **(1) walltime/backfill** — `sbatch
> --test-only` returns the *same* start estimate for 2 h, 4 h, 12 h and 2 d requests, so
> the job's shape is not the constraint and shortening it buys nothing; **(2)
> `short-a100`** — same physical nodes saion-gpu[23-26], and `PriorityTier=1` against
> `gpu-a100`'s 10, so it is strictly *more* preemptible and cannot preempt the tier-10
> work occupying the nodes; **(3) v100/p100** — no v100 variant of
> `run_v3_goal_ae_ablation_big_a100.sbatch` exists (v100/p100 need the 4-GPU Route-A
> native-conv setup, and a naive port of an A100 script dies in under a minute), and both
> partitions are at our 8-GPU cap running the Director baselines anyway. Nothing of ours
> is idle: every GPU we are entitled to is in use.
>
> **The v100 fallback was suspected of a correctness risk; it was TESTED and is CLEAN.**
> The worry: on a100 a tilt run uses 1 GPU and nothing is sharded, but on v100 it needs 4,
> and `partition_rules` is unset so params default to `P()` (replicated) while the batch is
> sharded. The count tables are `nj.Variable`s updated by **accumulation, not gradients**
> (`add = deposit(ids).sum(0) * weight`), so `.sum(0)` reduces over the sharded axis into a
> replicated result and XLA *must* insert a cross-device all-reduce. Local partial sums
> would make `select_mass` 1/4 of the truth, a per-device full sum 4× — silently, in the
> quantity the arm is studying, and `select_mass` has already caught two real accounting
> bugs. **Measured on 4 simulated CPU devices: single-device 16.0, 4-way sharded 16.0,
> expected 16.0 — exact.** The gap that made this worth checking was real: all 34 tests in
> `test_mgr_novel.py` had **zero** device/shard/mesh coverage. Now covered permanently by
> **`embodied/tests/test_mgr_novel_sharded.py`** (4 tests: sharded-vs-single mass at two
> batch sizes, the `1/train_ratio` env-step weighting, and `ucb_bonus` agreeing across
> shards). Run it alone — JAX fixes the device count at first use:
> `XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
> embodied/tests/test_mgr_novel_sharded.py`; it skips rather than passing vacuously in a
> full-suite run (verified: 34 passed, 4 skipped). **So the v100 move is a genuine
> priority trade after all** — 2 tilt seeds displacing e680/e681 — with the caveat that
> this validates the count-table accounting under sharding, not the whole 4-GPU training
> loop. **Moot in the event: a100 freed within hours and e708/e709 started, so no seeds
> were displaced.** |

> **TILTED SAMPLING, FIRST RESULT (interim, 2026-08-29, e708/e709 at ~500k of 4M).**
> Sustained-50 crossing on pinpad five: **224k and 256k**, against Director (n=5) at
> 144k / 160k / 176k / 256k / 496k. Both sit **inside** Director's band and e709 lands
> exactly on its 4th seed, so **the corrected rule does not delay learning** — reproducing
> the earlier interim claim (176k/208k) that had been measured under the *buggy* sharpening
> rule. Last-15 at 500k: **158.7 / 218.7**, scoring on 61%/57% of episodes.
> **How much this is worth: not much yet.** n=2 against a baseline whose own seeds span
> 3.4× (144–496k), so "inside the band" is a weak statement — the band swallows almost any
> arm. This is the n=2 resolution problem noted above, not a result that settles it.
>
> **The mechanism diagnostics are the more interesting part.** Trajectory of
> `ucb_novelty_gain` (chosen candidate's bonus over the 8-candidate mean):
> e708 1.002 → 1.021 → 1.038 → 1.034 → **1.056**; e709 1.002 → 1.019 → 1.067 → 1.130 →
> **1.097**, while `select_bonus_mean` falls 0.584 → 0.546 → **0.527** and `eff_frac`
> falls 0.53 → 0.088 → **0.047**. So the bonus **level** anneals as designed while mass
> **concentrates**, and the spread — which is what actually tilts the choice — is still
> drifting up. That is the failure mode the old notes predicted for a 4-bin grid: with
> 65,536 cells and only ~500k decisions a never-visited candidate always exists, so the
> tilt can sharpen over training instead of fading. Smearing (`select_h` 1.0) was the
> countermeasure and it is doing something (eff_cells in the tens of thousands, not ~1).
>
> **UPDATE at 1M — BIN THE SERIES, do not compare single samples.** Per-row `gain` is
> noisy enough (±0.02–0.05) that single readings suggested first a clean rise, then a
> turnover, then neither; spread comparisons at one step were also misleading because they
> mixed pinpad five and six. Binned at 100k (~20 samples/bin) the answer is unambiguous and
> **all four seeds rise monotonically on two tasks**:
> `e708 (pp5) 1.010 1.032 1.022 1.032 1.051 1.059 1.071 1.077 1.080 1.114 1.111`;
> `e709 (pp5) 1.008 1.042 1.075 1.104 1.123 1.107 1.152 1.146 1.148 1.156 1.170`;
> `e710 (pp6) 1.008 1.071 1.158 1.232`; `e711 (pp6) 1.002 1.036 1.066 1.088`.
> Rises of +0.09 to +0.22 over their ranges, far above bin-to-bin wobble. **Method note for
> anyone re-deriving this: bin before concluding.**
>
> **CONFIRMED ACROSS FOUR TASKS (2026-08-30, 200k bins, ~40 samples/point).** Net change in
> binned gain from first to last bin: **e708 pp5 +0.106, e709 pp5 +0.117, e716 hopper
> +0.089, e717 hopper +0.088, e713 cartpole +0.048, e714 cheetah +0.048, e715 cheetah
> −0.011**. **Six of seven rise, one is flat** (e715's −0.011 is an absence of rise, not a
> fall, and its first bin is its noisiest); mean **+0.069**. So the sharpening is **not
> pinpad-specific** — it holds on dense-reward tasks too, though the two sparse pinpad five
> seeds show the largest rises. **`select_bonus_mean` falls on every run over the same
> span** (e713 → 0.512, e715 → 0.496), so level-anneals-while-spread-grows is the general
> behaviour of this grid, not a property of one task.
>
> **NO TURNOVER THROUGH 3M (2026-08-31, 500k bins, ~100 samples/point).** The stated test
> was whether gain turns over by ~2M; it does not, and it still does not by 3M. Binned
> series, first bin → last: **e714 cheetah 1.039→1.196 (+0.157)**, **e713 cartpole
> 1.034→1.181 (+0.147)**, **e716 hopper 1.046→1.175 (+0.129)**, **e717 hopper
> 1.074→1.170 (+0.096)**, **e715 cheetah 1.057→1.109 (+0.052)**. All five rise and four of
> five are still rising in the final bin, at mean count ~4.8/cell against the ~7.6 projected
> at 4M. Only the last 1M remains for the predicted fade, which would make it an
> end-of-training effect rather than the annealing the design claims. **Do not read the
> heartbeat for this**: a single interval showed 4-of-5 falling and the next showed 3-of-5
> rising, both pure noise against the binned trend.
>
> **NO TURNOVER AT 4M EITHER — e713 IS THE FIRST RUN OF THE BATCH TO FINISH
> (2026-08-31).** `e713_..._j4702050` COMPLETED cleanly at 3,995,992 steps, 27 h 26 m on
> one A100, exit 0. It answers the pre-registered question for one seed, and answers it in
> the strongest direction: **the final 500k bin is the highest of the whole run.** Full
> 500k-binned `ucb_novelty_gain`: **1.034 1.057 1.137 1.173 1.147 1.119 1.153 1.221**,
> trailing-20 mean at the end **1.214**. **The series is not monotonic and the wobble is
> real, not noise:** per-bin SE is 0.001–0.009 (n≈102/bin), so the 1.173 → 1.119 dip across
> 2.0–3.0M is ~10 SE and genuine — it simply was not the start of the fade, since the next
> two bins recover and overshoot. Within-bin sd also grows 0.016 → 0.088, so the tilt gets
> **more variable** as well as stronger.
> **The "n is still small" defence is now spent:** mean count landed at **7.61 cells⁻¹**,
> exactly the ~7.6 the 4-bin grid was budgeted for, so the tilt reached its designed sample
> count without fading. What *did* anneal is the **level** — `select_bonus_mean` falls
> monotonically 0.513 → **0.458** — while the **spread**, the part that actually moves the
> choice, ended at its maximum. Level-anneals-while-spread-grows is now confirmed over a
> complete run, not extrapolated from a partial one.
> **Still not sharpening, though.** `ucb_logp_shift` is **negative for the entire run**
> (final bin −0.158, final row −0.195) — the tilt keeps picking goals the manager rated
> *less* likely, which is the intended direction. The alarm was set at > +0.25; nothing
> approached it. `select_eff_cells` runs 12,785 (first bin, when the table is near-empty and
> smearing dominates) → ~6,100, so about 9% of the 65,536 cells carry the mass.
> **Score: no signal, as expected.** last-15 **659.1** (last-50 699.2, last-100 648.5),
> against Director cartpole BIG (e502–e505) 655.2 / 753.1 / 747.3 / 859.0, mean 753.7. That
> is just above the weakest baseline seed and below the mean — but **n=1** (e712, the seed-0
> partner, is still held), and cartpole was already ruled unresolvable at a baseline spread
> of 204. Read this as a mechanism result, not a performance one. The learning curve is
> healthy and flat at the end: 500k bins **227.6 479.3 601.2 663.3 701.2 708.4 735.5
> 707.3**. e714–e717 reach 4M within ~a day and will say whether the no-turnover finding is
> general or a cartpole seed.
>
> **CONFIRMED ON TWO MORE TASKS — e715 AND e716 ALSO FINISHED (2026-08-31, both COMPLETED
> clean, ~27.5 h each).** Both land at the designed budget (7.62 and 7.61 counts/cell) and
> **neither fades**; both peak in the *second*-to-last bin and hold there, so the last bin
> is at the maximum rather than past it. 500k-binned `ucb_novelty_gain`:
> **e715 cheetah s1** 1.057 1.073 1.077 1.092 1.089 1.124 **1.152** 1.138 (+0.081,
> trailing-20 1.197); **e716 hopper s0** 1.046 1.077 1.096 1.152 1.163 1.200 **1.241**
> 1.233 (+0.187, trailing-20 1.262). With e713 that is **three tasks, three runs, no
> turnover at the count budget the grid was sized for** — the annealing claim is now
> falsified on the arm's own terms, not merely unconfirmed. The **level** anneals in all
> three (`select_bonus_mean` e715 0.524 → 0.439, e716 0.522 → 0.449, both monotonic), so
> the level/spread split is the general behaviour.
> **`ucb_logp_shift` stays negative everywhere — and on hopper it *grows*.** e715 wobbles
> near zero (−0.159 → +0.042 by bin, final row −0.161); e716 goes steadily more negative,
> −0.041 → **−0.280** by bin, final row **−0.338**, i.e. more than twice e713's. The tilt is
> pushing *harder* against the manager's preference as training goes on. Still the safe
> direction (the +0.25 sharpening alarm is for the opposite sign), but it is the same
> "spread grows" story read through the log-prob rather than the bonus.
> **Mass concentration is task-dependent, which is new.** `select_eff_cells` first → last
> bin: e713 cartpole **12,785 → 6,100** (concentrating), e715 cheetah **5,972 → 9,638** and
> e716 hopper **13,319 → 14,261** (both spreading). So cartpole funnels its goal
> distribution while cheetah and hopper broaden it, on identical settings — worth a look
> when interpreting the per-task gain differences.
> **Scores.** **e715 cheetah last-15 584.4** (last-50 612.4, last-100 610.4) against
> Director cheetah BIG (e554–e557) 244.6 / 517.1 / 510.9 / 497.7, mean 442.6 — **above
> every baseline seed on the same metric** (last-15 is the convention for this whole family,
> §"score (last-15)" table — and the claim does not depend on the window, since last-50 and
> last-100 are both *higher* than last-15 here, which is the volatility caveat at line 277
> cutting the safe way for once). Treat it as promising and nothing more: n=1, the
> baseline spread is 272.5, and the run's own 500k score bins peak mid-run and come back
> down (275 506 522 580 653 748 744 **615**), so the last bin is ~18% off its own peak.
> **e716 hopper_hop last-15 161.1** (bins 14 89 163 188 190 112 163 163) — hopper_hop was
> retired as a comparison task (bimodal, ~1% power at 5 seeds) so this orders nothing, but
> it is worth recording that it is **not** the dead floor the masked/var-K arms gave here
> (0.0 in every run, see [[hrl-hopper-goal-locality]]); it just cannot be compared.
>
> **ALL FIVE DENSE-TASK RUNS ARE IN — FINAL VERDICT ON THE ANNEALING CLAIM (2026-08-31).**
> e713–e717 all COMPLETED clean (exit 0, ~27.5 h each, all archived + removed from `/work`),
> all landing at the designed budget of **7.61–7.62 counts/cell**. 500k-binned
> `ucb_novelty_gain`, first bin → last bin, with the peak bin marked:
>
> | run | task | first → last | Δ | peak at | final score (last-15) |
> |---|---|---|---|---|---|
> | e714 | cheetah s0 | 1.039 → **1.294** | **+0.255** | last bin | **795.4** |
> | e716 | hopper s0 | 1.046 → 1.233 | +0.187 | bin 7/8 | 161.1 |
> | e713 | cartpole s1 | 1.034 → **1.221** | +0.147 | last bin | 659.1 |
> | e717 | hopper s1 | 1.074 → 1.162 | +0.088 | **bin 5/8** | 172.5 |
> | e715 | cheetah s1 | 1.057 → 1.138 | +0.081 | bin 7/8 | 584.4 |
>
> **Every run ends well above where it started; not one returns toward 1.0.** The `level`
> anneals in all five (`select_bonus_mean` ≈ 0.52 → 0.43 everywhere, monotone), so the
> level/spread split is now the confirmed general behaviour of this grid.
> **e717 is the single genuine turnover, and it is small.** Its peak is 1.205 at 2.0–2.5M
> falling to 1.162 — a 0.043 drop, **7.9 SE**, so real and not the per-row noise the earlier
> false alarms came from. But it is −3.6% off its own peak and still **+0.088 above its first
> bin**: a slight rounding-over, not annealing. Four of five never turn over at all, and
> e714 is still climbing steeply in its final bin.
> **`ucb_logp_shift` binned, first → last:** e714 −0.085 → **−0.294** (final row −0.308),
> e716 −0.041 → −0.280, e713 −0.072 → −0.158, e717 −0.169 → −0.139, e715 −0.159 → +0.042.
> Negative everywhere it matters — the tilt keeps selecting goals the manager rated *less*
> likely, and on cheetah s0 and hopper s0 it pushes ~4× harder at the end than at the start.
> Nothing anywhere approached the +0.25 sharpening alarm. **Read these binned:** e717's
> *final row* is +0.026, which alone would look like a sign flip, while its final bin is
> −0.139.
>
> **THE CHEETAH RESULT IS THE ONE TO LOOK AT. Both seeds beat every Director seed.**
> Arm: **795.4 (e714) / 584.4 (e715)**, mean 689.9. Director cheetah BIG (e554–e557):
> 244.6 / 517.1 / 510.9 / 497.7, mean 442.6. Both arm seeds are above all four baseline
> seeds, i.e. **complete separation with no overlap at n=2 vs n=4** — which under the
> project's own convention is the **1/15 permutation floor (p ≈ 0.067)**, the same "p at the
> 1/15 floor with no seed overlap" language used for e592–e599. Not significant at 0.05, and
> it cannot be with these seed counts; it is the strongest cheetah signal the project has.
> **Robustness:** e714 is flat at the top (last-15 795.4, last-50 791.6, last-100 787.5) and
> its score bins *end at their maximum* (220 400 535 620 520 608 654 **728**), so it is not
> an end-of-run spike. e715 is the weaker case — its bins peak at 748 mid-run and end at 615.
> **Next step if this is worth pursuing: two more cheetah seeds**, which would take a
> no-overlap result to p ≈ 0.014 (1/70) and out of the floor.
> **A correlation worth noting and not yet believing:** on cheetah the stronger tilt is also
> the better score (e714 gain 1.294 / 795.4 vs e715 1.138 / 584.4), but on hopper it inverts
> (e716 1.233 / 161.1 vs e717 1.162 / 172.5). n=2 per task, and a "gain tracks performance"
> hypothesis already dissolved once on this arm — do not build on it.
> **Concentration is task-dependent** (`select_eff_cells` first → last bin): cartpole
> **12,785 → 6,100** is the only one that funnels; cheetah **6,716 → 8,931** (e714) and
> **5,972 → 9,638** (e715), hopper **13,319 → 14,261** (e716) and **10,839 → 14,944** (e717)
> all broaden, on identical settings.
>
> **Two measurement lessons, both learned the hard way here.** (1) Per-row `gain` carries
> ±0.02–0.05 noise; three separate "trends" read off single samples (a clean rise, a
> turnover, a task ordering, and a "gain is high where the agent fails" hypothesis) all
> dissolved on the next reading. Only binned or trailing-mean series are worth
> interpreting. (2) The watcher's "inert" alarm was set at gain < 1.02, *inside* that noise
> band, and duly fired a false positive on e714 at 1.0164 when its trailing mean was 1.0359
> — the same error already fixed once for `logp_shift`. Both now use a trailing 20-row mean.
>
> **Older single-sample framing (kept because the level/spread split is the mechanism):**
> Measuring at *matched* steps — which needed a fix, since the milestone label fires on the
> first poll after the threshold and e708's "@50k" line was really taken at ~80k — four
> seeds at ~200k give gain **1.038 / 1.067 / 1.099 / 1.049**, i.e. a between-seed spread of
> about **±0.03**. From 200k to 1M, e708 goes **1.038 → 1.108** and e709 **1.067 → 1.174**:
> rises of +0.07 and +0.11, **2–3× that spread**, in the same direction on both seeds.
> `select_bonus_mean` over the same interval falls 0.546 → **0.504**. So the level anneals
> as designed while the spread — the part that actually tilts the choice — grows. This is
> the 4-bin-grid failure the old notes predicted, now with evidence above the noise floor.
> **Not yet fatal:** mean count is 1.92/cell heading for ~7.6, and `1/sqrt(n+1)` flattens
> as n grows, so a turnover in the second half is still possible — that is what the 2M and
> 3M milestones are for. **The tilt also remains weak in absolute terms** (gain 1.11–1.17 =
> the chosen goal's bonus 11–17% above the candidate average, against the 1.3–1.5× rarity
> the `c=10` calibration targeted). Scores at 1M are strong: last-15 **274.7 / 274.0**,
> versus Director's *final* pinpad-five last-15 of 20–107 — but pinpad oscillates hugely,
> so that comparison is indicative only.
>
> **Mass accounting checks out.** `select_mass * K / step` read 0.905 at 43k and ~0.98 by
> 357k — the signature of a fixed ~4,064-step replay warmup decaying away (the table only
> accrues in train steps), not a proportional loss, which would have held 0.905 flat. The
> watcher alarms if it is still under 0.97 past 500k. |
| e624–e625 (cont.) | 4698962–63 | **pinpad six** / BIG | Director | 0–1 | **4M → 6M** | *resumed 2026-08-24* | not new experiments — same run dirs (`…_j4697904` / `…_j4697905`), same commit `6494119`, same config with `RUN_STEPS=6000000`; resumes from the 4M checkpoint and the intact 1M-step replay, and appends to the same wandb runs (id = md5 of logdir). ~2M steps ≈ 12–13h on one A100, fits one 48h slice. **The arm seeds e626–e627 are still capped at 4M** — extend them the same way before comparing at 6M. |
| e502–e505 | 4676883–6 | cartpole_swingup / BIG | pure Director | 0–3 | 4M | 655.2 / 753.1 / 747.3 / 859.0 — **mean 753.7, std 83.3, spread 203.8** | unimodal; every seed over the 600 bar. Spread 204 vs "< 200" predicted, i.e. on target. |
| e506–e509 | 4676887–90 | hopper_stand / BIG | pure Director | 0–3 | 4M | 824.8 / 817.6 / 821.5 / 825.2 — **mean 822.3, std 3.5, spread 7.6** | unimodal and extraordinarily tight — spread 7.6 against a "< 300" prediction. |

| e510–e549 | 4677164–4677819 | cartpole_swingup + hopper_stand / BIG | 5 goal-AE arms | 0–3 | 4M | see §2 FINAL table | **No arm beats Director on either task.** Straight-through arms tie on hopper (−5.9 / −9.7 / +13.3); estimator-free arms reliably worse (−115.9, −54.4, both p=0.029). Cartpole unresolvable (baseline spread 204). §6 second branch fired. |
| e550–e553 | 4677820–26 | cartpole_swingup_**sparse** / BIG | pure Director | 0–3 | 4M | 781.0 / 787.7 / 785.6 / 748.3 — **mean 775.7, std 15.9, spread 39.4** | complete 2026-08-17. Tight, and the sparse variant is no harder for Director than the dense one (753.7 on e502–e505). Baseline for e558–e561. |
| e554–e557 | 4677821–27 | cheetah_run / BIG | pure Director | 0–3 | 4M | 244.6 / 517.1 / 510.9 / 497.7 — **mean 442.6, std 114.2, spread 272.5** | complete 2026-08-17. One low seed (e554, 245) against three near 500; peaks 261 vs 538–631, so e554 is genuinely worse and not just noisy at the end. Wide baseline — a small effect will not resolve here. Baseline for e562–e565. |

**Both finished 2026-08-10 ~20:00.** These are now the reference distributions
for e510–e549, and the first `→` branch of the §6 e502–e509 entry fired: both
tasks are unimodal, they are the standard comparison substrate, and four seeds
per arm is the minimum for a claim.

`hopper_stand`'s spread of **7.6 points** is the notable number. `hopper_hop`
was retired for being bimodal with ~1% power at five seeds; the same body on a
different task gives four seeds inside 8 points of each other. It can resolve
differences an order of magnitude smaller than cartpole can, and any arm effect
on hopper_stand will be far easier to establish than the pre-registered 850 bar
assumed. **That bar stays where it was pre-registered** — restating it now that
the spread is known would be exactly the move pre-registration exists to
prevent — but it is worth recording, as a post-hoc observation, that a 30-point
difference here would be ~4 standard deviations and would not need the 850 bar
to be believed.

---

## 6. Pre-registered hypotheses (live board)

_Written before results exist. One entry per live-board row._

### e502–e509 — pure-Director reference distributions on two usable tasks

**Question.** What does this implementation's Director arm do, across seeds, on
tasks that can actually resolve a difference?

**Hypothesis.** Both tasks are within Director's demonstrated range and this
implementation is sound (walker walk: 923.7 / 897.2 at ~850k). So both should
be **unimodal across seeds**, unlike `hopper_hop`.

**Quantitative expectation (pre-registered).**
- cartpole_swingup (e502–e505): all four seeds ≥ 600 by 2M steps; seed spread
  (max − min of last-100-episode means) < 200. Reference: the archived
  single-seed BIG cartpole runs sat at 601–749.
- hopper_stand (e506–e509): all four seeds ≥ 300 by 2M; spread < 300. There is
  no prior number for hopper_stand here — this run *is* the reference.

**Branches.**
- → If both tasks come out unimodal with spreads inside those bounds, they
  become the standard comparison substrate, and four seeds per arm is the
  minimum for any later claim.
- → If cartpole is unimodal but hopper_stand is bimodal like hopper_hop, the
  problem is the hopper body, not sparse reward, and hopper is dropped
  entirely rather than swapped for another hopper task.
- → If cartpole itself is bimodal, the variance is not task-specific and the
  implementation goes back under audit — that would contradict walker walk and
  would be the strongest evidence yet for a real defect.

### e582–e586 — can the manager exploit the ordering the SOM built?

**Question.** F19–F22 established that the SOM-on-a-line makes the code index a
real coordinate: goal similarity falls smoothly with index distance, and the
code "knows how far it moved". But the manager has never been able to use that.
Its per-block head emits 8 free logits over classes it treats as unordered
labels, so REINFORCE credit for class 4 says nothing about class 5. The
geometry is there and the policy is blind to it. Does closing that gap help?

**Mechanism.** `mgr_poisson` (Zhu et al. 2024, Eqs. 8–10). Each block's head
emits two scalars instead of eight logits — a Poisson rate λ and a temperature
τ — and the class distribution is `softmax_j((j log λ − λ − log j!)/τ)`. That
distribution is unimodal by construction, so probability decays away from the
mode on both sides and credit for class j necessarily lifts j±1 more than
distant classes. The mode moves monotonically 0→7 with λ, so the manager is
choosing a *position on the line*, not a label.

**Hypothesis.** If the ordering is real and usable, matching the policy's
inductive bias to it should help most where credit assignment is hardest —
sparse reward and long-horizon tasks — and least where the task is easy enough
that the unordered head already solves it.

**Quantitative expectation (pre-registered).**
- Trains without pathology on all five: no NaNs, `mgr_ent_norm_skill_mean`
  stays inside (0.2, 0.95) rather than pinning, and the `mgr_actent` multiplier
  settles instead of railing at its 1e2 cap.
- cartpole_swingup (e582) and hopper_stand (e583) are the two tasks where
  `som_line` already has 4 seeds (e510–e517), so they are the only honest
  same-arm comparisons. Expect **within the seed spread** on both — cartpole's
  baseline spread is 204, so nothing short of a huge effect can show there.
- The interesting cells are e584 (sparse) and e585 (cheetah), where SOM+LiP
  beat Director and cheetah carried the whole +8.2% aggregate.

**Power warning, stated up front.** One seed per task. This cannot resolve
anything on its own — hopper_hop was retired precisely because 5 seeds could
not resolve a 15× difference there. Read e582–e586 as a screen: does it train,
does it look broken, is any cell far enough from baseline to be worth 4 seeds.
No claim about "better" can come out of this batch.

**Early-health result (2026-08-19, all five past 175k steps).** The first
pre-registered expectation is met: no NaNs, no railing, no collapse.

| run | task | step | ent | mult | blk/change | score |
|---|---|---|---|---|---|---|
| e582 | cartpole_swingup | 266k | 0.531 | 3.8e-3 | 0.358 | 251.1 |
| e583 | hopper_stand | 263k | 0.629 | 1.8e-3 | 0.235 | 6.5 |
| e584 | cartpole_swingup_sparse | 214k | 0.519 | 2.0e-3 | 0.367 | 23.0 |
| e585 | cheetah_run | 176k | 0.535 | 1.4e-2 | 0.237 | 155.1 |
| e586 | hopper_hop | 186k | 0.521 | 2.1e-3 | 0.214 | 0.29 |

The one thing worth flagging beforehand was whether a **single** temperature
scalar could sharpen as fast as C free logits. It can: entropy went 0.92 → 0.52
on the same timescale as the baselines (which go 1.00 → ~0.55 between 31k and
100k) and then held at the 0.5 target. Multipliers are 1e-5 to 2e-2, nowhere
near the 1e2 cap. Block-change rates 0.21–0.37 match the baselines' 0.25–0.39,
so the goal code is not frozen.

Note the Poisson head starts at **0.92** normalized entropy, not 1.00 — that is
the value of the family at lam=3.5, tau=1, not a sign of early convergence.
Any entropy comparison against a categorical head before ~100k is meaningless
for this reason; `tools/check_poisson_health.py` encodes that.

On hopper_hop, the only same-task comparison available this early, e586 scores
0.449 over 150–200k against Director's 0.64–2.20 and SOM+LiP's 0.10–4.61.
Everything is ~0 there this early; this rules out divergence, nothing more.

**Branches.**
- → If a cell lands clearly outside its baseline's seed spread, that task gets
  4 seeds and becomes the real test.
- → If everything lands inside the spread, the honest reading is "the ordering
  is real but the manager was not the bottleneck", and the next question is
  whether the worker, not the manager, is what fails to exploit it.
- → If the entropy controller rails or the rate collapses to one class on
  every block, the parameterization is too restrictive at C=8 and the finding
  is about the policy class, not the geometry.

### e510–e549 — does a geometry-preserving goal code buy anything?

**Question.** motivation.tex measures Director's goal code losing goal-space
geometry (hard-code r = 0.36–0.62 across four tasks, *declining* over training)
and argues this confounds the manager's REINFORCE advantage: a one-block edit
can land anywhere in goal space, so the advantage credited to that edit is
noise. A SOM topology makes adjacent codebook entries decode to nearby goals; a
Lipschitz bound caps how far the decoder can move the goal per unit of code
change. Do either, or both, convert into worker success and task return?

**Hypothesis.** Yes, and mostly through the worker: the mediator to read first
is `wkr_goal_rew`, then `mgr_extr_adv`, then return. Prior evidence is
single-seed and on the now-retired `hopper_hop`, but it is consistent — the
`som_lipvq` family produced that task's best numbers in the whole project
(e481 `som_lipvq_line` 281.1 last-15 / 393.2 peak at 4M, against 176.8 median
for the five-seed Director baseline e495–e499), and `som_lipvq` on cartpole
reached 825.7 (e420).

**Quantitative expectations (pre-registered).** Reference is the e502–e509 mean
of four seeds at 4M, and its seed spread is the yardstick — a difference smaller
than the baseline's own spread is not a difference.

- **cartpole_swingup.** Baseline ≈ 730–800 with a spread of ~240. An arm
  "beats" it only at ≥ 900 mean over four seeds. e420's single-seed 825.7 sits
  *inside* the baseline spread, so it predicts a tie, not a win.
- **hopper_stand.** Baseline ≈ 800 with a spread of ~30 — a tight distribution
  that can resolve much smaller effects. An arm beats it at ≥ 850 mean.
- **Codebook health, all VQ arms.** `goal/perplexity` ≥ 6 of 8 and
  `goal/used_frac` ≥ 0.95 by 1M steps. Every healthy archived arm sat at
  7.1–7.7; the collapse signature is ≤ 2 with reconstruction error climbing.
  Early degeneracy is expected and is not the failure: perplexity ≈ 1.01 at 6k
  steps in the pre-launch smoke of all five arms, and e415 recovered 1.28 at
  577k to 7.55 at 2.28M.
- **The Lipschitz bound must actually move.** `goal/lip_bound_max` starts at
  ~28.7 and under the old `logprod` form moved 0.55 over 3.7M steps, i.e. the
  constraint was inert. Under `prod` with a per-layer bound the penalty reaches
  the whole layer, and at lr 4e-5 over ~250k gradient steps it can move ~10. If
  `lip_bound_max` is still ≈ 28 at 2M, the `_prod` arms did not test anything
  the `logprod` arms had not already tested, and that is the finding.
- **Straight-through.** `som_line` vs `som_orig_line` is a clean one-factor
  contrast on the estimator. Prior: e486 (`som_orig_line`, hopper_hop) ended at
  `rec_q` 40.0 and perplexity 4.08 against 8–13 and 7.1–7.7 for every
  straight-through arm. Expectation: the STE-off arms reconstruct worse and use
  less of the codebook on these tasks too.

**Branches.**
- → **Any arm clears its bar on both tasks.** The geometry fix is real; it
  becomes the recipe and the next question is which component carries it
  (`lipvq_prod` vs `som_line` vs the two combined answer that directly).
- → **Arms tie the baseline but the post-hoc geometry measurement improves.**
  Then geometry preservation is achievable and *not* what limits return, which
  contradicts motivation.tex's argument as stated and is the more interesting
  negative result. It must be written up as such, not buried: the paper's
  Sec. "Goal Code Learning" would need the claim narrowed from "this confound
  costs performance" to "this confound exists".
- → **Arms tie and geometry does not improve either.** The mechanism did not
  engage; read `lip_bound_max` and `perplexity` to say which, and neither the
  claim nor the architecture is tested.
- → **Arms lose.** Most likely via codebook collapse (perplexity ≤ 2 with
  `rec_q` climbing), which is a property of the quantizer and not of the
  geometry hypothesis; report it as a bottleneck-swap cost.

**Power.** Four seeds against four is the whole design, and it bounds what can
be claimed: an exact two-sided permutation test on 4 vs 4 has a minimum
attainable p of 2/70 = 0.029, reached only when the two groups do not overlap at
all. Differences will therefore be reported as effect size with the per-cell
seed spread, and p-values quoted with that floor stated, rather than leaning on
a significance threshold that this n cannot support.

#### RESOLVED 2026-08-16 — the second branch fired

> "**Arms tie the baseline but the post-hoc geometry measurement improves.**
> Then geometry preservation is achievable and *not* what limits return, which
> contradicts motivation.tex's argument as stated and is the more interesting
> negative result. It must be written up as such, not buried."

Exactly what happened, and it has been written up that way
(`26_04_HRL-paper` commit `4fd2d64`, §2.1.2–2.1.3). Point by point against what
was pre-registered:

- **"An arm beats it only at ≥ 900 (cartpole) / ≥ 850 (hopper)."** None did. Best
  cartpole 838.1, best hopper 835.6.
- **"Codebook health ≥ 6 perplexity / ≥ 0.95 used by 1M."** Held for every
  straight-through arm; failed for both estimator-free arms, which is the F19
  mechanism and the source of their score loss.
- **"If `lip_bound_max` is still ≈ 28 at 2M the `_prod` arms tested nothing."**
  Did not fire — the bound fell to 25–28 and the realized decoder ratio dropped
  in every pairing where the penalty was added.
- **"The STE-off arms reconstruct worse and use less of the codebook."** Held,
  and more strongly than expected: reconstruction 15–42 against 5–15, and the
  collapse costs 116 points of return on the task that can measure it.

The one pre-registered expectation that was *wrong* is worth recording: the
hypothesis said the mediator to read first would be `wkr_goal_rew`, on the
theory that better geometry would show up as worker success before return. The
geometry moved by 4–10x and worker success did not track it. Whatever the code's
local isometry buys, it is not delivered through the worker's ability to reach
its goal.

---
## 7. Config flags & metrics reference

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
