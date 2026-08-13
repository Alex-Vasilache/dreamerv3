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

### Two seeds in, 2026-08-13 (complete runs only)

| arm | cartpole (n) | hopper_stand (n) |
|---|---|---|
| *Director* | *753.6 ± 83.3 (4)* | *822.3 ± 3.5 (4)* |
| som_line | 780.4 ± 50.2 (2) | **825.5 ± 5.4** (2) |
| som_orig_line | 490.4 ± 293.0 (2) | 801.0 ± 4.7 (2) |
| lipvq_prod | 823.5 (1) | 799.3 (1) |
| som_lipvq_line_prod | 704.5 ± 177.5 (2) | **823.6 ± 10.3** (2) |
| som_orig_lipvq_line_prod | 843.4 (1) | 755.6 (1) |

**The shape that is emerging is a tie on return.** On hopper_stand, where the
baseline resolves to ±3.5, the two straight-through arms land at 825.5 and
823.6 against 822.3 — inside the noise, not above it. Cartpole is too wide to
say anything at two seeds. Meanwhile F19 shows the geometry *is* measurably
different: the same arms turn a flat displacement curve into a ramp and cut the
one-class edit from 6.3 to 1.4.

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

**This corrects the 08-12 note that called locality "the robust claim".**
Director's k=1 is 1.47 — as local as `som_line`'s 1.25. Director is not bad at
small edits; it has *no gradation*, every edit moving the goal about the same
distance whatever the index says (ratio 1.1). What the straight-through SOM arms
add over Director is **ordering** — ratio 1.1 → 4.2–9.9, consistent across both
seeds and both tasks. What the STE-off arms lose relative to *both* is locality:
their nearest edit is 3.3–7.2, and their decoder expands by 26–39x against
Director's 1.9. Two distinct claims, against two different references; the
earlier note conflated them.

Seed 0–1; remaining seeds measured as they finish. The effect sizes (4–10x
versus 1.1x) are far outside anything the seed spread has produced, and
batch-to-batch noise on these quantities is 0.002–0.006.

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
| e502–e505 | 4676883–6 | cartpole_swingup / BIG | pure Director | 0–3 | 4M | 655.2 / 753.1 / 747.3 / 859.0 — **mean 753.7, std 83.3, spread 203.8** | unimodal; every seed over the 600 bar. Spread 204 vs "< 200" predicted, i.e. on target. |
| e506–e509 | 4676887–90 | hopper_stand / BIG | pure Director | 0–3 | 4M | 824.8 / 817.6 / 821.5 / 825.2 — **mean 822.3, std 3.5, spread 7.6** | unimodal and extraordinarily tight — spread 7.6 against a "< 300" prediction. |

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
