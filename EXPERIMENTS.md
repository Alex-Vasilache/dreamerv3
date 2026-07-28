# DreamerV3 HRL — Experiment Log

Director-style hierarchical RL on DreamerV3, studying whether the manager can act
**sparsely**: editing only part of a persistent goal code per decision (masked edits) and
choosing how long each goal is held (variable durations). Companion working paper:
`paper/main.pdf` (self-contained method + results). Full narrative history of everything
below is preserved verbatim in `EXPERIMENTS_ARCHIVE_20260713.md` (and git); this file is
the restructured, maintained log. `HYPERPARAMETER_COMPARISON.csv` holds the full
cross-cell hyperparameter comparison table (e46–e341), split out to keep this file
condensed (§2).

**Structure of this file:**
§1 Conventions · §2 Current state & live board · §3 Settled findings · §4 Dead ends ·
§5 Experiment ledger (all runs) · §6 Pre-registered hypotheses for the live board ·
§7 Technical notes · §8 Config flags & metrics reference.

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
  its job, compress it here and rely on `EXPERIMENTS_ARCHIVE_20260713.md` + git for the
  full text. Experiment numbers are global and monotonic; never reuse or renumber.

---

## 2. Current state (2026-07-27 — e326–e341 16-cell batch finished @4M, plus
e294/e295/e242/e246 finals and e312/e313 live pull. Headline: the tensor-level
equivalence proof (below) does NOT translate into trained equivalence — e334/e335
(block-pooled var-K pinned to `goal_duration_fixed=8`) trains far below true Director
even under both bugfixes. Full raw pull and hypothesis-by-hypothesis verdicts follow;
older narrative for how the two bugs were found/fixed is preserved below under
"Prior state".)

**e350–e357 launched (2026-07-27) — the e342–e349 batch relaunched under FIVE new
block-pooled fixes, plus same-code Director baselines.** e342–e349 were cancelled ~2h in
(archived to the bucket, deleted from `/work`) after a differential Director-equivalence
audit found that the block-pooled var-K path diverges from fixed-K Director in five
places, none of them in the pooling arithmetic the earlier tensor-level proof covered —
all of them in the **reductions taken over the padded decision axis**
(`downsample_at_switch_mask` packs decisions into a static H+1=17-wide buffer holding
**2 real decisions** at H=16/K=8 and forward-fills the tail). Summary, full detail in §7:

| # | Defect | Measured effect | Scope |
|---|---|---|---|
| 1 | `Agent.loss` reduces manager losses with `v.mean(1)` over the **buffer width**, not the decision count | manager policy + both critic losses **exactly 8× smaller** than Director's on an identical rollout; ~7× on the replay side | every block-pooled run (e326–e349, e44, e62); weaker form in every var-K run |
| 2 | `mgr_retnorm`(meanstd)/valnorm/advnorm and the entropy `AutoAdapt`s got the padded tensor unmasked | ~13 of 16 samples were copies of one decision → shrunken return spread → inflated advantages | same |
| 3 | `mgr_switch` counted a switch on the rollout's final step as trainable | trains a decision owning zero transitions (fixed-K drops it as the bootstrap anchor) | same |
| 4 | Director's windowed `split_traj` worker credit was gated on `not variable_goal_length` | the duration-pinned **control** trained its worker with a different algorithm than its baseline | duration-pinned runs (e334/e335, e342/e343) |
| 5 | Pinned duration head still trained via the entropy regularizer and the E[dur] prior | nuisance gradient into the shared manager trunk that Director never carries | same |

All five are fixed and guarded by `embodied/tests/test_block_pooled_equivalence.py`
(35/35 pass; 8 of those tests fail on the pre-fix code). The suite asserts what the
helper-level tests could not: under `goal_duration_fixed=K` the **reduced losses that
reach the optimizer**, the **arguments handed to each running normalizer**, and the
**gradients** w.r.t. the manager's parameters all match fixed-K Director exactly.
Training smoke: `sbatch/run_smoke_variable_goals.sbatch` (two new legs — the
duration-pinned control and a pure fixed-K Director leg).

| Exp | Job | Task | Recipe | Question |
|---|---|---|---|---|
| e350 | 4671354 | hopper hop | plain_vark, **pinned** K=8, block-rew, mean | does the equivalence control now reach e124's 279.4? |
| e351 | 4671355 | cheetah run | plain_vark, **pinned** K=8, block-rew, mean | does it now reach e191's 290.0 (was 25.2 as e335)? |
| e352 | 4671356 | cheetah run | pure Director, K=8 | same-code baseline: removes code drift as an explanation |
| e353 | 4671357 | hopper hop | pure Director, K=8 | same-code baseline |
| e354 | 4671358 | hopper hop | plain_vark, lagr τ8, sum, relabel ON | rerun of e326/e344 under the fixes |
| e355 | 4671359 | cheetah run | plain_vark, lagr τ8, sum, relabel ON | does the late collapse (trail50 7.3) go away? |
| e356 | 4671360 | hopper hop | plain_vark, lagr τ8, sum, relabel OFF | rerun of e330/e346 |
| e357 | 4671361 | cheetah run | plain_vark, lagr τ8, sum, relabel OFF | does the best cheetah cell (124.7) improve? |

Deviation from a 1:1 resubmit: the e348/e349 reruns are dropped to make room for
e352/e353, since e191 (cheetah 290.0, itself noisy: 225–471, cancelled at 64% budget)
predates many commits. All BIG scale, `gpu-a100`, 4M steps, SEED=0;
`sbatch/submit_e350_e357_director_equiv.sh`. **Pre-registered readout:** if defects 1–5
were the cause, e351 tracks e352 within seed noise; if e351 stays near its old 25.2 while
e352 reproduces ~290, the entire padded-axis family of explanations is ruled out and the
λ-return construction itself is the remaining suspect. Live invariant check for the
pinned control (verified green on the smoke leg): `tools/check_pinned_director_invariants.py`
asserts `mgr_valid_decisions==2`, `mgr_duration_mean==8`, `mgr_duration_std==0`, and no
duration-prior loss.

**e342/e343 launched (2026-07-27) — e334/e335 rerun under TWO NEW agent.py fixes,
full 4M-step budget.** A deeper pipeline audit of the e334/e335 gap (tensor/gradient-
level tests, not training curves — `embodied/tests/test_variable_goals.py`, 23/23
passing) found and fixed two real issues, neither previously known:
1. **Duration-head REINFORCE noise.** `imag_loss_mgr` summed EVERY manager head's
   log-prob into the REINFORCE term unconditionally, including `duration` even when
   `goal_duration_fixed>0` makes its sample provably unable to affect switch timing
   (`_duration_steps` already overrides it). The duration head was still being pushed
   every decision by an advantage caused entirely by the skill/code choice — a
   nuisance signal true fixed-K Director never carries (no duration head at all).
   Confirmed live via a gradient test (nonzero gradient, 0.0728, into the duration
   head's own params, reaching the shared trunk too). **Fix:** new
   `manager_reinforce_policy(manager_policy, duration_fixed)` (agent.py) excludes
   `'duration'` from the REINFORCE sum whenever `goal_duration_fixed>0`; wired via a
   new `duration_fixed` param on `imag_loss_mgr`. Gradient into the duration head is
   now exactly 0.0; skill-head/trunk gradients unaffected (bit-identical to a
   skill-only computation).
2. **Replay-side trailing-state asymmetry (the stronger candidate).** Fixed-K's
   replay downsample (`downsample_manager_states`) always appends the training
   window's TRUE final state as an extra bootstrap anchor, even when it isn't a real
   decision boundary. Block-pooled var-K's replay downsample
   (`downsample_at_switch_mask`) had no such provision — it only ever forward-fills
   the LAST REAL SWITCH's stale state into that slot instead. Since `batch_length=64`
   essentially never divides evenly by an 8-step hold (only 1 of 8 possible phases
   aligns — see the modular-arithmetic derivation in conversation), this fires on
   ~7/8 of replay-side manager value updates, every training step, unlike the
   duration-head issue which only touches imagination. **Fix:** new
   `patch_trailing_replay_state(x_down, x_full, switch_mask)` (agent.py) overwrites
   the stale slot with the window's true final value; applied to both `feat_down`
   (the value-prediction input) and `last_down` (the episode-boundary flag) at the
   real replay call site. Verified correct on the original bug fixture, a no-op on
   the exact-multiple (imagination) case, per-row-correct on a heterogeneous batch,
   correctly propagates a genuine terminal flag, and composes cleanly with the
   existing (2026-07-24) truncated-duration relabeling. Explicitly NOT fixed: the
   trailing partial segment's own real reward is still not pooled as its own
   credited block (a separate, larger design question, left alone deliberately).

Both fixes are tensor/gradient-level verified only — neither has yet been shown to
move a real training score. **e342 (hopper) / e343 (cheetah)** relaunch e334/e335's
exact config (`RECIPE=plain_vark, DUR_MODE=fixed, DUR_FIXED=8, DUR_TARGET=8.0,
DUR_ACTENT_TARGET=0.0, MGR_REWARD_AGG=mean, VARIABLE_GOAL_BLOCK_REW=True, SEED=0`,
BIG scale, `gpu-a100`, `RUN_STEPS=4000000`) under both fixes, to test whether hopper
(target: e124's 279.4) and cheetah (target: e191's 290.0) close the gap that e334/e335
showed (0.06 and 25.2 respectively). Job ids: e342=4671225, e343=4671226, both
RUNNING as of launch. The original e334/e335 runs' own trajectories already settled
by 750k–1M steps (hopper flat-dead from 250k on; cheetah oscillating in a ~15–26 band
from 750k on, barely different from its 4M-step final value of 25.2) — so an early
interim pull well before 4M should already be informative, even though the full
budget was requested.

**e344–e349 launched (2026-07-27) — reruns of six of the e326–e341 matrix cells under
the replay-trailing-state fix, full 4M budget.** All six have `goal_duration_fixed=0`
(duration is learned/variable, not pinned), so **the duration-head REINFORCE fix is a
no-op for every one of these six** — only `patch_trailing_replay_state` can change
their behavior, since all six use `variable_goal_block_rew=True`. Exact 1:1 config
reruns (verified against each original's `job.env`, differing only in job id/`RUN_DIR`):

| New | Job | Old | Old score (trail300, peak) | Config |
|---|---|---|---|---|
| e344 | 4671258 | e326 (hopper) | 0.08 (9.2), dead floor | sum, relabel ON, lagr τ8, imag16 |
| e345 | 4671259 | e327 (cheetah) | 50.8 (196.0), collapsed late (trail50 7.3) | same |
| e346 | 4671260 | e330 (hopper) | 0.27 (11.9), dead floor | sum, relabel OFF, lagr τ8, imag16 |
| e347 | 4671261 | e331 (cheetah) | **124.7 (216.6), best cheetah cell in the matrix** | same |
| e348 | 4671262 | e332 (hopper) | 0.23 (13.3), dead floor | no dur-reg (reg0), dur-entropy default target |
| e349 | 4671263 | e333 (cheetah) | 6.9 (113.4), worst cheetah cell (duration collapsed to 3.55) | same |

All RUNNING on `gpu-a100` as of launch. Hypotheses: (i) does the trailing-state fix
rescue hopper in any of the three configs (unlikely, given hopper's project-wide
pattern of failing under every mechanism tried except the unrestricted single-head
design, but worth checking since credit-assignment fixes are exactly the category
that previously mattered for hopper, e.g. Finding 6/F19); (ii) does e345 (cheetah,
relabel ON) stop collapsing late in training now that its replay-side value target is
correct — this was the single most-visibly-broken trajectory in the original matrix;
(iii) does e347's already-best cheetah cell hold or improve; (iv) does e349's
duration-collapse-driven weak cheetah score change at all, given the fix touches the
replay value target, not the (still-unfixed) mechanism that let duration collapse to
3.55 in the first place.

**e326–e341 (16 cells, `gpu-a100`, all COMPLETED exit 0, full 4M-step budget) — final
numbers, from each run's `scores.jsonl`/`metrics.jsonl` in `/work` (not yet archived).**
Score = trailing-300-episode mean (trailing-50 in parens where it diverges sharply from
peak, noted in the text); peak = single highest raw episode score reached at any point.
`blk/step = L / mgr_duration_mean` (`L=8`, mask off in every cell here so every switch
edits the full code) computed from each run's final `goal/mgr_duration_mean`.

| Exp | Job | Task | Config vs. e326 baseline | Score (trail300, peak) | Final `mgr_duration_mean` | blk/step | Verdict |
|---|---|---|---|---|---|---|---|
| e326 | 4669107 | hopper hop | baseline: sum, relabel ON, lagr τ8, imag16 | 0.08 (9.2) | 8.06 | 0.99 | Dead floor — hopper unrescued (again) |
| e327 | 4669108 | cheetah run | same | 50.8 (196.0); trail50 only 7.3 | 8.03 | 1.00 | Alive but collapses late in training |
| e328 | 4669109 | hopper hop | imag_length=32 | 0.25 (14.8) | 7.90 | 1.01 | Dead floor, longer horizon no rescue |
| e329 | 4669110 | cheetah run | imag_length=32 | 46.1 (123.4), stable | 7.91 | 1.01 | Alive, stable — no late collapse (unlike e327) |
| e330 | 4669111 | hopper hop | relabel OFF | 0.27 (11.9) | 8.08 | 0.99 | Dead floor, relabel-off no rescue |
| e331 | 4669112 | cheetah run | relabel OFF | **124.7 (216.6)**, stable/rising | 8.07 | 0.99 | **Best cheetah cell in the matrix** |
| e332 | 4669130 | hopper hop | no dur-reg (reg0), dur-entropy ctrl on | 0.23 (13.3) | 5.04 | 1.59 | Dead floor; duration collapses below τ8 |
| e333 | 4669131 | cheetah run | no dur-reg, dur-entropy ctrl on | 6.9 (113.4) | 3.55 | 2.25 | **Worst cheetah cell** — duration collapse tanks score |
| e334 | 4669122 | hopper hop | `goal_duration_fixed=8`, mean agg (equivalence check) | 0.06 (13.7) | 8.00 (exact) | 1.00 | **EQUIVALENCE CHECK FAILS** — still dead vs. e242's 90.8 |
| e335 | 4669123 | cheetah run | `goal_duration_fixed=8`, mean agg (equivalence check) | 25.2 (58.7) | 8.00 (exact) | 1.00 | **EQUIVALENCE CHECK FAILS** — 25 vs. e246's 352.4 (7×) |
| e336 | 4669124 | hopper hop | mean agg, relabel ON | 0.18 (18.9) | 7.95 | 1.01 | Dead floor, mean-agg no rescue |
| e337 | 4669125 | cheetah run | mean agg, relabel ON | 60.2 (153.8), still climbing | 7.93 | 1.01 | Alive, stable |
| e338 | 4669126 | hopper hop | mean agg, relabel OFF | 0.16 (14.2) | 7.97 | 1.00 | Dead floor |
| e339 | 4669127 | cheetah run | mean agg, relabel OFF | 14.3 (49.2), still climbing | 7.92 | 1.01 | Alive but weak |
| e340 | 4669128 | hopper hop | worker countdown ON | 1.04 (17.7) | 8.03 | 1.00 | Nominally highest hopper score, still ~dead |
| e341 | 4669129 | cheetah run | worker countdown ON | 57.4 (164.1), decayed from early (600k) peak | 8.04 | 1.00 | Alive, countdown ~no effect vs. e326/e327 |

**Hypothesis verdicts (vs. the (v)–(vii) stated at launch, §2 prior-state below):**
- **(v) e334/e335, block-pooled var-K pinned to `goal_duration_fixed=8` trains
  indistinguishably from true Director — REFUTED.** Hopper stays at the dead floor
  (0.06 vs. e242's 90.8) and cheetah reaches only 25.2 vs. e246's 352.4 (≈7× below) —
  even below the non-block-pooled `plain_vark` τ8 Lagrangian comparison cell (e303,
  194.5) that shares every other setting. The tensor-level proof that the pooled-return
  *arithmetic* matches fixed-K exactly (prior-state section below) does not imply the
  *trained* model matches — some further discrepancy in the block-pooled λ-return
  construction, not caught by the unit test, remains. This is the batch's central
  negative result and the next debugging priority before block-pooled var-K can be
  trusted for anything.
- **(vi) sum vs. mean reward aggregation, both relabel ON/OFF — duration-collapse
  pressure is NOT specific to `sum`, and in fact doesn't appear at all once both bugs
  are fixed.** e326/e330 (sum) and e336/e338 (mean) all track τ=8 cleanly
  (`mgr_duration_mean` 7.90–8.08 across all four) — the runaway/collapse behavior seen
  in the pre-fix e304–e311 pull was an artifact of the two credit-assignment bugs, not
  of the aggregation choice. The one condition that DOES collapse duration is removing
  the prior anchor entirely (e332/e333, see below) — orthogonal to sum-vs-mean.
- **(vii) worker countdown (`worker_timed_goals`) re-tested under the fixed credit path
  — still no effect.** e340/e341 vs. e326/e327 (their only-difference twins): hopper
  1.04 vs. 0.08 (both at the dead floor, not a real rescue) and cheetah 57.4 vs. 50.8
  (within noise of each other, though e341 decays less catastrophically than e327's
  crash to trail50 7.3). Confirms the earlier (pre-fix, F13/e185) null result still
  holds now that the credit-assignment bugs are fixed.
- **New finding — removing the duration-prior anchor collapses duration SHORTER, not
  longer, and costs cheetah most of its score.** e332/e333 (`DUR_REG=0`, entropy
  controller only, no Lagrangian target) let `mgr_duration_mean` fall to 5.04 (hopper)
  / 3.55 (cheetah) — well below τ=8, the opposite direction the `sum`-aggregation
  long-hold incentive would predict. Cheetah craters to 6.9 (trail50 4.35), the weakest
  cheetah cell in the whole matrix, roughly 4–9× below every anchored cheetah cell
  (46–125). Without an explicit target, the manager does not drift toward exploiting
  `sum`'s pro-length reward incentive — if anything it prefers shorter, more frequent
  goal changes, at real cost to task performance.
- **New finding — truncated-hold relabeling REVERSES direction under the fix, and its
  effect is now a stability story, not a bias-correction story.** The pre-fix, live
  pull tentatively read relabel-ON as helping cheetah (e305 > e309). Under the fixed
  code the opposite holds cleanly: relabel-OFF (e331, 124.7, stable/rising, no late
  collapse) is the single best cheetah cell in the entire 16-cell matrix, while its
  relabel-ON twin (e327) collapses late in training (trail300 50.8 but trail50 only
  7.3 — a real crash, not noise). Tentatively implicates the relabeling correction
  itself as a source of late-training instability in this credit path, not the
  truncation bias it was designed to fix.
- **New finding — `imag_length=32` avoids the late cheetah collapse seen at 16, but
  does not rescue hopper.** e328/e329 (imag32) vs. e326/e327 (imag16): hopper stays
  dead either way (0.25 vs. 0.08), but cheetah is stable/no-collapse at imag32 (46.1,
  trail50 47.4 ≈ trail300) vs. imag16's late crash (50.8 trail300, trail50 7.3) —
  suggests the imag16 collapse is a horizon-length artifact of this specific credit
  path, worth using imag32 as the default for any future block-pooled var-K cell on
  cheetah-like dense tasks.

**e242/e246 (struct-only ablation, resumed to full 4M) — final.** e242 (hopper, BIG,
`RECIPE=director`, struct 200+adapt target 0.01, no reuse term) finishes at trailing-300
**90.8** (peak 219.9 raw episode, reached late at 2.34M then decayed) — well below the
e124 pure-Director baseline (≈300) and, notably, DOWN from the 115.1 interim read at its
original 1M-step cutoff: more budget did not close the gap, it reopened it. e246
(cheetah, same ablation) finishes at **352.4** (peak 420.9) — inside/above the e123-class
250–435 band, holding parity with Director. Struct-only is safe for cheetah but does not
rescue hopper even given 4× the original training budget — refines "struct is a safe,
no-cost stabilizer" (F18) to "safe and neutral for cheetah; does not by itself rescue
hopper," consistent with hopper's project-wide pattern of failing under every mechanism
except the unrestricted single-head design (Finding 6).

**e294/e295 (Family-C recipe on cartpole/acrobot) — final; e312/e313 (struct-only
ablation of the same) — live pull.** e294 (cartpole) finishes at **639.5** (peak 754.4),
just inside cartpole's ≥650 Director-baseline band — "costs nothing" holds, but only
marginally (down from the 754.4 peak, and below e312's live read, next). e295 (acrobot)
fully decays from its 347.6 interim peak to **19.2** — a clean fourth confirmation of
F12's acrobot-specific failure, not a rescue. e312 (cartpole, struct-only, reuse term
OFF), read at 66% of budget, is **already ahead of e294's finished score** (695.1 vs.
639.5) — a concrete, if not yet final, signal that the forced-reuse term costs cartpole
a small amount of score rather than being free (neither cell is anywhere near cartpole's
dead floor, so this is about the *last mile* of score, not survival). e313 (acrobot,
struct-only), read at 58% of budget, decays earlier and harder than e295 (trail300 4.6
vs. e295's 19.2, despite less training) — struct correlation alone does not rescue
acrobot either, ruling out the reuse term specifically as acrobot's problem.

**e296–e303 (plain var-K Lagrangian isolation, τ∈{4,8}) — final at cancellation
(58–75% of budget, cancelled once the pattern was unambiguous).** hopper dead both τ
(0.09/1.21); acrobot decaying from an early peak both τ (2.8/22.3), never recovering;
cartpole and cheetah alive both τ with **τ8 clearly beating τ4 on both tasks**
(cartpole 850.0 vs. 616.2; cheetah 194.5 vs. 107.6, ≈1.4× on both) — exactly replicates
F12's prediction (switch-timing variability alone is sufficient to kill hopper/acrobot,
independent of masking/struct/reuse) and gives the cleanest τ4-vs-τ8 read in the project:
longer, precisely-tracked commitments help dense tasks by a wide, consistent margin.

### Prior state (2026-07-24, later still — a SECOND block-pooled var-K bug found
and fixed while building a direct tensor-level equivalence test (not just a training
curve) against fixed-K Director, per the project's own "100% certainty" bar. e314–e321
CANCELLED a second time (they only had the first fix) and superseded by a 16-cell batch,
e326–e341, all under both fixes. `run_v3_prior_vargoal_big_a100.sbatch` gained a second
new knob, `DUR_FIXED` (wired to `--agent.goal_duration_fixed`, pre-existing in `agent.py`/
`configs.yaml` but never exposed by this script before).

**Bug 2: empty-segment continuation was `0.0` ("episode ended"), starving the last real
decision's bootstrap in every block-pooled rollout.** `aggregate_mgr_cont_variable`
(`agent.py:259`) pools continuation per real segment via `segment_min`; for a segment with
zero real steps — either genuine padding far past the last real decision, or (the common
case) the LAST real decision's own trailing hold when its switch happens to land on the
rollout's final timestep — it filled that with `0.0`. A `0.0` continuation reads as
`term=1` to `lambda_return`, which caps whatever real decision immediately precedes it to
only the `(1-λ)` fraction of its bootstrap value, not the full λ-weighted blend. Fixed-K's
own arrays never hit this: they're sized exactly to their real content (no padding at
all), so `lambda_return`'s base case (`rets[-1]=boot[-1]`) lands on the true terminal value
directly, at full weight — this bug is structurally impossible there. Confirmed this is
block-pooled-var-K-specific (not present in fixed-K or in the full-resolution/non-pooled
var-K path) via direct tensor comparison, not training curves.

**Fix (2026-07-24, `agent.py`):** empty segments now get the correct empty-product
identity, `1.0`, not `0.0` (`aggregate_mgr_cont_variable`, line ~281). Also removed the
redundant `* block_mask` re-masking of the pooled continuation in
`variable_block_director_tensors` (was undoing the identity fix) — safe because a segment
id `>= n_sw` can never occur (`variable_segment_ids` is bounded by `cumsum(switch_mask)-1`),
so `aggregate_mgr_cont_variable`'s own output is already correct at every position past
the real content; `block_mask` is still needed (and applied) on the reward side.

**Verification — genuine tensor-level equivalence, not just a training-curve match.**
Two new tests in `embodied/tests/test_variable_goals.py` (10/10 pass): with a
deterministic τ=8 switch mask on a T=17 rollout (3 decisions: t=0, 8, 16 — exactly what
`goal_duration_fixed=8` produces at `imag_length=16`), block-pooled var-K's returns for
the two fully-real decisions now match fixed-K's **exactly, to float precision**
(`[28.1386, 29.2875]` both), and correctly bootstraps the third (degenerate, zero-trailing-
step) decision to exactly its own value (`30.0`, was `0.0` pre-fix) — a decision fixed-K's
own array construction structurally never even creates (its arrays end at 3 states → 2
exposed returns, not 3; see conversation for the fuller "why 3 vs 2" explanation, not
reproduced here). A training curve alone could never prove this with certainty (different
code paths consume RNG and accumulate float rounding differently even under a fixed seed);
this tensor-for-tensor check is the actual proof.

**Ground-truth baselines changed scope.** e322–e325 (pure fixed-K `RECIPE=director`
baselines, launched then re-launched earlier this session) were explicitly descoped —
not needed; the real equivalence claim is proven at the tensor level above, and the
BIG-scale check that matters is the `goal_duration_fixed=8` cells (e334/e335) below, which
run the actual var-K/block-pooled code path pinned to behave like Director, not the
separate never-touches-var-K Director implementation itself.

**Current batch — 16 cells, all under both fixes, `gpu-a100` (8-GPU/128-CPU cap; 8 running,
8 queued behind on CPU headroom):**

| Exp | Job | Task | Reward agg | Relabel | Dur mode | Worker countdown | `imag_length` | Purpose |
|---|---|---|---|---|---|---|---|---|
| e326 | 4669107 | hopper hop | sum | ON | lagrangian τ8 | off | 16 | original 8-cell matrix (§ below), post-fix |
| e327 | 4669108 | cheetah run | sum | ON | lagrangian τ8 | off | 16 | same |
| e328 | 4669109 | hopper hop | sum | ON | lagrangian τ8 | off | 32 | same |
| e329 | 4669110 | cheetah run | sum | ON | lagrangian τ8 | off | 32 | same |
| e330 | 4669111 | hopper hop | sum | OFF | lagrangian τ8 | off | 16 | same |
| e331 | 4669112 | cheetah run | sum | OFF | lagrangian τ8 | off | 16 | same |
| e332 | 4669130 | hopper hop | sum | ON | none (`DUR_REG=0`), dur-entropy **on** (default target) | off | 16 | same |
| e333 | 4669131 | cheetah run | sum | ON | none (`DUR_REG=0`), dur-entropy **on** (default target) | off | 16 | same |
| e334 | 4669122 | hopper hop | mean, `goal_duration_fixed=8` | n/a | fixed@8 (deterministic, no controller) | off | 16 | BIG-scale equivalence check vs. Director |
| e335 | 4669123 | cheetah run | mean, `goal_duration_fixed=8` | n/a | fixed@8 | off | 16 | same |
| e336 | 4669124 | hopper hop | mean | ON | lagrangian τ8 | off | 16 | `sum` vs `mean` isolation |
| e337 | 4669125 | cheetah run | mean | ON | lagrangian τ8 | off | 16 | same |
| e338 | 4669126 | hopper hop | mean | OFF | lagrangian τ8 | off | 16 | same |
| e339 | 4669127 | cheetah run | mean | OFF | lagrangian τ8 | off | 16 | same |
| e340 | 4669128 | hopper hop | sum | ON | lagrangian τ8 | **on** | 16 | first block-rew × `worker_timed_goals` combo |
| e341 | 4669129 | cheetah run | sum | ON | lagrangian τ8 | **on** | 16 | same |

All 16: `RECIPE=plain_vark`, `VARIABLE_GOAL_BLOCK_REW=True`, BIG scale, `SEED=0`,
`RUN_STEPS=4000000`. e332/e333 relaunched once more mid-batch (jobs 4669113/4669114 →
4669130/4669131) to flip `DUR_ACTENT_TARGET` back to its default (`-1.0`, entropy
regularizer **on**) instead of the `0.0`-disabled ablation originally specified — isolates
"no duration-target prior, but the entropy controller still fighting collapse" as its own
distinct cell, rather than "no regularization of any kind." **[Superseded — all 16 cells
finished @4M on 2026-07-27; final numbers and hypothesis verdicts are at the top of §2.]**
e326–e333 had already accrued ~10–15 min under the FIRST fix only before being cancelled
and relaunched fresh under both, so none of their prior progress carried over.

**New hypotheses this batch adds on top of (i)–(iv) above:** (v) e334/e335 — does
block-pooled var-K pinned to a constant duration=8 actually train indistinguishably from
plain Director (e124/e123-class scores), confirming the tensor-level equivalence holds
end-to-end, not just for the pooling math in isolation? (vi) e326/e330 vs. e336/e338
(sum vs. mean, both relabel ON/OFF) — does the duration-collapse pressure identified in
the earlier e305 investigation (duration Lagrangian + entropy controller both railing)
appear under `mean` too, or is it specific to `sum`'s pro-length incentive? (vii) e340/e341
— does giving the worker its remaining-hold-time as an input change anything now that the
credit-assignment path itself is correct (previously tested only under the OLD, buggy
credit assignment via e185/F13, which found no effect — worth re-checking since the
worker's own value target depends on the same bootstrap machinery just fixed).

### Prior state (2026-07-24, root-caused and fixed the e304–e311
cheetah/duration anomalies to a real off-by-one bug in the block-pooled manager return;
e304–e311 CANCELLED and superseded by e314–e321 under the fix.) Investigating e305's
"low expl return" and railed `mgr_ent_loss`/`mgr_actent_duration_scale_mean` (both
flagged during live analysis of the e304–e311 pull above) traced to
`variable_block_director_tensors` (`agent.py:300`, the `variable_goal_block_rew` credit
path e304–e311 all use): `n_blocks = jnp.maximum(n_sw - 1, 1)` (was line 311) drops the
LAST real manager decision of every imagined rollout from the pooled reward/continuation
fed to the critic, so `mgr_cont` reads "episode ends" one decision early and the λ-return
loses essentially all of its value-function bootstrap. Confirmed via the return/reward
ratio: e303 (no block-rew, full-resolution path) ≈300–500×, e294 (fixed-K Director,
same general block-pooled mechanism but the correct, non-buggy convention) ≈110–160×,
e305/e307/e309/e311 (all four `variable_goal_block_rew` cells checked) ≈1.3–1.5× — a
near-total collapse of the bootstrap horizon, not just a smaller value. This single bug
plausibly explains the whole cluster of anomalies found this session: the runaway
duration drift (§7 "duration-prior skyrocket"), the railed duration-entropy controller
(`mgr_actent_duration_scale_mean` → its 100.0 ceiling), and cheetah running below the
plain-var-K comparison cells despite fixing the two credit-assignment bugs (relabel +
`sum` agg) that this same batch was built to test. **Fix (2026-07-24):**
`n_blocks = jnp.maximum(n_sw, 1)` — matches the fixed-K analog `aggregate_mgr_cont`
(`agent.py:135`), which never subtracts 1 from its block count. Smoke-tested clean
(`run_smoke_variable_goals.sbatch`, job 4668996, all 5 legs incl. both block-rew legs
PASSED, zero NaN/Inf) — the smoke debug net trains only 800 steps so reward is still
~0 everywhere, making the return/reward ratio itself uninformative at that scale; real
confirmation of the magnitude fix requires the BIG-scale relaunch below. **e304–e311
CANCELLED** (were running the pre-fix code the whole time, all 8 cells compromised for
their original credit-assignment question) and superseded by **e314–e321** (jobs
4668997–4669004, all RUNNING on `gpu-a100`, launched immediately after cancellation
freed the 8-GPU cap):

| Exp | Job | Task | Reward agg | Relabel | Dur mode | Dur-entropy ctrl | `imag_length` |
|---|---|---|---|---|---|---|---|
| e314 | 4668997 | hopper hop | sum | ON | lagrangian τ8 | default (target 0.5) | 16 |
| e315 | 4668998 | cheetah run | sum | ON | lagrangian τ8 | default | 16 |
| e316 | 4668999 | hopper hop | sum | ON | lagrangian τ8 | default | 32 |
| e317 | 4669000 | cheetah run | sum | ON | lagrangian τ8 | default | 32 |
| e318 | 4669001 | hopper hop | sum | OFF | lagrangian τ8 | default | 16 |
| e319 | 4669002 | cheetah run | sum | OFF | lagrangian τ8 | default | 16 |
| e320 | 4669003 | hopper hop | sum | ON | none (`DUR_REG=0`) | **off** (`DUR_ACTENT_TARGET=0.0`, new knob) | 16 |
| e321 | 4669004 | cheetah run | sum | ON | none (`DUR_REG=0`) | **off** | 16 |

All 8: `RECIPE=plain_vark`, `VARIABLE_GOAL_BLOCK_REW=True`, `RUN_STEPS=4000000`, `SEED=0`,
BIG scale, under the `n_blocks` fix. e320/e321 exercise a new script knob,
`DUR_ACTENT_TARGET` (`run_v3_prior_vargoal_big_a100.sbatch`, wired to
`--agent.manager_actent_duration_target`) — set to `0.0` to neutralize the duration-head
entropy regularizer (`mgr_dur_actent`) as an ablation: with an unreachably-low target the
controller's multiplier just relaxes to its floor and never pushes back. No boolean
disable flag exists for it in `agent.py`, so this is the practical off-switch within the
current flag surface.

**Hypotheses:** (i) e314/e315 vs. the OLD (buggy) e305/e309/e311 pulls — does hopper stay
dead and does cheetah's score/duration-drift/entropy-collapse signature go away now that
the critic sees every real decision? (ii) e314 vs. e316 (hopper, imag16 vs. imag32) —
re-test of the original imag_length question now that the credit signal itself isn't
truncated. (iii) e314 vs. e318 (relabel ON vs. OFF, both now under the fix) — does the
relabel fix's own effect (previously entangled with the n_blocks bug) look different once
isolated. (iv) e320/e321 vs. e314/e315 — with both the duration-target Lagrangian AND the
duration-entropy controller neutralized, does cheetah's duration still drift under `sum`
agg, isolating whether the `sum`-reward pull itself (not the controllers failing to
resist it) is the primary force, now that the return-construction bug no longer amplifies
it.

### Prior state (2026-07-24, earlier same day — live pull across the four batches spawned by F20.
e278–e293 unchanged (F20, final, §3). e294/e295 (task-generalization) now 76–83%/4M:
cartpole (e294) holds its ≥650 baseline band with no sign of cost; acrobot (e295) has
now fully decayed from its 347.6 peak to a trail300 of 19.4, confirming the 07-23
tentative "declining, not stabilizing" read as a third acrobot failure under F12's
pattern. e296–e303 (plain-var-K Lagrangian isolation, τ∈{4,8}) stayed **CANCELLED**
07-23 at 58–75%/4M — final pre-cancellation numbers below replicate F12 exactly (hopper
dead both τ, acrobot decaying both τ, cartpole/cheetah alive with τ8>τ4 at both). e304–
e311 (block-pooled var-K credit, two implementation fixes) now ~55%/4M: hopper dead in
all 4 cells regardless of relabel/agg setting; cheetah alive in all 4 but running
noticeably *below* the plain-var-K comparison cells (e302/e303) at a matched training
fraction — an unexpected below-par reading, not yet explained, worth watching to 4M
before concluding the fixes help. e312/e313 (struct-only ablation) and the e242/e246
resumes are still `PENDING` (`AssocGrpGRES`, queued behind e294/e295 on `gpu-v100`) —
no data yet.

| Exp | Job | Task | Config | Progress | Score (trail300/trail50) | Peak (step) | Reuse/overlap | vs. hypothesis |
|---|---|---|---|---|---|---|---|---|
| e294 | 4668658 | cartpole swingup | `RECIPE=director`, `MGR_FREQ=8`, `STRUCT_ADAPT` target 0.01, `MGR_COND_GOALCODE=True`, `GOAL_SOFT_REUSE_ADAPT` target 0.5 (vel 3.9e-6) | 3.33M/4M (83%) | **625.7** / 624.6 | 754.4 @1.37M | ov 0.512/0.50, struct_corr 0.918 | Confirms: e286/e290 recipe still costs nothing on an easy dense task |
| e295 | 4668659 | acrobot swingup | identical to e294 | 3.03M/4M (76%) | **19.4** / 19.4 | 347.6 @1.36M | ov 0.527/0.50, struct_corr 0.869 | Refutes rescue: full decay from peak, matches F12's acrobot-specific failure, not a fourth F20-style save |
| e296 | 4668663 | hopper hop | `RECIPE=plain_vark`, `DUR_MODE=lagrangian`, τ=4, `STRUCT_W=0`, `MGR_COND_GOALCODE=False` | 2.95M/4M (74%), CANCELLED at pull | **0.09** / 0.03 | 8.1 @921k | n/a | Confirms F12: dead floor regardless of τ |
| e297 | 4668664 | hopper hop | same, τ=8 | 2.99M/4M (75%), CANCELLED | **1.21** / 0.87 | 15.5 @2.07M | n/a | Confirms F12: τ8 no better than τ4 |
| e298 | 4668665 | acrobot swingup | same, τ=4 | 2.93M/4M (73%), CANCELLED | **2.83** / 0.88 | 153.9 @2.08M | n/a | Confirms F12: decaying from early peak, acrobot's own slow failure |
| e299 | 4668666 | acrobot swingup | same, τ=8 | 2.79M/4M (70%), CANCELLED | **22.3** / 27.7 | 165.3 @224k | n/a | Same decay pattern as e298, τ has not rescued it |
| e300 | 4668667 | cartpole swingup | same, τ=4 | 2.41M/4M (60%), CANCELLED | **616.2** / 630.5 | 662.9 @1.77M | n/a | Alive, dense-task control as expected |
| e301 | 4668668 | cartpole swingup | same, τ=8 | 2.35M/4M (59%), CANCELLED | **850.0** / 851.4 | 865.5 @2.02M | n/a | Alive, τ8 clearly beats τ4 (850 vs. 616) |
| e302 | 4668669 | cheetah run | same, τ=4 | 2.37M/4M (59%), CANCELLED | **107.6** / 95.2 | 169.7 @2.15M | n/a | Alive, mid-pack |
| e303 | 4668670 | cheetah run | same, τ=8 | 2.31M/4M (58%), CANCELLED | **194.5** / 231.7 | 245.0 @2.30M | n/a | Alive, τ8 clearly beats τ4 (195 vs. 108), same pattern as cartpole |
| e304 | 4668889 | hopper hop | `RECIPE=plain_vark`, block-rew `sum`, relabel **ON**, `imag_length=16` | 2.19M/4M (55%) | **0.68** / 0.01 | 15.8 @1.93M | n/a | Confirms hyp. (i): credit-assignment fix does not rescue hopper |
| e305 | 4668890 | cheetah run | same | 2.22M/4M (55%) | **61.4** / 74.3 (rising) | 80.0 @2.22M (still climbing) | n/a | Alive but well below e302/e303 at matched %, unexpectedly weak so far |
| e306 | 4668891 | hopper hop | same, `imag_length=32` | 2.06M/4M (51%) | **0.08** / 0.08 | 12.0 @921k | n/a | Confirms hyp. (i)/(ii): longer imagination horizon doesn't rescue hopper either |
| e307 | 4668892 | cheetah run | same | 2.03M/4M (51%) | **56.3** / 55.4 | 79.3 @521k | n/a | Alive, close to e305 — `imag_length` 16 vs. 32 not yet differentiating |
| e308 | 4668893 | hopper hop | block-rew `sum`, relabel **OFF**, `imag_length=16` | 2.20M/4M (55%) | **3.86** / 3.31 | 89.8 @1.32M | n/a | Still dead but nominally highest of the 4 hopper cells — watch, not yet a rescue |
| e309 | 4668894 | cheetah run | same | 2.20M/4M (55%) | **32.8** / 29.7 | 53.1 @1.74M | n/a | Alive but weaker than the relabel-ON legs (e305/e307) — tentative relabel-helps-cheetah signal |
| e310 | 4668895 | hopper hop | block-rew `mean`, relabel **OFF**, `imag_length=16` | 2.21M/4M (55%) | **0.49** / 0.60 | 14.7 @929k | n/a | Dead, in line with hopper's other 3 cells |
| e311 | 4668896 | cheetah run | same | 2.19M/4M (55%) | **26.4** / 27.5 | 45.4 @1.67M | n/a | Weakest of the 4 cheetah cells so far — tentative `sum`-beats-`mean` signal, consistent with hyp. #1 in the launch note |

Pulled directly from each run's live `scores.jsonl` in `/work` (not yet archived). None
of e304–e311 are final; the cheetah below-baseline reading and the relabel/agg orderings
above are provisional until 4M. Full per-cell hyperparameters (struct weight, mask mode,
duration mode, reward agg, countdown, ratchet) for these and every prior experiment are
in `HYPERPARAMETER_COMPARISON.csv`, updated through e341 in this same pass (e322–e325
marked descoped; e314–e321 marked cancelled/superseded a second time).

**e296–e303 cancelled (2026-07-23), e304–e311 launched — block-pooled var-K credit,
with two implementation fixes.** Investigating the block-pooled manager credit path
(`variable_goal_block_rew=True`, never exercised post-fix, see §7 "Block-pooled
var-K credit assignment") surfaced two concrete implementation gaps in
`imag_loss_mgr`/`_imagine_with_manager`, both fixed same-day:
1. **`mgr_reward_agg=mean`** (the existing default) gives the manager literally zero
   reward-side signal that a longer hold accrued more value — a 4-step and 16-step
   block earning reward every step both pool to the same mean. Worked example: same
   total reward, `mean` reports identical `G_t` for 1×16-step vs 2×8-step splits;
   `sum` correctly favors more total reward. This is the likely mechanistic
   explanation for e62's old, previously-unexplained "duration drifted to 2.48, block-
   rew alone did not fix no-prior collapse" result (`mgr_reward_agg` was at its
   default `mean` there too).
2. **Horizon-truncated holds were mislabeled.** When a sampled duration doesn't fit in
   the remaining `imag_length` (structurally likely near the tail of every rollout,
   and guaranteed whenever `goal_duration_max == imag_length`), the manager was
   credited with the *sampled* duration class against a reward that only covers the
   *realized* (shorter) segment — a spurious bias against long durations sampled late
   in a fixed-length window. Fixed via hindsight relabeling
   (`agent.relabel_truncated_last_duration`, new): only the last real decision in a
   rollout can be truncated (every earlier one is, by construction, followed by a
   genuine switch); its `duration` class gets swapped to the realized length before
   REINFORCE runs. New flag `goal_duration_relabel_truncated` (default `True`;
   `False` = old, biased behavior) makes this ablatable. The duration Lagrangian
   prior and the worker's countdown are both deliberately left untouched by this fix
   (see §7 for why each is a different case).

Both fixes smoke-tested clean before launch: `run_smoke_variable_goals.sbatch` LEG 4
(job 4668864, relabel ON, `goal_duration_max=16=imag_length` to force frequent
truncation) and a standalone LEG 5 rerun (job 4668885, relabel OFF) — both
`ALL SMOKE LEGS PASSED`/`OK`, exit 0, zero NaN/Inf, duration stats sane
(mean≈8.2–8.8 vs τ=8, spread across all quartiles incl. 16–34% in the 13–16 bucket).

| Exp | Job | Task | Config | Status |
|---|---|---|---|---|
| e304 | 4668889 | hopper hop | block-rew, `sum`, relabel **ON**, `imag_length=16` | RUNNING, gpu-a100 |
| e305 | 4668890 | cheetah run | block-rew, `sum`, relabel **ON**, `imag_length=16` | RUNNING, gpu-a100 |
| e306 | 4668891 | hopper hop | block-rew, `sum`, relabel **ON**, `imag_length=32` | RUNNING, gpu-a100 |
| e307 | 4668892 | cheetah run | block-rew, `sum`, relabel **ON**, `imag_length=32` | RUNNING, gpu-a100 |
| e308 | 4668893 | hopper hop | block-rew, `sum`, relabel **OFF**, `imag_length=16` | RUNNING, gpu-a100 |
| e309 | 4668894 | cheetah run | block-rew, `sum`, relabel **OFF**, `imag_length=16` | RUNNING, gpu-a100 |
| e310 | 4668895 | hopper hop | block-rew, `mean`, relabel **OFF**, `imag_length=16` | RUNNING, gpu-a100 |
| e311 | 4668896 | cheetah run | block-rew, `mean`, relabel **OFF**, `imag_length=16` | RUNNING, gpu-a100 |

All 8: `RECIPE=plain_vark`, `DUR_MODE=lagrangian`, `DUR_TARGET=8.0` (held constant
across all 8 so the `imag_length` 16-vs-32 contrast isn't confounded by also changing
target duration), `RUN_STEPS=4000000`, `SEED=0`, BIG scale. `imag_length=32` (e306/
e307) was previously hardcoded to 16 at this scale ("won't fit BIG" per the original
script comment) — now overridable; watch for OOM, since BIG's activation memory at
32 imagined steps is untested at this scale.

**Hypotheses:** (i) e304 vs e308 vs e310 (hopper, relabel-ON/OFF/mean-baseline at
fixed `imag_length=16`) isolates whether the relabeling fix and the `sum`-vs-`mean`
choice each independently move the needle, replicating the F12-style dead-floor
result or not. (ii) e304 vs e306 (hopper, `imag_length` 16 vs 32 at `sum`+relabel-ON)
tests whether more decisions-per-rollout (per the "~2 decisions at τ=8, H=16" limit
discussed this session) changes the outcome. (iii) Cheetah legs (e305/e307/e309/e311)
are the dense-task controls — expected alive regardless, per every prior var-K run —
so a collapse there would flag a bug in the new code paths rather than a genuine
finding.

**Default changed.** `dreamerv3/configs.yaml` `defaults.agent` now ships with the e286
(hopper BIG)/e290 (cheetah BIG) recipe on by default — `goal_soft_reuse_adapt: True`,
`goal_soft_reuse_target: 0.5` (ratcheted from `goal_soft_reuse_target_init: 0.0` at
`goal_soft_reuse_target_vel: 3.9e-6`, the BIG-scale calibration), `goal_struct_adapt:
True`, `goal_struct_adapt_target: 0.01`. This is currently the best-performing cell of
the whole project: e286 (hopper BIG) finishes clearly *above* its Director baseline
(330.5 trail300 vs. ≈300, F20) and e290 (cheetah BIG) finishes inside the top of its own
noisy baseline band (325.9 vs. ≈300–435 — walked back from the 07-22 interim's "above,"
which was read off a since-decayed peak, see F20), and it's the first reuse/sparsity
mechanism family that doesn't collapse hopper outright. `mgr_cond_goalcode` stays `False`
in the YAML (unchanged) since `agent.py`
already force-enables it whenever `goal_soft_reuse_adapt` is set
(`self.mgr_cond_goalcode = True` at the point the flag is read, `agent.py:547`) —
confirmed by reading the code, not just the comment, so no config-level side effect
there. All existing `run_v3_prior_vargoal_*.sbatch` templates pass every one of these
flags explicitly per-launch (own `False`/old-target fallbacks), so this default change is
inert for every past and already-scripted experiment; it only changes behavior for a bare
`--configs defaults` invocation with no overrides (interactive runs, new sbatch scripts
that don't set these vars explicitly). Small-scale runs should still override
`goal_soft_reuse_target_vel` to `1.0e-6` (the small-scale calibration) — the shipped
default is BIG-calibrated, matching "the params in e286 and e290" literally.

**e294/e295 launched (2026-07-22)** — same recipe as e286/e290, ported to two untested
tasks, using explicit flags (not relying on the new ambient default, for the project's
usual explicit-repro convention):

| Exp | Job | Task | Scale | Config | Status (2026-07-23 pull) | Hypothesis |
|---|---|---|---|---|---|---|
| e294 | ~~4668645~~ → **4668658** (original a100 submission CANCELLED at 0:00 elapsed — never dispatched, `sacct` shows `CANCELLED+`/`None assigned`; resubmitted same-day to `gpu-v100`) | cartpole swingup | BIG | identical to e286/e290 (`RECIPE=director`, `MGR_FREQ=8`, `STRUCT_ADAPT=True` target 0.01, `MGR_COND_GOALCODE=True`, `GOAL_SOFT_REUSE_ADAPT=True` target 0.5 ratcheted from 0 at vel 3.9e-6) | RUNNING, `gpu-v100`, 1.41M/4M steps (35%): trail300 **635.8**, trail50 727.4, peak 754.4 @1.37M — in range of cartpole's ≥650 baseline family, no sign of the mechanism taxing an easy task | The best cell found on hopper/cheetah generalizes to a dense, already-easy task without cost — cartpole's own Director/masked baselines are all ≥650 (§1), so this checks the mechanism doesn't quietly tax an easy task even where sparsity pressure isn't needed to survive |
| e295 | ~~4668646~~ → **4668659** (same cancel/resubmit as e294) | acrobot swingup | BIG | identical to e294 | RUNNING, `gpu-v100`, 1.11M/4M steps (28%): trail300 **40.6**, trail50 35.0 (falling), peak 264.6 @1.03M then decayed — early trend leans toward "acrobot fails again," not a clean rescue, but too early (28%) to call | Acrobot has failed under every restricted recipe tried so far (F12, e176/e177/e192/e199, alive only under unrestricted Director e180) — this is the first test of the reuse-mechanism family on acrobot specifically; a clean readout either extends the "hopper needed the credit-assignment fix, not sparsity" story (Finding 6/F19) to a third sparse-ish task, or shows acrobot has its own distinct failure mode as F12 already flagged |

Both were originally queued behind the 8/8-GPU `gpu-a100` cap but that submission never
dispatched and was cancelled outright (`sacct`: `CANCELLED+`, 0:00 elapsed, no node ever
assigned — not a preemption, the jobs simply never left the queue); both were resubmitted
the same day to `gpu-v100` (jobs 4668658/4668659) and are running normally there.
`RUN_STEPS=4000000`, `SEED=0`, single seed each.

**e312/e313 launched (2026-07-23) — struct-only ablation of e294/e295, queued behind
them on `gpu-v100`.** Identical config to e294/e295 (`RECIPE=director`, `MGR_FREQ=8`,
`STRUCT_ADAPT=True` target 0.01, `MGR_COND_GOALCODE=True`) with the single change
`GOAL_SOFT_REUSE_ADAPT=False` (was `True` target 0.5, ratcheted from 0 at vel 3.9e-6) —
isolates the struct-correlation term alone, with no forced-implicit-sparsity (reuse)
loss active. `mask_sparsity_*`/masking is inactive either way under `RECIPE=director`
(no `masked_goals` config block), so this is a clean single-flag ablation of the reuse
mechanism only, not a masking-vs-no-masking comparison. **Originally launched as
e304/e305** but renumbered within minutes of submission after an unrelated concurrent
session claimed e304–e311 for the block-pooled var-K credit matrix (see entry above,
jobs 4668889–4668896, submitted 15:19:36 vs. this batch's original 15:16:14) — the
first pair (jobs 4668886/4668887) was cancelled before it ever started (still `PENDING`,
no `RUN_DIR` created) and resubmitted clean as e312/e313 to avoid a duplicate-number
collision in this log.

Same struct-only ablation extended to hopper/cheetah, **resumed from the existing
e242/e246 checkpoints** (BIG, `struct` condition, part of the e234–e249 implicit-sparsity
matrix, §7) rather than started fresh — those stopped at ~995k/1M steps on 2026-07-15/16,
before `goal_soft_reuse_adapt` existed, so this simply continues them to the same
4M-step budget as e294/e295/e312/e313 with the (now-ambient-default-since-07-22) reuse
term explicitly held off. Checkpoints restored from
`/bucket/.../results/dreamerv3/{e242_dmc_hopper_hop_director_BIG_j4664287,e246_dmc_cheetah_run_director_BIG_j4664291}/`
back to `/work` (rsync, 900MB each, `logdir/ckpt/latest` verified present pre-launch) and
`RUN_DIR` pointed at the restored path so `main.py` resumes from checkpoint in place;
config flags otherwise copied verbatim from each run's original `job.env` (notably
`STRUCT_W=200.0`, carried forward even though `RECIPE=director`'s own script default is
`0.0` — the original launch set it explicitly and resuming preserves the exact
loss-scale history rather than silently changing it mid-run).

| Exp | Job | Task | Scale | Config | Status (2026-07-23 launch) | Hypothesis |
|---|---|---|---|---|---|---|
| e312 | 4668901 | cartpole swingup | BIG | identical to e294 except `GOAL_SOFT_REUSE_ADAPT=False` | PENDING (`gpu-v100`, `AssocGrpGRES` — queued behind e294/4668658 and e295/4668659, dispatches automatically as GPUs free) | If e294 (struct+reuse) and e312 (struct-only) land in the same ≥650 range, the reuse/implicit-sparsity term isn't doing anything on cartpole either way (consistent with cartpole being easy/dense enough that neither mechanism matters); a gap would isolate which term (struct correlation vs. forced reuse) drives any difference from plain Director |
| e313 | 4668902 | acrobot swingup | BIG | identical to e295 except `GOAL_SOFT_REUSE_ADAPT=False` | PENDING (`gpu-v100`, `AssocGrpGRES` — queued behind e294/4668658 and e295/4668659, dispatches automatically as GPUs free) | e295 (struct+reuse) trended toward acrobot failing again (F12 pattern); e313 tests whether struct correlation alone, without the reuse loss, changes that outcome — separates "does struct correlation help/hurt acrobot" from "does the forced-reuse mechanism help/hurt acrobot" |
| e242 (resumed) | 4668898 | hopper hop | BIG | struct-only, resumed from ~995k/4M steps (was 1M-budget original) | PENDING (`gpu-v100`, `AssocGrpGRES` — queued behind e294/e295 and e312/e313) | e242 (struct, no controller) was alive but still climbing at its original 1M cutoff (115/116, well below the ~250–305 Director-at-1M reference, §7); extending to 4M checks whether it keeps closing that gap given more budget, now under the same full-budget comparison as e294/e295/e312/e313 |
| e246 (resumed) | 4668899 | cheetah run | BIG | struct-only, resumed from ~996k/4M steps (was 1M-budget original) | PENDING (`gpu-v100`, `AssocGrpGRES` — queued behind e294/e295 and e312/e313) | e246 (struct, no controller) was already within noise of Director-at-1M (330/335 vs. e123-class 250–305, §7); extending to 4M checks whether that parity holds or decays over a full run, same comparison basis as e294/e295/e312/e313 |

`RUN_STEPS=4000000`, `SEED=0`, single seed each. Both queue behind e294/e295 on the
8/8-GPU `gpu-v100` account cap (no explicit `--dependency`, same natural-queueing
precedent as e294/e295's own a100→v100 move above) — SLURM holds them `PENDING` with
reason `AssocGrpGRES` and dispatches as soon as GPUs free.

**e279/e283/e288/e289 cancelled (2026-07-22, ~84–98%/4M budget) — all 4 hopper-small
cells, obviously dead, freed for e294/e295's queue.** Live pull immediately before
cancelling: `e279` (family A, target 0.8) trail300 **0.3**, trail50 0.4, peak only 14.9;
`e283` (family B) trail300 **0.9**, trail50 0.5, peak 28.6 (early, decayed); `e288`
(family C struct+ratchet) trail300 **0.0**, trail50 0.1, peak 15.6 (early, decayed);
`e289` (family C ratchet-only) trail300 **1.2**, trail50 1.8, peak 76.2 (reached mid-run,
fully decayed back down) — all indistinguishable from the pre-existing small-hopper dead
floor (e213 ≈1, F15) regardless of mechanism or target, at 84–98% of budget with no
recovery in sight. The other 4 small-scale cells (e281, e285, e292, e293, all cheetah)
and all 6 remaining BIG-scale cells (e278, e280, e282, e284, e286, e287, e290, e291) are
clearly alive (91–353 trailing) and were left running — including e278, which has
declined from its 327 peak to ~91–99 but is nowhere near the dead floor these four show.
Archived to `/bucket/.../results/dreamerv3/` (`SKIP_REPLAY=1`, slurm logs included,
copies verified before deleting from `/work`); this does not free `gpu-a100` capacity for
e294/e295 (the cancelled cells were on `gpu-v100`), only `gpu-v100` slots.

**e296–e303 launched (2026-07-22) — plain var-K Lagrangian isolation, τ∈{4,8}, all 4
tasks, BIG.** Fills a gap F12/F7 left open: `DUR_MODE=lagrangian` (e171's exact duration
mechanism — `goal_duration_lagrange True`, `goal_duration_reg 0.0`, `goal_duration_adapt
False`) has only ever been run bundled with masking+struct200 (`RECIPE=vark_masked`,
e170–e177) or, pre-fix, in the e144–e159 symmetry matrix (group C, cartpole-only, 557,
predates the 07-10 forward-fill/one-sided-controller fixes). It has never been isolated
post-fix with `RECIPE=plain_vark` (struct forced to 0, no mask flags at all) the way the
fixed-prior duration mechanism already was in e166–e169/e178/e179. `run_v3_prior_vargoal_big_a100.sbatch`'s
`plain_vark` branch + `DUR_MODE=lagrangian` is exactly "pure Director + only variable
goal length, e171-style, nothing else":

| Exp | Job | Task | τ target | Config | Status (2026-07-23 pull, all RUNNING on `gpu-a100`) |
|---|---|---|---|---|---|
| e296 | 4668663 | hopper hop | 4 | `RECIPE=plain_vark, DUR_MODE=lagrangian, DUR_TARGET=4.0` | 2.26M/4M (56%): trail300 **0.07**, trail50 0.01, peak only 8.1 @921k — already indistinguishable from the F12/F15 dead floor |
| e297 | 4668664 | hopper hop | 8 | same, `DUR_TARGET=8.0` | 2.28M/4M (57%): trail300 **0.32**, trail50 0.03, peak 15.5 @2.07M (transient, decayed) — same dead floor, τ8 no better than τ4 |
| e298 | 4668665 | acrobot swingup | 4 | `RECIPE=plain_vark, DUR_MODE=lagrangian, DUR_TARGET=4.0` | 2.22M/4M (55%): trail300 **2.8**, trail50 4.0, falling from a 154 peak @2.08M — decaying toward the dead floor, not yet at it |
| e299 | 4668666 | acrobot swingup | 8 | same, `DUR_TARGET=8.0` | 2.10M/4M (52%): trail300 **4.8**, trail50 6.9, falling from a 165 peak @224k (very early) — same decay pattern as e298 |
| e300 | 4668667 | cartpole swingup | 4 | `RECIPE=plain_vark, DUR_MODE=lagrangian, DUR_TARGET=4.0` | 1.71M/4M (43%): trail300 **601.0**, trail50 579.7, peak 651.9 @961k — clearly alive, dense-task baseline range |
| e301 | 4668668 | cartpole swingup | 8 | same, `DUR_TARGET=8.0` | 1.66M/4M (41%): trail300 **748.7**, trail50 754.4, peak 849.0 @1.37M — alive and stronger than τ4 |
| e302 | 4668669 | cheetah run | 4 | `RECIPE=plain_vark, DUR_MODE=lagrangian, DUR_TARGET=4.0` | 1.67M/4M (42%): trail300 **97.2**, trail50 113.6 (rising), peak 137.9 @1.30M — alive, mid-training |
| e303 | 4668670 | cheetah run | 8 | same, `DUR_TARGET=8.0` | 1.61M/4M (40%): trail300 **184.9**, trail50 196.3 (rising), peak 219.3 @1.60M (still climbing) — alive and stronger than τ4, same τ4<τ8 pattern as cartpole |

All 8: `MGR_FREQ` n/a (variable-K), `STRUCT_W=0.0` (plain_vark default), no mask flags,
`MGR_COND_GOALCODE=False` (plain_vark default — unconditioned manager, unlike the reuse
family), `RUN_STEPS=4000000`, `SEED=0`. Job ids in `e296_303_job_ids.tsv`.
**Early readout (40–57% through budget, not final):** hopper is already at the dead
floor at both τ, matching F12's prediction exactly. Acrobot has NOT reached the floor yet
but is falling steadily from an early peak at both τ — consistent with F12's "acrobot has
its own (slower) failure mode," not yet distinguishable from "will eventually flatten
somewhere above zero." Cartpole and cheetah are both alive and, at both tasks, τ8 clearly
beats τ4 (cartpole 748.7 vs. 601.0; cheetah 184.9 vs. 97.2) — the first clean signal on
the τ4-vs-τ8 question this matrix was designed to answer, though still early enough that
cheetah's τ4 leg is trending up and could close the gap.

**Hypothesis:** given F12 (masking-alone and var-K-alone were each independently
sufficient to kill hopper/acrobot in the *fixed-prior* isolation, e168/e169/e178/e179) and
F17 (the dual-head credit-assignment confound, not sparsity/duration-variability per se,
was the actual culprit for masking), the Lagrangian duration controller alone should
behave like the fixed-prior one did: alive on cartpole/cheetah (dense-reward, e46-style),
dead or near-dead on hopper/acrobot regardless of τ — since nothing here touches the
dual-head goal-content mechanism F17 implicated, only the switch-timing mechanism F12
already showed was independently fatal. τ4 vs τ8 tests whether shorter/more frequent
hold-length variation (closer to e169's dead τ4) is worse than longer (closer to e178's
weak-pulse τ8) under the tighter Lagrangian tracking (F7: ~20× tighter than the reg
prior) — if τ8-lagr survives where τ8-reg (e178) only pulsed, tracking precision itself
would be implicated as a lever, not just target length.

**Paper updated (2026-07-23):** added a new §"Variable-duration option: mechanism
reference" to `paper/main.tex` (after "The combined recipe", before "Experimental
setup") — a self-contained diagram + equations + per-network I/O/loss table for the
var-K machinery alone (decision-timing recurrence, the two manager credit-assignment
resolutions, the duration prior/Lagrangian), plus a "Status" subsection answering the
"is this a bug?" question raised by e296–e299 (hopper/acrobot dead again under plain
var-K post-fix): no — three points argue against an implementation defect (reproduced
across two independent code revisions; the countdown/`worker_timed_goals` fix rescues
a collapsing *dense*-task var-K run 5.4× but leaves hopper/acrobot exactly as dead;
the same tasks are solvable by an unrestricted manager). Flags var-K × single-head
(F17) as the one still-untested combination that could still change this reading.

---

### FINAL results (2026-07-23) — e278–e293, all 16 cells COMPLETED the full 4M-step
budget. Supersedes the 07-22 interim pull below; full finding is **F20** (§3).

All 12 non-cancelled cells finished (`sacct`: `COMPLETED`, exit 0:0); the 4 hopper-small
cells (e279/e283/e288/e289) were cancelled 07-22 (already dead, see above) and archived.
`Score` = mean `episode/score` over the trailing 300 logged episodes at the 4M-step
finish; `Trail50` = trailing 50 (volatility/endpoint check); `Peak` unchanged from the
interim pull. Pulled directly from each run's final `scores.jsonl`/`metrics.jsonl` in
`/work` (not yet archived to `/bucket` as of this write-up).

| Exp | Job | Task | Scale | Family | Score (trail300) | Trail50 | Peak (step) | Reuse/overlap final | Lagrange scale final | vs. interim (07-22) | vs. Director baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e278 | 4668382 | hopper | BIG | A: code+decoded input | **163.0** | 222.3 | 326.8 (@2.71M) | sim 0.81/0.80 | 0.0001 (floor) | 240.6 → 163.0 (down, noisy — see chunked trace below) | below ≈300, alive not collapsed |
| e280 | 4668384 | cheetah | BIG | A | **296.6** | 291.7 | 333.0 (@2.87M) | sim 0.88/0.80 | 0.0 (floor) | 309.1 → 296.6 (stable) | at low end of ≈300–435 |
| e281 | 4668385 | cheetah | small | A | **292.9** | 288.4 | 351.7 (@3.75M) | sim 0.94/0.80 | 0.0 (floor) | 142.2 → 292.9 (up, still rising) | well above ≈100 |
| e282 | 4668386 | hopper | BIG | B: decoded-only input | **230.2** | **1.6** | 347.3 (@2.64M) | sim 0.81/0.80 | 0.13 (spiked from floor) | 287.5 → 230.2/**1.6** (TERMINAL COLLAPSE, see F20) | trail300 near baseline but trail50 is the honest read: dead |
| e284 | 4668388 | cheetah | BIG | B | **198.2** | 208.2 | 226.6 (@4.0M, still climbing at cutoff) | sim 0.96/0.80 | 0.0 (floor) | 146.7 → 198.2 (up) | below ≈300–435 but alive and rising |
| e285 | 4668389 | cheetah | small | B | **180.1** | 235.0 | 289.3 (@3.73M) | sim 0.93/0.80 | 0.0 (floor) | 196.2 → 180.1 (trail50 recovering to 235) | above ≈100 |
| e286 | 4668390 | hopper | BIG | C: struct+ratchet | **330.5** | 325.7 | 430.6 (@3.69M) | ov 0.50/0.50 | 0.17 | 358.3 → 330.5 (declined from peak, stable plateau, no collapse) | **above ≈300 — best hopper cell in the project** |
| e287 | 4668391 | hopper | BIG | C: ratchet-only | **199.4** | 185.6 | 298.8 (@3.84M) | ov 0.50/0.50 | 0.36 | 186.7 → 199.4 (stable, mild recovery) | below ≈300, alive, stable plateau |
| e290 | 4668394 | cheetah | BIG | C: struct+ratchet | **325.9** | 332.4 | 449.4 (@3.06M) | ov 0.50/0.50 | 1.21 | 404.5 → 325.9 (down from peak) | inside ≈300–435, not clearly above once baseline noise is considered |
| e291 | 4668395 | cheetah | BIG | C: ratchet-only | **160.6** | 167.5 | 214.8 (@1.13M) | ov 0.50/0.50 | 15.6 (still elevated, not at floor) | 132.6 → 160.6 (up) | well below ≈300–435, weakest BIG cheetah cell |
| e292 | 4668396 | cheetah | small | C: struct+ratchet | **107.0** | 149.7 | 191.8 (@1.38M) | ov 0.50/0.50 | 23.9 (elevated, oscillating hard all run) | 80.3 → 107.0 (up) | roughly matches ≈100 |
| e293 | 4668397 | cheetah | small | C: ratchet-only | **259.2** | 277.8 | 342.1 (@3.68M) | ov 0.50/0.50 | 0.28 | 201.6 → 259.2 (up) | well above ≈100, strongest small-scale cell |

Baselines unchanged from the interim pull: hopper BIG ≈300 (e124), cheetah BIG ≈300–435
(e123/e191, noisy), cheetah small ≈100 (e212), hopper small ≈1 (e213, dead even
unrestricted).

**Headline correction to the interim readout: e282 (family B, hopper BIG) did not
survive to the finish.** Its `episode/score` sat on a stable 200–230 plateau through
step 3.94M (98.6% of budget), matching the interim's "rising, near-baseline" read. At
step 3,939,344 `train/goal/reuse_adapt_scale_mean` — flat at 0.0002–0.0005 (its floor)
for the preceding ~3.9M steps, like every other cell in the table — jumped to 0.18, then
0.21 by step 3,943,600, then 0.09 by 3,947,776; `train/wkr_goal_rew` fell in lockstep
(0.53 → 0.35 → 0.33) and `episode/score` crashed to ~0 starting the very next logged
episode (step 3,947,937) and stayed there (occasional 1–17-point blips, mean 1.6 over the
last 50 episodes) through the 4M cutoff — roughly the last 1.4% of training, no time
to recover before the budget ran out. This is the *same* signature F18/F19 already
named (a railed Lagrange multiplier dominating the loss and destroying the policy) —
just triggered ~2.9M steps later than any F19 cell, and well past the point the interim
pull judged the multiplier "settled near its floor" (it had been, for essentially the
entire run, until it wasn't). All 15 other cells show smaller scale excursions
throughout training (see raw `metrics.jsonl` — jumps to 1–20+ are common and self-correct
within tens of thousands of steps) but none of the other 15 fails to recover before 4M.
Full chunked (10-bucket) score trajectories for the 6 BIG hopper/cheetah-A/C cells, and
the reuse-scale spike detection, are in the analysis scripts used for this pull
(not checked in; re-derivable from `scores.jsonl`/`metrics.jsonl` directly).

**Struct-stacking reversal (F19 → interim → final) holds at BIG scale but NOT at small
scale — scale-dependent, not just target-dependent.** BIG: struct+ratchet still clearly
beats ratchet-only on cheetah (e290 325.9 vs. e291 160.6), same direction as the interim
pull. Small: ratchet-only clearly beats struct+ratchet (e293 259.2 vs. e292 107.0) — the
*original* F19 (0.7-target) ordering, not the reversed one. The interim pull's own note
already flagged small scale as "the one place the old ordering survives" using
partial numbers (e292 80.3 vs. e293 201.6); the final numbers confirm that read rather
than reversing it further. Net: struct-stacking's cost/benefit depends on both target
*and* scale — do not generalize a single "struct helps/hurts" rule across cells.

**e278 (family A, hopper BIG) is volatile, not a clean rescue.** Chunked into 10 equal
step-windows, its per-window mean score is 42 → 119 → 259 → 289 → 135 → 100 → 264 → 139
→ 68 → 142 — repeatedly rising above 250 and falling back below 100, never settling.
Ends at a mediocre 163 (trail300) despite a trailing-50 of 222 (the last 50 episodes
happen to sit in an up-swing). This is qualitatively different from e286's trajectory
(monotonic rise to a ~350–390 plateau, then a mild, stable decline) — family A "survives"
hopper in the sense of never reaching the dead floor, but does not deliver a stable
policy the way family C does.

**Bottom line for F20:** of the 6 BIG-scale hopper/cheetah cells that were dead or
near-dead under F19's 0.95/0.7 targets, 5 are genuinely alive and stable at the lower
0.8/0.5 targets by the 4M-step finish (e278 noisy-but-alive, e280, e284, e286, e287,
e290, e291 all clear of the floor); the 6th (e282) demonstrates the underlying
magnitude-domination fragility F18/F19 identified is still latent in family B — it did
not go away at the lower target, it just took longer to trigger. Family C
(`goal_soft_reuse_adapt`, the promoted config default) is the only family with zero
collapse events across all 6 of its cells (hopper+cheetah × BIG+small × struct/no-struct)
and the only one with a cell (e286) finishing clearly above its Director baseline —
the strongest evidence yet that the discrete block-overlap loss, not the continuous
decoded-goal-similarity loss (families A/B), is the mechanism worth building on.

---

### Prior state (2026-07-22, morning — e278–e293 INTERIM at 73–89%/4M steps: the lower,
better-calibrated target rescues almost every cell, reversing F19's headline claim for
BIG-scale hopper). Retest of the same 3 reuse-mechanism families as e258–e277, with a
looser ratchet (~2M steps instead of ~1M) and lower targets (`goal_reuse_target=0.8`, was
0.95; `goal_soft_reuse_target=0.5`, was 0.7), testing F19's hypothesis that the collapse
is caused by the target being too aggressive relative to each mechanism's own achievable
range, not by the mechanism itself. All 16 RUNNING (`gpu-v100` small cells + `gpu-a100`
BIG cells), no job has crashed or been requeued. Design mirrors e258–e277 exactly (same
task×scale×family×struct grid, `RECIPE=director`, `MGR_FREQ=8`, `RUN_STEPS=4000000`,
`SEED=0`), only the two targets and their ratchet velocities changed:
`GOAL_REUSE_TARGET_VEL` = 6.3e-6 (BIG) / 1.7e-6 (small) for the 0→0.8 ratchet;
`GOAL_SOFT_REUSE_TARGET_VEL` = 3.9e-6 (BIG) / 1.0e-6 (small) for the 0→0.5 ratchet.

**Readout logic (pre-registered), restated:** compare each cell directly against its
e258–e277 counterpart on (1) whether the Lagrange scale still rails at its ceiling — if
it now settles interior, the target is within the mechanism's reachable range; (2) task
score relative to the ≈300/≈300–435/≈100/≈1(dead) baselines; (3) whether hopper stays
dead at the lower target too, or whether its failure was target-magnitude-dependent.

**Interim results (2026-07-22, pulled directly from each run's live `scores.jsonl` /
`metrics.jsonl`; all 16 cells 73–89% through the 4M-step budget, none finished).** `Score`
= mean `episode/score` over the trailing 300 logged episodes as of this pull; `Trend` =
that trailing-300 mean vs. the preceding 300-episode window (rising/flat/falling — note a
"falling" cell can still be far above the dead floor); `Scale/target` = the mechanism's
own Lagrange multiplier and its current ratchet target.

| Exp | Job | Task | Scale | Family | Score (trail300) | Trend | Peak (step) | Reuse-or-overlap / target | Lagrange scale | vs. e258–e277 counterpart |
|---|---|---|---|---|---|---|---|---|---|---|
| e278 | 4668382 | hopper | BIG | A: code+decoded input | **240.6** | rising | 327 (@2.71M) | sim 0.860/0.80 | 0.05 (near floor) | e258 **0.001 → 240.6** |
| e279 | 4668383 | hopper | small | A | **0.07** | rising (noise) | 12.6 (@3.06M) | sim 0.810/0.80 | 0.02 (near floor) | e260 0.002 → 0.07 (still dead) |
| e280 | 4668384 | cheetah | BIG | A | **309.1** | rising | 333 (@2.87M) | sim 0.884/0.80 | ~0 (floor) | e262 **1.85 → 309.1** |
| e281 | 4668385 | cheetah | small | A | **142.2** | flat | 201 (@2.56M) | sim 0.928/0.80 | ~0 (floor) | e264 **3.47 → 142.2** |
| e282 | 4668386 | hopper | BIG | B: decoded-only input | **287.5** | rising | 347 (@2.64M) | sim 0.797/0.80 | 0.01 (near floor) | e274 **0.004 → 287.5** |
| e283 | 4668387 | hopper | small | B | **0.99** | falling | 28.6 (@0.86M) | sim 0.805/0.80 | 0.09 | e275 0.001 → 0.99 (still dead) |
| e284 | 4668388 | cheetah | BIG | B | **146.7** | flat | 188 (@2.66M) | sim 0.954/0.80 | ~0 (floor) | e276 **7.24 → 146.7** |
| e285 | 4668389 | cheetah | small | B | **196.2** | flat | 236 (@2.80M) | sim 0.975/0.80 | ~0 (floor) | e277 **7.57 → 196.2** |
| e286 | 4668390 | hopper | BIG | C: struct+ratchet | **358.3** | falling (still high) | 424 (@2.91M) | ov 0.503/0.50 | 0.06 (near floor) | e266 **1.75 → 358.3** |
| e287 | 4668391 | hopper | BIG | C: ratchet-only | **186.7** | falling (still high) | 283 (@2.55M) | ov 0.498/0.50 | 0.19 | e267 **0.19 → 186.7** |
| e288 | 4668392 | hopper | small | C: struct+ratchet | **0.16** | flat | 15.6 (@0.37M) | ov 0.510/0.50 | 0.01 (floor) | e268 0.30 → 0.16 (still dead) |
| e289 | 4668393 | hopper | small | C: ratchet-only | **0.60** | falling | 76.2 (@1.11M) | ov 0.510/0.50 | 0.18 | e269 0.004 → 0.60 (still dead) |
| e290 | 4668394 | cheetah | BIG | C: struct+ratchet | **404.5** | rising | 449 (@3.06M) | ov 0.501/0.50 | 0.43 | e270 **70.4 → 404.5** |
| e291 | 4668395 | cheetah | BIG | C: ratchet-only | **132.6** | flat | 215 (@1.13M) | ov 0.517/0.50 | 20.0 (elevated) | e271 **182.0 → 132.6 (down)** |
| e292 | 4668396 | cheetah | small | C: struct+ratchet | **80.3** | falling | 192 (@1.38M) | ov 0.510/0.50 | 0.01 (floor) | e272 73.3 → 80.3 |
| e293 | 4668397 | cheetah | small | C: ratchet-only | **201.6** | flat, near peak | 298 (@3.08M) | ov 0.494/0.50 | ~0 (floor) | e273 **111.3 → 201.6** |

Baselines for reference (unchanged): hopper BIG ≈300 (e124), cheetah BIG ≈300–435
(e123/e191), cheetah small ≈100 (e212), hopper small ≈1 (e213, dead even unrestricted).

**Interim readout — branch 1 (Lagrange scale) fires clean, 14/16 cells:** in every cell
except e287/e289 (moderate, 0.18–0.19) and e291 (elevated, 20.0), the multiplier has
settled at or within a few multiples of its numerical floor (1e-5–0.4), not railed at its
100.0 ceiling the way every e258–e277 cell was. The 0.8/0.5 targets are within reach at
essentially no pressure — confirming the pre-registered "target was too aggressive, not
the mechanism" branch, decisively.

**Branch 2 (task score) fires clean for BIG scale, both tasks, all three families:**
every one of the 6 BIG-scale cells that was dead or near-dead in e258–e277 is now within
20–35% of, or above, its Director baseline: e278 240.6 (was 0.001), e280 309.1 (was
1.85, now essentially at the ≈300–435 baseline), e282 287.5 (was 0.004), e284 146.7 (was
7.24), e286 358.3 (was 1.75, now *above* the ≈300 baseline), e290 404.5 (was 70.4, now
*above* the ≈300–435 baseline). Small-scale cheetah improves similarly (e281 142.2,
e285 196.2, e293 201.6 — all above the ≈100 baseline; e292 80.3 roughly matches it). The
lone regression is e291 (cheetah BIG, family C ratchet-only): 182.0 → 132.6, *down*
despite the looser target — its Lagrange scale (20.0) is also the only one of the 6
"healthy" BIG cells not near its floor, suggesting this specific cell hasn't yet found the
cheap equilibrium the others have; worth rechecking once it finishes.

**Branch 3 (hopper) is scale-dependent, not resolved uniformly:** hopper BIG is alive in
all 6 cells across all three families for the first time under any reuse/sparsity
mechanism in the project (187–358, vs. F19's 0.001–1.75) — directly overturning F19's
"hopper fails under every mechanism except the unrestricted single-head design" for the
BIG-scale, well-calibrated-target case. Hopper SMALL stays dead in all 4 cells
(0.07–0.99), indistinguishable from the small-hopper noise floor under pure unrestricted
Director (e213 ≈1, F15) — this reads as the pre-existing small-scale hopper limitation,
not a reuse-mechanism failure, and was never claimed to be mechanism-specific.

**Why this reads as a real reversal, not a transient:** in e258–e277, every cell that
eventually collapsed did so once its ratchet finished tightening (~1M steps) and never
recovered over the remaining 3M steps. These interim cells are 2.6–3.6M steps in — 1.6–2.6M
steps past their own (slower, ~2M-step) ratchet completion — and show no such collapse
signature; several are still rising. The 4M-step checkpoints (all 16 cells) will confirm,
but the diagnostic window that mattered in the prior batch has already been crossed here
without incident.

**One finding likely needs revision once complete:** F19 stated struct is "safe only as
the sole sparsity lever" because struct+ratchet halved cheetah's score vs. ratchet-only
under the 0.7 target (e270 70 vs. e271 182). At the 0.5 target the direction **flips**:
struct+ratchet now clearly beats ratchet-only on cheetah BIG (e290 404.5 vs. e291 132.6)
and roughly matches it on cheetah small (e292 80.3 vs. e293 201.6, the one place the
old ordering survives). Struct's cost/benefit against a direct reuse loss looks
target-dependent, not fixed — do not generalize F19's struct-stacking conclusion beyond
the 0.7-target regime until this batch finishes.

**Not yet closed:** whether the "healthy at floor" BIG-scale cells hold their score
through to 4M steps (esp. e286/e287, currently the two *falling*-trend cells, albeit from
a very high level); `goal_delta_mode`'s own marginal contribution, still untested at
scale (both its scheduled A/Bs were cancelled pre-readout, §7); a second seed on any
promoted configuration before this enters the paper as a finding.

---

### Prior state (2026-07-21 — e258–e277 (16 cells, 3 mechanism families) all
COMPLETED the full 4M-step budget; nothing currently running — **F19**: every
differentiable reuse/sparsity mechanism tested reproduces F18's magnitude-domination
collapse on hopper (16/16 hopper-adjacent... see below, all 4 hopper cells per family
dead); cheetah survives partially, and only under the discrete block-overlap loss
(family C), best when NOT stacked with struct. Full readout in the per-batch sections
below and Finding F19 in §3. e266–e273 (`goal_soft_reuse_adapt`, direct/differentiable
block-overlap sparsity) × {hopper,cheetah} × {small,BIG} × {struct+ratchet,
ratchet-only}, full 4M-step budget, all 8 COMPLETED. e258/e260/e262/e264
(`goal_reuse_adapt` "direct" cells, continuous goal-space Lagrangian, ratcheted 0→0.95
over ~1M steps, no struct, `mgr_cond_goalcode=True`) × {hopper,cheetah} × {small,BIG},
full 4M-step budget, all 4 COMPLETED. e259/e261/e263/e265 (the `goal_delta_mode` "delta"
counterparts) were CANCELLED ~3h10m in, no readout, superseded by e274–e277 (a
manager-input ablation instead of a delta-mode comparison) (a first submission with the
target fixed at 0.95 from step 0 was cancelled within ~2 minutes and relaunched
ratcheted). e274–e277 (`goal_reuse_adapt`, `mgr_cond_goalcode=False`, manager sees
decoded goal only, no raw code) × {hopper,cheetah} × {small,BIG}, full 4M-step budget,
all 4 COMPLETED. e250–e257 (struct/ratchet retest under `goal_delta_mode`, full 4M-step
budget), launched earlier the same day, were CANCELLED ~3.5h in and archived (no
readout), superseded by e258–e265. e234–e249, the implicit-sparsity matrix, COMPLETED
07-15/16 at the full 1M-step budget: struct-only survives in 4/4 cells, the REINFORCE
implicit-sparsity controller (ratchet/direct/struct+ratchet) collapses 12/12 cells —
**F18**. e230–e233 cancelled 07-15 at 15–17% of budget to free hardware for the e234
launch; final partial numbers below suggest the looser 0.5 target substantially recovers
hopper vs. the 0.3-target predecessors, but the runs never finished. Prior state:
e178–e211 cancelled/archived 07-14, replaced by the single-head mask campaign
e212–e229; e226 was the project's first alive sparse-task cell (F17))

### Results board — 16 cells, all COMPLETED at the full 4M-step budget, all
`RECIPE=director` (fixed K=8, plain whole-code goal blocks, no masking/var-K), `SEED=0`.
Three mechanism families, all targeting some form of goal reuse/sparsity without the
abandoned explicit-mask design. `Score` = mean `episode/score` over the last ~300
logged episodes (`Peak` = best mean-episode score reached at any point, with the step it
was reached); `Reuse@end` = the mechanism's own achieved-similarity/overlap metric,
averaged over the same trailing window, vs. its target. Director-recipe reference
baselines (no reuse mechanism, same hardware): hopper BIG ≈300 (e124), cheetah BIG
≈300–435 (e123/e191, noisy), cheetah small ≈100 (e212), hopper small ≈1 (e213, dead even
unrestricted — F15). Dead floor for both tasks is score ≈0–1.

| Exp | Job | Task | Scale | Family | Config vs. family default | Score (last~300) | Peak (step) | Reuse@end / target | Hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| e258 | 4664904 | hopper | BIG | A: `goal_reuse_adapt`, code+decoded input | `MGR_COND_GOALCODE=True` (manager sees raw one-hot code **and** decoded goal) | **0.001** | 51 (@480k) | sim 0.973 / 0.95 | Continuous decoded-goal-space similarity loss alone (no code-level mechanism) reaches ~0.95 mean similarity at reasonable task return, without needing `goal_delta_mode`'s code recombination |
| e260 | 4664906 | hopper | small | A | vel 3.9e-6 (small-scale calibration) | **0.002** | 17 (@176k) | sim 0.917 / 0.95 | same, at small scale |
| e262 | 4664908 | cheetah | BIG | A | — | **1.85** | 203 (@945k) | sim 0.953 / 0.95 | same, cheetah |
| e264 | 4664910 | cheetah | small | A | vel 3.9e-6 | **3.47** | 138 (@881k) | sim 0.957 / 0.95 | same, cheetah/small |
| e274 | 4664976 | hopper | BIG | B: `goal_reuse_adapt`, decoded-only input | `MGR_COND_GOALCODE=False` (manager sees **only** decoded goal + world state, no raw code) | **0.004** | 138 (@504k) | sim 0.949 / 0.95 | Manager doesn't need the exact discrete code to satisfy the similarity target — decoded goal alone is sufficient input; compare directly against e258 (same config otherwise) |
| e275 | 4664978 | hopper | small | B | vel 3.9e-6 | **0.001** | 28 (@256k) | sim 0.917 / 0.95 | same, vs. e260 |
| e276 | 4664977 | cheetah | BIG | B | — | **7.24** | 175 (@737k) | sim 0.932 / 0.95 | same, vs. e262 |
| e277 | 4664979 | cheetah | small | B | vel 3.9e-6 | **7.57** | 132 (@817k) | sim 0.946 / 0.95 | same, vs. e264 |
| e266 | 4664964 | hopper | BIG | C: `goal_soft_reuse_adapt`, struct+ratchet | `STRUCT_ADAPT=True` (target 0.01) | **1.75** | 209 (@625k) | overlap 0.594 / 0.7 | Differentiable block-overlap loss on the manager's own softmax (no REINFORCE, no sample recombination) reaches the 0.7 overlap target at reasonable return — resolves F18 via a cheaper mechanism than `goal_delta_mode`; struct-adapt isolates whether the known-safe indirect lever still helps on top |
| e267 | 4664965 | hopper | BIG | C: ratchet-only | `STRUCT_ADAPT=False` | **0.19** | 51 (@440k) | overlap 0.603 / 0.7 | isolates the ratchet controller alone vs. e266's struct+ratchet |
| e268 | 4664968 | hopper | small | C: struct+ratchet | vel 2.9e-6 | **0.30** | 14 (@3.56M) | overlap 0.589 / 0.7 | same as e266, small scale |
| e269 | 4664969 | hopper | small | C: ratchet-only | vel 2.9e-6 | **0.004** | 44 (@416k) | overlap 0.559 / 0.7 | same as e267, small scale |
| e270 | 4664966 | cheetah | BIG | C: struct+ratchet | — | **70.4** | 113 (@336k) | overlap 0.559 / 0.7 | same as e266, cheetah |
| e271 | 4664967 | cheetah | BIG | C: ratchet-only | — | **182.0** | 365 (@808k) | overlap 0.569 / 0.7 | same as e267, cheetah |
| e272 | 4664970 | cheetah | small | C: struct+ratchet | vel 2.9e-6 | **73.3** | 151 (@545k) | overlap 0.599 / 0.7 | same as e266, cheetah/small |
| e273 | 4664971 | cheetah | small | C: ratchet-only | vel 2.9e-6 | **111.3** | 155 (@3.09M) | overlap 0.601 / 0.7 | same as e267, cheetah/small |

**Family defaults:** A/B share `GOAL_REUSE_ADAPT=True`, `GOAL_REUSE_TARGET=0.95`
ratcheted from `GOAL_REUSE_TARGET_INIT=0.0` (vel 1.5e-5 BIG unless noted),
`GOAL_DELTA_MODE=False`, `STRUCT_W=0.0`, `IMPL_MODE=none`. C shares
`GOAL_SOFT_REUSE_ADAPT=True`, `GOAL_SOFT_REUSE_TARGET=0.7` ratcheted from
`GOAL_SOFT_REUSE_TARGET_INIT=0.0` (vel 1.1e-5 BIG unless noted), `MGR_COND_GOALCODE=True`,
`STRUCT_W=0.0`, `GOAL_REUSE_ADAPT=False`, `IMPL_MODE=none`.

**Cross-family readout (2026-07-21, all 16 cells at full budget):**
- **A vs. B** (raw code channel load-bearing?): no. Removing the one-hot code channel
  from the manager's input (B) does not rescue any cell and does not clearly hurt either
  — both hit the same collapse. If anything B's cheetah cells hold a slightly higher
  score at the end (e276 7.24 vs e262 1.85; e277 7.57 vs e264 3.47) and slightly higher
  peaks on hopper (e274 138 vs e258 51; e275 28 vs e260 17), but both are dead by any
  task-relevant standard. **Verdict: the raw code channel is not load-bearing — but moot,
  since the mechanism it's an input to collapses regardless of whether it's present.**
- **A/B vs. C** (continuous goal-space loss vs. discrete block-overlap loss): C is
  decisively better on cheetah (70–182 vs. 1.85–7.57, i.e. roughly an order of magnitude
  closer to the ≈300–435 Director baseline) and no worse on hopper (both families dead:
  A/B 0.001–0.004, C 0.004–1.75 — C's hopper numbers are nominally higher but still 2–3
  orders of magnitude below the ≈300 baseline, not a real rescue). **Verdict: the
  discrete block-overlap loss (family C) is the least-bad mechanism, but still fails to
  reach its own target (overlap plateaus 0.56–0.60 vs. a 0.7 target, scale railed at/near
  its ceiling — same signature as A/B's railed `reuse_adapt_scale_mean` and F18's railed
  REINFORCE multiplier) and still fails hopper outright.**
- **struct+ratchet vs. ratchet-only within C** (does the known-safe F18 struct lever
  still help once a direct reuse loss is applied?): no — it actively hurts cheetah.
  Ratchet-only beats struct+ratchet at both scales (e271 182 vs. e270 70 BIG; e273 111
  vs. e272 73 small), roughly halving cheetah's score when struct is stacked on. On
  hopper the direction flips (struct+ratchet e266/e268 > ratchet-only e267/e269) but both
  members of each pair are already dead, so this is noise near the floor, not a real
  benefit. **Verdict: struct is not a free stabilizer once a direct differentiable reuse
  loss already targets the same quantity — stacking two constraints on the same code
  compounds rather than helps, refining F18's "struct is safe" to "safe only as the sole
  lever."**

### 2026-07-16 · e266–e273 — `goal_soft_reuse_adapt`: direct (REINFORCE-free)
implicit sparsity via a block-overlap loss

**Motivation.** F18 root-caused the original `impl_sparsity_mode` controller's
collapse to having no differentiable representation of "kept" under plain
sampling — the only channel was a stop-gradiented reward penalty through
REINFORCE, the same mechanism Finding 1 already showed rails and kills reward.
`goal_delta_mode` fixed this by recombining the SAMPLE itself with the past
(votes added to the previous distribution before sampling). This is a third,
cheaper mechanism that leaves sampling completely untouched — plain/direct
prediction, mutually exclusive with `goal_delta_mode` — and instead: (1) feeds
the manager the previous decision's own soft distribution `p_{t-1}` as an input
(replacing the one-hot channel, same input-side trick `goal_delta_mode` uses),
and (2) adds an ordinary differentiable loss on the block-level OVERLAP between
this decision's plain softmax `p_t` and `p_{t-1}`:
`overlap = mean_blocks(sum_classes p_t * p_{t-1})`, bounded in [0,1] by
Cauchy-Schwarz (the probability an independent draw from each distribution
lands on the same class). No sampling, no decoder pass, no reward shaping —
backprops straight into both steps' own logits through their softmax. A
dual-ascent Lagrange multiplier (`goal_soft_reuse_adapter`, `inverse=True`,
same sense as `goal_reuse_adapter`) holds mean overlap AT a target, ratcheted
0→0.7 over ~1M steps exactly like the original `impl_sparsity_target` ramp.
Implementation: `agent.py` `goal_soft_reuse_adapt` (`_emit_manager`'s plain-
softmax branch, `_mgr_input`'s soft conditioning channel, the new
`preedit_eff['skill_probs']` threading through all 3 imagination-loss branches,
and the new loss block after `impl_sparsity_mode`'s); `configs.yaml`
`goal_soft_reuse_*`; `run_v3_prior_vargoal_{small,big_a100}.sbatch`
`GOAL_SOFT_REUSE_*` env-var flags. Smoke-tested (`run_smoke_goal_soft_reuse.sbatch`,
job 4664962, PASSED): 3 legs (mechanism off — regression check for the shared
`skill_probs` plumbing now also gating on this flag; on with a fixed target;
on with a fast-vel ratchet) all crash-free, `goal/soft_reuse_overlap_mean`
correctly bounded in [0,1], `goal/soft_reuse_target_now` ratchets 0.2→0.7
monotonically within the 1000-step smoke budget, `losses['goal_soft_reuse']`
matches the expected `-scale·overlap` value exactly (-12.52 at scale≈100,
overlap≈0.125), no NaN/Inf anywhere, and the off-leg carries zero `soft_reuse`
keys (clean gating).

| Exp | Job | Task | Scale | Condition |
|---|---|---|---|---|
| e266 | 4664964 | hopper | BIG | struct-adapt 0.01 + ratchet→0.7 |
| e267 | 4664965 | hopper | BIG | ratchet→0.7 only |
| e268 | 4664968 | hopper | small | struct-adapt 0.01 + ratchet→0.7 |
| e269 | 4664969 | hopper | small | ratchet→0.7 only |
| e270 | 4664966 | cheetah | BIG | struct-adapt 0.01 + ratchet→0.7 |
| e271 | 4664967 | cheetah | BIG | ratchet→0.7 only |
| e272 | 4664970 | cheetah | small | struct-adapt 0.01 + ratchet→0.7 |
| e273 | 4664971 | cheetah | small | ratchet→0.7 only |

All 8: `RECIPE=director`, `MGR_FREQ=8`, `MGR_COND_GOALCODE=True`,
`GOAL_SOFT_REUSE_ADAPT=True`, `GOAL_SOFT_REUSE_TARGET=0.7`,
`GOAL_SOFT_REUSE_TARGET_INIT=0.0` (ratchet from 0), `GOAL_SOFT_REUSE_TARGET_VEL`
= `1.1e-5` (BIG) / `2.9e-6` (small) — reusing the exact `impl_sparsity_target_vel`
calibration from e250–e257 (identical 0→0.7 distance, identical
`MGR_FREQ=8`/`RECIPE=director` train-call-to-env-step ratio per scale), `SEED=0`,
`RUN_STEPS=4000000` (full budget, not a 1M-step probe like e234–e249).

**Readout logic (pre-registered):**
→ If these cells train reasonably (comparable to the e234–e249/e258–e265
Director-class controls) while reaching the 0.7 overlap target: the direct
block-overlap loss is a working, REINFORCE-free sparsity lever — resolves F18
without needing `goal_delta_mode`'s sampling recombination at all.
→ If they collapse similarly to the old `impl_sparsity_mode` REINFORCE cells
(F18) despite having a fully differentiable gradient path: the failure was
never really about differentiability specifically, but about something else
common to all three "push kept-fraction toward 0.7" attempts (e.g. the target
itself being too aggressive relative to the Director-unprompted band of
0.32–0.45, Table `tab:implbaseline`) — would refocus blame from "no gradient"
onto "target too far from the natural operating point."
→ struct+ratchet vs. ratchet-only isolates whether the (known-safe, e234–e249)
indirect struct lever helps or is neutral/redundant once a direct differentiable
lever is already applied.

Single seed; not yet compared against `goal_reuse_adapt` (e258–e265, decoded-goal
space) or `goal_delta_mode` (e250–e257, cancelled) at the same target/budget —
those are the natural next A/Bs once this batch lands.

**Outcome (2026-07-21, all 8 COMPLETED at the full 4M-step budget).** All 8 trained
reasonably early (peaks 14–365 between 336k and 3.6M steps — see §2 results board) and
the overlap target ratchet completed on schedule, but **none reached the 0.7 target**:
`goal/soft_reuse_overlap_mean` plateaus at 0.56–0.60 across every cell regardless of
task/scale/struct, with the Lagrange multiplier (`goal_soft_reuse_scale_mean`) railed at
or near its 100.0 ceiling throughout — the same magnitude-domination signature as F18's
REINFORCE controller and as A/B's `goal_reuse_adapt_scale_mean`, just reached via a
genuinely differentiable path. **The second pre-registered branch fired**: this was
never about differentiability specifically. Task outcome is task-dependent, though: all
4 hopper cells collapse to the dead floor (0.004–1.75, vs. the ≈300 baseline) exactly
like every other reuse mechanism tried (F18, A, B) — hopper appears uniquely fragile to
*any* secondary manager-side objective, not specifically to non-differentiable ones. But
cheetah **survives substantially** (70–182, roughly 20–60% of the ≈300–435 baseline),
an order of magnitude better than families A/B's cheetah cells (1.85–7.57) — this is the
first reuse mechanism of the four tried (F18's REINFORCE controller, A, B, C) where
cheetah does not fully collapse. The **third branch (struct+ratchet vs. ratchet-only)**
also fired cleanly and unexpectedly: ratchet-only beats struct+ratchet on cheetah at
both scales (e271 182 vs. e270 70; e273 111 vs. e272 73) — struct is not neutral here,
it actively costs cheetah roughly half its score once a direct reuse loss already
targets the same code, refining F18's "struct is safe" to "safe only as the sole lever."
See Finding F19 (§3) for the cross-family synthesis.

### 2026-07-16 · e250–e257 — struct/ratchet retest under `goal_delta_mode`

**Motivation.** F18 found the `impl_sparsity_mode=reinforce` controller catastrophic
in every one of 12 tested cells (e235/e236/e237/e239/e240/e241/e243/e244/e245/e247/
e248/e249), root-caused (§7) to the controller being architecturally forced into a
non-differentiable, stop-gradiented reward-penalty mechanism — plain Director had no
way to express "keep this block" except by coincidence. `goal_delta_mode` (implemented
today, §7) fixes exactly that: the manager's votes combine additively with the
previous decision's real confidence-weighted distribution (`skill_probs`), giving
reuse an explicit, gradient-carrying channel, and the actor loss is re-derived under
the same combined distribution so the REINFORCE gradient itself is taken correctly.
This batch re-runs the same struct/ratchet cells from e234–e249 (target 0.7 kept,
ratchet 0→0.7, `struct_adapt_target=0.01`) with `goal_delta_mode=True` layered on top,
at the full 4M-step budget (e234–e249 were 1M-step probes) — the question is whether a
differentiable reuse pathway changes the outcome, or whether the REINFORCE-reward
mechanism itself is the problem regardless of what it acts on.

**Ratchet calibration fix.** e234–e249 used the same `impl_sparsity_target_vel=1.1e-5`
at both scales; checking the actual trajectories (this session) shows this was
correctly calibrated for BIG (ratchet completes at ≈1.02M env-steps, extrapolated from
the linear ramp) but **not** for small (completes at ≈262k steps — 4× too fast, never
spent most of training at the intended target-approach dynamics). Recalibrated small to
`impl_sparsity_target_vel=2.9e-6` (≈1.1e-5 × 262k/1M) so the ratchet takes the full ~1M
env-steps at both scales, matching the request and matching the earlier
`mask_sparsity_target_vel` calibration convention (2.8e-6 small / 1.12e-5 BIG for the
same 0.7-distance/1M-step ramp — the near-identical small value cross-checks the
derivation). BIG's vel is unchanged (already correct).

**Template changes.** Both `run_v3_prior_vargoal_{small,big_a100}.sbatch` gained
`GOAL_DELTA_MODE`/`GOAL_DELTA_CLIP` env-var-driven flags (default `False`/`1.0`,
no-op unless set), following the existing `IMPL_MODE`/`MGR_COND_GOALCODE` convention.

| Exp | Job | Task | Scale | Condition | struct_adapt_target | impl_target_vel |
|---|---|---|---|---|---|---|
| e250 | 4664853 | hopper | BIG | struct+ratchet+delta | 0.01 | 1.1e-5 |
| e251 | 4664854 | hopper | BIG | ratchet+delta | — | 1.1e-5 |
| e252 | 4664855 | hopper | small | struct+ratchet+delta | 0.01 | 2.9e-6 |
| e253 | 4664856 | hopper | small | ratchet+delta | — | 2.9e-6 |
| e254 | 4664857 | cheetah | BIG | struct+ratchet+delta | 0.01 | 1.1e-5 |
| e255 | 4664858 | cheetah | BIG | ratchet+delta | — | 1.1e-5 |
| e256 | 4664859 | cheetah | small | struct+ratchet+delta | 0.01 | 2.9e-6 |
| e257 | 4664860 | cheetah | small | ratchet+delta | — | 2.9e-6 |

All 8: `RECIPE=director`, `MGR_FREQ=8`, `IMPL_MODE=reinforce`, `IMPL_TARGET=0.7`,
`IMPL_TARGET_INIT=0.0`, `IMPL_ONE_SIDED=True`, `MGR_COND_GOALCODE=True`,
`GOAL_DELTA_MODE=True`, `GOAL_DELTA_CLIP=1.0`, `RUN_STEPS=4000000`, `SEED=0`.

**Readout logic (pre-registered):**
→ If these cells train reasonably (comparable to or above e234–e249's struct-only
cells — cheetah BIG ≈330, hopper BIG ≈115 at 1M steps, presumably higher by 4M) while
also reaching the 0.7 kept-target: `goal_delta_mode` resolves F18 — the REINFORCE
penalty was never inherently catastrophic, it only had nothing differentiable to act
on before.
→ If these cells still collapse to near-zero score similarly to e234–e249's
ratchet/struct+ratchet cells: the magnitude-domination failure is independent of
whether a differentiable pathway exists for the thing being penalized — the
REINFORCE-reward mechanism itself is the problem, and F18's fix needs to happen at the
controller/loss level (e.g. a differentiable loss reading `skill_probs` directly
instead of a stop-gradiented reward penalty), not just at the sampling-mechanism
level.
→ A split outcome (e.g. struct+ratchet cells collapse but ratchet-only cells don't, or
hopper differs from cheetah) would separate the struct term's own contribution from
the ratchet controller's, and is itself informative — not pre-empted by either branch
above.

Single seed; `goal_delta_mode` itself is smoke-tested only (not yet validated at scale)
going into this launch — treat as a combined validation of both goal_delta_mode at
scale AND the F18 follow-up question.

**Outcome: cancelled 2026-07-16 at ~3.5h (well short of the 4M-step budget), superseded
by e258–e265's `goal_reuse_adapt` direct/delta comparison, which targets the same
question with a cleaner, continuous-target design. Archived to
`/bucket/.../results/dreamerv3/` (`SKIP_REPLAY=1`); no readout — none of the
pre-registered branches above were reached.**

### 2026-07-16 · e258–e265 — `goal_reuse_adapt` (continuous goal-space target),
no struct, ratcheted 0→0.95 over ~1M steps

**Motivation.** Direct follow-up to F18/`goal_reuse`: code-level reuse (block class
identity, targeted by `goal_delta_mode` and the old `impl_sparsity_mode` controller) is
only a proxy for what the worker actually needs — a stable *decoded* goal
(`goal_reward_cosine_max(goal_deter, feat)` is computed entirely in continuous deter
space, never in code space). This batch drops the code-level reuse controller
(`IMPL_MODE=none`) and the struct correlation term entirely (`STRUCT_ADAPT=False`,
`STRUCT_W=0.0` — explicitly not needed if the thing we actually care about, the decoded
goal, is targeted directly) in favor of `goal_reuse_adapt`, a dual-ascent Lagrange
multiplier holding mean decoded-goal similarity at a target — see §7 for the mechanism
(gradient into the manager only, decoder gradient blocked; verified in isolation and via
matched `opt/goal_grad_norm` in the full pipeline).

**Target ratcheted, not fixed from step 0.** The first submission of this batch (jobs
4664890–4664897) set `goal_reuse_target=0.95` fixed from step 0 and was cancelled within
~2 minutes (no meaningful training elapsed, run dirs deleted, not archived) before
relaunch: correctly mirrors `impl_sparsity_target`'s own ratchet — start at
`goal_reuse_target_init=0.0` (no pressure; realized similarity at random init is
~0.2–0.3, already above 0.0, so the adapter's scale starts at its floor) and ramp to the
final `goal_reuse_target=0.95` over ~1M env-steps, giving the manager time to establish
basic task competence before the similarity constraint tightens. `goal_reuse_target_vel`
recalibrated for the larger distance (0.95 vs. the impl_sparsity ratchet's 0.7) from the
same per-scale env-steps-per-train-call relationship: `1.5e-5` (BIG) / `3.9e-6` (small) —
scaled directly from the already-validated `1.1e-5`/`2.9e-6` (§ e250–e257) by the
distance ratio `0.95/0.7`. Verified via a fast-vel smoke leg
(`run_smoke_goal_reuse_ratchet.sbatch`, job 4664903, PASSED): `goal/reuse_target_now`
ramped `0.175 → 0.950` monotonically within a 1000-step run, correctly capped at 0.95.

| Exp | Job | Task | Scale | Mechanism |
|---|---|---|---|---|
| e258 | 4664904 | hopper | BIG | direct (plain Director) |
| e259 | 4664905 | hopper | BIG | delta (`goal_delta_mode`) |
| e260 | 4664906 | hopper | small | direct |
| e261 | 4664907 | hopper | small | delta |
| e262 | 4664908 | cheetah | BIG | direct |
| e263 | 4664909 | cheetah | BIG | delta |
| e264 | 4664910 | cheetah | small | direct |
| e265 | 4664911 | cheetah | small | delta |

All 8: `RECIPE=director`, `MGR_FREQ=8`, `STRUCT_ADAPT=False`, `STRUCT_W=0.0`,
`IMPL_MODE=none`, `GOAL_REUSE_ADAPT=True`, `GOAL_REUSE_TARGET=0.95`,
`GOAL_REUSE_TARGET_INIT=0.0`, `MGR_COND_GOALCODE=True`, `RUN_STEPS=4000000`, `SEED=0`;
`GOAL_DELTA_MODE={False,True}` per mechanism column, `GOAL_DELTA_CLIP=1.0` where
applicable.

**Readout logic (pre-registered):**
→ If "direct" cells (e258/e260/e262/e264) reach high similarity (~0.95) at reasonable
task return, comparable to their `goal_delta_mode` counterparts: the differentiable
goal-space loss alone is sufficient, and `goal_delta_mode`'s code-level machinery is
unnecessary for reuse specifically (though it may still matter for other reasons, e.g.
sample efficiency of reaching a given code).
→ If "delta" cells clearly outperform "direct" cells (higher achieved similarity at the
same task-return cost, or the same similarity at lower cost): the code-level
differentiable-reuse pathway (`goal_delta_mode`) and the goal-space loss are
complementary, not redundant — delta mode gives the manager an easier gradient path to
actually *reach* high code-level reuse, which the goal-space loss alone has to discover
through the straight-through gradient of a plain categorical.
→ Watch `goal/reuse_adapt_scale_mean` for railing at its ceiling (100.0, matching the
smoke test's random-init behavior) persisting long after the target ratchet completes
(~1M steps) — that would mean 0.95 is unreachable at reasonable task-return cost, the
same magnitude-domination shape as F18, just via a genuinely differentiable path this
time (gradient magnitude domination in the combined weighted loss sum, not
reward-poisoning) rather than a structurally different failure mode.

Single seed; combined validation of `goal_reuse_adapt` at scale (smoke-tested only going
into this launch) and the "is code-level reuse still needed once goal-space reuse is
targeted directly" question.

**Delta cells cancelled 2026-07-16 at ~3h10m** (e259 job 4664905, e261 job 4664907, e263
job 4664909, e265 job 4664911) — no readout taken (short of the 4M-step budget);
superseded by e274–e277 below, a manager-input ablation on the surviving "direct" cells
rather than a delta-mode comparison.

**Outcome (2026-07-21, all 4 "direct" cells COMPLETED at the full 4M-step budget).** The
target was reached — `goal/reuse_sim_mean` sits at 0.92–0.97 against the 0.95 target in
all 4 cells, close enough that the third pre-registered watch-for ("scale railed at
ceiling persisting long after ratchet completes") is the operative reading: 3 of 4 cells
(e260 hopper-small, e262 cheetah-BIG, e264 cheetah-small) have `reuse_adapt_scale_mean`
pinned at its 100.0 ceiling; e258 (hopper-BIG) sits at 67.6, not fully railed but still
very high. **All 4 cells collapse in task score regardless**: hopper 0.001–0.002 (vs.
≈300 baseline — total collapse) and cheetah 1.85 (BIG) / 3.47 (small) (vs. ≈300–435 /
≈100 baselines — collapse to 1–4% of baseline). Trajectories (see §2 Peak column) show
this is not a failure to ever learn: every cell rises to a real peak early (51–203,
reached between 176k and 945k steps, i.e. before or right around the ~1M-step point the
similarity ratchet finishes tightening) and then **collapses to the floor and stays
there for the remaining 3M steps** — the same shape as F18's REINFORCE cells and the
duration-reg cliff (§7): a soft prior that is fine while loose becomes catastrophic once
the target fully engages. This is the "reach 0.95 unreachable at reasonable task-return
cost" branch, confirmed, via a genuinely differentiable gradient path this time (no
REINFORCE) — direct falsification of the batch's central hypothesis that a continuous
goal-space loss alone (no code-level mechanism) would be a cheap, safe reuse lever.
Since the `goal_delta_mode` "delta" counterparts (e259/e261/e263/e265) were cancelled
without a readout, whether code-level recombination would have changed this outcome is
untested — see F19 (§3).

### 2026-07-16 · e274–e277 — `goal_reuse_adapt`, manager sees decoded goal only (no raw code)

**Motivation.** e258/e260/e262/e264 (kept running) already pass BOTH the raw one-hot
running code (`mgr_cond_goalcode=True`, 64-dim) and the decoded previous goal
(`mgr_cond_decgoal`, forced on by `goal_reuse_adapt=True`) to the manager — see
`_mgr_input` (`agent.py:1379`). This makes it hard to attribute reuse behavior: the
manager could be keying off the exact discrete code (a trivial identity/copy channel)
rather than actually using the continuous goal-space signal the `goal_reuse_adapt` loss
targets. e274–e277 replace those 4 BIG/small direct cells with an otherwise-identical
config but `MGR_COND_GOALCODE=False`, so the manager's input is *only*
`feat2tensor(feat)` (world-model deter+stoch) concatenated with the decoded previous
goal (`goal_dec` output, stop-gradiented) — no raw code channel at all. Everything else
(RECIPE=director fixed-K8 plain goal-blocks, `GOAL_DELTA_MODE=False`,
`GOAL_REUSE_TARGET=0.95` ratcheted from 0.0, vel/struct/impl settings, batch/imag/
train_ratio/envs/steps/seed/hardware) is identical to e258/e260/e262/e264.

| Exp | Job | Task | Scale | Manager input |
|---|---|---|---|---|
| e274 | 4664976 | hopper | BIG | decoded goal + world state (no code) |
| e275 | 4664978 | hopper | small | decoded goal + world state (no code) |
| e276 | 4664977 | cheetah | BIG | decoded goal + world state (no code) |
| e277 | 4664979 | cheetah | small | decoded goal + world state (no code) |

All 4: `RECIPE=director`, `MGR_FREQ=8`, `STRUCT_ADAPT=False`, `STRUCT_W=0.0`,
`IMPL_MODE=none`, `GOAL_DELTA_MODE=False`, `GOAL_REUSE_ADAPT=True`,
`GOAL_REUSE_TARGET=0.95`, `GOAL_REUSE_TARGET_INIT=0.0`, `GOAL_REUSE_TARGET_VEL=1.5e-5`
(BIG) / `3.9e-6` (small), **`MGR_COND_GOALCODE=False`**, `RUN_STEPS=4000000`, `SEED=0`.

**Readout logic (pre-registered):** compare against e258/e260/e262/e264 (same recipe,
`MGR_COND_GOALCODE=True`) on task return and achieved `goal/reuse_sim_mean` at the
`goal_reuse_target` ratchet. If e274–e277 reach comparable similarity/return to their
code-conditioned counterparts, the raw discrete code channel is not load-bearing for
reuse — the decoded-goal signal alone is sufficient for the manager to satisfy the
continuous target. If they lag noticeably (lower achieved similarity, or collapsed
return), the manager needs the exact discrete code (not just its lossy decoded
projection) to reliably reproduce a block.

**Outcome (2026-07-21, all 4 COMPLETED at the full 4M-step budget).** The **first
pre-registered branch fired**: e274–e277 reach comparable (if anything marginally
higher) similarity to their code-conditioned counterparts — `reuse_sim_mean` 0.92–0.95
vs. e258/e260/e262/e264's 0.92–0.97 — and comparable task collapse. The raw discrete
code channel is not load-bearing: hopper is dead either way (e274 0.004 / e275 0.001 vs.
e258 0.001 / e260 0.002 — no meaningful difference, both total collapse vs. ≈300 and
≈1(dead) baselines) and cheetah is dead either way, though B is consistently a bit less
dead than A (e276 7.24 vs. e262 1.85 BIG; e277 7.57 vs. e264 3.47 small — roughly 2–4×
higher, but still 2–15% of the ≈300–435/≈100 baselines, i.e. still collapsed by any
task-relevant standard). Peaks tell the same story: B's hopper-BIG cell (e274) peaks
notably higher than A's (138 @504k vs. 51 @480k), and B's other 3 cells peak within noise
of A's, before both collapse to the floor by ~1M steps. **Conclusion:** the manager does
not need the exact discrete code as input to satisfy a continuous similarity target —
but this ablation ends up moot, since the mechanism it's isolating (`goal_reuse_adapt`)
collapses task learning regardless of what conditions the manager. See F19 (§3).

### 2026-07-15 · Pivot to *implicit* sparsity (masking abandoned) + e212–e233 cancelled

**Design change.** Explicit block-masking/abstain is dropped. Rationale: masking was
only ever a *proxy* for "keep part of the goal unchanged", and the discarded-content /
abstain machinery hurts credit assignment. Instead the manager **regenerates the whole
goal code each decision** (plain Director), and we simply **measure** how much it chose to
keep — *implicit* (effective) sparsity — and add incentives to raise it:
- New metrics (all HRL runs, incl. plain Director), `agent.py` imagine path:
  `goal/implicit_sparsity_block` = fraction of goal-code blocks whose class is identical to
  the previous goal (1 = regenerated identical, 0 = all-new); `goal/implicit_sparsity_cont`
  = decoded-goal cosine_max to the previous goal, mapped to [0,1]. Measured at
  manager-decision steps only.
- `mask_viz` yellow overlay now marks blocks **changed vs the previous goal** (new
  `last_change_mask` carry), not the edit selector.
- New **implicit-sparsity controller** (`agent.impl_sparsity_mode=reinforce`,
  `impl_sparsity_target/_init/_vel/_one_sided`): a mask-free per-decision REINFORCE cost
  `λ·change_frac` (λ dual-ascent toward the annealed kept-target) drives measured implicit
  block sparsity to a target. `direct` = fixed target; `ratchet` = anneal 0→target,
  one-sided λ.
- `mgr_cond_goalcode` **ungated from masked goals**: plain Director now conditions the
  manager on the previous goal code so it can *deliberately* reuse blocks (else keeping is
  incidental). All e234+ runs use it.

**e212–e233 (18 runs) cancelled + archived 2026-07-15** (single-head mask campaign,
superseded by the implicit-sparsity design): e212 cheetah-director-small, e213
hopper-director-small, e214–e221 small joint sweep, e222/e223/e226/e227 BIG joint,
e230–e233 BIG ratchet retest. All rsynced to bucket (`SKIP_REPLAY=1`) then deleted from
`/work`.

**Director baseline implicit sparsity (measured 2026-07-15** at final checkpoints, new
code, `impl_sparsity_block` = kept-block fraction, `_cont` = decoded-goal cosine):

| Baseline | scale | kept (block) | cont |
|---|---|---|---|
| e212 cheetah (Director) | small | 0.445 | 0.971 |
| e213 hopper  (Director) | small | 0.414 | 0.887 |
| e123 cheetah (Director) | BIG   | 0.316 | 0.980 |
| e124 hopper  (Director) | BIG   | 0.367 | 0.849 |
| e125 acrobot (Director) | BIG   | 0.395 | 0.770 |
| e115 cartpole | BIG | **N/A** | — |

A *trained* Director already keeps ~0.32–0.45 of blocks per decision unprompted (vs ~1/8
random early); small keeps more than BIG. e115 is `director_match masked_goals variable`
(not a clean Director) **and** its pre-redesign masked checkpoint no longer shape-matches
current code (chex shape assert on load) — not measured.

**e234–e249, launched 2026-07-15, COMPLETED 07-15/16** (jobs 4664279–4664294) — the
implicit-sparsity matrix: pure Director copy (`RECIPE=director`, `MGR_FREQ=8`,
`mgr_cond_goalcode=True`, 1M steps) × {hopper, cheetah} × {small (v100) e234–e241, BIG
(a100) e242–e249} × 4 conditions: **struct** (struct-adapt target 0.01, no impl
controller), **struct+ratchet**, **ratchet** (impl controller ratchet kept 0→0.7 over the
run, one-sided), **direct** (impl controller fixed kept-target 0.7 from step 0). **Target
0.7 kept = 0.3 change** (chosen above the ~0.32–0.45 baseline so the controller actively
*raises* sparsity). Pre-registered hypothesis: struct raises implicit sparsity indirectly
(temporal code correlation) while the ratchet/direct REINFORCE controller raises it
directly; question was which reaches ~0.7 kept at least task-return cost.

**Result: the question as posed doesn't arise — struct is the only lever that doesn't
break the task.** Full-budget (1M step) numbers, `train/goal/*` tail-5 averages at the
final checkpoint:

| exp | env | scale | condition | last15 (peak15) | kept `s` final | `s` target | λ (impl_sparsity_scale) | struct_loss |
|---|---|---|---|---|---|---|---|---|
| e234 | hopper | small | struct | 1.9 (46.5) | 0.357 | n/a | n/a | 0.0103 |
| e235 | hopper | small | struct+ratchet | 0.0 (0.7) | 0.563 | 0.70 | **5.00 (railed)** | 0.0104 |
| e236 | hopper | small | ratchet | 0.0 (0.6) | 0.560 | 0.70 | **5.00 (railed)** | — |
| e237 | hopper | small | direct | 0.0 (0.5) | 0.562 | 0.70 | **5.00 (railed)** | — |
| e238 | cheetah | small | struct | 117.8 (159.0) | 0.259 | n/a | n/a | 0.0103 |
| e239 | cheetah | small | struct+ratchet | 3.0 (5.0) | 0.561 | 0.70 | **5.00 (railed)** | 0.0102 |
| e240 | cheetah | small | ratchet | 2.2 (4.6) | 0.510 | 0.70 | **5.00 (railed)** | — |
| e241 | cheetah | small | direct | 1.4 (4.2) | 0.563 | 0.70 | **5.00 (railed)** | — |
| e242 | hopper | BIG | struct | **115.1 (115.7)** | 0.379 | n/a | n/a | 0.0100 |
| e243 | hopper | BIG | struct+ratchet | 0.0 (0.2) | 0.530 | 0.68 | **5.00 (railed)** | 0.0101 |
| e244 | hopper | BIG | ratchet | 0.0 (1.0) | 0.485 | 0.68 | **5.00 (railed)** | — |
| e245 | hopper | BIG | direct | 0.0 (0.6) | 0.510 | 0.70 | **5.00 (railed)** | — |
| e246 | cheetah | BIG | struct | **329.8 (335.4)** | 0.195 | n/a | n/a | 0.0103 |
| e247 | cheetah | BIG | struct+ratchet | 3.4 (6.1) | 0.515 | 0.68 | **5.00 (railed)** | 0.0101 |
| e248 | cheetah | BIG | ratchet | 4.1 (8.6) | 0.534 | 0.68 | **5.00 (railed)** | — |
| e249 | cheetah | BIG | direct | 3.4 (7.9) | 0.551 | 0.70 | **5.00 (railed)** | — |

**Per-cell readout (H: as stated above for the whole matrix):**
- **e234/e238/e242/e246 (struct, no controller) — H partially confirmed, reframed.** Task
  return survives in all 4 cells: cheetah BIG (e246, 330/335) lands within noise of the
  contemporaneous pure-Director-at-1M-steps comparators (e123 runs read 250–305 at the same
  step count), i.e. near-zero cost; cheetah small (e238, 118/159) and hopper BIG (e242,
  115/116, still climbing at the 1M cutoff) are clearly alive but well below their
  Director-at-1M references (e123-class ≈250–305, e124 ≈295 at 1M though e124 itself later
  *declined* to 196 by its own final checkpoint) — a real, partial cost, not a free lunch.
  Hopper small (e234, 1.9/46.5) stays at noise floor, consistent with F15 (hopper never
  learns at small scale under any recipe) and uninformative about struct specifically. The
  reframe: struct's own *measured* implicit sparsity (`kept` 0.195–0.379) is **not** higher
  than the unprompted Director baselines (0.32–0.45, §2 table above) — on 3 of 4 cells it is
  *lower* — so on the discrete block-identity metric struct does not clearly raise reuse at
  all (caveat: those baselines were measured on checkpoints trained without
  `mgr_cond_goalcode`, so the comparison is suggestive, not a controlled ablation). What
  struct reliably buys is task-safety, not demonstrated sparsity.
- **e235–e237, e239–e241, e243–e245, e247–e249 (any cell with the REINFORCE controller —
  12/12) — H refuted, decisively.** Every controller-bearing cell collapses to a score of
  0.0–8.6 (vs. struct-only's 1.9–330 in the matched cell), regardless of task, scale, or
  whether struct is stacked on top. The controller does not even reach its own objective:
  `impl_sparsity_scale_mean` (the dual multiplier λ) rails to its ceiling (5.0) in every one
  of the 12 cells, typically within the first 30–50% of training, while the realized change
  fraction plateaus at 0.44–0.52 — well short of the 0.3 implied by the 0.7 kept-target.
  This is the same **magnitude-domination/railing** failure mode already seen twice before
  in this project (the duration-reg cliff, §7; one-sided edit/switch costs, F2) — a per-step
  REINFORCE cost on `(1 − kept_frac)` is too strong relative to the task-return gradient at
  any of the weights explored, and dual ascent's only response to being short of target is
  to keep growing the multiplier, which does not help once other manager gradients are
  already drowned out. struct vs. no-struct makes no difference once the controller is
  active (struct+ratchet ≈ ratchet ≈ direct in every cell) — struct's stabilizing role
  established elsewhere (F11) does not rescue a controller-induced collapse.

**Implication for the project.** The pivot's original question — "does implicit sparsity
cost anything, and can we raise it" — is answered asymmetrically: *measuring* it is free
and informative (kept the metrics), *nudging* it via struct is safe but unproven as a
sparsity lever, and *forcing* it via REINFORCE is destructive at every setpoint/task/scale
tried so far, mirroring F2/F17's ratchet result under the old single-head design. No further
implicit-sparsity-controller cells are planned pending a much weaker controller weight or a
different mechanism (e.g. clipping λ well below 5.0, or switching to a KL-to-prior penalty
instead of REINFORCE). struct-only is now the safe default for future sparsity-adjacent
runs; the controller (`agent.impl_sparsity_mode`) should stay off by default.

**Dense tasks: solved at Director-comparable cost, and countdown found a second,
simpler route there.** Best cartpole: combined recipe BIG → 833/869 (e157), reproduced
post-fixes at 776/844 (e171), at blk/step 0.6–1.1. Best small: e57 (masked var-K + soft
prior τ_d=4) → best-15 839 / max 854. Best cheetah: combined (Lagrangian controllers)
465/508 (e175) ≈ 2× the fixed-penalty version. New this campaign: plain var-K τ4 (no
masking, no struct) + worker countdown conditioning turns e166's collapsed 130 into a
stable 697/757 (e187) — countdown alone rescues the exact failure struct used to be
needed for, at small scale. Director-matched controls for both dense tasks are running
(e190 714/747 @63%, still climbing; e191 non-monotonic, peak ~500) to pin the denominators.

**Sparse tasks: no longer a uniform dead end — the single-head design breaks the pattern.**
Every dual-head (skill + mask) restricted variant ever run died on hopper/acrobot (20+
cells, F12). The single-head redesign (`agent.mask_joint_edit`: one C+1-way categorical per
block, class 0 = abstain — removes the discarded-content REINFORCE confound by
construction) changes this: **e226** (single-head, struct off, ratchet off, BIG, fully
free) is alive on hopper — reward_rate 0.245 (vs. the 1e-2 "alive" bar and vs. e178's dead
0–5e-3 pulse under the old design), score climbing to a 280–286 last-15 plateau by 1.6–2.1M
steps, ahead of even the pure-Director reference (e124, 196/326). It does this without
actually editing sparsely: `goal/mask_frac_mean` sits at 0.94–0.97 throughout training — left
free, the manager never chooses to abstain much; the win is from the single-head mechanism,
not from achieving sparsity. Forcing sparsity costs performance in every cell tested: struct
(e227, single-head + struct) nearly halves the score (141/165) but — unlike e189 under the
old design — no longer kills it outright; the mask-sparsity-target ratchet (e228, forces
abstain-rate up to a 0.3 non-abstain target by ~1.1M steps) **refutes** the exploration-
starvation hypothesis it was built to test — score stays flat 1–20 the whole run, well below
the free variant, and struct+ratchet combined (e229) produces an outright collapse (208 peak
@500k → <10 for the remaining 2M+ steps). The pattern replicates on cheetah (dense-hard):
free/struct-on single-head cells (e222 434 peak, e223 455 peak) beat Director-comparable
levels, ratchet-on cells plateau 2–3× lower (e224 143, e225 201) — the ratchet is not
dense-regression-neutral either. The rescue is **BIG-scale-only so far**: at small scale,
hopper stays dead under every single-head cell (e218–e221, all rise-then-decay or flat,
matching the small pure-Director control e213's own floor) — consistent with F15. Acrobot
was not re-run under single-head this campaign; still open. Full per-cell numbers: single-head
campaign table below; supersedes F12, see new finding F17 (§3).

**New mechanism, 07-14: mask-sparsity-target ratchet (e204–e211).** Motivating question:
under the mask, is the manager's exploration-reward advantage (`_mgr_expl_reward`, dense
goal-deter reconstruction error, block-pooled exactly like the extrinsic reward) scoped to
the *edited* blocks, or is it one scalar over the whole resulting goal applied uniformly to
every manager head? Confirmed the latter — with a sparse mask (target 0.3, 2–3 of 8 blocks
edited/decision), the manager gets the same-magnitude exploration credit whether it edited 1
block or 8, with no per-block attribution telling the mask head *which* edit caused the
novelty. Hypothesis: this starves goal-space exploration early in training, before the
manager has any signal about which blocks are worth editing — compounding on top of F12
(masking alone already kills hopper/acrobot at fixed K8, e164/e165). Fix (this session):
`agent.Ratchet` (`embodied/jax/utils.py`) — a deterministic, velocity-capped, open-loop
ramp of the `mask_sparsity_adapter`'s **target** itself (not just its Lagrange weight, which
was already adaptive) from `mask_sparsity_target_init=1.0` (edit the whole goal — full
goal-space exploration, mask has no effect early on) down to `mask_sparsity_target=0.3` at a
rate tuned to ~1M env-steps (`mask_sparsity_target_vel`: 2.8e-6/train-call small-scale,
1.12e-5/train-call BIG-scale — both scales' batch/train_ratio give different env-steps per
call, so the raw `vel` differs but the wall-clock/env-step ramp length matches). Fully
opt-in (`target_init == target` by default → no-op, byte-identical to prior runs unless set).
Smoke-tested (`run_smoke_mask_ratchet.sbatch`, job 4662316): ratchet observed 0.8→0.3
monotonic + correctly clipped, full combined-recipe smoke leg (prob_entropy + var-K
Lagrangian + struct-adapt + countdown + ratchet together) trains without error. e204–e211
(full stack + ratchet on the old dual-head design) were cancelled+archived 07-14 before
producing results, to free hardware for the single-head campaign — the ratchet hypothesis
was tested there instead, as the "R1" cells of e212–e229's 2×2 (§2 single-head table). Verdict
(2026-07-15): **refuted**. `mask_frac_mean` confirms the ratchet ramps correctly (0.89→0.30 by
~1.1M steps on both e224/cheetah and e228/hopper) but every ratchet-on cell underperforms its
free (R0) counterpart — e228 (22/52) vs e226 (214/302) on hopper, e224 (134/143) vs e222
(181/434) on cheetah — and stacking it with struct causes outright collapse (e229: 208 peak
@500k → <10 for the rest of training). The exploration-starvation mechanism the ratchet was
designed to fix does not appear to be what was killing sparse tasks under the old design; see
F17.

### Best known configs

```bash
# Combined recipe, BIG (e157/e171-class) — via template:
sbatch -J eNNN_<task>_BIG --export=ALL,EXP_TAG=eNNN,TASK=dmc_<task>,RECIPE=vark_masked,\
DUR_TARGET=4.0,MASK_MODE=prob_entropy,DUR_MODE=lagrangian run_v3_prior_vargoal_big_a100.sbatch
# e57 small-scale champion: same via run_v3_prior_vargoal_small.sbatch with MASK_MODE=prob,
# DUR_MODE=fixed (reg 0.01), DUR_TARGET=4.0.
# Optional new knobs (defaults preserve old behavior): WORKER_TIMED_GOALS=True,
# STRUCT_ADAPT=True, STRUCT_ADAPT_TARGET=0.006 (default 0.005; big_a100 script gained
# both flags 07-14 -- previously only run_v3_prior_vargoal_small.sbatch had them),
# MGR_REWARD_AGG={mean,sum}, MGR_EXPL_W, RECIPE={vark_masked,mask_fixedk,plain_vark}.
```

### Live board (31 rows, 19 running: 4 cancelled 07-14 and replaced by e200–e203; 8 more
cancelled 07-14 (checkpoints resumable) and replaced by e204–e211)

| Exp | Job | Recipe | Task | Scale | Question | Signal as of 2026-07-14 (interim, % of 4M budget) |
|---|---|---|---|---|---|---|
| e160 | 4660123 | combined, struct=0 | cartpole | BIG short-a100 | struct needed inside full recipe? | COLLAPSED: 138→22 @2.19M (55%); mask_frac drifted 0.30→0.51, mgr_extr_adv≈0 |
| e178 | 4660138 | plain var-K τ8 | hopper | BIG | hold length alone | flat 0–5e-3 through 3.2M (80%), score ≈0 — the "rising" read at 0.6M did not hold. **CANCELLED 07-14** to free A100 for e202 |
| e179 | 4660139 | plain var-K τ8 | cartpole | BIG | stability without struct at τ8 | 740/755 @3.25M (81%), stable since 1.8M — struct not needed at τ8. **CANCELLED 07-14** (checkpoint kept, resumable) to free A100 for e205 |
| e180 | 4660140 | pure Director | acrobot | BIG | task-difficulty control | ALIVE: rate 0→0.19, score 176/262 @3.34M (83%), still climbing. **CANCELLED 07-14** (checkpoint kept, resumable) to free A100 for e211 |
| e185 | 4660298 | e178 + countdown | hopper | BIG | horizon observability | dead, ≈e178 (wkr 0.21 vs e178's 0.16); dur_std blew up to 6.6. **CANCELLED 07-14** to free A100 for e202 |
| e186 | 4660299 | combined + countdown | cartpole | small | countdown inside best recipe (vs e170: 668/786) | 695/766 @2.08M (52%) ≈ e170; wkr_goal_rew 0.57, above e170's level. **CANCELLED 07-14** (checkpoint kept, resumable) to free V100 for e204 |
| e187 | 4660300 | plain var-K τ4 + countdown | cartpole | small | sharpest worker-confusion test (vs e166: 130) | 697/757 @2.11M (53%) — 5.4× e166, stable since 0.7M. **CANCELLED 07-14** (checkpoint kept, resumable) to free V100 for e206 |
| e188 | 4660301 | mask-only K8 + countdown | cartpole | small | phase observability at fixed K (vs e162: 272/575) | 254/347 @2.13M (53%), below e162's 565 peak so far. **CANCELLED 07-14** (checkpoint kept, resumable) to free V100 for e208 |
| e189 | 4660308 | plain var-K τ8 + struct200 | hopper | BIG | struct: stabilizer or locality killer? | dead, weaker than e178; struct_corr 0.94 — locality face confirmed. **CANCELLED 07-14** to free A100 for e202 |
| e190 | 4660309 | pure Director | cartpole | BIG | Director-matched control | 714/747 @2.52M (63%), still climbing toward e171's 776. **CANCELLED 07-14** (checkpoint kept, resumable) to free A100 for e207 |
| e191 | 4660310 | pure Director | cheetah | BIG | control redo (e123 stopped 2.6M, peak 425) | non-monotonic: 471 peak@1.4M → dip 225@2.27M → 293/499 @2.55M (64%). **CANCELLED 07-14** (checkpoint kept, resumable) to free A100 for e209 |
| e192 | 4660311 | plain var-K τ8 + countdown | acrobot | BIG | best-guess sparse recipe vs e180 | dead (no trend) while e180 climbs — acrobot ≠ hopper failure mode. **CANCELLED 07-14** to free A100 for e203 |
| e193 | 4660312 | plain var-K τ4 + struct200 | cartpole | small | duration×struct 2×2 (with e46/e166/e194) | 42/203 @2.09M (52%) — peaked early (240k), stuck low since. **CANCELLED 07-14** (checkpoint kept, resumable) to free V100 for e210 |
| e194 | 4660313 | plain var-K τ8, struct0 | cartpole | small | 2×2; small mirror of e179 | 152/285 @2.11M (53%) — no collapse, no climb either |
| e195 | 4660317 | plain var-K τ4 + struct-adapt | cartpole | small | struct-as-Lagrangian pilot | 175/352 @2.09M (52%) — beats e193, still ≪ e46's 724 |
| e196 | 4660324 | **full stack**: combined+countdown+struct-adapt | cartpole | small | everything-adaptive cell vs e170/e186 | 648/683 @2.01M (50%) ≈ e170/e186, dipped-and-recovered at 1.1M |
| e197 | 4660325 | full stack | hopper | small | — (small hopper never learned; long shot) | dead, 0.23/3.1 @2.0M (50%) — expected long shot |
| e198 | 4660326 | full stack | cheetah | small (P100) | vs e174 (166) | 100/122 @2.31M (58%) — below e174, flat since 250k |
| e199 | 4660327 | full stack | acrobot | small (P100) | vs e176 (0) | 3.4/22 @2.36M (59%) — still dead vs e176 |
| e200 | 4662288 | full stack (struct-adapt target 0.006) | cartpole | BIG | scale e196 cell to BIG/τ8 with a lower, never-tried struct-adapt target | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e201 | 4662289 | full stack (struct-adapt target 0.006) | cheetah | BIG | scale e198 cell to BIG/τ8 | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e202 | 4662290 | full stack (struct-adapt target 0.006) | hopper | BIG | replaces e178/e185/e189 — last untried sparse-task cell (struct-adapt+countdown+τ8 together) | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e203 | 4662291 | full stack (struct-adapt target 0.006) | acrobot | BIG | replaces e192 — same combined cell for acrobot | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e204 | 4662318 | full stack + mask ratchet (init 1.0→0.3, vel 2.8e-6/call) | cartpole | small | dense regression check: ratchet should be return-neutral | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e205 | 4662322 | full stack + mask ratchet (init 1.0→0.3, vel 1.12e-5/call) | cartpole | BIG | dense regression check vs e200 | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e206 | 4662319 | full stack + mask ratchet | cheetah | small | dense regression check | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e207 | 4662323 | full stack + mask ratchet | cheetah | BIG | dense regression check vs e201 | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e208 | 4662320 | full stack + mask ratchet | hopper | small | exploration-starvation test (never learned at small scale under any recipe) | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e209 | 4662324 | full stack + mask ratchet | hopper | BIG | **key test**: does full-goal-edit early rescue hopper vs e202 (dead)? | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e210 | 4662321 | full stack + mask ratchet | acrobot | small | exploration-starvation test | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |
| e211 | 4662325 | full stack + mask ratchet | acrobot | BIG | **key test**: does full-goal-edit early rescue acrobot vs e203 (dead)? | **CANCELLED+archived 07-14** (freed for e212+ single-head campaign) |

### Single-head mask campaign (e212–e229, launched 07-14)

New component test of the **single-head ("joint") masked manager** (`agent.mask_joint_edit`:
one categorical per block over C+1 classes, class 0 = abstain, 1..C = overwrite — removes the
dual-head discarded-content REINFORCE confound by construction). Focused on the two decisive
envs (**cheetah** dense-hard, **hopper** sparse-hard) at two scales. Each single-head run starts
from the **pure Director base** (fixed manager_sample_freq=8, no var-K/durreg) + `mask_joint_edit`,
then sweeps a 2×2: **struct** {off | adaptive dual-MSE, target 0.006} × **mask ratchet** {off
(fully free) | on: P(abstain) annealed 0→0.7 over ~1M steps via the rate-Lagrange + target
Ratchet on the non-abstain probability}. Small = size6m/v100·p100; BIG = director_match/a100.
BIG Director baselines reused: cheetah **e191**, hopper **e124**. Cell code Sx Ry = struct x /
ratchet y.

| exp | job | recipe | env | size | hypothesis / expected | status (2026-07-15 interim, last15/peak15, % of 4M) |
|---|---|---|---|---|---|---|
| e212 | 4662391 | pure Director (fixed-K) baseline | cheetah | small | small Director control for the campaign | 100/111 @71% |
| e213 | 4662392 | pure Director (fixed-K) baseline | hopper | small | small Director control | 1/40 @71% — small hopper dead even under the unrestricted base |
| e214 | 4662393 | **single-head** S0R0 (struct off, ratchet off — fully free) | cheetah | small | minimal single-head vs Director e212: clean per-block credit ⇒ ≥ baseline | 111/157 @54%, rising — **H confirmed**, ≥ e212 with less training |
| e215 | 4662394 | single-head S1R0 (struct-adapt 0.006, ratchet off) | cheetah | small | +struct locality on top of single-head | 174/194 @54%, still rising — best small-cheetah cell |
| e216 | 4662395 | single-head S0R1 (struct off, ratchet on) | cheetah | small | +ratchet: full-edit early → sparse; dense regression-neutral | 145/238 @54%, early spike (181) then decays to ~145 — **H refuted**, not regression-neutral |
| e217 | 4662396 | single-head S1R1 (struct-adapt 0.006 + ratchet on) | cheetah | small | struct×ratchet combined | 50/67 @54% — weakest cheetah cell at any scale |
| e218 | 4662397 | single-head S0R0 | hopper | small | single-head on sparse-hard small (long shot) | 1/125 @54%, rises to 83 @545k then decays to ~1 — long shot fails, mirrors e213's floor |
| e219 | 4662398 | single-head S1R0 | hopper | small | +struct | 7/117 @54%, same rise-then-decay shape as e218 |
| e220 | 4662399 | single-head S0R1 | hopper | small | +ratchet: does full-goal-edit early aid sparse exploration? | 24/81 @64%, low flat trace, no transient bump — **H refuted** |
| e221 | 4662400 | single-head S1R1 | hopper | small | struct×ratchet | 10/21 @64% — weakest hopper cell at small scale |
| e222 | 4662401 | single-head S0R0 | cheetah | BIG | single-head vs Director e191 at scale | 181/434 @63%, peaked 413 @1.5M then noisy — ≥ Director-comparable |
| e223 | 4662402 | single-head S1R0 | cheetah | BIG | +struct | 205/455 @65%, cleanest trajectory, plateau 320–434 from 0.5–1.8M — best BIG-cheetah cell |
| e224 | 4662403 | single-head S0R1 | cheetah | BIG | +ratchet | 134/143 @64%, flat plateau ~110–140 from 700k — **H refuted**, ~3× below e222/e223. **CANCELLED+archived 07-15** to free A100 for e232 |
| e225 | 4662404 | single-head S1R1 | cheetah | BIG | struct×ratchet | 79/201 @64%, flat low plateau + late decline. **CANCELLED+archived 07-15** to free A100 for e230 |
| e226 | 4662405 | single-head S0R0 | hopper | BIG | **key**: single-head vs Director e124 on sparse-hard | **214/302 @65%**, rises to 280–286 plateau by 1.6–2.1M; reward_rate 0.245, mask_frac 0.95 — **H confirmed decisively: first alive sparse-task cell in the project** (F17). Kept running (campaign champion) |
| e227 | 4662406 | single-head S1R0 | hopper | BIG | +struct | 141/165 @65%, smooth plateau, no collapse — struct costs ~half of e226 but no longer fatal (contra F11 under the old design). Kept running (comparator) |
| e228 | 4662407 | single-head S0R1 | hopper | BIG | **key**: ratchet rescue of hopper? | 22/52 @65%, flat/noisy all run, mask_frac tracks target to 0.30 by 1.1M and never recovers — **H refuted**: forcing sparsity is worse than never forcing it. **CANCELLED+archived 07-15** to free A100 for e233 |
| e229 | 4662408 | single-head S1R1 | hopper | BIG | struct×ratchet | 4/268(peak) @65% — tracks e226 to 208 @500k then **collapses** to <10 for the remaining 2M+ steps once the ratchet target locks in. **CANCELLED+archived 07-15** to free A100 for e231 |

Verified at launch (07-14): all 18 RUNNING; e216 (ratchet) step 4544 shows
`mask_sparsity_target_now`≈1.0 annealing down, `mask_frac_mean`≈0.89 (full-edit early),
`loss/mask_sparsity` finite — ratchet behaving as designed.

GPU occupancy (post 07-14 single-head relaunch): A100 8/8 (e222–e229 BIG single-head) ·
V100 8/8 (e212–e219 small) · P100 2/8 (e220/e221, pinned `saion-gpu[11-14]`). Prior
occupants e194–e211 all CANCELLED+archived 07-14.
Cancelled at 0.6M for flatness (07-13): e181 (agg=sum), e182/e184 (combined τ8 hop/acrobot),
e183 (10× expl), e161 (struct-0 hopper). Cancelled 07-14 (dead, freed A100s for e200–e203):
e178/e185/e189 (hopper), e192 (acrobot). Cancelled 07-14 (alive but not "just started";
checkpoints preserved and resumable, freed 4×A100 + 4×V100 for the mask-ratchet cell
e204–e211): e179/e180/e190/e191 (A100) and e186/e187/e188/e193 (V100). Detailed hypotheses
& readout logic: §6.

### Ratchet-target 0.5 / struct-adapt 0.01 retest (e230–e233, launched 07-15)

Follow-up to the single-head campaign's R1 cells (e224/e225/e228/e229), which all
underperformed their free/struct-only siblings and were cancelled+archived 07-15 (65% of
budget) to free their 4 A100 slots for this batch. Motivating question: were the old
ratchet(target 0.3, vel tuned to ~1.1M steps)/struct-adapt(target 0.006) settings simply
too aggressive, or does forcing edit-fraction sparsity cost performance at any setpoint
under the single-head design? This batch loosens both: mask-sparsity-target ratchet now
anneals `mask_sparsity_target_init=1.0` → `mask_sparsity_target=0.5` (was 0.3) — half the
prior distance (0.5 vs 0.7) — over the same ~1M-env-step horizon, via a recalibrated
`mask_sparsity_target_vel=8.0e-6` (`= 1.12e-5 × 0.5/0.7`, BIG scale; old velocity/distance
ratio preserved). Struct-adapt target loosened `0.006` → `0.01` (dual-ascent, two-sided,
default `goal_struct_adapt_one_sided=False`). Both on together (S1R1 cell) — extended to
**all 4 tasks** this time (cartpole, acrobot new; cheetah, hopper repeat at the new
setpoints), since e230/e231 give the first single-head data point on cartpole/acrobot at
any hyperparameters.

| Exp | Job | Task | Hypothesis / expected | Status |
|---|---|---|---|---|
| e230 | 4664145 | cartpole | dense control: does S1R1 at looser targets close the gap to e222/e223-class free/struct performance? First single-head cartpole data point. | **CANCELLED 07-15 @17% (680k/4M)** to free A100 for e234 launch — final: 435/479, mask_frac 0.68 (still ramping toward 0.5), still climbing when cut |
| e231 | 4664146 | acrobot | first single-head data point on acrobot at any setting; H: fails like e224/e225's dual-head predecessors (e176/e177), or single-head's hopper rescue mechanism generalizes | **CANCELLED 07-15 @15% (594k/4M)** — final: 54/84, mask_frac 0.71 — weak but non-zero, unlike every prior dual-head acrobot cell (e176/e177 ≈0–2.5); too early to call, run never finished |
| e232 | 4664147 | cheetah | vs e224/e225 (0.3/0.006, dead-ish plateau ~110–140): does 0.5/0.01 recover toward e222/e223's 400+ plateau, or does any forced sparsity cap cheetah near the same ceiling regardless of setpoint? | **CANCELLED 07-15 @16% (649k/4M)** — final: 173/178, mask_frac 0.69 — clears the 0.3-target predecessors' ~110–140 plateau already, partial recovery toward e222/e223, run never finished |
| e233 | 4664148 | hopper | **key**: vs e228/e229 (0.3/0.006, flat ~22 / collapsed): does a looser ratchet target let the manager keep enough of the free-run edit behavior (e226: mask_frac 0.95) to preserve the hopper rescue, or does *any* imposed sparsity target — regardless of how loose — reproduce the same collapse/flatline? | **CANCELLED 07-15 @17% (665k/4M)** — final: **215/216**, mask_frac 0.68 — far above e228's final 22/52 and inside e226's alive plateau (280–286) at less than a fifth of the training budget; **the "any imposed target collapses" branch looks refuted at 0.5**, but the run was cut before a stable late-training verdict (recall e226/e227 also looked alive at a similar fraction of budget before declining, §2 correction note) |

GPU occupancy after this swap (07-15): A100 8/8 (e222/e223/e226/e227 single-head champions
+ e230–e233 this batch, until e230–e233 were themselves cancelled 07-15 to free all 4 A100s
for the e234–e249 implicit-sparsity launch — see above). e224/e225/e228/e229 archived to
`/bucket/DoyaU/vasilache/bucket/results/dreamerv3/`, deleted from `/work` (~42GB freed);
e230–e233 archived the same way when cancelled.

**Caveat on all four e230–e233 numbers:** none of these runs reached even 20% of budget,
and the single-head board's own history (e226/e227) shows an early-looks-alive cell can
decline substantially by 3M+ steps. Treat e233's promising early read as a lead for a
follow-up run, not a settled rescue.

### Hyperparameter comparison table

Moved to its own file, `HYPERPARAMETER_COMPARISON.csv` (2026-07-24, updated through
e313 — e312/e313 are placeholder rows, still `PENDING`/no data) — a bare cross-cell
table, no legend or prose.

---

## 3. Settled findings

**Mask mechanism**
- **F1. `mask_sparsity_mode=prob` (differentiable rate target) is the only working sparsity
  penalty.** `sample` (no gradient) drifts dense/destabilizes (e1/e3/e7, e26–e32);
  `reinforce` rails and kills reward (e10/e11); fixed-weight→0 freezes the goal → score 0
  (e12–e15, also the clean "manager is load-bearing" control).
- **F2. One-sided costs never create interior equilibria; two-sided targets do.** Edit
  costs freeze (mask→0, score 0) or explode (mask→0.95+) depending on magnitude with no
  stable middle (e70–e77); switch costs rail K to max (e67–e69); achievement-gated costs
  collapse to "never edit" (e78, e81). The prob target's two-sided restoring force is what
  keeps masks at 0.3 (this reasoning replaced the earlier "slack multiplier" misreading).
- **F3. `prob_entropy` (rate + per-block entropy anti-collapse) is the best-tracking mask.**
  Entropy-only works at small scale but railed dense at BIG with a goal-KL divergence
  (e144 vs e145); the rate target is load-bearing at BIG (e145 collapse; e149 tight at 0.31–0.35).
- **F4. Per-block Q-credit creates a genuine per-block edit threshold** (e82/e88: block
  edited iff its leave-one-out advantage exceeds the cost) — the only working "emergent
  sparsity" mechanism — but no freedom variant beat the target-based recipes on score.

**Duration / K**
- **F5. K is the dominant lever and a tradeoff:** fixed K=1 → best score, loose reaching
  (e8 800, cosine 0.34); K=8 → tight reaching, score collapse for masked recipes (e9 ~1,
  e20–e23). No fixed K gets both; e57's learned K=4 was the first to get both on cartpole.
- **F6. Variable K needs a soft two-sided duration prior.** No prior → drifts short (~2.2,
  score ~100–200: e24/e52/e54); reg cliff: 0.001–0.03 fine, 0.1 collapses via magnitude
  domination (§7). Cartpole target ordering 4 >> 16 > 8 >> 32 (e57–e60) — but τ=4 is
  cartpole-tuned and kills sparse tasks (F12).
- **F7. The duration Lagrangian (dual ascent on |E[dur]−τ|, own loss key) tracks ~20×
  tighter than the fixed prior** (std 0.0012 vs 0.024 at BIG) and self-relaxes to λ=0 in
  tolerance; return-neutral alone (e153 557) but compounds in the combined recipe (e157 833).
  The first version regulated raw duration one-sidedly and railed — fixed 07-10 (§7).
- **F8. `mgr_reward_agg=sum` drives K short and scores below mean** (e66, falsified the
  "averaging under-credits long blocks" theory); it also did NOT rescue hopper (e181 dead) —
  duration-return decoupling is not the sparse-task mechanism.

**Recipes & transfer**
- **F9. The combined recipe is superadditive.** Cartpole BIG: combined 776–844 ≫ mask-only
  360/419 ≫ plain var-K τ4 97 (collapsed from 658). Historically masked-K8 ≈ 1 (e9) and
  plain var-K without struct ≤ 200 — neither component ever worked alone.
- **F10. Lagrangian controllers transfer where fixed penalties don't:** cheetah 253 (fixed
  targets, e142) → 465/508 (Lagrangians, e175), cartpole unchanged. `hrl_auto`'s fully
  adaptive targets fixed corner-railing off-cartpole but not learning (e126–e138) — the
  fixed-target-with-adaptive-multiplier middle ground is what works.
- **F11. Struct (`goal_struct_weight`) is independent of masking and has two faces.**
  Not a score lever on small cartpole (w400 best of family, e33–e40) but load-bearing for
  plain var-K: with it 724 (e46); without it, weak or rise-then-collapse (e166 130, e167
  97 after peaking 658). Collapse anatomy (e167): manager advantage spikes 4× then dies to
  ≈0 permanently, exactly when worker goal-reward drops to run-low and the goal-AE
  reconstruction drifts with no anchor — a coordination collapse, NOT entropy (entropies
  pinned at targets throughout). Whether the locality face hurts sparse tasks: e189 decides.
- **F12. Sparse tasks die under every restricted variant; the base does not.** Pure
  Director learns hopper ~300 (e124) and acrobot (e180 in progress); ALL masked and/or
  var-K variants sit at rewrate ≈1e-4 forever. Clean diff: e169 vs e124 = only the var-K
  package (τ4, reg 0.01) → 0 vs 300. Masking alone at fixed K8 also kills hopper
  (e164/e165) despite HIGH worker action entropy (+3.1) — not an exploration-entropy
  problem. τ8 gives a weak pulse only (e178). Combined-recipe rescues (τ8, 10× expl)
  failed (e182–e184).
- **F13. The worker is blind to its deadline** — input is only (feat, goal); its λ-return
  is cut at switches, so under var-K it faces a random unobservable horizon (variance in
  the critic + ~half of e124's reach-time at τ4). HiTS (Gürtler et al. 2021) fixes exactly
  this with countdown-conditioned lower levels; our var-K manager already implements the
  manager half of timed subgoals (durations fixed in advance, never interrupted early).
  Implemented as `worker_timed_goals` (07-13); testing e185–e188/e192/e196–e199.
- **F14. History corrections that changed interpretations:** e162's "failed anchor" was a
  misremembered comparison — its true anchor e9 scored ~1, so e162's 272 beats history;
  e46's 724 was WITH struct+τ8 so e166's 130 is in-family. No code regression; the controls
  were valid (2026-07-13 correction).

**Scale & baselines**
- **F15. Scale matters for anything beyond cartpole:** Director-default batch (16×64) +
  cadence doubled cheetah vs small (e118 346 vs e111 167); hopper only ever learned at BIG.
  Native cuDNN conv ≈2× faster (A100 always; V100 via the Route-A env clone). e88-style
  freedom recipes proved cartpole-specific when ported (e110–e114).
- **F16. Base-hierarchy reproduction:** pure Director beats the full ported recipe on the
  same hardware (e123 cheetah 435 max > e118 372; e124 hopper 300 vs e122's 0.1). Ten
  behavioral diffs enumerated in the archive ("regression analysis", 2026-07-07); the
  six suppression mechanisms it identified (off-manifold partial edits, var-K variance,
  untuned reward-unit regularizers, larger action space, sum agg, cond input) have each
  since been tested — the survivors are the duration/horizon/struct threads of §2.
- **F17. The single-head masked manager rescues hopper; F12 was diagnosing the dual-head
  design, not sparsity itself (2026-07-15).** `agent.mask_joint_edit` (one C+1-way
  categorical per block, class 0 = abstain — removes the discarded-content REINFORCE
  confound of the old skill+mask dual-head design by construction) is alive on hopper at
  BIG scale when left fully free (e226: reward_rate 0.245, score 214/302, rising to a
  280–286 plateau by 1.6–2.1M) — the first alive sparse-task cell after 20+ dead dual-head
  cells (F12). It does this **without** editing sparsely: `mask_frac_mean` sits at
  0.94–0.97 throughout — left free, the manager never chooses to abstain much, so the win
  is attributable to the single-head credit-assignment mechanism, not to achieving
  sparsity. F12's individual claims stand (dual-head masking/var-K did kill hopper), but
  its implicit generalization — "sparse tasks die under every restricted variant" — no
  longer holds; scope F12 to the dual-head design. Forcing sparsity still costs
  performance under the new design too: struct (e227) nearly halves the score (141/165)
  but, unlike e189 under the old design, no longer kills it outright; the mask-sparsity
  ratchet (e228, forces the edit fraction down to a 0.3 target by ~1.1M steps) stays flat
  at 22/52 the whole run — refuting the exploration-starvation hypothesis it was built to
  test (§2) — and struct+ratchet combined (e229) collapses outright (208 peak @500k → <10
  for the remaining 2M+ steps), the same collapse signature as F11's dual-head cells. The
  rescue is **BIG-scale-only**: at small scale every single-head hopper cell
  (e218–e221) rises transiently then decays or stays flat, matching the small
  pure-Director control's own dead floor (e213: 1/40) — consistent with F15. Acrobot
  untested under single-head so far. Single seed; re-run e226/e223 (best cheetah cell)
  with a second seed before promoting to the default recipe.
- **F18. A REINFORCE cost on implicit sparsity rails and kills the task in every cell
  tested; only the indirect (struct) lever is safe (2026-07-16, e234–e249, full 1M-step
  budget, 16/16 cells).** Testing whether to *raise* measured implicit goal-code reuse
  (`goal/implicit_sparsity_block`, the un-masked analogue of mask_frac introduced with the
  pivot away from explicit masking, §2) costs task return, and whether an indirect lever
  (struct-adapt) or a direct one (a dual-ascent REINFORCE penalty on `1 − kept_frac`,
  modes `ratchet`/`direct`) is cheaper: struct-only preserves learning in all 4 cells
  tried (hopper/cheetah × small/BIG) — cheetah BIG lands within noise of its
  contemporaneous Director control, hopper BIG and cheetah small pay a real but partial
  cost, hopper small stays at the F15 small-hopper noise floor uninformatively. Every one
  of the other 12 cells (ratchet, direct, or struct+ratchet, both scales, both tasks)
  collapses to a score of 0–8.6, regardless of struct being stacked on or not. The
  controller does not even reach its own target: `impl_sparsity_scale_mean` (the dual
  multiplier) rails to its configured ceiling (5.0) in every one of the 12 cells, usually
  within the first third of training, while the realized change-fraction plateaus at
  0.44–0.52 — well short of the 0.3 implied by the 0.7 kept-target. Same
  magnitude-domination signature as the duration-reg cliff (§7) and the one-sided
  edit/switch costs of F2, and consistent with the single-head mask ratchet's failure
  under the old design (F17, e228/e229) — this is now the third independent mechanism
  where forcing a sparsity metric via an aggressive per-step penalty destroys learning,
  while a softer/indirect lever (soft prior, struct) does not. Caveat: struct's own kept
  fraction (0.20–0.38) is not clearly *higher* than the unprompted Director baseline
  (0.32–0.45, measured on pre-`mgr_cond_goalcode` checkpoints, so not a fully controlled
  comparison) — struct is established here as *safe*, not as a working sparsity lever.
  `agent.impl_sparsity_mode` should stay off (`none`) by default pending a much weaker
  controller weight or a non-REINFORCE mechanism.

  **Root cause (2026-07-16, code-level): the implicit controller is architecturally
  forced into the one mechanism F1 already found broken.** Under the explicit single-head
  mask (`mode='prob'`/`prob_entropy`, what e224–e229 used), abstain is class 0 of each
  block's own categorical, and the sparsity penalty is built from `self._mask_prob()`
  (`agent.py:1281`) — `sigmoid(logit)`, differentiable — added as a genuine loss term
  (`agent.py:2483–2506`) that backprops directly into each block's own logit: exact,
  low-variance, per-block credit, and a dedicated "keep" outcome one bit away. Under the
  implicit controller there is no abstain action — every block always redraws a real
  class — so "kept" can only be measured post-hoc as `argmax(Z_t) == argmax(Z_{t-1})`
  (`agent.py:2534–2540`), which has **no gradient by construction**. The code marks this
  explicitly with `sg(chg)` and instead subtracts `impl_scale * chg` from the manager's
  extrinsic reward (`agent.py:2555–2560`) — the *only* remaining channel is REINFORCE
  through the actor-critic return, sharing the same reward stream as the task signal
  itself. This is structurally the same mechanism as the explicit-mask framework's own
  `mode='reinforce'` (`agent.py:2448–2462`), which **F1 already found rails and kills
  reward**, in the earliest bring-up runs (e10/e11) — dropping the mask didn't just
  remove the discarded-content confound (the intended win of the pivot), it also removed
  access to a differentiable sparsity proxy, forcing the new controller into the one mode
  already known broken. Consistent with this: raw scores show the implicit-controller
  cells are dead from step 0 (e243/e244 hopper BIG: 0.0, peak ≤1.0, no initial rise),
  whereas the explicit-mask ratchet cells at least achieved partial or transient learning
  before degrading (e228 22/52; e229 rose to 208 before collapsing @500k) — the
  differentiable per-block loss shaped behavior carefully early on; the reward-mediated
  scalar penalty never got the chance to.

- **F19. Making the reuse pressure genuinely differentiable does not fix F18 — it
  reproduces the same magnitude-domination collapse via a cleaner gradient path
  (2026-07-21, e258–e277, three mechanism families, 16/16 cells at the full 4M-step
  budget).** F18 blamed REINFORCE specifically: the implicit-sparsity controller had no
  gradient path for "kept," so its only lever was a stop-gradiented reward penalty, the
  same mechanism F1 already found rails and kills reward. Two follow-up mechanisms give
  the manager a real, backprop-carrying channel instead — `goal_reuse_adapt` (dual-ascent
  Lagrangian on continuous decoded-goal cosine similarity, families A/B) and
  `goal_soft_reuse_adapt` (dual-ascent Lagrangian on a discrete block-overlap loss on the
  manager's own softmax, family C) — and both still collapse task learning in every
  hopper cell tried (8/8: A+B, 0.001–0.004 vs. an ≈300 baseline) once their similarity
  ratchet fully engages (~1M steps): every cell rises to a real peak early (17–365,
  reached 176k–945k steps) and then falls to the floor and stays there for the remaining
  3M steps, with the Lagrange multiplier railed at or near its configured ceiling
  throughout — the identical shape as F18's REINFORCE multiplier and the duration-reg
  cliff (§7), just reached with an unambiguous gradient path this time. **Differentiability
  was never the root cause; a soft prior that behaves while loose becomes catastrophic
  once its target fully engages, regardless of which mechanism carries the pressure.**
  Family C is the partial exception and the most promising lead so far: it does not
  reach its own 0.7 overlap target either (plateaus 0.56–0.60, scale still railed), and
  still fails hopper outright, but cheetah survives at 20–60% of baseline (70–182 vs.
  1.85–7.57 for families A/B) — the first reuse mechanism of the four tried (F18's
  REINFORCE controller, A, B, C) where cheetah does not fully collapse. Two secondary
  ablations, both clean: (1) the raw one-hot code channel is not load-bearing for the
  manager to satisfy a continuous similarity target (family B, no code input, matches or
  slightly beats family A on every cell) — moot, since A collapses regardless; (2)
  stacking the known-safe F18 struct lever on top of a direct reuse loss is not neutral,
  it actively costs cheetah roughly half its score (family C struct+ratchet vs.
  ratchet-only: 70 vs. 182 BIG, 73 vs. 111 small) — refines F18's "struct is safe" to
  "safe only as the sole sparsity lever; stacking constraints on the same code
  compounds." **Net state:** hopper has now failed under every sparsity/reuse mechanism
  tried project-wide except the unrestricted single-head design (F17) — it reads as
  uniquely fragile to *any* secondary manager-side objective, not specifically to
  non-differentiable or REINFORCE-based ones. Cheetah's partial survival under family C
  is the one live thread: worth a direct A/B against a lower overlap target (matching the
  unprompted 0.32–0.45 band from Table `tab:implbaseline` rather than 0.7) before
  concluding the mechanism itself is exhausted. `goal_reuse_adapt` (either manager-input
  variant) should stay off by default pending a much weaker target/multiplier ceiling;
  `goal_soft_reuse_adapt` ratchet-only (no struct) is the least-bad configuration found
  to date for cheetah specifically. `goal_delta_mode`'s own contribution is still
  untested at scale (its two scheduled A/Bs, e250–e257 and e259/e261/e263/e265, were both
  cancelled before a readout) — open question, not closed by F19.

- **F20. Lowering the reuse/overlap target rescues 5 of 6 dead BIG-scale F19 cells at the
  full 4M-step budget — but the 6th shows the same collapse can still fire, just later,
  so "settled near floor" is not a stability guarantee (2026-07-23, e278–e293, all 16
  cells COMPLETED).** Direct retest of F19's three mechanism families at a lower,
  better-calibrated target (decoded-similarity 0.8 vs. 0.95; block-overlap 0.5 vs. 0.7,
  same ratchet shape stretched to ~2M steps instead of ~1M): confirms the 07-22 interim
  read for 5 of the 6 BIG-scale hopper/cheetah cells that were dead or near-dead under
  F19 — e278 163.0 (noisy, never settles, but never returns to the floor either), e280
  296.6, e284 198.2, e286 330.5, e287 199.4, e290 325.9, e291 160.6 (all vs. F19's
  0.001–70.4 for the same cells) — with the Lagrange multiplier finishing at or near its
  numerical floor in most of them, confirming the interim's "target was too aggressive,
  not the mechanism" diagnosis for these cells specifically. **The 6th cell, e282
  (family B, hopper BIG), reproduces F18/F19's magnitude-domination collapse late instead
  of avoiding it**: `episode/score` held a stable 200–230 plateau through 98.6% of the
  4M-step budget, then `train/goal/reuse_adapt_scale_mean` — flat at its 0.0002–0.0005
  floor for the preceding ~3.9M steps, same as every other cell in the batch — jumped to
  0.09–0.21 within a single ~8k-step logging window at step 3,939,344, `wkr_goal_rew`
  dropped in lockstep (0.53→0.33), and `episode/score` crashed to a trailing-50 mean of
  1.6 with no recovery before the run ended ~50k steps later. This is not evidence the
  lower target is unsafe in general — the other 15 cells show comparably large scale
  excursions throughout training and self-correct within tens of thousands of steps — but
  it does mean a cell that "looks settled" at 73–98% of budget is not thereby proven
  stable, and the interim readout's confidence (branch 1: "in every cell... the
  multiplier has settled... not railed") should have been scoped to "as of this pull,"
  not stated as a durable property of the retest. Struct-stacking's cost/benefit is
  confirmed scale-dependent as well as target-dependent: at BIG scale struct+ratchet
  still beats ratchet-only on cheetah (e290 325.9 vs. e291 160.6, same direction as the
  interim pull, reversing F19's 0.7-target ordering), but at small scale ratchet-only
  still wins (e293 259.2 vs. e292 107.0) — the *original* F19 ordering survives at small
  scale even under the lower target. **Net effect on F19:** "hopper fails under every
  reuse/sparsity mechanism except the unrestricted single-head design" no longer holds as
  a project-wide statement — family C (`goal_soft_reuse_adapt`) keeps hopper alive and
  stable (e286, e287, no collapse anywhere in its 6-cell family) — but "a soft prior that
  behaves while loose becomes catastrophic once its target fully engages" (F19's causal
  claim) is *not* refuted, only shown to depend on the target being low enough relative
  to the mechanism's own reachable range for a given family+task+scale, and even then
  not to be a permanent guarantee within a fixed training budget. Family C
  (`goal_soft_reuse_adapt`, now the config default alongside `goal_struct_adapt`, §2) is
  the only one of the three families with zero collapse events across all 6 of its
  BIG/small × hopper/cheetah × struct/no-struct cells, and the only one with a cell
  (e286) finishing clearly above its Director baseline — the strongest evidence yet that
  the discrete block-overlap loss is the more robust mechanism of the three, not just the
  better-performing one at the old target. `goal_reuse_adapt` (families A/B) should stay
  off by default even at the lower target given e282's late failure; `goal_soft_reuse_adapt`
  is the recommended mechanism pending a second seed. Full per-cell table and the e282
  collapse trace: §2.

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

One row per experiment; **bold** = campaign best. Scores: last-15 (peak) unless noted.
Full narratives: archive file.

### e1–e15 · Masking bring-up (small cartpole)
Root cause of early failures: the soft penalty had zero gradient to the mask head; `prob`
mode fixed it (→F1).

| Exp | Config | Score | Outcome |
|---|---|---|---|
| e1/e3/e7 | sample-mode sparsity ± struct, K8/K1 | 610–840 | mask collapses dense |
| e2/e4/e6 | mask_topk=3 | ~0–100 | topk strangles (F4-dead-end) |
| e5 | Director size6m 64×64 | ~720 | 64×64 baseline OK; report segfault (§7) |
| **e8** | **prob 0.3 + struct200, K=1** | **~800** | WIN; mask 0.15–0.21 sparse, stable |
| e9 | prob 0.3, K=8 | ~1 | over-sparsified inert — the true masked-K8 anchor |
| e10–e15 | reinforce / fixed→0 | ~0–19 | broken modes; manager-load-bearing control |

### e16–e23 · Manager conditioning {K1,K8}×{targets}×{Z, Z+achieve}
Conditioning doesn't break the K tradeoff (F5). Best: **e17** K1/Z-only **777**. K8 cells:
tight reaching (0.65–0.76) but scores 3–504.

### e24–e40 · Var-K intro + struct variants
e24/e25: no-prior var-K drifts short/unstable. e26–e32: sample-mode is the destabilizer
(→F1). e33–e40 struct family: **e33 w400 744**; feat/margin/adaptive variants hurt
(211→96); struct ≠ score lever at small scale (→F11).

### e41–e56 · Duration-prior tuning (plain var-K + masked)
| Exp | Duration shaping | Score | Verdict |
|---|---|---|---|
| e41/e43/e49 | reg 0.1 (± repval fix) | 104–156 | collapse; repval not the cause |
| e42 | fixed K=8 control | **674** (758) | plain fixed-K baseline |
| **e46** | reg 0.01→8 | **724** (761) | steadiest var-K; WITH struct200 |
| e47/e51 | reg 0.001 / 0.03 | 601 / 614 | reg cliff mapped (→F6) |
| e48/e53 | adaptive prior | 160–189 | didn't rescue |
| e50 | fixed-K8 via duration bypass | 133 | var graph degrades even pinned |
| e52/e54 | no prior | 103/193 | drift short |
| e56 | adaptive→16, masked | 671 (766) | best masked var-K pre-e57 |

### e57–e65 · Soft-prior masked var-K + first ports
| Exp | Config | Score | Verdict |
|---|---|---|---|
| **e57** | masked var-K, τ4, prob 0.3, struct200 | best-15 **839**, max 854 | small-cartpole champion; blk/step 0.64 |
| e58/e59/e60 | τ8 / τ16 / τ32 | 511 / 701 / 119 | target ordering 4>>16>8>>32 |
| e61 | walker τ8 | 549 (685) @3M | directional, cancelled |
| e62 | block-rew, no prior | 216 | credit pooling ≠ prior substitute |
| e63/e64 | dur-entropy / min-hold | 181 / 155 | no-prior fixes fail |
| e65 | antmaze-M port, 10M | 0 @2.7M | no reward encountered; abandoned |

### e66–e90 · "Manager freedom" campaign (priced costs, per-block credit)
Conclusion: one-sided costs cannot create interior equilibria (F2); per-block credit is
the only emergent-sparsity mechanism that works (F4); nothing beat e57.

| Exp | Mechanism | Score | Note |
|---|---|---|---|
| e66 | sum agg | 691 | K→3, falsifies avg-under-credit theory (F8) |
| e67–e69 | switch cost 0.02–0.1 | 713–**761** | K rails 13–14 (corner) |
| e70–e77 | edit costs (2 sweeps) | 0–753 | freeze↔explode knife-edge (F2) |
| e78/e79 | ach-gated cost / sparsemax | 0 / 373 | corner / too restrictive |
| e80 | sum+switch combo | 750 | K rails ~13.6 anyway |
| e81/e82 | per-block credit T1/T2 | 0 / **727** | T2 works for task credit, mask dense 0.69 |
| e83–e90 | perblock cost sweep + combos | 508–**764** | **e88** (combo 0.3, durmax32): 764, mask 0.35, K~18, blk/step 0.16 — emergent on both axes (target inert) |

### e91–e122 · Ports & scale (freedom recipe)
e91–e98 VOID (misconfigured, no data). e99–e114: e88-recipe ports to 5 DMC tasks mostly
fail (best: quadruped 233, cheetah 167) → cartpole-specific (F15). e115 director_match
64×64 cartpole: 528 @2.2M (cancelled). e116/e117: throughput work (native conv ~2×;
train_ratio 8) — e117 cancelled early. **e118** cheetah at TRUE Director batch: **346**
(2× the small port; F15). e119–e122 multi-V100 ports: acrobot/hopper ran to 4M with NO
learning (long-hold drift K≈15); pendulum/quadruped OOM'd; wandb silently off (§7).

### e123–e125 · Pure Director baselines
e123 cheetah: max 435 @2.6M (> e118's 372; stopped by walltime — redo = e191).
**e124 hopper: ~300 by 1.5M — the reference every sparse-task comparison uses.**
e125 acrobot: spiked 339 then collapsed (unstable, 4×V100); clean A100 redo = e180.
Regression analysis (archive): the full ported recipe was suppressing the base (F16).

### e126–e138 · `hrl_auto` (fully adaptive targets)
Interior K/mask off-cartpole (corner-railing fixed) but cheetah/hopper dead → adaptivity
alone ≠ transfer (F10). e126-vs-e132 A/B: entropy mode's value is dip-*recovery*
(anti-collapse), not level. e133–e138 restart (K-entropy 0.5, KL off) cancelled for e139+.

### e139–e159 · Fixed-prior ports + Lagrangian symmetry matrix
e139–e143 (e57 recipe off-cartpole/BIG): cheetah 51 small / **253 BIG**, hopper 0, cartpole
BIG 709 — fixed penalties transfer poorly (F10). e144–e159 (4 groups × task × scale):

| Group | Mechanism | Cartpole BIG | Verdict |
|---|---|---|---|
| A entropy-only | no rate target | 86 (railed 0.72–0.80 + KL divergence) | rate target load-bearing at BIG (F3) |
| B prob_entropy | rate + anti-collapse | 710, mask tight 0.31–0.35 | best mask tracking |
| C dur-Lagrangian | duration dual ascent | 557, dur std 20× tighter | mechanism ✓, return-neutral (F7) |
| **D combined** | B + C | **833 (869)** | only cell beating the 709 baseline (F9) |

All 8 hopper cells dead. Post-mortem (07-10) found two env-symmetric bugs (§7: forward-fill
recency bias; one-sided duration controller) → fixed; e145's entropy-mode failure carries a
goal-KL divergence confound (unresolved).

### e160–e177 · Controls + combined-recipe retest on fixed code
e160/e161: struct-0 ablation on short-a100 (e161 cancelled 07-13 — moot on all-dead hopper;
e160 continues). e162–e177 (post-fix code, 07-10→07-12):

| Exp | Recipe | Task/Scale | last10 (peak) | Note |
|---|---|---|---|---|
| e162/e163 | mask-only K8 | cartpole s/BIG | 272 (575) / 360 (419) | beats true anchor e9 (~1) — F14 |
| e164/e165 | mask-only K8 | hopper s/BIG | 0 | dead despite wkr entropy +3.1 (F12) |
| e166/e167 | plain var-K τ4 struct0 | cartpole s/BIG | 130 (192) / 97 (**658→collapse**) | F11 collapse anatomy |
| e168/**e169** | plain var-K τ4 struct0 | hopper s/BIG | 0 / **0** | **e169-vs-e124 clean diff (F12)** |
| e170/**e171** | combined τ4 | cartpole s/BIG | 668 (786) / **776 (844)** | 07-10 fixes performance-neutral |
| e172/e173 | combined τ4 | hopper s/BIG | 0 | dead |
| e174/**e175** | combined τ4 | cheetah s/BIG | 166 / **465 (508)** | 2–3× fixed-prior (F10) |
| e176/e177 | combined τ4 | acrobot s/BIG | 0 (16) / ~1 (37) | dead; e177 lost 19h to preemption (§7) |

### e178–e203 · Current campaign (see §2 live board, §6 hypotheses)
Launched 07-13. Triaged at 0.6M: e181–e184 cancelled (flat; see §4). e185–e188 countdown
cells; e189/e193–e195 struct cells; e190/e191 controls; e192 acrobot best-guess;
e196–e199 full stack (combined + countdown + struct-adapt) on 4 small tasks.
Smokes passed pre-launch: countdown ×3 recipes + flag-off regression (4660289–91),
struct-adapt (4660314), full stack (4660322).
e200–e203 (launched 07-14): full-stack cell (masked var-K τ8 lagrangian + struct-adapt,
now `run_v3_prior_vargoal_big_a100.sbatch`-native via new `STRUCT_ADAPT`/
`STRUCT_ADAPT_TARGET` env knobs + countdown) at BIG scale on all 4 tasks, struct-adapt
target lowered to **0.006** (untried at any scale before this). Replaces the 4 dead
hopper/acrobot BIG cells e178/e185/e189/e192, cancelled the same day to free the A100s.
e178–e211 all cancelled+archived 07-14 to free hardware for the single-head campaign below
(see §2 for cancellation reasons per run).

### e212–e229 · Single-head mask campaign (see §2 table, §3 F17)
Launched 07-14, single-head (`mask_joint_edit`) 2×2 struct×ratchet on cheetah/hopper ×
small/BIG, against pure-Director controls (e212/e213 small; e124/e191 BIG, reused). Interim
readout below is 2026-07-15, 54–72% of a 4M-step budget — single seed, still training.

| Exp | Cell | Task/Scale | last15 (peak15) | Note |
|---|---|---|---|---|
| e212/e213 | Director baseline | cheetah/hopper small | 100 (111) / 1 (40) | small-scale denominators; small hopper dead even unrestricted |
| e214/e215 | single-head S0R0/S1R0 | cheetah small | 111 (157) / 174 (194) | ≥ baseline; struct helps |
| e216/e217 | single-head S0R1/S1R1 | cheetah small | 145 (238) / 50 (67) | ratchet costs plateau; combined worst |
| e218/e219 | single-head S0R0/S1R0 | hopper small | 1 (125) / 7 (117) | transient rise, decays — small-scale ceiling holds regardless of design |
| e220/e221 | single-head S0R1/S1R1 | hopper small | 24 (81) / 10 (21) | ratchet suppresses even the transient bump |
| e222/e223 | single-head S0R0/S1R0 | cheetah BIG | 181 (434) / **205 (455)** | ≥ Director-comparable; struct is the best BIG-cheetah cell |
| e224/e225 | single-head S0R1/S1R1 | cheetah BIG | 134 (143) / 79 (201) | ratchet ~3× below free/struct; not regression-neutral |
| **e226** | single-head S0R0 | hopper BIG | **214 (302)**, rising to ~285 | **first alive sparse-task cell (F17)**; beats e124 196(326) |
| e227 | single-head S1R0 | hopper BIG | 141 (165) | struct costs ~half but no longer fatal (contra old-design F11) |
| e228 | single-head S0R1 | hopper BIG | 22 (52) | **ratchet-rescue hypothesis refuted** — flat all run |
| e229 | single-head S1R1 | hopper BIG | 4 (268) | rises to 208 @500k then **collapses** to <10 once ratchet target locks in |

e224/e225/e228/e229 cancelled+archived 07-15 (65%) to free A100s for e230–e233, a
looser-setpoint (ratchet target 0.5, struct-adapt target 0.01) retest on all 4 tasks —
see §2.

### e230–e233 · Looser ratchet-target retest, single-head design (see §2)
Launched 07-15, cancelled 07-15 at 15–17% of a 4M-step budget to free hardware for the
e234–e249 launch — no final verdict, but the partial trend is informative.

| Exp | Cell | Task | Step (% budget) | last15 (peak15) | Note |
|---|---|---|---|---|---|
| e230 | single-head S1R1, target 0.5/0.01 | cartpole | 680k (17%) | 435 (479) | still climbing when cut; first single-head cartpole datum |
| e231 | single-head S1R1, target 0.5/0.01 | acrobot | 594k (15%) | 54 (84) | weak but non-zero — every dual-head acrobot cell before this was ≈0–2.5 |
| e232 | single-head S1R1, target 0.5/0.01 | cheetah | 649k (16%) | 173 (178) | already above e224/e225's 0.3-target plateau (~110–140) |
| e233 | single-head S1R1, target 0.5/0.01 | hopper | 665k (17%) | **215 (216)** | far above e228's final 22(52); inside e226's alive-plateau range at <1/5 the budget |

### e234–e249 · Implicit-sparsity matrix (see §2 table, §3 F18)
Launched 07-15, **completed 07-15/16 at the full 1M-step budget** (all 16 jobs COMPLETED,
no cancellations). Pure Director base (`mgr_cond_goalcode=True`) × {hopper, cheetah} ×
{small, BIG} × {struct, struct+ratchet, ratchet, direct}. Full numbers and per-cell
readout: §2. One-line verdict: struct-only survives in 4/4 cells; every REINFORCE
implicit-sparsity-controller cell (12/12) collapses — new finding F18.

| Exp | Cell | Task/Scale | last15 (peak15) | Note |
|---|---|---|---|---|
| e234/e238 | struct | hopper/cheetah small | 1.9 (46.5) / 118 (159) | hopper small uninformative (F15 floor); cheetah small alive, below Director-at-1M |
| e235–e237 | struct+ratchet/ratchet/direct | hopper small | 0.0 (0.7) / 0.0 (0.6) / 0.0 (0.5) | dead; λ railed to ceiling (5.0) in all three |
| e239–e241 | struct+ratchet/ratchet/direct | cheetah small | 3.0 (5.0) / 2.2 (4.6) / 1.4 (4.2) | dead; λ railed in all three |
| e242/e246 | struct | hopper/cheetah BIG | 115 (116) / **330 (335)** | cheetah BIG ≈ Director-at-1M control; hopper BIG partial, still climbing at cutoff |
| e243–e245 | struct+ratchet/ratchet/direct | hopper BIG | 0.0 (0.2) / 0.0 (1.0) / 0.0 (0.6) | dead; λ railed in all three |
| e247–e249 | struct+ratchet/ratchet/direct | cheetah BIG | 3.4 (6.1) / 4.1 (8.6) / 3.4 (7.9) | dead; λ railed in all three |

### e250–e277 · Differentiable reuse mechanisms, three families (see §2 results board, §3 F19)
e250–e257 (`goal_delta_mode` struct/ratchet retest) launched 07-16, **cancelled ~3.5h in,
no readout** (superseded by e258–e265). e259/e261/e263/e265 (`goal_delta_mode` "delta"
counterparts of e258–e265) launched 07-16, **cancelled ~3h10m in, no readout**
(superseded by e274–e277). The remaining 16 cells across three mechanism families —
e258/e260/e262/e264 (A: `goal_reuse_adapt`, code+decoded manager input), e274–e277
(B: `goal_reuse_adapt`, decoded-only manager input), e266–e273 (C: `goal_soft_reuse_adapt`,
discrete block-overlap loss, ×{struct+ratchet, ratchet-only}) — all launched 07-16,
**all 16 COMPLETED 07-21 at the full 4M-step budget.** Full numbers and per-cell readout:
§2. One-line verdict: every family collapses hopper to the dead floor (8/8 A+B: 0.001–
0.004; 4/4 C: 0.004–1.75, all vs. an ≈300 baseline); families A/B additionally collapse
cheetah (1.85–7.57 vs. ≈300–435/≈100 baselines); family C's cheetah cells partially
survive (70–182), best at ratchet-only without struct — new finding F19.

| Exp | Cell | Task/Scale | Score last~300 (peak) | Note |
|---|---|---|---|---|
| e258/e260 | A: `goal_reuse_adapt`, code+decoded | hopper BIG/small | 0.001 (51) / 0.002 (17) | dead; sim 0.92–0.97 achieved, scale railed near ceiling |
| e262/e264 | A | cheetah BIG/small | 1.85 (203) / 3.47 (138) | dead; sim ≈0.95 achieved, scale railed |
| e274/e275 | B: `goal_reuse_adapt`, decoded-only | hopper BIG/small | 0.004 (138) / 0.001 (28) | dead; ≈ family A, code channel not load-bearing |
| e276/e277 | B | cheetah BIG/small | 7.24 (175) / 7.57 (132) | dead but ~2–4× less dead than family A |
| e266/e268 | C struct+ratchet | hopper BIG/small | 1.75 (209) / 0.30 (14) | dead; overlap 0.59 vs. 0.7 target, scale railed |
| e267/e269 | C ratchet-only | hopper BIG/small | 0.19 (51) / 0.004 (44) | dead; struct+ratchet slightly less dead here |
| e270/e272 | C struct+ratchet | cheetah BIG/small | 70.4 (113) / 73.3 (151) | partial survival, but struct costs ~half vs. ratchet-only |
| e271/e273 | C ratchet-only | cheetah BIG/small | **182.0** (365) / **111.3** (155) | best cheetah cells of all 16 — 40–60% of baseline |

### e278–e293 · Lower-target reuse/overlap retest, same 3 families (see §2 FINAL results, §3 F20)
Same design as e250–e277 (`RECIPE=director`, `MGR_FREQ=8`, `SEED=0`, `RUN_STEPS=4000000`)
with the reuse/overlap target lowered (decoded-similarity 0.8 vs. 0.95; block-overlap 0.5
vs. 0.7) and the ratchet stretched to ~2M steps instead of ~1M. Launched 07-21, 4
hopper-small cells (e279/e283/e288/e289) cancelled 07-22 at 84–98% of budget (already at
the F15 dead floor, no recovery in sight); the other 12 all COMPLETED 07-23 at the full
4M-step budget. Full numbers, per-cell trajectory notes, and the e282 collapse trace: §2.
One-line verdict: F19's "hopper dead in every family" reverses for 5/6 BIG cells (family
A noisy-alive, family B alive-then-terminal-collapse in the hopper cell, family C
stable-and-above-baseline); struct-stacking's cheetah cost/benefit is confirmed both
target- and scale-dependent — new finding F20.

| Exp | Cell | Task/Scale | Score trail300 (trail50) | Peak | Note |
|---|---|---|---|---|---|
| e278 | A: `goal_reuse_adapt`, code+decoded | hopper BIG | 163.0 (222.3) | 326.8 | alive, never settles — repeatedly 250+ then <100 across training |
| e279 | A | hopper small | — | 14.9 | CANCELLED 07-22 @84%, dead floor (F15) |
| e280 | A | cheetah BIG | 296.6 (291.7) | 333.0 | stable, low end of ≈300–435 baseline |
| e281 | A | cheetah small | 292.9 (288.4) | 351.7 | well above ≈100 baseline, still rising at cutoff |
| e282 | B: `goal_reuse_adapt`, decoded-only | hopper BIG | 230.2 (**1.6**) | 347.3 | **terminal collapse @3.94M (98.6% of budget)** — see F20 |
| e283 | B | hopper small | — | 28.6 | CANCELLED 07-22 @84%, dead floor (F15) |
| e284 | B | cheetah BIG | 198.2 (208.2) | 226.6 | alive, still rising at 4M cutoff |
| e285 | B | cheetah small | 180.1 (235.0) | 289.3 | alive, above ≈100 baseline |
| e286 | C struct+ratchet | hopper BIG | **330.5** (325.7) | 430.6 | **above ≈300 baseline — best hopper cell in the project**; config default |
| e287 | C ratchet-only | hopper BIG | 199.4 (185.6) | 298.8 | alive, stable plateau below baseline |
| e288 | C struct+ratchet | hopper small | — | 15.6 | CANCELLED 07-22 @84%, dead floor (F15) |
| e289 | C ratchet-only | hopper small | — | 76.2 | CANCELLED 07-22 @98%, dead floor (F15) despite transient peak |
| e290 | C struct+ratchet | cheetah BIG | 325.9 (332.4) | 449.4 | inside ≈300–435 band; config default |
| e291 | C ratchet-only | cheetah BIG | 160.6 (167.5) | 214.8 | weakest BIG cheetah cell; Lagrange scale still elevated (15.6) at finish |
| e292 | C struct+ratchet | cheetah small | 107.0 (149.7) | 191.8 | roughly matches ≈100 baseline; struct costs vs. e293 (small-scale reversal, F20) |
| e293 | C ratchet-only | cheetah small | **259.2** (277.8) | 342.1 | well above ≈100 baseline; strongest small-scale cell |

### e294–e303 · Task-generalization + plain-var-K Lagrangian isolation (RUNNING, see §2)
e294/e295 (cartpole/acrobot BIG, e286/e290's exact recipe ported to two untested tasks)
launched 07-22; original `gpu-a100` submissions (jobs 4668645/4668646) never dispatched
and were cancelled (`sacct`: 0:00 elapsed, no node assigned), resubmitted same-day to
`gpu-v100` as jobs 4668658/4668659. e296–e303 (`plain_vark` + `DUR_MODE=lagrangian`,
τ∈{4,8} × {hopper,acrobot,cartpole,cheetah}, isolating the duration-Lagrangian mechanism
alone with no masking/struct/reuse) launched 07-22 on `gpu-a100`. All 10 cells RUNNING as
of 2026-07-23, 28–57% through their 4M-step budgets — see §2 for current trailing scores
and per-cell status; not final.

---

## 6. Pre-registered hypotheses & readout logic (live board, written 2026-07-13 pre-results)

Reference values in §1. Read mediators before scores: `wkr_goal_rew` (countdown cells),
`struct_corr`/`rec_mean` (struct cells), `mgr_extr_adv` (collapse).

**e178 — plain var-K τ8, hopper.** H: short commitment was the primary var-K killer.
Expected if fully true: rewrate >1e-2 by ~1–1.5M, score ≥100 by 2.5M.
→ Alive & scores: τ4 default retired off-cartpole; var-K machinery exonerated.
→ Plateaus 1e-3–1e-2 without score: commitment necessary not sufficient → residual is
horizon observability (e185) or duration-head variance.
→ Dies to 1e-4: pulse was noise; var-K machinery itself indicted.
→ **INTERIM (07-14, 3.2M/80%):** rewrate noisy 0–5e-3 the whole run, never sustains above
1e-2; score ≈0. **"Plateaus without score" branch fired** — necessary-not-sufficient.
Corrects the 07-13 "rising" read: it did not keep rising over the next 2.5M steps.

**e179 — plain var-K τ8, cartpole (stability sentinel).** H: e167's collapse was unanchored
code drift, not τ4 — so τ8 alone does NOT prevent it.
→ Collapse (adv→0 + wkr dip + rec climb): struct/anchor REQUIRED for BIG var-K → carry
struct(-adapt) in all var-K runs; e46-vs-e166 resolves toward struct.
→ Stable ≥600 @4M: τ8 suffices without struct on dense; struct demoted.
→ Flat/low without collapse event: τ8 costs cartpole score (echoes e58) → adaptive duration.
→ **INTERIM (07-14, 3.25M/81%):** 740/755, stable since 1.8M, no collapse. **"Stable ≥600"
branch fired** with margin before 4M — struct demoted at τ8, at BIG scale. Note e194 (same
recipe, small scale) is stuck at 152 — struct's dispensability at τ8 looks BIG-scale-only
(see e193/e194 below).

**e180 — pure Director, acrobot.** Already learning (0.125 rate @0.6M). Its plateau becomes
the acrobot denominator. If it dies late: acrobot marginal even for the base; e192 weakens
to "no worse than baseline".
→ **INTERIM (07-14, 3.34M/83%):** rewrate rose 0→0.19 over the run, still climbing; score
177/262. Base is genuinely alive on acrobot — sharpens e192's failure into a real
HRL-specific problem rather than a shared task-difficulty artifact.

**e185 — e178 + countdown, hopper (flagship).** H: unobservable random deadline is a second
independent killer. Expected: wkr_goal_rew persistently above e178's; rewrate ≥3× e178 @1M.
→ e185 ≫ e178: countdown default in all var-K; sparse recipe = τ8 + countdown ± struct (e189).
→ e185 ≈ e178 (both weak): horizon observability not binding on hopper → pivot to
manager-side (non-local proposals, worker task-reward mixing).
→ e185 < e178: countdown input harmful → try duration-at-switch conditioning instead.
→ **INTERIM (07-14, 2.55M/64%):** wkr_goal_rew 0.21 vs e178's 0.16 (both collapse-zone, not
"persistently above"); rewrate flat ~0, score ≈0. **"≈e178, both weak" branch fired** —
horizon observability is not hopper's binding constraint; pivot manager-side per the global
rule. Side note: duration std jumped to 6.6 (vs e178's 1.9) — countdown may be destabilizing
the duration head, worth a second look before calling it purely inert.

**e186 — combined + countdown, cartpole small (vs e170 668/786).** Even a null score with
wkr_goal_rew ↑ supports adoption. If wkr_goal_rew unchanged: masking keeps goals near so
deadlines rarely bind → restrict flag to plain var-K.
→ **INTERIM (07-14, 2.08M/52%):** 695/766 ≈ e170's 653/785 at matched training fraction — no
score win. wkr_goal_rew 0.57, well above e170's level and the healthy band. **"Null score,
wkr_goal_rew↑" branch fired** — keep countdown in the combined recipe; it measurably
improves worker reliability even though masking already keeps deadlines from binding here.

**e187 — plain var-K τ4 + countdown, cartpole small (vs e166 130).** Sharpest single test —
short+random windows, no struct. NOTE e166's worker reward was already healthy (0.46), so
failure here implicates the manager side at small scale, redirecting to struct cells.
→ ≥300 by 2M: worker confusion is a primary var-K cost.
→ Flat ≈130: not binding at small scale. Judge on pre-2M slope; tail may collapse without
struct either way.
→ **INTERIM (07-14, 2.11M/53%):** 697/757, stable since 0.7M, no collapse. **"≥300" branch
fired by a wide margin** — 5.4× e166's score off the byte-identical config. Worker confusion
is a real, primary var-K cost at small scale; candidate for the default recipe pending a
BIG-scale replicate. (Also revises the "worker reward already healthy" premise: a healthy
per-step cosine average coexisted with a policy confused about *when* the goal would change.)

**e188 — mask-only K8 + countdown (vs e162 272/575).** H: stationary window stats → small
effect. If ≥400 sustained: even fixed-K Director hides load-bearing phase info — standalone
finding; countdown default everywhere.
→ **INTERIM (07-14, 2.13M/53%):** 254/347, plateaued since 950k. Below the ≥400 bar and not
yet clearly ahead of e162 (272/565) at matched fraction. **Default "small effect" branch**,
provisionally — e162 itself needed the full run to reach 565, so recheck near 4M.

**e189 — plain var-K τ8 + struct200, hopper (struct's two faces, decisive).**
→ Pulse killed (→1e-4) with struct_corr ≈0.97: locality face dominates sparse tasks →
struct must be gated/scheduled/adaptive, and the combined recipe's sparse deaths are
mechanistically explained (masks + struct both localize).
→ Pulse survives/strengthens: struct is primarily a stabilizer → include everywhere;
locality story carried by masks alone.
→ Unchanged: struct orthogonal on hopper.
→ **INTERIM (07-14, 2.51M/63%):** struct_corr 0.935 (close to the ≈0.97 predicted band);
rewrate dead — actually weaker than e178's own noisy trace. **"Pulse killed" branch fired**:
struct's locality face dominates and actively suppresses the weak τ8 signal. Gate/schedule
struct off for sparse tasks; keep it on for dense combined cells (e160/e179 below — not a
contradiction, F11's two faces).

**e190/e191 — Director controls, cartpole/cheetah.** Define the paper's denominators.
e190 ≈800 & combined ≈ it at blk/step 0.6–1.1 → "sparser at no dense cost" claim holds;
e190 ≥900 → real deficit, report it. e191 ~500 vs e175's 465/508 → combined ≈ base on
cheetah; ≫550 → gap. Either control badly under published Director → reproduction caveat
on ALL comparisons.
→ **INTERIM (07-14):** e190 (2.52M/63%) at 714/747, still climbing (reward rate flat
0.63–0.66 since 1.1M, but score still ticking up) — already near e171's 776 with 1.5M steps
left; too early to call ≈800 vs ≥900, recheck near 4M. e191 (2.55M/64%) is non-monotonic:
peaked 471–500 by 1.4–2M, dropped to 225 @2.27M, partial recovery to 293/499. Peak10/15
(~500) sits at the "≈500, combined≈base" boundary, but the control's own instability means
any e175/e191 or e198/e191 comparison should cite e191's peak window, not last15 — on
last15 alone e191 (293) already reads below e175 (463), which would overstate the win.

**e192 — plain var-K τ8 + countdown, acrobot.** H: sparse-task fix generalizes.
→ Alive & tracks e180 (with lag): revised recipe ready for 4-task × ≥2-seed evaluation.
→ Dead while e180 climbs and hopper cells pulse: acrobot has an additional failure mode
(impulsive timing?) — split tasks in the analysis.
→ **INTERIM (07-14, 2.48M/62%):** rewrate pure noise 0–5e-3, no trend, while e180 rose
0.08→0.20 over the identical window; score 8.1/19.4. **"Dead while e180 climbs" branch
fired** — confirms acrobot needs its own diagnosis thread (the "impulsive timing" guess is
worth prioritizing over porting more hopper-cluster fixes here).

**e193/e194 — cartpole small 2×2 completion** (with e46 τ8+struct=724, e166 τ4+none=130):
e193 high & e194 low → struct load-bearing on dense var-K; reverse → duration; both high →
either suffices (drop struct where locality risks); both low → superadditive fragility
(e46 was the unique corner) → adaptive controllers matter more. e194-vs-e179 is also a
scale check (small worker 0.46 vs BIG 0.15 goal-reward).
→ **INTERIM (07-14, ~2.1M/52–53%):** e193 (τ4+struct200) 42/203 — peaked early (240k) and
never returned. e194 (τ8+struct0) 152/285 — no collapse but no climb toward 700+ either.
**"Both low" branch fired (provisionally)** — e46's 724 looks like a unique corner needing
struct **and** τ8 together, neither alone suffices at small scale. The e194-vs-e179 scale
check also resolved: e179 (τ8, struct0, BIG) is stable at 740 while e194 (same recipe,
small) is stuck at 152 — struct's dispensability at τ8 appears to be a BIG-scale-only
effect, not general.

**e195 — struct-adapt pilot (τ4, init 200 → loss target 0.01).** Watch the multiplier:
settles interior & ≥e193 → adopt adaptive struct, promote to symmetric corr-target
Lagrangian; rails at max → target too tight; →0 with collapse → loss-target is the wrong
form, use a corr floor (inequality constraint).
→ **INTERIM (07-14, 2.09M/52%):** struct_corr 0.955, not railed; score 175/352, beats e193
(42/203) on both last15 and peak. **"Settles interior, ≥e193" branch fired** — a real if
modest edge for adaptive over static struct at τ4 — but still well short of e46's 724, so
not yet strong enough on its own to promote to the symmetric Lagrangian without more steps.

**e196–e199 — full stack (combined + countdown + struct-adapt), 4 small tasks.**
H: the all-adaptive stack is the best transferable recipe. e196 vs e170/e186 isolates the
stack's cartpole cost/benefit; e198 vs e174 (166) is the meaningful transfer cell; e197
(hopper small — never learned at small scale under ANY recipe incl. pure Director-unknown)
and e199 are long shots included for coverage.
→ If e196 ≈ e170 and e198 > e174: the full stack is safe + transfers → becomes the default
small-scale recipe pending BIG confirmation.
→ If e196 ≪ e170: some interaction among the three additions hurts — factorize before
adopting (e186 and e195 identify which).
→ **INTERIM (07-14):** e196 (2.01M/50%) 648/683 ≈ e170/e186 — cartpole cost looks
acceptable so far (dipped to 152 @1.12M, self-recovered). e198 (2.31M/58%) 100/122, flat
since 250k, **below its own e174 comparator (165)** — transfer fails, not just falls short.
e197 (2.0M/50%) 0.23/3.1 and e199 (2.36M/59%) 3.4/22 — both dead as expected. **"e196≪e170"
side not clean (e196 is fine) but e198 fails outright** → per the pre-registered logic,
factorize via e186 (countdown alone, also ≈e170) and e195 (struct-adapt alone, partial win)
before concluding which addition is responsible for the cheetah failure; mask_frac is
inflated in both failing full-stack cells (e197 1.07, e198 1.17 blk/step) but *not* in e199
(0.60) — see the hyperparameter table note above.

**e160 — combined, struct0, cartpole BIG.** H: masks already localize, so struct is
redundant inside the full recipe on dense tasks. ≈e171 → drop struct from the dense
combined recipe; ≪e171 or collapse → struct load-bearing even under masks.
→ **INTERIM (07-14, 2.19M/55%):** near-zero through 1.5M → real rise to 138 @1.95M →
collapsed to 22 @2.19M; mask_frac drifted 0.30→0.51 (blk/step 1.04, above Director's own
dense baseline) and mgr_extr_adv≈0. **"Collapse" branch fired** — struct is load-bearing
even under masking, not just for plain var-K as F11 currently states; extend F11 rather
than open a new finding once this run lands. Caution on the mask_frac-inflation reading,
though: e171/e175 (campaign champions) also finish at blk/step 1.05–1.13, so high *final*
mask_frac alone isn't diagnostic — what marks e160 (and e197/e198) is elevated mask_frac
concurrent with still-low score mid-run, not drift after success.

**Global rule:** first alive sparse cell defines the revised recipe → rerun 4 tasks × ≥2
seeds against e190/e191 before any claim enters the paper as a result. If the whole board
fails on sparse tasks: next interventions are code-level — worker task-reward mixing, then
periodic non-local goal proposals (mask-free decisions every Nth switch), in that order.

---

## 7. Technical notes (implementation facts that bit us)

- **Codebase consolidation (2026-07-28).** `agent.py` had grown to 4535 lines
  carrying every mechanism ever A/B'd. Cut down to the paths that are production
  defaults, then split into a `dreamerv3/hrl/` package.
  **Removed** (all previously-superseded arms): the entire explicit goal-mask head
  (`use_masked_goals`, `mask_topk`, `mask_sparsemax`, `mask_joint_edit`,
  `mask_perblock_credit`, every `mask_sparsity_mode`, `mask_actent`/`mask_kl`),
  the priced-edit costs (`goal_edit_cost*`, `perblock_edit_cost`),
  `impl_sparsity_mode` (F18: catastrophic), `goal_delta_mode`, and
  `goal_reuse_weight`/`goal_reuse_adapt` (F19/F20 losing families); the
  full-resolution variable-K credit path and the `mgr_decision_mean_rescale=False`
  control; `goal_duration_adapt`.
  **Kept**: implicit sparsity via `goal_soft_reuse_adapt` (+ `goal_struct_adapt`),
  block-pooled variable-K credit (now unconditional under `variable_goal_length`,
  the flag is gone), fixed-K Director as the baseline, and
  `goal_duration_{reg,lagrange,fixed}`.
  **Structure**: `agent.py` 4535 -> ~1.5k lines (`Agent` = network wiring,
  `policy`/`train`/`loss`); helpers now in `dreamerv3.hrl.{tensors,heads,losses,
  video}` (pure functions) and `dreamerv3.hrl.{goals,manager,reporting}` (mixins).
  `configs.yaml` lost 65 dead keys plus the `masked_goals`/`hrl_auto` blocks.
  Behavior on the retained paths is unchanged: the 191 surviving agent-side tests
  pass identically, and a fixed-seed CPU smoke of the four production
  configurations reproduces the pre-refactor `metrics.jsonl` losses.
  *Gotcha found doing this*: two helpers (`goal_reward_cosine_max`,
  `pairwise_cosmax`) fell between extraction ranges and vanished silently --
  `pyflakes` cannot see across modules, so only the test import caught it.
  `embodied/tests/test_hrl_package.py` now guards submodule imports, `__all__`
  resolution, and mixin-method reachability.

- **Variable-K repval (fixed 06-18, `9716a78`).** Replay manager-value loss mirrors
  imagination (per-step rewards on full timeline via `_switch_mask_from_skills`), not
  fixed-K downsampling. Correct but lowered var-K peaks.
- **Block-rew bugs (fixed 06-19).** `variable_segment_ids` off-by-one +
  `aggregate_mgr_cont_variable` segment_max→segment_min. Benign on cartpole, wrong for
  terminating tasks; e44 predates the fix. Tests: `embodied/tests/test_variable_goals.py`
  — run on a V100, not the login node (glibc/Py3.11).
- **Duration-reg magnitude domination.** Fixed `reg·(E[dur]−τ)²` inside `mgr_policy` at
  reg=0.1 swamps the normalized REINFORCE under one grad-clipped optimizer. reg ≤0.03 OK.
- **Terminal flags derived from a continuation PROBABILITY (fixed 07-28) — the big
  one.** Block-pooled replay built its terminal flags as
  `term_down = (1.0 - repl_mgr_cont).astype(term.dtype)`. `is_terminal` is an
  `elements.Space(bool)`, and `repl_mgr_cont` is a *probability* (~0.997), so this is
  `(1.0 - 0.997).astype(bool)` → **`True` at essentially every manager decision**.
  `lambda_return` uses `live = 1 - term`, so **every bootstrap term was zeroed** and the
  manager's replay-side value target collapsed to the immediate pooled block reward:
  measured **7.22 → 0.30 (24×)** on a normal non-terminating window. Fixed-K uses the
  real flags (`term[:, idx_down]`) and non-block var-K uses `term` directly — this was
  exclusive to block pooling, i.e. to every `variable_goal_block_rew` run ever
  (e44, e62, e326–e359).
  **Signature in the logs** (e351 pinned vs e352 same-code Director, 1.4–2.0M):
  `mgr_extr_ret` 1.59 vs 16.72, `mgr_extr_val` 1.44 vs 16.88, `mgr_extr_adv`
  **+1.40 vs −0.26** (chronically positive — a critic trained on de-bootstrapped
  targets can never catch imagination's properly-bootstrapped returns),
  `opt/ac_grad_norm` 2.44 vs 0.46.
  **This is the mechanism behind the project-wide "hopper is 0 in every var-K run"
  pattern (F12 / hopper-goal-locality).** Sparse-reward tasks carry nearly all of their
  signal in the bootstrap, so deleting it is fatal; dense cheetah still had the immediate
  block reward and reached 126 vs Director's 337. Fix mirrors `last_down`: downsample the
  real flags at switch positions, then `patch_trailing_replay_state`.
  Tests: `test_bool_cast_of_a_continuation_complement_reads_as_terminal`,
  `test_replay_terminal_flags_are_not_derived_from_continuation`,
  `test_downsampled_terminal_flags_round_trip_through_the_bool_cast`.
  **Method note:** no shape or value check could catch this — the numbers stayed finite
  and plausible and JAX raised nothing. It was a *dtype* crossing. It surfaced from
  fuzzing the pooling helpers against a longhand numpy reference
  (`embodied/tests/test_block_pooled_stepwise.py`) plus ranking every shared metric
  between the pinned run and a same-code Director baseline by divergence.
- **Padded decision axis silently down-weighted the whole manager (fixed 07-27).** THE
  block-pooled var-K bug — affects every `variable_goal_block_rew` run to date (e326–e349
  and earlier), and in weaker form every var-K run. `downsample_at_switch_mask` packs a
  data-dependent number of decisions into a **static, full-width** buffer and forward-fills
  the tail, so at BIG scale (H=16, K=8) the manager tensors are **17 columns wide with 2
  real decisions**. `Agent.loss` then reduces every manager loss with `v.mean(1)` — over
  the buffer width, not the decision count — so `mgr_policy`, `mgr_extr_value` and
  `mgr_expl_value` came out **exactly 8× smaller** than fixed-K Director's for a
  bit-identical rollout (measured, `test_block_pooled_manager_losses_match_fixed_k`). With
  fixed loss scales that is an 8× cut to the manager's effective learning rate against the
  world model and the worker; under true variable K the factor drifts with the realized
  hold length. Same bug on the replay side (9 real decisions in a 63-wide buffer at
  `batch_length=64`). Fix: `decision_mean_rescale` multiplies the masked weights by
  `n_cols / n_valid` per row, so the caller's `.mean(1)` is a per-decision mean; no-op
  under fixed K.
- **Padded slots contaminated the manager's running statistics (fixed 07-27).** Companion
  to the above: `mgr_retnorm` (`meanstd`), `valnorm`, `advnorm` and the adaptive
  entropy `AutoAdapt`s were handed the *whole* padded tensor with no mask, so ~13 of 16
  columns were repeats of one decision — shrinking the return spread the advantage is
  divided by (advantages inflated) and making the entropy controller track a single
  decision. `Normalize`/`AutoAdapt` now take an optional `weights` argument (weighted
  meanstd; `nanpercentile` for `perc`), and `imag_loss_mgr` passes the decision mask.
  Logged manager metrics (`mgr_adv`, `mgr_*_ret`, `mgr_ent/*`, `mgr_*_rew`) are now
  decision-weighted too, so **var-K manager metrics before/after 07-27 are not comparable**.
- **Empty trailing decision was trained (fixed 07-27).** `mgr_switch` counted *all*
  switches, including one landing on the rollout's final step. Rewards are indexed
  `rew[:, 1:]`, so that decision pools an empty segment: fixed-K keeps that column purely
  as a bootstrap anchor and drops it with `[:, :-1]`. `variable_block_director_tensors`
  now counts `switch_mask[:, :-1]`.
- **Worker credit used a different algorithm under a pinned hold (fixed 07-27).** The
  Director `split_traj` windowed worker loss was gated on `not variable_goal_length`, so
  the `goal_duration_fixed=8` *control* — whose rollout switches on exactly the fixed-K
  grid — silently trained its worker with the dense fallback instead (boundary state
  conditioned on the NEW goal, one lambda-return with resets, different per-window
  weighting). New `worker_split_window` selects the windowed path whenever the hold is
  statically regular (fixed K, or `goal_duration_fixed>0`); genuinely variable holds still
  use the dense path.
- **Pinned duration head still trained (fixed 07-27, completing the 07-27 REINFORCE fix).**
  `manager_reinforce_policy` removed 'duration' from the REINFORCE sum under
  `goal_duration_fixed`, but the adaptive duration-entropy regularizer and the
  `goal_duration_reg` prior on E[dur] kept pushing the head — and the shared trunk —
  with signals that cannot affect the trajectory and that Director (no duration head at
  all) never carries. Both are now skipped when the hold is pinned; the manager loss is
  then bit-identical to a skill-only manager's.
  Tests for all five: `embodied/tests/test_block_pooled_equivalence.py` (Director-
  equivalence differential suite; run on a compute node, not the login node).
- **Packed-tensor recency bias (fixed 07-10).** Under var-K, `forward_fill_packed` made
  every unweighted mean over decision slots ~75% weighted to the LAST imagined decision —
  affected all mask-sparsity losses/metrics up to e159. Now valid-slot-weighted; pre/post
  mask stats are not comparable.
- **One-sided duration-Lagrangian (fixed 07-10).** First version regulated raw mean
  duration with inverse=False → railed at max, acted as a stiff fixed prior. Now regulates
  switch-weighted |E[dur]−τ| against `goal_duration_lagrange_tol` (0.1), symmetric,
  self-relaxing (e171: multiplier → 0.0 with K still pinned at 4.0).
- **Worker countdown implementation (07-13, `worker_timed_goals`).** Normalized
  steps-left appended to worker policy+value inputs at ALL sites: imagination scan (new
  per-step output), both imagination-loss paths, replay value (`_manager_skills_on_sequence`
  outputs a `countdown` key — pop it before tree-ops on skills), real acting (from
  `mgr_step` carry), bootstrap action. Report/viz paths default to full budget. Flag off =
  byte-identical inputs (regression-smoked).
- **Where to run tests (07-27).** The account's GPU quota is shared, so while a full
  8-job A100 batch is in flight a `-p gpu-v100`/`gpu-a100` test job pends forever on
  `AssocGrpGRES`/`AssocGrpCpuLimit`. The **`intel` CPU partition has no GrpTRES limit for
  us and its nodes are el8**, so `dreamerv3_env` imports there and the JAX suites run on
  CPU in <1 min: `sbatch/run_pytest_cpu.sbatch`, or
  `srun -p intel -c 8 --mem=24G -t 0:20:00 bash -lc '...'`. The login node still cannot
  (glibc too old).
- **64×64 report segfault (e5).** Report-compile crashes at 64×64 regardless of conv impl.
  Use `--agent.report False` or lighten report.
- **`mask_sparsity_mode=none` bring-up.** Loss/scale key-set assert requires popping the
  scale for `none` (`agent.py:761`); same class of assert applies to any new loss key.
- **Requeue safety (07-13).** All prior_vargoal templates + the Director-baseline script:
  job-id-keyed RUN_DIR fallback (requeues keep the id → same dir → checkpoint resume),
  `--requeue`, `--open-mode=append`, USR1-trap self-requeue before walltime (training
  backgrounded + `wait` so the trap fires). Motivation: e177 lost 19h to a fresh-mktemp
  requeue after being PREEMPTED on plain gpu-a100 (short-a100 preempts it in practice);
  e160/e161 stalled on short-a100's terminal 2h TIMEOUT (only preemption auto-requeues).
- **Walltime.** Small V100 template now 48h (24h truncated every 4M run at ~2.8M).
- **wandb silently off** if a script activates a venv directly without sourcing
  `source_dreamerv3_env.sh` (`WANDB_API_KEY` unset → jsonl-only): bit e119–e122.
- **Native cuDNN conv** (`unset DREAMERV3_CONV_IMPL`): ~2× faster; A100 always OK; V100
  only via the Route-A clone env (jax 0.4.33 + cuDNN 9.1.0.70 + CUDA 12.2).
- **P100 partition:** 8-GPU quota; only `saion-gpu[11-14]` are el8 — always pin
  `--nodelist=saion-gpu[11-14]` for Py3.11 or segfault at import.
- **V100 16GB OOM** at director_match batch16×64 on 4×V100 (e120/e121): shrink to
  batch 12 / imag 12 or use A100-80GB; device 0 is double-loaded (policy+train).
- **Struct-adapt retargeted, `AutoAdapt` gained an optional one-sided mode
  (07-14, three revisions same day; settled on dual-sided).** e195–e199 (and any
  earlier struct-adapt pilot) ran under the *original* mechanism: `AutoAdapt`
  regulated the raw struct MSE/margin loss toward 0.01, two-sided (grew above,
  shrank below) — diagnosed via e198 vs e174: the multiplier collapsed from
  init 200 to ~20–30 within the first ~280k steps because raw loss cleared 0.01
  almost immediately, leaving `struct_corr` stuck at 0.91–0.94 for the rest of
  training vs. e174's fixed-200 run reaching 0.966–0.971. A first fix
  retargeted onto `struct_corr` directly (target 0.97); pulling e171/e175's own
  trajectories under a *fixed*, never-throttled weight of 200 showed BIG-scale
  runs never clear 0.97 at all — they plateau at 0.90–0.92 (cartpole, e171) /
  0.94 (cheetah, e175), and MSE actually drifts back *up* late in training
  (e171: 0.0068 @445k → 0.0183 @3.1M) even at fixed weight, vs. small-scale
  runs reaching 0.0032–0.0048. A 0.97 corr target would have ratcheted the
  BIG-scale multiplier to `goal_struct_adapt_max=2000` and pinned it there for
  the whole run. Settled: regulate on the raw struct loss again (as
  originally), target lowered to **0.005** (below every historical run's floor,
  both scales), **two-sided/dual-ascent** (`goal_struct_adapt_one_sided: False`,
  the default). `AutoAdapt` (`embodied/jax/utils.py`) permanently gained the
  `one_sided` param as an opt-in for future use (drops the shrink branch, so
  the scale only ever grows to correct a violation and holds — never relaxes —
  once cleared; relevant if a target is later found to be re-violated after
  clearing, as e171's late-training MSE drift suggests could happen here too).
  Whether 0.005 is reachable at either scale without pinning at the weight
  ceiling — and, if reachable, whether dual-ascent shrinks the weight back down
  early and then lags on the later re-violation the way it did at 0.01 — is now
  an open empirical question for the next struct-adapt run to answer, not an
  assumption. Old struct-adapt runs (e195–e199) are **not** directly comparable
  to any struct-adapt run launched after this date — re-run before drawing
  conclusions about the adaptive-vs-static struct comparison.
- **mask_viz "changed" overlay was masked-goals-only; plain-Director +
  implicit-sparsity runs (e234–e249) never rendered white (kept) blocks
  (fixed 2026-07-16).** `_manager_skill_step`'s sticky `last_change_mask` — the
  carry the mask_viz panel reads for its yellow-vs-white overlay (§2) — was only
  populated `if self.use_masked_goals`; the plain-Director path hard-coded
  `edit = ones(...)` ("every block treated as changed"), a holdover from before
  `mgr_cond_goalcode` gave plain Director the ability to deliberately reuse a
  block. Fixed by tracking `last_change_mask` (and the `prev_code`/`base_code`
  reference it needs) unconditionally for any HRL run, masked or plain —
  `argmax(new code) != argmax(previous code)`, refreshed only on a manager
  switch, cleared at episode reset. `agent.py`: `init_policy`/`_unpack_carry`
  (unconditional carry alloc), `_manager_skill_step` (unified `prev_code`
  capture + change computation), `policy()`'s panel builder (drops the
  `use_masked_goals` branch, always reads `last_change_mask`). Byte-identical
  for masked-goals runs; changes ONLY the rendered overlay for plain-Director
  runs (no effect on training, since mask_viz is a `log/`-only diagnostic).
- **`goal_delta_mode`: differentiable "reuse" for plain-Director goal
  generation (2026-07-16, F18 follow-up).** F18 found the implicit-sparsity
  REINFORCE controller catastrophic because "keep this block" had no
  differentiable representation — reuse was only ever a coincidence, measured
  post-hoc via a stop-gradiented `argmax==argmax` comparison. `goal_delta_mode`
  (opt-in, mutually exclusive with `use_masked_goals`) changes *how the goal
  code is sampled*, per block (all `L=8` blocks handled independently — the
  combine step below is purely elementwise over `(..., L, C)`, no cross-block
  coupling; only the shared network trunk producing the logits can correlate
  blocks): the manager's per-class logits pass through an **unnormalized,
  independent sigmoid** ("votes", NOT softmax — the point is that all classes
  can be driven to ~0 simultaneously, which a normalized distribution can
  never do) that gets **added** to the previous goal code's distribution
  before the categorical sample (`agent._delta_combine_skill`). All-~0 votes
  for a block ⇒ its combined distribution is numerically unchanged from
  before ⇒ the same class stays most likely — an explicit, gradient-carrying
  default, as opposed to actively re-voting the same class (which still
  shifts its margin and so still carries gradient, keeping "lazy default" and
  "active re-confirmation" distinguishable to any downstream loss).
  `goal_delta_clip` (default 1.0, a no-op at sigmoid's natural ceiling) caps a
  single vote before the add. Implies `mgr_cond_goalcode=True` (forced in
  `__init__`) — the manager needs to see what it's adding to. Applied
  consistently at all 9 `_emit_manager` call sites (online policy, both
  imagination-rollout paths, replay re-derivation) via a new `prev_mgr_skill`
  parameter, AND re-derived identically at the actor-loss reconstruction site
  (`agent.py` ~2456) so REINFORCE logp/entropy are computed under the
  distribution that was actually sampled from, not the raw pre-combination
  network output — this second site is easy to miss and was the trickiest part
  to get right. Also carries `minent`/`maxent` over from the original head
  (needed by the adaptive entropy regularizer; a freshly-built `OneHot` has
  neither).
  **Refined same day: the additive base is the previous decision's real
  pre-sample probabilities (`skill_probs`), not its collapsed one-hot.** A
  one-hot is a lossy view of the previous decision — a 55%-confident pick and
  a 99%-confident pick both collapse to an identical one-hot once sampled —
  so basing "how much to trust the standing choice" on the one-hot silently
  maxes out trust at 1.0 regardless of how contested the decision actually
  was, capping every single-vote switch at a coin-flip tie no matter what.
  Using the real `skill_probs` instead makes a block's stickiness scale with
  its actual prior confidence: a decisively-won class (prob ≈1.0) still caps
  a fresh vote at a tie (unchanged from before), but a contested class (e.g.
  0.55/0.45) can be outright overtaken by one strong vote on the alternative
  (0.45+~1.0 clearly beats 0.55+~0) — a genuinely decisive switch, not just a
  coin flip. Falls back to the one-hot when no `skill_probs` exists yet
  (synthetic seed dicts at episode start / report-time proposals, which
  aren't a real manager decision — treating those as a firm commitment is the
  right default). This reuses the SAME `skill_probs` carry field introduced
  for the input-conditioning follow-up below, now doing double duty as both
  the manager's next-step input AND the next decision's combination base.
  The manager's own INPUT conditioning channel (`mgr_cond_goalcode` in
  `_mgr_input`) was switched, under delta mode, from the collapsed one-hot to
  this same soft pre-sample distribution (new carry field `skill_probs`,
  threaded through `_emit_manager`'s result and `skill_switch` like any other
  skill field) — strictly richer (the one-hot is a lossy argmax of it), and
  additionally tells the manager how *contested* its last decision was, not
  just which class won.
  **Smoke-tested on cartpole, PASSED** (`run_smoke_goal_delta.sbatch`, job
  4664845 — final, `skill_probs`-based-combination version; two earlier
  submissions superseded: job 4664816 failed on a `skill_switch`
  dict-key-mismatch from reading `agent.py` mid-edit, not a real bug; job
  4664824 validated the pre-refinement one-hot-based combination). Three
  legs, all COMPLETED (exit 0): baseline (delta off, byte-identical-path
  check), delta-on at clip 1.0, delta-on at clip 0.5 — all trained ~1000
  steps crash-free through the online policy path, both imagination-rollout
  paths, AND the actor-loss re-derivation site, with zero NaN/Inf across
  every logged scalar. `goal/implicit_sparsity_block` ≈0.13 in both delta
  legs, close to baseline's ≈0.13 (vs. ≈0.30 under the pre-refinement
  one-hot-based combination) — expected at random init, since `skill_probs`
  starts near-uniform (1/8) rather than a hard one-hot, so early behavior
  resembles a fresh decision rather than an artificially sticky one; not yet
  informative about the mechanism's effect once trained. Not yet run as a
  real experiment — next step is an e-numbered A/B against the existing
  plain-Director + `impl_sparsity_mode` cells (e234–e249).
- **`goal_reuse_adapt`/`goal_reuse_weight`: differentiable CONTINUOUS-goal-space
  reuse loss (2026-07-16, targets the thing that actually matters instead of a
  proxy for it; refined same day — decoder gradient blocked, made adaptive).**
  Both F18's original REINFORCE controller and `goal_delta_mode` above operate
  at the goal-CODE level (block class identity). But code-level reuse is only
  a proxy: an unchanged code doesn't guarantee an unchanged DECODED goal
  (`goal_dec` need not be locally smooth there), and a changed code doesn't
  guarantee a changed one — while worker success is entirely a function of the
  DECODED goal (`goal_reward_cosine_max(goal_deter, feat)`), never the code
  directly. `goal_struct_weight` already tries to address this, but
  indirectly and globally (forces code-space distances to correlate with
  deter-space distances across the whole manifold); `goal_reuse` targets the
  SPECIFIC quantity that matters — this decision's decoded goal vs. the
  previous one — directly, every decision, with a real gradient path.
  Mechanism: `goal/implicit_sparsity_cont` already computes almost exactly
  this similarity as a *metric*, but is fed from `goals`, which is
  deliberately stop-gradiented twice over (`sg(self._goals_from_skills(
  jax.tree.map(sg, mgr_skills), bdims=2))`) so worker-target conditioning
  never leaks gradient into the manager — same disease as F18's root cause, a
  real signal with no gradient path. Fix: decode the sampled code
  (`post_code_impl = self._running_goal_code(mgr_skills)`, which — unlike
  `goals` — was never separately stop-gradiented, so it still carries the
  straight-through gradient into the manager's logits, regardless of which
  goal-generation mechanism produced it) a SECOND time, through
  `self._decode_goal_no_decoder_grad(post_code_impl, 2)`.
  **Decoder-gradient block (refined 07-16):** the fresh decode initially used
  a plain `self.goal_dec(...)` call, which — since `MultiOptimizer` does one
  combined backward pass and routes gradient to each module's own optimizer
  purely by which params the forward computation touched, with NO other
  isolation — meant `goal_dec`'s own weights also got a gradient contribution
  from this manager-shaping loss. Per instruction, this is now blocked:
  `_decode_goal_no_decoder_grad` implements `g(params, x) = f(sg(params), x)`
  by temporarily swapping `goal_dec`'s live parameter values for
  stop-gradiented copies of the SAME values (`.values`/`.write()`, the same
  primitives `embodied.jax.utils.SlowModel` already uses for a different
  purpose), decoding, then restoring — gradient into the decoder's params is
  now exactly zero, gradient into the code (hence the manager) is unchanged.
  **Verified in isolation** (`test_freeze_grad.py` + `run_smoke_freeze_grad.sbatch`,
  job 4664883, a minimal ninjax module outside the full agent): forward value
  matches the unfrozen call exactly, gradient w.r.t. the decoder's parameter
  is exactly `0.0`, gradient w.r.t. the input matches the unfrozen call
  exactly (`36.0` vs `36.0` in the test's toy example) — the three properties
  that together prove the construction does exactly what it's supposed to.
  **Adaptive Lagrangian (refined 07-16):** mirrors `goal_struct_weight`/
  `goal_struct_adapt`'s existing dual-mode pattern. `goal_reuse_weight`
  (fixed, unconditional push toward similarity 1) is kept as a simple
  ablation; the new, recommended mode is `goal_reuse_adapt` (bool) +
  `goal_reuse_target` (target similarity, default 0.9) +
  `goal_reuse_adapt_{init,min,max,vel,one_sided}` — a dual-ascent Lagrange
  multiplier (`embodied.jax.AutoAdapt`, `inverse=True`, the same sense used
  for entropy regularizers: "push up when below target") that grows while
  mean realized similarity sits below `goal_reuse_target` and shrinks while
  above, self-tuning the pressure to HOLD similarity at the target rather
  than driving it unconditionally to 1. The two modes are mutually exclusive
  (raises if both set), matching struct's convention exactly. Loss key
  `goal_reuse` (own scale in `loss_scales`, conditionally registered — off
  leaves the scales/losses key sets byte-identical to before). New metrics:
  `goal/reuse_sim_mean` (realized similarity), `goal/reuse_adapt_scale_mean`
  (the Lagrange multiplier, adapt mode only).
  **Manager input:** `mgr_cond_decgoal` ("concat decoded previous goal") already
  existed but was gated to `use_masked_goals` only (same over-restriction
  `mgr_cond_goalcode` had before the delta-mode fix) — un-gated, and either
  `goal_reuse_weight > 0` or `goal_reuse_adapt` now forces it on (the manager
  needs to see the previous decoded goal to reason about how far it's moving
  it).
  **Smoke-tested on cartpole, PASSED** (`run_smoke_goal_reuse.sbatch`, final
  version job 4664884, superseding the pre-refinement job 4664875): 4 legs,
  all COMPLETED exit 0 — baseline (off, no key leakage), `goal_reuse_weight`
  fixed mode, `goal_reuse_adapt` (target 0.9), and adapt mode combined with
  `goal_delta_mode` (compatibility). All trained ~1000 steps crash-free with
  `goal/reuse_sim_mean` finite and in `[-1,1]`, `goal/reuse_adapt_scale_mean`
  finite and within `[min,max]` in both adapt legs, zero NaN/Inf anywhere.
  Not yet run as a real experiment.
- **`goal_soft_reuse_adapt`: direct (REINFORCE-free) implicit sparsity via a
  block-overlap loss (2026-07-16, third F18 follow-up, see §2 e266–e273).**
  Leaves sampling exactly as plain/direct Director (mutually exclusive with
  `goal_delta_mode`, which recombines the sample itself) and instead: feeds the
  manager `p_{t-1}` (previous decision's own softmax, not the one-hot) as an
  input via the SAME `mgr_cond_goalcode` channel `goal_delta_mode` repurposes
  (`_mgr_input`, now gated on `_mgr_needs_skill_probs = goal_delta_mode or
  goal_soft_reuse_adapt` instead of `goal_delta_mode` alone), and adds an
  ordinary loss on `overlap = mean_blocks(sum_classes p_t * p_{t-1})` (bounded
  [0,1], Cauchy-Schwarz), backpropagating through both steps' own softmax —
  no sampling, no decoder pass, no reward shaping. `_emit_manager` computes
  `skill_probs` as the manager's plain softmax (not a delta-combined
  distribution) whenever this flag is set; the online/imagination carry init
  sites (`init_policy`, `_unpack_carry`, `_imagine_with_manager`) all broadened
  their `skill_probs` zero-init from `goal_delta_mode`-only to
  `_mgr_needs_skill_probs`, so `mgr_skills['skill_probs']` is a genuine
  per-decision stacked field (threaded through the scan like `goal_code`) —
  no onehot fallback needed at the loss-recomputation site, unlike
  `goal_delta_mode`'s own (separately, not fixed here) `preedit_eff`
  construction. Dual-ascent Lagrange multiplier `goal_soft_reuse_adapter`
  (`inverse=True`, same sense as `goal_reuse_adapter`) + `Ratchet`-annealed
  target (`goal_soft_reuse_target_{init,vel}`), mirroring `impl_sparsity`'s own
  ratchet exactly. Loss key `goal_soft_reuse` (own scale). New metrics:
  `goal/soft_reuse_overlap_mean`, `goal/soft_reuse_target_now`,
  `goal/soft_reuse_scale_mean`. **Smoke-tested, PASSED**
  (`run_smoke_goal_soft_reuse.sbatch`, job 4664962): 3 legs (off/regression,
  fixed target, fast-vel ratchet), all crash-free, overlap correctly bounded,
  ratchet monotonic 0→0.7, loss value matches `-scale·overlap` exactly, zero
  NaN/Inf, zero key leakage on the off leg.
- **Block-pooled var-K credit assignment (`variable_goal_block_rew=True`) had two
  unaddressed gaps, both fixed 2026-07-23 (see §2 e304–e311).** This path pools
  reward/continuation per realized hold segment (`aggregate_mgr_extr_rew_variable`
  / `aggregate_mgr_cont_variable`, mirroring fixed-K `abstract_traj`) and had only
  ever been tested with `mgr_reward_agg=mean` and no truncation handling (e44
  pre-bugfix, e62 masked/no-prior — both dead ends, §4).
  1. **`mean` gives zero reward-side incentive for longer productive holds** — a
     4-step and 16-step block earning the same per-step reward pool to an
     identical number. `sum` fixes this (couples pooled reward to how much was
     actually earned); a properly-discounted sum (`Σγⁿrₙ`, not implemented) would
     be more correct still but the flat `sum` in the codebase already closes most
     of the gap in a worked numeric check.
  2. **Truncated holds were mislabeled.** A decision sampling duration `d` but
     only getting `avail < d` real steps before `imag_length` cut it off was
     credited via REINFORCE against the *sampled* class, not the *realized*
     one — biasing against long durations sampled late in a rollout (structural,
     not rare, since every rollout has a fixed length). Fixed by
     `relabel_truncated_last_duration` (agent.py, called from the
     `variable_goal_block_rew` branch of the manager-loss body): only the final
     real decision in a rollout can be truncated (every earlier one is followed
     by a genuine switch, so it always ran its full sampled duration); its
     `duration` class gets swapped to `clip(avail, dur_min, dur_max)` before
     `align_skill_events`/`head_logp_time` compute the REINFORCE log-prob.
     Gated by new flag `goal_duration_relabel_truncated` (default `True`).
     Deliberately does **not** touch: the duration Lagrangian prior (already
     reads `E[d]` off the softmax directly, never the sample, so truncation
     doesn't reach it) or the worker's countdown input (relabeling it would
     teach a false "decision imminent" signal that doesn't hold at real acting
     time, since `imag_length` is a training-time artifact, not a task
     property).
  Smoke-tested both settings of the new flag before launch: `ALL SMOKE LEGS
  PASSED` (`run_smoke_variable_goals.sbatch` job 4668864, relabel ON, LEG 4 with
  `goal_duration_max=16=imag_length` to force frequent truncation) and a
  standalone rerun of the OFF leg (job 4668885) — both exit 0, zero NaN/Inf,
  `mgr_duration_mean` tracking τ=8 with healthy spread across all quartiles.

## 8. Config flags & metrics reference

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
