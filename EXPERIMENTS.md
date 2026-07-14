# DreamerV3 HRL — Experiment Log

Director-style hierarchical RL on DreamerV3, studying whether the manager can act
**sparsely**: editing only part of a persistent goal code per decision (masked edits) and
choosing how long each goal is held (variable durations). Companion working paper:
`paper/main.pdf` (self-contained method + results). Full narrative history of everything
below is preserved verbatim in `EXPERIMENTS_ARCHIVE_20260713.md` (and git); this file is
the restructured, maintained log.

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

## 2. Current state (2026-07-14, interim readout — 18/19 live-board runs still training)

**Dense tasks: solved at Director-comparable cost, and countdown found a second,
simpler route there.** Best cartpole: combined recipe BIG → 833/869 (e157), reproduced
post-fixes at 776/844 (e171), at blk/step 0.6–1.1. Best small: e57 (masked var-K + soft
prior τ_d=4) → best-15 839 / max 854. Best cheetah: combined (Lagrangian controllers)
465/508 (e175) ≈ 2× the fixed-penalty version. New this campaign: plain var-K τ4 (no
masking, no struct) + worker countdown conditioning turns e166's collapsed 130 into a
stable 697/757 (e187) — countdown alone rescues the exact failure struct used to be
needed for, at small scale. Director-matched controls for both dense tasks are running
(e190 714/747 @63%, still climbing; e191 non-monotonic, peak ~500) to pin the denominators.

**Sparse tasks: the open problem, now sharper.** Hopper and acrobot score ~0 under EVERY
restricted variant ever run (20+ hopper cells), while pure Director learns hopper (180/326,
e124) and is climbing on acrobot (197/295 @83%, e180). Cleanest attribution used to be
"variable duration alone kills hopper" (e169 vs e124, τ_d=4, 0 vs ~300) — that still holds,
but the mask-only fixed-K8 pair (e164/e165, 0/4.5) shows masking alone, with the *same*
fixed schedule as the alive baseline, is **independently** also sufficient to kill it. Raising
τ_d to 8 gives a real-but-weak, non-rising pulse (e178: noisy 0–5e-3 through 3.2M, never
sustains above the 1e-2 alive bar). Countdown does not rescue hopper (e185 ≈ e178) or
acrobot (e192 dead while e180 climbs) — it is a dense-task-only fix so far. Struct shows
both faces on live cells: kills the hopper pulse further (e189, struct_corr 0.94) but is
load-bearing for the *dense* combined recipe too, not just plain var-K (e160 collapses
138→22 with struct=0). Remaining live cells: struct-adapt (e195), full stack (e196–e199,
transfers on cartpole but not cheetah so far), and the still-open struct×countdown hopper
cell. See §6 for per-experiment interim readouts and the paper's Outlook for the revised
escalation order (worker task-reward mixing, then non-local goal proposals) if these
also fail.

### Best known configs

```bash
# Combined recipe, BIG (e157/e171-class) — via template:
sbatch -J eNNN_<task>_BIG --export=ALL,EXP_TAG=eNNN,TASK=dmc_<task>,RECIPE=vark_masked,\
DUR_TARGET=4.0,MASK_MODE=prob_entropy,DUR_MODE=lagrangian run_v3_prior_vargoal_big_a100.sbatch
# e57 small-scale champion: same via run_v3_prior_vargoal_small.sbatch with MASK_MODE=prob,
# DUR_MODE=fixed (reg 0.01), DUR_TARGET=4.0.
# Optional new knobs (defaults preserve old behavior): WORKER_TIMED_GOALS=True,
# STRUCT_ADAPT=True, MGR_REWARD_AGG={mean,sum}, MGR_EXPL_W, RECIPE={vark_masked,mask_fixedk,plain_vark}.
```

### Live board (19 runs)

| Exp | Job | Recipe | Task | Scale | Question | Signal as of 2026-07-14 (interim, % of 4M budget) |
|---|---|---|---|---|---|---|
| e160 | 4660123 | combined, struct=0 | cartpole | BIG short-a100 | struct needed inside full recipe? | COLLAPSED: 138→22 @2.19M (55%); mask_frac drifted 0.30→0.51, mgr_extr_adv≈0 |
| e178 | 4660138 | plain var-K τ8 | hopper | BIG | hold length alone | flat 0–5e-3 through 3.2M (80%), score ≈0 — the "rising" read at 0.6M did not hold |
| e179 | 4660139 | plain var-K τ8 | cartpole | BIG | stability without struct at τ8 | 740/755 @3.25M (81%), stable since 1.8M — struct not needed at τ8 |
| e180 | 4660140 | pure Director | acrobot | BIG | task-difficulty control | ALIVE: rate 0→0.19, score 176/262 @3.34M (83%), still climbing |
| e185 | 4660298 | e178 + countdown | hopper | BIG | horizon observability | dead, ≈e178 (wkr 0.21 vs e178's 0.16); dur_std blew up to 6.6 |
| e186 | 4660299 | combined + countdown | cartpole | small | countdown inside best recipe (vs e170: 668/786) | 695/766 @2.08M (52%) ≈ e170; wkr_goal_rew 0.57, above e170's level |
| e187 | 4660300 | plain var-K τ4 + countdown | cartpole | small | sharpest worker-confusion test (vs e166: 130) | 697/757 @2.11M (53%) — 5.4× e166, stable since 0.7M |
| e188 | 4660301 | mask-only K8 + countdown | cartpole | small | phase observability at fixed K (vs e162: 272/575) | 254/347 @2.13M (53%), below e162's 565 peak so far |
| e189 | 4660308 | plain var-K τ8 + struct200 | hopper | BIG | struct: stabilizer or locality killer? | dead, weaker than e178; struct_corr 0.94 — locality face confirmed |
| e190 | 4660309 | pure Director | cartpole | BIG | Director-matched control | 714/747 @2.52M (63%), still climbing toward e171's 776 |
| e191 | 4660310 | pure Director | cheetah | BIG | control redo (e123 stopped 2.6M, peak 425) | non-monotonic: 471 peak@1.4M → dip 225@2.27M → 293/499 @2.55M (64%) |
| e192 | 4660311 | plain var-K τ8 + countdown | acrobot | BIG | best-guess sparse recipe vs e180 | dead (no trend) while e180 climbs — acrobot ≠ hopper failure mode |
| e193 | 4660312 | plain var-K τ4 + struct200 | cartpole | small | duration×struct 2×2 (with e46/e166/e194) | 42/203 @2.09M (52%) — peaked early (240k), stuck low since |
| e194 | 4660313 | plain var-K τ8, struct0 | cartpole | small | 2×2; small mirror of e179 | 152/285 @2.11M (53%) — no collapse, no climb either |
| e195 | 4660317 | plain var-K τ4 + struct-adapt | cartpole | small | struct-as-Lagrangian pilot | 175/352 @2.09M (52%) — beats e193, still ≪ e46's 724 |
| e196 | 4660324 | **full stack**: combined+countdown+struct-adapt | cartpole | small | everything-adaptive cell vs e170/e186 | 648/683 @2.01M (50%) ≈ e170/e186, dipped-and-recovered at 1.1M |
| e197 | 4660325 | full stack | hopper | small | — (small hopper never learned; long shot) | dead, 0.23/3.1 @2.0M (50%) — expected long shot |
| e198 | 4660326 | full stack | cheetah | small (P100) | vs e174 (166) | 100/122 @2.31M (58%) — below e174, flat since 250k |
| e199 | 4660327 | full stack | acrobot | small (P100) | vs e176 (0) | 3.4/22 @2.36M (59%) — still dead vs e176 |

GPU occupancy: A100 8/8 · V100 8/8 · P100 2/8 (pinned `saion-gpu[11-14]`) · short-a100: e160.
Cancelled at 0.6M for flatness (07-13): e181 (agg=sum), e182/e184 (combined τ8 hop/acrobot),
e183 (10× expl), e161 (struct-0 hopper). Detailed hypotheses & readout logic: §6.

### Interim hyperparameter comparison table (2026-07-14, live campaign + controls)

All current-campaign rows (no `†`) are **still running** (50–83% of a 4M-step budget) —
read as a snapshot, not a landed result. `†` rows are finished/archived comparators pulled
in for contrast. `works?`: ✅ alive/on-reference, ❌ dead or collapsed, ~ mixed/partial/still
resolving. `score` = last-15 (peak-15) episode return. `blk/step` = realized
`mask_frac×8/duration` (masking off ⇒ whole code re-written every switch, so it reduces to
`8/duration`; Director's own fixed-K8 baseline = 1.00). `goal length` = `fix 8` (Director's
fixed switch interval) or `var τ{4,8} (reg|lagr)` (soft fixed-prior vs. duration-Lagrangian
control mode). Struct `+adapt` = `goal_struct_adapt` Lagrangian on top of the listed init
weight.

| exp | env | size | works? | score | blk/step | struct | goal mask | goal length | reward agg | wkr countdown |
|---|---|---|---|---|---|---|---|---|---|---|
| e46† | cartpole | small | ✅ | 724 (755) | 1.00 | 200 | off | var τ8 (reg) | mean | no |
| e124† | hopper | BIG | ✅ | 196 (326) | 1.00 | 0 | off | fix 8 | mean | no |
| e162† | cartpole | small | ~ | 270 (565) | 0.31 | 200 | prob | fix 8 | mean | no |
| e163† | cartpole | BIG | ~ | 364 (412) | 0.31 | 200 | prob | fix 8 | mean | no |
| e164† | hopper | small | ❌ | 0.0 (3.7) | 0.30 | 200 | prob | fix 8 | mean | no |
| e165† | hopper | BIG | ❌ | 0.3 (3.9) | 0.31 | 200 | prob | fix 8 | mean | no |
| e166† | cartpole | small | ❌ | 129 (185) | 2.00 | 0 | off | var τ4 (reg) | mean | no |
| e167† | cartpole | BIG | ❌ | 98 (650) | 2.00 | 0 | off | var τ4 (reg) | mean | no |
| e168† | hopper | small | ❌ | 0.3 (3.1) | 2.02 | 0 | off | var τ4 (reg) | mean | no |
| e169† | hopper | BIG | ❌ | 0.0 (2.4) | 2.01 | 0 | off | var τ4 (reg) | mean | no |
| e170† | cartpole | small | ✅ | 653 (785) | 0.61 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e171† | cartpole | BIG | ✅ | 776 (843) | 1.13 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e172† | hopper | small | ❌ | 0.7 (4.3) | 1.03 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e173† | hopper | BIG | ❌ | 0.0 (3.9) | 1.33 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e174† | cheetah | small | ~ | 165 (171) | 1.05 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e175† | cheetah | BIG | ✅ | 463 (502) | 1.05 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e176† | acrobot | small | ❌ | 1.9 (12.3) | 0.54 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e177† | acrobot | BIG | ❌ | 2.5 (25.9) | 0.75 | 200 | prob_entropy | var τ4 (lagr) | mean | no |
| e161† | hopper | BIG | ❌ cancelled | 0.4 (1.5) | 1.17 | 0 | prob_entropy | var τ4 (lagr) | mean | no |
| e181† | hopper | BIG | ❌ cancelled | 0.0 (3.0) | 1.00 | 0 | off | var τ8 (reg) | **sum** | no |
| e182† | hopper | BIG | ❌ cancelled | 0.1 (1.6) | 0.50 | 200 | prob_entropy | var τ8 (lagr) | mean | no |
| e183† | hopper | BIG | ❌ cancelled | 0.0 (0.4) | 0.94 | 200 | prob_entropy | var τ4 (lagr) | mean, expl_w=1.0 (10×) | no |
| e184† | acrobot | BIG | ❌ cancelled | 0.4 (17.6) | 0.30 | 200 | prob_entropy | var τ8 (lagr) | mean | no |
| e160 | cartpole | BIG | ❌ | 22 (182) | 1.04 | 0 | prob_entropy | var τ4 (lagr) | mean | no |
| e178 | hopper | BIG | ❌ | 0.03 (3.1) | 1.00 | 0 | off | var τ8 (reg) | mean | no |
| e179 | cartpole | BIG | ✅ | 740 (755) | 1.00 | 0 | off | var τ8 (reg) | mean | no |
| e180 | acrobot | BIG | ~ (climbing) | 177 (262) | 1.00 | 0 | off | fix 8 | mean | no |
| e185 | hopper | BIG | ❌ | 0.03 (0.9) | 1.00 | 0 | off | var τ8 (reg) | mean | **yes** |
| e186 | cartpole | small | ✅  | 695 (766) | 0.72 | 200 | prob_entropy | var τ4 (lagr) | mean | **yes** |
| e187 | cartpole | small | ✅ | 697 (757) | 2.00 | 0 | off | var τ4 (reg) | mean | **yes** |
| e188 | cartpole | small | ~ | 254 (347) | 0.29 | 200 | prob | fix 8 | mean | **yes** |
| e189 | hopper | BIG | ❌ | 0.0 (1.7) | 1.00 | **200** | off | var τ8 (reg) | mean | no |
| e190 | cartpole | BIG | ✅ (climbing) | 714 (747) | 1.00 | 0 | off | fix 8 | mean | no |
| e191 | cheetah | BIG | ~ (non-monotonic) | 293 (499) | 1.00 | 0 | off | fix 8 | mean | no |
| e192 | acrobot | BIG | ❌ | 8.1 (19.4) | 1.00 | 0 | off | var τ8 (reg) | mean | **yes** |
| e193 | cartpole | small | ❌ | 42 (203) | 2.00 | 200 | off | var τ4 (reg) | mean | no |
| e194 | cartpole | small | ❌ | 152 (285) | 1.00 | 0 | off | var τ8 (reg) | mean | no |
| e195 | cartpole | small | ~ | 175 (352) | 2.00 | 200+adapt | off | var τ4 (reg) | mean | no |
| e196 | cartpole | small | ✅ | 648 (683) | 0.71 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** |
| e197 | hopper | small | ❌ | 0.23 (3.1) | 1.07 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** |
| e198 | cheetah | small | ❌ | 100 (122) | 1.17 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** |
| e199 | acrobot | small | ❌ | 3.4 (21.9) | 0.60 | 200+adapt | prob_entropy | var τ4 (lagr) | mean | **yes** |

Reading the columns together: every ✅ dense-task row sits at blk/step ≥1.0 by the end of
training (Director-dense or denser) except e170/e186/e196 (0.61–0.72) — high final blk/step
is **not** on its own a collapse signature (e171/e175 are campaign champions at 1.05–1.13);
what distinguishes e160/e197/e198 is that mask_frac is already inflated mid-run while score
is still low, not just a late-training drift after success. The τ4-vs-τ8 columns line up
cleanly with the struct column: every working τ4 var-K cell (e46 doesn't apply — that's τ8;
e170/e171/e174/e175/e187/e196) either has struct on or has countdown standing in for it
(e187); every dead-or-weak τ4 cell without either (e166/e167/e169/e176/e193/e195) stays low.
τ8 cells (e178/e179/e181/e185/e189/e192/e194) are struct-independent on cartpole (e179/e194
split by scale, see §6) but never rescue hopper/acrobot regardless of struct or countdown.

The mask-only fixed-K8 cluster (e162–e165) sharpens the sparse-task question further. On
cartpole, masking alone (no var-K at all) gets a real but modest score (270–412) — below
the full combined recipe (653–843) but clearly non-zero, confirming masking is an
independent, if weaker, dense-task mechanism (blk/step 0.31, properly sparse). On hopper,
the *same* fixed-K8-plus-masking recipe (e164/e165) is dead (0.0–0.3), even though it uses
the exact fixed duration (K=8) that keeps e124/e180 alive. Combined with e169/e168/e173/e172
(masking off or on, var-K on, also all dead on hopper) and e178/e179's split (var-K alone is
fine on cartpole, dead on hopper), the honest reading is: **masking alone and variable
duration alone are each independently sufficient to kill hopper/acrobot**, while vanilla
fixed-K8 whole-code Director (neither mechanism) is the only thing that survives. This is a
stronger claim than "variable duration is the culprit" (§6, e160 interim note) — sparse
tasks look fragile to *any* deviation from vanilla Director, not to one specific mechanism.

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

### e178–e199 · Current campaign (see §2 live board, §6 hypotheses)
Launched 07-13. Triaged at 0.6M: e181–e184 cancelled (flat; see §4). e185–e188 countdown
cells; e189/e193–e195 struct cells; e190/e191 controls; e192 acrobot best-guess;
e196–e199 full stack (combined + countdown + struct-adapt) on 4 small tasks.
Smokes passed pre-launch: countdown ×3 recipes + flag-off regression (4660289–91),
struct-adapt (4660314), full stack (4660322).

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

- **Variable-K repval (fixed 06-18, `9716a78`).** Replay manager-value loss mirrors
  imagination (per-step rewards on full timeline via `_switch_mask_from_skills`), not
  fixed-K downsampling. Correct but lowered var-K peaks.
- **Block-rew bugs (fixed 06-19).** `variable_segment_ids` off-by-one +
  `aggregate_mgr_cont_variable` segment_max→segment_min. Benign on cartpole, wrong for
  terminating tasks; e44 predates the fix. Tests: `embodied/tests/test_variable_goals.py`
  — run on a V100, not the login node (glibc/Py3.11).
- **Duration-reg magnitude domination.** Fixed `reg·(E[dur]−τ)²` inside `mgr_policy` at
  reg=0.1 swamps the normalized REINFORCE under one grad-clipped optimizer. reg ≤0.03 OK.
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

## 8. Config flags & metrics reference

All flags default to DreamerV3/pre-HRL behavior.

- **Masking:** `use_masked_goals`, `mask_sparsity_mode {prob,sample,reinforce,none,entropy,prob_entropy}`,
  `mask_sparsity_target/_max/_fixed_weight`, `mask_topk`, `mask_actent_*` (entropy
  anti-collapse), `mask_kl_enable` + `mask_kl_*` (KL-to-sparse-prior, off by default),
  `mask_perblock_credit`, `perblock_edit_cost`, `mask_sparsemax`.
- **Durations:** `variable_goal_length`, `goal_duration_{min,max,target,reg,fixed}`,
  `goal_duration_adapt(_max)`, `goal_duration_lagrange(_impl,_min,_max,_vel,_init,_tol)`
  (own loss key `goal_duration_prior`; mutually exclusive with adapt),
  `variable_goal_block_rew`, `goal_switch_cost`, `goal_edit_cost(_ach)`,
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
- **Template env knobs** (`run_v3_prior_vargoal_{small,big_a100,short_a100}.sbatch`):
  `RECIPE={vark_masked,mask_fixedk,plain_vark}`, `DUR_TARGET`, `DUR_MODE={fixed,lagrangian}`,
  `MASK_MODE`, `STRUCT_W`, `STRUCT_ADAPT`, `WORKER_TIMED_GOALS`, `MGR_REWARD_AGG`,
  `MGR_EXPL_W`, `MGR_FREQ`, `SEED`, `RUN_STEPS`, `RUN_DIR` (fixed = resumable).
- **Key metrics:** `goal/mask_frac_mean`, `goal/mask_prob_mean`, `goal/struct_corr`,
  `goal/rec_mean`, `goal/mgr_duration_mean/std`, `goal/mgr_switch_rate`, `wkr_goal_rew`,
  `wkr_ent/action`, `mgr_extr_rew(_block)`, `mgr_extr_adv`, `epstats/reward_rate`,
  `mgr_duration_lagrange_scale_mean`, `goal/mask_actent_scale_mean`,
  `goal/struct_adapt_scale` (pilot), plus §1's derived blk/step.
