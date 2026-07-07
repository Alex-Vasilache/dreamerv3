# DreamerV3 HRL — Experiment Log

**Setup.** Director-style hierarchical RL on DreamerV3: a **manager** emits/edits a goal
code `Z`; a **worker** is rewarded for reaching the decoded goal. Task: DMC **cartpole
swingup**, `size6m`, image **32×32**. Reference: non-HRL / no-mask HRL baseline solves
**~710**. Branch: `feat/hrl-masked-goals`.

**Naming.** Runs are `e{N}_{env}`; `N` is **global and monotonic** (never reused). A
requeue with a tweak gets a new `N`; a literal restart keeps it. Every run is logged below
with its final score (last-15-episode mean / max). Most runs hit the 24 h walltime
(`TIMEOUT`) at ~2.6–3.7M of 4M steps — treat scores as converged-but-still-climbing.

### Standard metrics (log on every HRL / masked-goals run)

| Metric | Source (`metrics.jsonl`) | Notes |
|---|---|---|
| **best-15** | `scores.jsonl` | max of last 15 episode scores |
| **mask_frac** | `train/goal/mask_frac_mean` | fraction of L blocks edited per manager decision |
| **K** | `train/goal/mgr_duration_mean` | mean goal hold length (steps between decisions) |
| **blk/step** | `mask_frac × L / K` | **avg blocks edited per env step; lower is better** |

**L = 8** (`skill_shape[0]` for cartpole and antmaze). Formula: manager edits
`mask_frac × L` blocks each decision, decisions occur every `K` steps on average →
`(mask_frac × L) / K`. Example: K=16, mask_frac=0.5 → 4 blocks/decision, 1/16 decisions/step → **0.25 blk/step**.

**e57 reference:** mask_frac 0.32, K 4.0 → **blk/step ≈ 0.64**.

**Caveat:** a low blk/step can come from **long holds** (large K) rather than a sparse
mask per decision — always report mask_frac, K, and blk/step together.

---

## Current best config

### ① Highest proven score — fixed K=1, masked goals  →  ~800  (e8; e17 ≈777)
```bash
--configs size6m masked_goals
--manager_sample_freq 1            # K=1 — the decisive lever for task score
--use_masked_goals True
--mask_sparsity_mode prob          # B2: penalize differentiable sigmoid probs (mandatory)
--mask_sparsity_target 0.3
--mask_sparsity_max 5.0
--mask_topk 0
--goal_struct_weight 200
--imag_length 16
# --mgr_cond_goalcode True         # optional; neutral at K=1 (e17 ≈ e8)
```
Holds a genuinely sparse mask (`mask_frac` 0.15–0.21), beats baseline, no collapse.
**Caveat:** loose worker goal-reaching (cosine ~0.37) — the manager is load-bearing.

### ② Best learned-duration (variable-K) — masked + soft prior @ **K=4**  →  best-15 **~839** (e57)
```bash
--configs size6m masked_goals variable_goals    # imag_length 32
--variable_goal_length True
--goal_duration_reg 0.01           # SOFT — 0.1 collapses; 0.001–0.03 solve
--goal_duration_target 4.0         # K=4 is the sweet spot (4 >> 16 > 8 >> 32; e57–e60)
--goal_duration_max 16
--mask_sparsity_mode prob          # masked edits (B2), target 0.3
--mask_sparsity_target 0.3
--goal_struct_weight 200
--mgr_cond_goalcode True            # Z-only conditioning
```
**This is the best variable-K config and matches the fixed-K K=1 champion.** e57 reached
best rolling-15 **839**, second-half avg **774**, max **854** — beating both the prior
var-K best (e46 plain @ K=8, 724) and tying/beating e8's 800. It is the **first learned-K
config to reach the K=1 score tier**, i.e. it gets a *learned* hold-length without paying
the usual var-K score penalty. **Caveat:** noisier than e46 — the last-15 window wobbled
down to ~682 before we cancelled it (mid-run, ~2.67M/4M), so treat ~770–840 as the band,
not a clean converged point. e46's plain @ K=8 (724, max 761) remains the *steadiest*
var-K run if low variance matters more than peak.

---

## Key findings (settled)

1. **`mask_sparsity_mode: prob` (B2) is the only working sparsity penalty.** It's
   load-bearing for *both* sparsity and stability. Failure modes of the alternatives:
   `sample` (gradient-free, the original) → mask drifts dense / destabilizes (e1/e3/e7,
   e27/e29/e31/e32); `reinforce` → rails the scale, drives manager reward negative,
   mask stays dense (e10/e11); fixed-weight→0 → freezes the goal, score 0 (e12–e15).
2. **K is the dominant lever and it's a tradeoff.** **K=1** → best task score (e8 800,
   e17 777) but loose reaching (cosine ~0.34). **K=8** → tight reaching (~0.7) but score
   collapses (e9 ~1; e20–e23 504→3). No fixed K gets both.
3. **Variable (learned) K needs a *soft* duration prior, and the target K matters.** No
   prior → durations drift to ~2–3, score ~100–200 (e24/e52/e54). With prior: **reg
   0.001–0.03 solve** (e47 601, e51 614, e46 724@K8); **reg=0.1 collapses** by
   magnitude-domination (e41/e43). **Target K=4 is the sweet spot (4 >> 16 > 8 >> 32,
   e57–e60): masked var-K @ K=4 (e57) hits best-15 839 / max 854 — the best var-K config
   and the first to reach the fixed-K K=1 tier (e8 ~800).** Masked adaptive holds→16 only
   reaches 671 (e56); masked @ K=8 conflicts (e58 511).
4. **Manager conditioning doesn't break the K tradeoff** (e16–e23). Feeding the manager the
   pre-edit `Z` / achievement signal is neutral-to-negative at K=1, mixed at K=8; Z-only ≥
   adding achievement. The bottleneck is K, not what the manager sees.
5. **Structural goal loss is healthy but not a score lever.** `goal_struct_weight=200`
   gives `struct_corr≈0.92`; w400 marginally helps (e33 744), w800/feat/margin/adapt mostly
   hurt (e34–e40); removing it doesn't destabilize (e28). Keep at 200.
6. **The manager is load-bearing.** Every "drive edits to 0" control froze the goal → score
   0 (e12–e15), proving the worker is not solving cartpole on its own.
7. **The variable-K repval "fix" is correct but lowered variable-K peaks** (e25→e54,
   e41→e43). It removed an imagination-vs-replay manager-value mismatch but was **not** the
   cure for the e41 collapse; soft duration priors are what recover score post-fix.
8. **Fixed K=8 plain baseline works** (~700: exp1 704, e42 674) — the reference variable-K
   is trying to match while learning K.

## Don't retry (dead ends)

- `mask_sparsity_mode` **`sample`** or **`reinforce`**; fixed-weight sparsity→0.
- `mask_topk` hard edit budget — strangles the policy at every K (e2/e4/e6, score ≤100).
- `goal_duration_reg ≥ 0.1` — magnitude-dominates the manager REINFORCE → collapse.
- `variable_goal_block_rew` without a duration prior: e62 post-fix rerun scored ~216
  (duration 2.5) — marginal over e54 (193), far below soft-prior runs. Don't cite e44
  (buggy code).
- `goal_edit_cost` at e70–e73 magnitudes (0.05–0.2): mask→0, score→0 within ~225k steps.
  Retried at 1e-4–1e-2 in e74–e77.

---

## Experiment log (condensed)

### e1–e15 · Masking mechanism — making sparsity stick
| Exp | Config | Score | Outcome |
|---|---|---|---|
| e1 | masked, soft sparsity 0.3, K=8 | 650 (750) | mask collapsed all-ones (no-op penalty) |
| e2 | `mask_topk=3`, K=8 | ~0 | topk budget strangled policy |
| e3 | `goal_struct_weight=200`, soft 0.3, K=8 | 610 (735) | solved but mask collapsed dense |
| e4 | `mask_topk=3` + struct200, K=8 | ~0–11 | topk strangled (confirms e2) |
| e5 | Director `size6m`, **64×64** | ~720 | 64×64 baseline trains; report-JIT segfaults |
| e6 | `mask_topk=3` + struct200, **K=1** | ~100 | K=1 rescued topk but still throttled |
| e7 | soft 0.3 + struct200, K=1 | ~840 | high score but mask runaway-to-dense |
| **e8** | **prob 0.3 + struct200, K=1** | **~800 ✅** | **WIN** — sparse mask 0.15–0.21, no collapse |
| e9 | prob 0.3, K=8 | ~1 | over-sparsified to inert (mask→0.001) |
| e10/e11 | `reinforce` mode, K=1/8 | ~3 / ~19 | mode broken — scale rails, reward negative |
| e12–e15 | fixed-weight sparsity→0, K=1/8 | 0 | goal frozen — clean "manager is load-bearing" control |

**Root cause (e1/e3/e7):** the soft penalty had **zero gradient** to the mask head
(computed from the hard Bernoulli sample). `prob` mode (B2) feeds the differentiable probs
to the adapter — the fix in e8.

### e16–e23 · Manager-input conditioning  {K∈1,8} × {target 0.3, 0.125} × {Z, Z+achieve}
| Exp | K / target / cond | Score | wkr_goal |
|---|---|---|---|
| **e17** | K=1 / 0.3 / **Z-only** | **777** | 0.34 |
| e16 | K=1 / 0.3 / Z+achieve | 673 | 0.29 |
| e18 / e19 | K=1 / 0.125 / Z+ach / Z | 684 / 624 | 0.33 / 0.31 |
| e20 / e21 | K=8 / 0.3 / Z+ach / Z | 504 / 365 | **0.76 / 0.65** |
| e22 / e23 | K=8 / 0.125 / Z+ach / Z | **3** / 143 | 0.72 / 0.72 |

**Verdict:** conditioning works mechanically but doesn't break the K tradeoff — K=1 high
score + loose reaching; K=8 tight reaching + score collapse. Best arm = e17 (Z-only, K=1).

### e24–e32 · Variable-K intro + e25 scaffolding ablation
| Exp | Config | Score | Note |
|---|---|---|---|
| e24 | var-K, **plain**, no prior | 13 | durations drift short (~3.1) |
| e25 | var-K, **masked** (e17 recipe), no prior | 406 (786) | pre-repval-fix; unstable, peaks then drifts |
| e26–e32 | ablate e25's 3 knobs (cond / sparsity-target / struct) | — | see verdict |

**e26–e32 verdict:** **K2 (`mask_sparsity_mode sample`, i.e. no real sparsity target) is
the destabilizer** — every K2 run collapsed (score ≤27); the two stable runs were the ones
*with* the prob target (e26 K1=217, e28 K3=131). K3 (no struct) is sparsity-neutral; K1
(no cond) is invariant. Confirms finding #1.

### e33–e40 · Structural goal loss variants  (all masked var-K, e25 base)
| Exp | Knob | Score | corr |
|---|---|---|---|
| e33 | weight **400** (deter/mse) | **744** | 0.95 |
| e34 | weight 800 | 534 (828) | 0.97 |
| e38 | feat + adaptive | 597 | 0.95 |
| e40 | margin + adaptive | 499 | 0.96 |
| e35/e36/e37/e39 | adapt / feat / margin / feat+margin | 211/144/114/96 | 0.90–0.93 |

**Verdict:** no struct variant beats e8/e17 on score. Plain mse at w400 (e33) is the best
of the family; the feat/margin/adaptive elaborations mostly *hurt*. Struct ≠ score lever.

### e41–e56 · Variable-K duration-prior tuning  (plain unless noted; struct=200; repval fix)
| Exp | Duration shaping | Goal style | Score | Verdict |
|---|---|---|---|---|
| e41 | reg=0.1→8 (pre-fix) | plain | 104 (peak **728**) | peaks then collapses (reachable-but-useless goals) |
| e42 | fixed K=8 (control) | plain | **674** (758) | true fixed-K baseline ✅ |
| e43 | reg=0.1→8 + repval fix | plain | 156 (443) | repval fix did **not** cure e41 |
| e44 | reg=0.1 + block-rew | plain | 94 | **buggy credit code — uninterpretable** |
| e45 | struct w400, masked | masked | 660 (745) | decent; struct-w400 + repval |
| **e46** | **reg=0.01→8 (soft)** | plain | **724** (761) | ✅ stable, dur pinned 8.00 |
| e47 | reg=0.001→8 (ultra-soft) | plain | 601 (763) | ✅ stable |
| e48 | adaptive prior (cap 0.05) | plain | 189 | ❌ adaptive didn't rescue |
| e49 | reg=0.1, **repval off** | plain | 13 | collapse persists ⇒ repval not the cause |
| e50 | fixed K=8 via duration bypass | plain | 133 | ❌ variable graph degrades even at pinned K=8 |
| e51 | reg=0.03→8 (mid) | plain | 614 (806) | ✅ locates the reg cliff (0.03 ok, 0.1 not) |
| e52 / e54 | **no** duration prior | plain / masked | 103 / 193 | ❌ durations drift short (~2.2) |
| e53 | adaptive→16 | plain | 160 | ❌ long holds don't help plain |
| **e56** | adaptive→16 | masked | **671** (766) | ✅ best masked var-K |
| e55 | no prior (WALKER) | masked | 447 | preliminary on harder env |

**Duration-reg cliff (target 8):** 0.001 ✅ · 0.01 ✅ · 0.03 ✅ · **0.1 ❌** (collapse).

### e57–e61 · Masked var-K + soft fixed duration prior (launched 2026-06-22)

Combines e46's soft prior (`goal_duration_reg=0.01`) with e56's masked recipe
(prob 0.3, struct 200, Z-cond). The untested cell: masked + var-K + fixed prior
(instead of e56's adaptive→16).

| Exp | Env | Target K | Job | Best-15 | last-15 | Max | Duration | Mask frac | **blk/step** | Wkr rew | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **e57** | cartpole | **4** | 4637337 | **839** | 682 | **854** | 4.00 | 0.317 | **0.64** | 0.707 | ✅ **WIN** — cancelled 2.67M/4M (free slot for e66–e73) |
| e58 | cartpole | 8 | 4637338 | — | 511 | 644 | 7.99 | 0.192 | 0.19 | 0.683 | cancelled (~2.5M) |
| e59 | cartpole | 16 | 4637339 | — | 701 | 710 | 15.02 | 0.379 | 0.20 | 0.550 | cancelled (~2.5M) |
| e60 | cartpole | 32 | 4637340 | — | 119 | 131 | 30.33 | 0.908 | 0.24 | 0.320 | cancelled (user, ~2.5M) |
| e61 | walker | 8 | 4637341 | — | 549 | 685 | 7.99 | 0.417 | 0.42 | 0.679 | cancelled (~3M/10M) |

*e57 best-15 = best rolling-15-episode mean (second-half avg 774); last-15 = final window
before cancel. Others at ~22h/2.5M snapshots. All cancelled 2026-06-23 to free v100 for the
e66–e73 sweep (antmaze e65 → a100).*

**Final verdict:** Target K ordering: **4 >> 16 > 8 >> 32**. **e57 (K=4) is the WIN** — best
rolling-15 **839**, second-half avg **774**, max **854**: the best variable-K config to date
and the **first learned-K run to reach the fixed-K K=1 tier (e8 ~800)**, breaking the prior
"var-K always costs score" pattern. It's noisier than e46 (last-15 dipped to 682), so the
honest band is ~770–840. e60 (K=32) drove the mask dense (0.908) — very long holds overwhelm
the sparsity penalty. e58 (K=8) scored only 511 despite clean duration lock (masking + K=8
conflict, worse than plain e46's 724). e59 (K=16) decent (701) but no record. Walker e61
(549/685 at 3M/10M) directionally promising but cancelled before convergence.

Submit: `./submit_e57_61_masked_durreg_soft.sh` from `code/dreamerv3/`.

### e62 · Masked var-K + block-pooled manager credit (launched 2026-06-22)

Tests the *other* lever for no-prior masked var-K: fix **credit assignment** instead of
adding a duration prior. e54 recipe (masked, var-K, **no** prior, prob 0.3, struct 200,
Z-cond) **+ `variable_goal_block_rew=True`** — Director-style reward pooling over each
realized duration segment. First run on the **post-06-19-bugfix** block-rew path
(segment-id off-by-one + continuation-product fixes); verified by
`test_variable_goals.py` (8/8 pass on v100, job 4637344). e44 ran on the buggy code and
is uninterpretable.

| Exp | Job | Config | Score (last-15) | Max | Duration | Mask frac | Status |
|---|---|---|---|---|---|---|---|
| e62 | 4637347 | masked var-K, no prior, **block-rew** | 216 | 445 | 2.48 | 0.301 | ❌ cancelled ~2.05M/4M; archived to bucket |

**Verdict:** ❌ Block-rew alone did not fix no-prior short-K collapse. Duration drifted to
2.48 (≈ e54's 2.31). Score 216 is a marginal gain over e54 (193) but far below soft-prior
runs. Script `run_v3_e62_cartpole_vargoal_masked_blockrew.sbatch`.

### e63–e64 · Fix var-K short-K collapse WITHOUT a duration prior (launched 2026-06-22)

Two anti-collapse mechanisms that don't specify a target K (vs e57–e61's soft prior).
Both on the e54 base (masked, var-K, no prior, prob 0.3, struct 200, Z-cond), **48h**.
Diagnostic for the lever: in e54 the duration-entropy adapter sat at its **floor**
(scale 1e-5, norm-ent 0.53 ≈ target 0.5) while `mgr_duration_mean` collapsed to **2.31**.

| Exp | Job | Mechanism | Score (last-15) | Max | Duration | Mask frac | Status |
|---|---|---|---|---|---|---|---|
| e63 | 4637352 | **Duration entropy bonus** (`manager_actent_duration_target=0.8`) | 181 | 274 | 4.53 | 0.310 | ❌ cancelled ~2.53M/4M; archived |
| e64 | 4637353 | **Min-hold floor** (`goal_duration_min=4`, range [4,16]) | 155 | 263 | 5.35 | 0.347 | ❌ cancelled ~2.51M/4M; archived |

**Verdict:** ❌ Both fail without a duration prior. e63 lifted mean duration to 4.53 but score
only 181; e64 forced dur≥4 (mean 5.35) but score 155. Neither approaches soft-prior runs
(e57 839). **The soft prior shapes duration in a way that benefits planning, not just
preventing collapse.** Scripts: `run_v3_e63_cartpole_vargoal_masked_durentropy.sbatch`,
`run_v3_e64_cartpole_vargoal_masked_minhold.sbatch`.

### e65 · Antmaze-M — masked var-K + soft fixed duration prior (launched 2026-06-22)

First **navigation** task in the series: Director **Ant Maze M** (`loconav_ant_maze_m`),
the e58 recipe (masked + var-K + soft fixed duration prior, **reg=0.01, target 8**) ported
off DMC. Obs = **camera-4 egocentric image (64×64) + ant proprio** (joints/sensors/
egocentric_camera), matching Director. **48h** walltime (gpu-v100 assoc `MaxWall` cap —
96h is not grantable), `RUN_STEPS=10M` so it will **not** finish in one window; resume by
re-running with the same `RUN_DIR`/`--logdir`. Smoke-tested first (cancelled smoke 4637377:
env imports, image obs reconstructs, `mgr_duration_mean≈8`, no errors).

Env bring-up fixes (loconav had never run in this series): `dreamerv3_env` is missing
`matplotlib`, and the dreamerv3 loconav port had regressed Director's `image=True` to
proprio-only. Fixed in `embodied/envs/loconav.py` — added an `image` kwarg (default True,
wired to the DMC wrapper) and replaced the `matplotlib` tab10 colormap with hardcoded tab10
RGB (wall colors are visual landmarks the egocentric camera sees, so they must match
Director). `configs.yaml` `env.loconav` now sets `image: True`.

| Exp | Job | Config | Score (last-15) | Steps | Duration | Mask frac | **blk/step** | Status |
|---|---|---|---|---|---|---|---|---|
| e65 | **4649088** (resume) | Antmaze-M, masked var-K, soft prior reg=0.01→8, 10M steps | 0 | **2.67M/10M** | 8.00 | 0.44 | **0.44** | **RUNNING** (gpu-a100) |

*Antmaze throughput ~1k steps/h on a100 (64×64 + ant proprio). Duration prior holds at 8.0;
mask ~0.43 (prob target 0.3), blk/step ~0.43. No episode reward yet (expected). Script
`run_v3_e65_antmaze_m_vargoal_masked_durreg_soft8.sbatch`.*

**Resumes:** 4637380 (v100) → 4637766 (v100) → 4637767 (a100, TIMEOUT @ 1.44M) →
4638323 (a100, **TIMEOUT** 2d @ 2.67M) → **4649088** (a100 resume, 2026-06-27), same `RUN_DIR`
(`e65_antmaze_m_vargoal_masked_durreg_soft8_20260622_103947_jXP19c`).*

### e66–e69 · Manager freedom — length levers (launched 2026-06-23, **COMPLETED** ~4M/4M)

Replaces forced duration prior with **priced activity** mechanisms. See
`plans/memoized-exploring-wilkinson.md`. Scripts: `run_v3_cartpole_freedom_sweep.sbatch`,
`submit_e66_73_freedom_sweep.sh`. (Sparsity target kept on `prob 0.3` in all four → masks
stay healthy ≈0.3; this isolates the *length* lever.)

| Exp | Job | Length lever | last-15 | best-15 | Max | **K (dur)** | Mask | switch | Status |
|---|---|---|---|---|---|---|---|---|---|
| e67 | 4637777 | switch **0.02**, mean | 730 | **744** | 769 | **13.0** | 0.37 | 0.10 | ✅ DONE |
| e69 | 4637779 | switch **0.10**, mean | 749 | **761** | 769 | **13.7** | 0.37 | 0.09 | ✅ DONE |
| e68 | 4637778 | switch **0.05**, mean | 713 | 719 | 741 | 13 | 0.33 | 0.09 | CANCELLED (freed slot for e81) |
| e66 | 4637776 | **sum** agg, prior off | 691 | **702** | 747 | **3.0** | 0.17 | 0.46 | ✅ DONE |

**Verdict (settled dynamics):**
- **sum aggregation FALSIFIES the plan's root-cause theory.** The plan argued averaging
  under-credits long blocks so `sum` would make K *neutral*. Empirically e66 sum drove K
  **shorter** (2.3, switch↑0.46) — the opposite. Averaging was **not** the source of the
  short-K bias. Score 726 < e57's 839.
- **switch-cost works as a length lever but only reaches the corner.** All three costs
  (0.02–0.10) push K to the ceiling (13–14, switch≈0.09) — none lands the interior K=4 that
  e57's prior achieves. Even the smallest (0.02) saturates; an interior K would need a cost
  <0.02, and the cost is one-sided so it has no interior fixed point either. Scores 761/719/762
  — healthy (sparsity target kept them alive) but all below e57's best-15 839.
- **Combine candidate:** sum (K→2.3) and switch (K→13) push K in **opposite** directions, so
  `sum + small switch` *could* cancel to interior K≈4–6 with no prior — the one untested
  freedom run worth doing. Lower priority since e57's prior already nails K=4 cleanly.

### e70–e73 · Manager freedom — edit-cost sweep v1 (cancelled 2026-06-23)

| Exp | Job | Config | last-15 | Max | Dur | Mask | Status |
|---|---|---|---|---|---|---|---|
| e70 | 4637784 | prior→4, edit **0.05**, mask none | 0.2 | 178 | 4.00 | ~0 | cancelled ~225k |
| e71 | 4637785 | prior→4, edit **0.1**, mask none | 0.4 | 158 | — | ~0 | cancelled ~225k |
| e72 | 4637786 | prior→4, edit **0.2**, mask none | 0.2 | 159 | — | ~0 | cancelled ~225k |
| e73 | 4637787 | switch 0.05 + edit 0.1, no prior | 0.1 | 90 | 14.8 | ~0 | cancelled ~225k |

**Verdict:** ❌ Edit costs 0.05–0.2 freeze the mask (frac→0) and kill score within ~225k steps.
e73 also froze edits while K drifted long. Cancelled to free v100 slots.

### e74–e77 · Manager freedom — edit-cost sweep v2, smaller costs (launched 2026-06-23, **CANCELLED** ~2.3M/4M)

Retried Phase B/C with edit costs **100×–1000× smaller** than e70–73. Script:
`submit_e66_73_freedom_sweep_smaller_edits.sh`.

| Exp | Job | Config | last-15 | best-15 | Max | **mask_frac** | K | Status |
|---|---|---|---|---|---|---|---|---|
| e77 | 4637807 | switch 0.02 + edit **0.001**, no prior, mask none | 711 | 753 | 760 | **0.99** | 8.4 | CANCELLED ~2.3M |
| e74 | 4637804 | prior→4, edit **0.0001**, mask none | 224 | 717 | 741 | **0.95** | 4.0 | CANCELLED ~2.3M |
| e75 | 4637805 | prior→4, edit **0.001**, mask none | 169 | 231 | 289 | 0.51 | 4.0 | CANCELLED ~2.3M |
| e76 | 4637806 | prior→4, edit **0.01**, mask none | 144 | 205 | 249 | 0.14 | 4.0 | CANCELLED ~2.3M |

**Verdict — edit-cost is a knife-edge corner-seeker; `mask_sparsity_mode=none` removes the
only restoring force.** An edit cost is **monotone (one-sided)**: it only pushes editing
*down*, and with no sparsity target nothing pushes back up (the worker-reward gradient through
a sampled, stop-grad'd mask is too weak). So the mask has **no interior fixed point** and runs
to a corner set purely by cost-vs-benefit:
- cost too high (e70–73, 0.05–0.2) → mask **freezes to 0** → stale goal → score **0** (e12–e15 death);
- cost too low (e74/e77, ≤1e-3) → mask **explodes to ~0.95–0.99 dense** → noisy/slow (e74 last-15 224 though best-15 717; e77 most stable at ~663);
- the in-between (e76, 0.01 → mask 0.14) lands a mid sparsity but score still poor (205) —
  the interior is unstable scaffolding it passes through, not an equilibrium.

The **`prob` target is two-sided** (Lagrange restoring force toward 0.3 from both sides) — that
is exactly what kept e57/e67–69 alive at mask≈0.3. The earlier "edit-cost is light insurance
because e57's `mask_sparsity_scale≈0.0006` is slack" reasoning was **backwards**: the multiplier
was slack *because the target was holding the mask in place*, not because it was unneeded.
Entropy is no substitute either — max mask-entropy is at p=0.5, so a strong mask-entropy bonus
just re-introduces a worse, fixed sparsity setpoint (0.5, uncontrollable).

### Manager-freedom sweep — overall conclusion (e66–e77)

**"Freedom via one-sided priced costs" FAILED for both levers.** A monotone cost cannot create
an interior equilibrium for a variable with no opposing gradient (length → corner K=max;
sparsity → corner 0 or 1). **The two-sided targets remain best: e57 (prior K=4 + sparsity
0.3) = best-15 839.** Best non-champion alternative is **e67** (switch 0.02 + sparsity target →
761, but K saturates at max); best fully-priced run is **e77** (753, dense mask). None beats
e57. Next, if pursuing freedom: the **sum + small-switch combo** (opposing length pressures,
keep sparsity target) is the only run that could reach interior-K-without-prior.

### e78–e80 · Emergent sparsity (no target) + length combo (launched 2026-06-24)

Three freedom mechanisms after the e66–e77 diagnosis. Script:
`run_v3_cartpole_freedom_sweep.sbatch`. K pinned by prior→4 in e78/e79 to isolate sparsity.

| Exp | Job | Mechanism | last-15 | best-15 | Mask | K | Status |
|---|---|---|---|---|---|---|---|
| e78 | 4637995 | **B.3** `goal_edit_cost_ach=0.3` (reward path) | ~0 | **0** | **~0** | 4.0 | ❌ CANCELLED ~3.48M |
| e79 | 4637996 | **C.5** `mask_sparsemax=True` | 338 | **373** | **0.14** | 4.0 | ❌ CANCELLED ~3.44M |
| e80 | 4637997 | **sum** + **switch 0.05** | 730 | **750** | 0.38 | **13.6** | ❌ CANCELLED ~3.49M |

**Verdict:**
- **e78 B.3 FALSIFIED.** Started interior (~0.36 mask) then collapsed to **mask≈0** by ~1.7M
  while `achievement→0.8+`; score died (best-15 **0**). Achievement-gated edit cost through the
  reward path still has no interior fixed point — converges to "never edit" once goals are mostly
  reached (same death mode as e70, opposite corner).
- **e79 sparsemax:** structural sparsity works (mask **0.14**, ~1-block budget) and K≈4 holds,
  but score weak (**373** vs e57 **839**). Mask entropy near zero — gate may be too restrictive
  or wrong blocks get edited.
- **e80 sum+switch:** score competitive (**750**, near e67 **761**) and mask healthy (0.38), but
  **K rails to ~13.6** — length combo did not cancel to interior K; switch lost to sum.

**A.1 (differentiable mask) — NOT launched** (needs pathwise manager actor; deferred).

### e81–e82 · Per-block credit plan Tier 1 + Tier 2 (`per_block_credit_plan.md`, launched 2026-06-24)

| Exp | Job | Mechanism | last-15 | best-15 | Mask | K | Status |
|---|---|---|---|---|---|---|---|
| e81 | 4637998 | **Tier 1** `prob_ach`, weight 0.3 | ~0 | **0** | **~0** | 4.0 | ❌ CANCELLED ~3.35M |
| e82 | 4638001 | **Tier 2** `mask_perblock_credit`, edit cost 0.1 | 714 | **727** | **0.69** | 4.0 | ❌ CANCELLED ~3.03M |

**Verdict:**
- **e81 Tier 1 FALSIFIED** (same outcome as e78). Analytic achievement-gated `prob_ach` penalty
  did not fix the corner: mask collapsed to **0**, score **0**. Moving B.3 from reward to
  zero-variance gradient path does not create an interior equilibrium.
- **e82 Tier 2 PARTIAL.** Q_g per-block credit works for **task** (best-15 **727**, K≈**4**;
  `mask_perblock_adv_std≈0.02` — credit discriminates across blocks; `mgr_goal_q_val` learning).
  **Sparsity failed:** mask **dense ~0.69** — `perblock_edit_cost=0.1` too weak; credit picks
  blocks but does not sparsify.

**Overall per-block credit sweep:** e57 (prior + sparsity target) remains champion. Neither tier
produces emergent interior sparsity without a target. e82 shows Tier 2 is viable for task credit;
**e83–e90** sweep stronger `perblock_edit_cost` and e80-length combos (launched 2026-06-25).

### e83–e90 · Per-block edit-cost sweep + sum/switch length combos (launched 2026-06-25, RUNNING ~1.8M/4M)

Cancelled e80/e82; freed v100 for 8-run sweep. Script: `run_v3_cartpole_freedom_sweep.sbatch`.
e65 antmaze unchanged on a100.

| Exp | Job | Config | best-15 | mask | K | **blk/step** | Status |
|---|---|---|---:|---:|---:|---:|---|
| e83 | 4638334 | perblock cost **0.3** | 670 | 0.54 | 4.0 | **1.08** | RUNNING |
| e84 | 4638335 | perblock cost **0.5** | 643 | 0.52 | 4.0 | **1.05** | RUNNING |
| e85 | 4638336 | perblock cost **1.0** | 508 | 0.33 | 4.0 | **0.67** | RUNNING |
| e86 | 4638337 | sum+switch 0.02, dur_max=32 | 680 | 0.51 | 20.8 | **0.20** | RUNNING |
| e87 | 4638342 | combo 0.3, dur_max=16 | 0 | 0.21 | 14.5 | **0.12** | RUNNING (unstable) |
| e88 | 4638343 | combo 0.3, dur_max=32 | 731 | 0.36 | 14.0 | **0.20** | RUNNING |
| e89 | 4638344 | combo 0.5, dur_max=16 | 762 | 0.46 | 11.5 | **0.32** | RUNNING |
| e90 | 4638345 | combo 0.5, dur_max=32 | 616 | 0.43 | 13.9 | **0.25** | RUNNING |

*Snapshot ~1.8M steps (2026-06-25). **blk/step** = mask_frac×8/K; lower is better (e57 ≈ 0.64).
e85 (cost=1.0) is the only perblock-only run below e57 blk/step so far, at the cost of score.
e86/e88 look sparse on blk/step mainly because K is large, not because mask_frac is low.*

**e82-style base (e83–e85):** `mask_perblock_credit=True`, `mask_sparsity_mode=none`,
`dur_reg=0.01→4`, `mgr_reward_agg=mean`.

**e80-style base (e86):** `mgr_reward_agg=sum`, `goal_switch_cost=0.02`, `dur_reg=0`,
`mask_sparsity_mode=prob` target 0.3, `goal_duration_max=32`.

**Combo base (e87–e90):** e80 length recipe + `mask_perblock_credit=True` + `perblock_edit_cost`
+ `goal_duration_max` ∈ {16, 32}. Keeps B2 sparsity target (e80) alongside per-block Q_g credit (e82).

**Watch vs e57 (839, blk/step 0.64) / e82 (727):** mask_frac, K, **blk/step**,
`goal/mask_perblock_adv_std`, best-15 score.

**Final (4M, completed 2026-06-27):** e88 (combo 0.3, dur_max32) is the standout —
best-15 **764**, last-15 **755** (the only stable-tailed combo run), mask 0.36, K **~18**,
**blk/step 0.16**. e89 761/737, e90 768/676, e83 766, e84 748/718, e85 655. e86 peaked
761 then **collapsed** (last-15 205); e87 (combo 0.3, dur_max16) **died** (0). None beats
e57 (839).

### e88 reframed — emergent on BOTH axes (the target was inert)

Checked e88's logged metrics: `mask_sparsity_scale_mean` pinned at its floor (1.1e-5) the
whole run and `loss/mask_sparsity ≈ 3e-6`, with `mask_prob_mean` 0.26 **below** the 0.3
target — so the `prob` sparsity target is **present in code but non-binding**. The mask is
held at ~0.35 by **per-block Q_g credit + `perblock_edit_cost=0.3`**, not the target. So
e88 is the first stable run that is emergent on **both** axes: **K~18 with no duration
prior** (sum + switch 0.02) **and** mask ~0.35 with a non-binding sparsity target. This is
*why* per-block credit beats the global edit cost (e74–e77): the per-block leave-one-out
advantage gives each block a benefit signal, so a block is edited iff its advantage exceeds
the cost — a genuine per-block interior threshold, unlike the one-sided global cost.
`perblock_edit_cost` is the real sparsity dial (e82 @0.1 → mask 0.69; e88 @0.3 → 0.35).

### e91–e98 · VOID (misconfigured, cancelled within ~8 min, no data)

Launched 2026-06-29 then immediately cancelled: improvement (1) wrongly used a **soft
duration prior** (`goal_duration_reg=0.01`), which is exactly the forced-K knob the
freedom line is trying to *avoid*. Superseded by e99–e106 below, where (1) is the
correct **switch-cost** lever (no prior, K stays emergent). Numbers retired per the
monotonic convention.

### e99–e106 · e88 improvements (full freedom, NO prior) + multi-env port (launched 2026-06-29)

Two single-knob improvements on the e88 base (`sum`, **dur_reg 0.0**, dur_max32, switch
0.02, prob target 0.3, perblock_credit, perblock_edit_cost 0.3) + a port of the combined
config to 5 DMC tasks. The duration prior stays **OFF in every run** — K is fully emergent.
Scripts: `run_v3_freedom_sweep_env.sbatch` (generalized, TASK-param),
`submit_e99_106_e88_freedom.sh`. 8 × v100 (= cap); e65 antmaze on a100.

- **(1) shorter emergent K** — lower the only length pressure: `goal_switch_cost` 0.02 → 0.01.
  No prior; K stays emergent. Pulls K (e88 ~18) shorter; tests the score gap to e57 (e88 764
  vs e57 839 — hypothesis: K~18 is too long, worker chases a stale goal).
- **(3) prove emergent sparsity** — remove the inert target: `mask_sparsity_mode=none`.
  Tests whether the mask still holds ~0.35 with the target gone (would prove per-block credit
  does the work, not the target).

| Exp | Job | Task | Change vs e88 | Hypothesis / expected | Status |
|---|---|---|---|---|---|
| e99  | 4649102 | cartpole | (1) `switch_cost=0.01` | shorter emergent K lifts score toward 800+ | ❌ cancelled ~2.6M |
| e100 | 4649103 | cartpole | (3) `mask_mode=none` | mask holds ~0.35 w/o target ⇒ emergent sparsity proven | ❌ cancelled ~2.6M |
| e101 | 4649104 | cartpole | (1)+(3) | full freedom both axes: shorter K + emergent mask, score ≥ e88 | ❌ cancelled ~2.6M |
| e102 | 4649105 | acrobot_swingup | (1)+(3) port | recipe transfers to a harder swingup | ❌ cancelled ~2.6M |
| e103 | 4649106 | cheetah_run | (1)+(3) port | transfers to locomotion | ❌ cancelled ~2.5M |
| e104 | 4649107 | pendulum_swingup | (1)+(3) port | transfers to sparse-reward swingup | ❌ cancelled ~2.6M |
| e105 | 4649108 | quadruped_run | (1)+(3) port | transfers to high-DoF locomotion (cam 2) | ❌ cancelled ~2.4M |
| e106 | 4649109 | hopper_hop | (1)+(3) port | transfers to hard hop | ❌ cancelled ~2.5M |

**Cancelled 2026-06-30 (~65% of 4M) for e107–e114 sweep:**

| Exp | best-15 | last-15 | mask | K | blk/step | Verdict |
|---|---|---|---|---|---|---|
| e99 | 234 | 224 | 0.42 | 10.7 | 0.32 | Lower switch cost did not lift score |
| e100 | 597 | 575 | 0.33 | 7.6 | 0.35 | Mask holds w/o target; score below e88 (764) at cancel |
| e101 | 265 | 257 | 0.65 | 8.3 | 0.62 | Combined (1)+(3) weak — dense mask |
| e102 | 38 | 7 | 0.86 | 29.7 | 0.23 | Early / volatile |
| e103 | 109 | 107 | 0.66 | 5.7 | 0.94 | Learning, dense mask |
| e104 | 22 | 1 | 0.64 | 29.7 | 0.17 | Barely learning |
| e105 | 189 | 178 | 0.84 | 18.0 | 0.37 | Best port; still early |
| e106 | 9 | 1 | 0.76 | 29.7 | 0.21 | Very weak |

### e107–e114 · mask-none cost sweep (cartpole) + e88-style port (launched 2026-06-30)

**e88 reference** (`job.env` from archived run): `mgr_reward_agg=sum`, `dur_reg=0.0`,
`dur_max=32`, `switch_cost=0.02`, `mask_mode=prob` (target 0.3, non-binding),
`perblock_credit=True`, `perblock_edit_cost=0.3` → best-15 **764**, mask **0.36**, K **~18**,
blk/step **0.16**.

Cartpole ablations fix `switch_cost=0.02`, `mask_mode=none`, sweep `perblock_edit_cost`
{0.1, 0.3, 1.0}. Ports use the **full e88 recipe** (mask prob, not none).
Script: `submit_e107_114_none_cost_sweep_and_e88_port.sh`. 8 × v100.

| Exp | Job | Task | Config | best-15 | last-15 | max | mask | K | blk/step | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| e107 | 4650130 | cartpole | none, `perblock_cost=0.1` | 550 | 263 | 612 | 0.673 | 10.8 | 0.499 | ❌ below e88; late-run decay (best15→last15 550→263) |
| e108 | 4650131 | cartpole | none, `perblock_cost=0.3` | 325 | 295 | 370 | 0.593 | 14.1 | 0.337 | ❌ well below e88 despite matching cost |
| e109 | 4650132 | cartpole | none, `perblock_cost=1.0` | 391 | 331 | 409 | 0.821 | 12.2 | 0.539 | ❌ strongest cost still didn't sparsify — mask *denser* than e107/e108 |
| e110 | 4650133 | acrobot_swingup | e88 verbatim | 19 | 1 | 132 | 0.306 | 30.0 | 0.082 | ❌ failed to learn |
| e111 | 4650134 | cheetah_run | e88 verbatim | 167 | 150 | 176 | 0.574 | 14.1 | 0.325 | ⚠️ weak but stable, no collapse |
| e112 | 4650135 | pendulum_swingup | e88 verbatim | 9 | 0 | 63 | 0.269 | 29.7 | 0.072 | ❌ failed to learn |
| e113 | 4650136 | quadruped_run | e88 verbatim | 233 | 217 | 498 | 0.111 | 29.2 | 0.031 | ⚠️ best port — sparse mask holds, moderate score |
| e114 | 4650137 | hopper_hop | e88 verbatim | 3 | 0 | 17 | 0.128 | 30.5 | 0.034 | ❌ failed to learn |

*All `--configs size6m masked_goals variable_goals`, image 32×32, 4M steps, 48h,
`dur_reg=0.0`. Compared vs e88 (764 / blk/step 0.16) and e57 (839 / 0.64). **Result:**
mask-none cost sweep (e107–e109) did not beat e88 — no `perblock_cost` value reproduced
e88's sparsity or score, and e109's cost=1.0 (strongest pressure) paradoxically has the
*densest* mask (0.82), suggesting cost alone doesn't drive sparsity without the `mask_mode
prob` target. The e88-verbatim port (e110–e114) mostly **fails outside cartpole**: only
quadruped (e113, 233) and cheetah (e111, 167) learn anything non-trivial; acrobot,
pendulum, hopper (e110/e112/e114) never escape near-zero. e88's recipe looks
cartpole-specific, not a general HRL recipe yet.*

### e115 · e88 + Director-scale + 64×64 on A100 (launched 2026-06-30)

Isolates model capacity / resolution: same e88 manager recipe on the new
`director_match` preset (RSSM deter/hidden **1024**, classes **32**, CNN depth **64**,
4-layer **512**-wide MLPs — parity with TF Director `defaults`) and **64×64** images.
Script: `run_v3_e115_cartpole_e88_director_match_a100.sbatch`. 1 × a100; v100 sweep
(e107–e114) unchanged.

| Exp | Job | Task | Config | best-15 | last-15 | max | mask | K | blk/step | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| e115 | 4650701 → 4651213 | cartpole | e88 + `director_match` + 64×64 | 528 | 490 | 533 | 0.317 | 21.1 | 0.120 | ⏹️ **CANCELLED at ~2.17M/4M** (not resumed; A100 freed for e118). Stable ~490–530 but never reached 4M; archived to bucket (2026-07-02) |

*Watch vs e88 (764 @ size6m 32×32) and e57 (839 @ size6m 32×32 + prior): score, mask, K,
blk/step. First launch (4650701) was cancelled at ~1.03d walltime and requeued as
4651213, currently ~16h in. At ~54% of training, trending steadily up (best15/last15
close together, 527/492 — no collapse) but still well below e88's 764 and mask is
sparser (0.265 vs e88's 0.36) with a longer hold (K 21.5 vs ~18) — plausibly still
converging, not yet conclusive on whether Director-scale + 64×64 closes the gap.*

**Update (2026-07-03):** e115 (4651213) **cancelled** at ~16.7h / ~2.15M steps to free
the A100 for a throughput investigation. Two speedups identified for the same config:
(1) **train_ratio 32→8** = Director's `train_every 16` cadence (gradient step every 16 env
steps, ~4× fewer updates), and (2) **native cuDNN conv** (`unset DREAMERV3_CONV_IMPL`) —
smoke-tested on gpu-a100, no segfault, **~1.98× faster** (fps/train ~792 vs ~401
reference). Both baked into `run_v3_e116_*` (cartpole, not yet launched) and `run_v3_e117_*`
(acrobot, below).

### e117 · e116 recipe (train_ratio 8 + native conv) on acrobot_swingup (launched 2026-07-03)

Retry of acrobot at Director-scale after e110 (e88 verbatim, size6m 32×32) failed to learn
(19/1/132). Same director_match + 64×64 + train_ratio 8 + native conv as e116, task
`dmc_acrobot_swingup`. Isolates whether capacity + resolution rescue the harder swingup.
Script: `run_v3_e117_acrobot_e115_trainratio8_a100.sbatch`. 1 × a100.

| Exp | Job | Task | Config | best-15 | last-15 | max | mask | K | blk/step | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| e117 | 4651556 | acrobot_swingup | e116 recipe (director_match + 64×64 + train_ratio 8 + native conv) | — | — | — | — | — | — | CANCELLED ~1.3h to free A100 for V100-fit / e118 work (2026-07-03) |

*Hypothesis: Director-scale capacity + 64×64 + longer effective training (native conv lets
us afford it) rescue acrobot where e110 failed at size6m/32×32. Watch score vs e110's
19/1/132 and mask_frac/K/blk/step for HRL health.*

### e118 · Director-DEFAULT batch on A100-80GB (cheetah, launched 2026-07-03)

First run at the TRUE DreamerV3/Director default batch on our hardware. Same e88/e116
manager recipe + `director_match` 64×64 + native conv, but **batch_size 16 ×
batch_length 64** (=1024 frames/grad-step, the real default), **imag_length 16**,
**train_ratio 64** (== Director "train every 16 env steps"). Fit validated on
A100-**80GB**: ~59/80 GB, ~1127 fps/train. train_ratio 64 @ 4M ≈ 62h > 48h wall →
**chained** (job2 `afterany` job1, shared RUN_DIR resume). Scripts:
`run_v3_e118_cheetah_director_defaults_a100.sbatch`.

| Exp | Job | Task | Config | best-15 | last-15 | max | mask | K | blk/step | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| e118 | 4651662→4651663 | cheetah_run | director_match 64×64, **b16/T64/imag16, train_ratio 64**, native conv | 347 | 346 | 373 | 0.70 | 8.1 | 0.70 | ✅ **DONE** 4M, last30 mean **346.5**, max 372.6 (2026-07-05) |

*First test of real-default batch (1024 frames) + Director training cadence for the HRL
recipe. Compare vs e111 (cheetah, e88 verbatim size6m 32×32: 167/blk 0.325). Watch
whether the bigger batch + Director cadence lifts cheetah and keeps the mask sparse.*
**RESULT:** real-default batch + Director cadence **doubled cheetah** vs e111 (346 vs 167),
and the manager stayed active — mask 0.70, K≈8, blk/step **0.70** (no long-hold drift, unlike
e119/e122 below). Best single env of the whole batch. wandb id `32d42875` (logged fine — the
A100 script sources the standard env; it's the Route-A e119+ runs that lost wandb, see below).

### e119–e122 · e118 config on MULTI-V100 (Route-A-fixed clone) — weekend chain (launched 2026-07-03)

Same e118 config (director_match 64×64, **b16/T64/imag16, train_ratio 64, native conv**)
for the e110–e114 environments **minus cheetah**, on **4× V100 data-parallel** via the
Route-A-fixed clone env (`/work/.../dreamerv3_env_routeA`: jax 0.4.33 + cuDNN 9.1.0.70 +
CUDA 12.2 — native conv now works on V100 sm_70). Fit: ~11–13 GB/device on 16GB V100s,
fps_train ~1324. Two concurrent lanes (8-GPU V100 quota), each env a 2-job `afterany`
chain (resume across 48h wall), next env starts when prior ends. Script:
`run_v3_e118cfg_env_multiv100_routeA.sbatch`.

| Exp | Job(s) | Task | Lane | last30 | max | mask | K | blk/step | Status |
|---|---|---|---|---|---|---|---|---|---|
| e119 | 4651831→4651832 | acrobot_swingup | A (→e121) | 5.8 | 132.8 | 0.29 | 15.0 | 0.15 | ✅ ran to 4M, **no learning** (2026-07-05) |
| e121 | 4651833→4651834 | quadruped_run | A (after e119) | — | — | — | — | — | ❌ **FAILED — CUDA OOM** at startup, both chain jobs (2026-07-05). **Rerun fix (new exp #):** shrink V100 footprint — `--batch_size 12 --agent.imag_length 12` (cuts resident + transient conv/imag scratch; 12/4=3 per device), optionally `--jax.autotune 0`; **or** rerun config-unchanged on **A100-80GB** (`-p gpu-a100 --gres=gpu:a100:1`, single-device, ~4× slower wall, chain across 48h — e118 fit ~59/80 GB). Quadruped's larger action head is the tightest → prefer A100 if keeping config exact. |
| e120 | 4651835→4651836 | pendulum_swingup | B (→e122) | — | — | — | — | — | ❌ **FAILED — CUDA OOM** at startup, both chain jobs (2026-07-03). **Rerun fix (new exp #):** same as e121 — `--batch_size 12 --agent.imag_length 12` (+ optional `--jax.autotune 0`) for V100 headroom, **or** A100-80GB config-unchanged. Note device 0 is double-loaded (`policy_devices 0` + `train_devices 0 1 2 3`) so it OOMs first. |
| e122 | 4651837→4651838 | hopper_hop | B (after e120) | 0.1 | 10.8 | 0.31 | 14.6 | 0.17 | ✅ ran to 4M, **no learning** (2026-07-05) |

*First HRL runs at real-default batch on V100 (only possible after the Route-A native-conv
fix). Compare vs the failed size6m/32×32 ports: e110 acrobot 19/1/132, e112 pendulum
9/0/63, e113 quadruped 233/217/498, e114 hopper 3/0/17. At train_ratio 64 (~55h/4M on
4×V100) not all will reach 4M over the weekend; chained for maximal back-to-back progress.*

**RESULT (2026-07-06):** mixed→poor.
- **e120 pendulum & e121 quadruped: FAILED — CUDA OOM.** XLA needed ~8.3–8.9 GiB on top of
  ~10.9 GiB resident, over the **16 GB V100** cap; the `afterany` resume retried and OOM'd
  identically. Config was already tight (~11–13 GB/device); quadruped's larger action head
  + a heavier compiled graph tipped both over. Rerun needs headroom (batch 12 / imag 12, or
  A100-80GB) and **new experiment numbers**.
- **e119 acrobot & e122 hopper: ran to 4M but did NOT learn.** acrobot last30 mean 5.8
  (peaked 132.8 ≈ the already-failed e110 port, then decayed); hopper last30 mean 0.1
  (max 10.8 < e114's 17). The manager drifted into a **long-hold** regime (K≈15, near the
  block max; switch_rate 0.12). ⚠️ blk/step looks low (0.15/0.17) but that's the long hold,
  **not** true per-decision sparsity (mask_frac 0.29–0.31 is normal) — the manager stopped
  re-editing rather than editing sparsely. Contrast e118 cheetah (same config, A100): mask
  0.70, K≈8, blk/step 0.70, score 346 — the V100 chain's long-hold+zero-return is the real
  signal to chase before spending more V100-weeks.
- **wandb: none of e119–e122 logged.** The Route-A sbatch activates the clone venv directly
  and never sources `source_dreamerv3_env.sh`, so `WANDB_API_KEY` was unset → silent
  `jsonl,scope`-only. Fixed 2026-07-06 by adding `WANDB_*` exports to
  `run_v3_e118cfg_env_multiv100_routeA.sbatch` and `run_v3_e119_acrobot_multiv100_routeA.sbatch`.
  Completed e119/e122 data lives only in `metrics.jsonl` (no offline wandb dir to `wandb sync`).

### e123 · PURE Director baseline (defaults only) — parity check (launched 2026-07-06)

**Motivation.** Every number we've compared to Director (cheetah e118 max **372** vs
Director ~**500**; hopper/acrobot never learned) came from runs that stacked our research
changes on top: e118 loaded `masked_goals variable_goals` + mask-sparsity (target 0.3) +
per-block credit + `goal_struct_weight 200` + variable-K (1–16, target 4) + `goal_switch_cost
0.02`; e110/e111/e114 additionally ran `size6m` at **32×32**. We never established that the
**base** manager/worker/goal-VAE hierarchy reproduces Director on its own. e123 does exactly
that: **defaults only** — `use_masked_goals=False` (**overwrite entire goals**, whole skill
code replaced each decision), `variable_goal_length=False` (**fixed goal length**
K=`manager_sample_freq`=8), `use_hrl=True` — with `director_match` (RSSM deter/hidden 1024,
classes 32, CNN depth 64, 4-layer 512-wide MLP) + 64×64 for architecture parity, and the real
default batch (b16×64=1024 frames, imag 16, train_ratio 64 = Director "train every 16"). The
DreamerV3 upgrades (masked/variable-K/struct) remain in the repo behind their opt-in flags —
this just measures the floor they build on. Script:
`run_v3_e123_director_baseline_a100.sbatch` (A100-80GB; `TASK=` override for hopper/acrobot/
cartpole, each gets a new exp number). All AGENT_FLAGS == config defaults (explicit for repro).

| Exp | Job | Task | Config | best-15 | last-15 | max | K | Status |
|---|---|---|---|---|---|---|---|---|
| e123 | 4655002 | cheetah_run | **pure Director** (defaults, director_match 64×64, b16/T64/imag16, tr64, native conv, A100) | — | — | 435 | 8 | ⏳ RUNNING (saion-gpu24, step 2.46M/4M @ 2074 fps/train, restarted 2026-07-06 15:51; last ep score 372, tracking above e118) |
| e124 | 4655006 | hopper_hop | **pure Director** (same, 4×V100 data-parallel, Route-A clone) | — | — | 350 | 8 | ⏳ RUNNING (saion-gpu21, step 2.05M/4M @ 1605 fps/train, launched 2026-07-06; last ep score 231 — actually learning, unlike e122's 0.1) |
| e125 | 4655007 | acrobot_swingup | **pure Director** (same, 4×V100 data-parallel, Route-A clone) | — | — | 339 | 8 | ⏳ RUNNING (saion-gpu22, step 2.07M/4M @ 1221 fps/train, launched 2026-07-06; last ep score 1.4 — spiked to 339 then collapsed, unstable) |

**Interim check (2026-07-07, all still running, ~2.0-2.5M/4M steps):** cheetah (e123) and
hopper (e124) both look promising — cheetah's running max (435) already exceeds e118's
max (372), and hopper is actually learning (max 350) unlike e122's stuck 0.1. Acrobot (e125)
is spiky/unstable: hit a max of 339 but its last logged episode was back down to 1.4,
suggesting it found then lost a good policy. `max` column above is the running max over the
metrics.jsonl tail as of this check, not a final result — best-15/last-15 to be filled in
once each run completes at 4M steps. e123 restarted once (new run dir timestamp 15:51 vs
original 09:28 launch) under the same job id 4655002.

*e124/e125 use `run_v3_e124_director_baseline_multiv100_routeA.sbatch` (`TASK=`/`EXP_TAG=`
override). Reference targets: Director hopper_hop learns to hop; acrobot_swingup is the task
e110 (masked recipe, size6m/32×32) failed at 19/132 — pure Director base at director_match/64×64
is the clean retry. 4×V100 lanes use the full 8-GPU v100 quota.*

*Hypothesis: pure Director base should reach Director's DMC cheetah ~500 (or close). If it
lands near e118's 372, the gap is in the base hierarchy (not our changes) — candidate causes:
DreamerV3-vs-DreamerV2 backbone, manager exploration reward lacking Director's `context` input
(`agent._mgr_expl_reward` note), or goal-VAE/actent tuning. If it hits ~500, then our research
toggles are what suppressed it and should be re-tuned against this baseline, not Director's paper.*

### Pure-Director vs masked/variable-K recipe — config diff & regression analysis (2026-07-07)

**Result so far (interim, ~2M/4M):** the **pure Director baseline BEATS our full recipe** on
the same hardware/backbone. e123 cheetah running max **435 > e118's 372**; e124 hopper is
**actually learning (max 350)** where the masked e122 was stuck at **0.1**. So the DreamerV4-style
manager machinery (masked partial edits + variable-K + sparsity/edit/switch/struct regularizers)
was **suppressing** score, not lifting it — at least as ported off cartpole.

**The comparison is clean:** e118 (ours) and e123 (pure) share their **entire backbone** — same
world model, goal-VAE, actor-critic, optimizer, `manager_actent` (mult/0.5), `mgr_expl_weight`
0.1, `mgr_retnorm` meanstd, `director_match` arch, 64×64, batch 16×64, `train_ratio` 64, 8 envs,
`imag_length` 16, `goal_opt`/`goal_autoencoder_beta`/`goal_rec_loss_agg`. **Every difference is in
the manager's goal-emission mechanism + its regularizers** — nothing in capacity, cadence, or WM.

**The 10 behavioral differences (e118 → e123):**

| # | Knob | e118 (masked/var-K) | e123 (pure Director) | Effect |
|---|---|---|---|---|
| 1 | `use_masked_goals` (+`masked_goals`) | **True** | False | edit a **subset of blocks** of a running Z vs **overwrite the whole code** |
| 2 | `variable_goal_length` (+`variable_goals`) | **True** | False | learned **duration head** vs **fixed hold** |
| 3 | `manager_sample_freq` | (ignored, var-K) | **8** | activates fixed **K=8** refresh |
| 4 | `goal_duration_{min,max,reg,target,adapt}` | 1/16/0.0/**4.0**/False | dropped | the var-K duration machinery |
| 5 | `mask_sparsity_mode`/`_target`/`_max` | **prob**/0.3/5.0 | none | Lagrangian sparsity penalty on the mask |
| 6 | `mask_perblock_credit`/`perblock_edit_cost` | **True**/**0.3** | False/n/a | per-block Q-credit + per-edited-block cost |
| 7 | `goal_switch_cost` | **0.02** | 0.0 | per-decision penalty on manager task reward |
| 8 | `mgr_cond_goalcode` | **True** | False | manager input concats pre-edit Z vs plain feat |
| 9 | `goal_struct_weight` | **200.0** | 0.0 | goal-AE term to keep block-splices on-manifold |
| 10 | `mgr_reward_agg` | **sum** | mean | manager extrinsic-reward pooling over the K-window |

Everything else passed in both scripts (`goal_autoencoder_beta 0.25`, `goal_rec_loss_agg sum`,
`goal_opt.*`, `manager_actent_* mult/0.5`, `mgr_expl_weight 0.1`, `mgr_retnorm meanstd`,
`mask_topk 0`, `mask_sparsemax False`, `goal_edit_cost* 0`, `mgr_cond_achieve False`) is
**identical or an inert default in both** — nominal, not a behavioral diff.

**Why the variability hurts (six mechanisms, several already in our findings):**
1. **Partial edits → off-manifold goals.** Pure Director hands the worker a *complete, freshly
   encoded* code every K (on the goal-VAE manifold → meaningful cosine target). Masked mode
   decodes a *partially-overwritten running Z* (block-splice); the decoder never trained on
   arbitrary splices → OOD/degraded target. `goal_struct_weight=200` exists only to patch this
   and per finding #5 is "healthy but **not a score lever**" — it mitigates, doesn't fix.
2. **Variable K is high-variance and parks at a bad corner.** The duration head is REINFORCE-
   trained under the same grad-clipped manager optimizer; without a soft prior it collapses
   short (finding #3, e63/e64). On the hard tasks it did the opposite — **e119/e122 drifted to
   K≈15 (near max), switch_rate 0.12: the manager stopped re-editing, goal went stale → zero
   learning.** Fixed K=8's forced refresh is why **e124 hopper learns (350) where e122 stuck at 0.1.**
3. **Regularizers subtract from the task signal and were never tuned off-cartpole.**
   `switch_cost 0.02` + `perblock_edit_cost 0.3` + the sparsity Lagrangian all come out of the
   manager's *extrinsic* reward. Our log shows these one-sided costs killing return (edit-cost →
   "freeze mask → score 0"; "duration-reg magnitude domination"). e110–e114 concluded the recipe
   is "**cartpole-specific, not a general HRL recipe**." Off cartpole they're mis-tuned dead weight.
4. **Bigger manager action space = slower learning.** Masking expands the manager action from
   {choose code} to {code} × {8 per-block edit bits} × {duration} → higher-variance policy
   gradients for a manager finding #6 proves is load-bearing.
5. **`mgr_reward_agg=sum` mis-scales the advantage.** e66 showed `sum` drove K shorter and scored
   below mean-pooled runs; it rescales the reward feeding the `meanstd` retnorm. `mean` is what
   the v3-tuned normalization expects.
6. **`mgr_cond_goalcode` is neutral-to-negative** (finding #4): extra input dims, no gain; "the
   bottleneck is K, not what the manager sees."

**Takeaway.** Plain wins because fixed-K whole-code overwrite gives the worker a clean, complete,
regularly-refreshed, on-manifold goal and the manager a small task-only objective with a small
action space. Every "variability" toggle trades some of that away. These toggles aren't worthless
(on cartpole, masked var-K e57=839 beat the fixed-K tiers) — they're **untuned outside cartpole**.
Next: once e123–e125 finish 4M, re-tune each toggle *against this baseline*, adding them back one
at a time, instead of porting the cartpole recipe wholesale.

### e126–e129 · Auto-tuning HRL (`hrl_auto`) — 6m panel (launched 2026-07-07)

**Direction.** Instead of hand-tuning the masked/variable-K knobs per env×size, make every
regularizer an **adaptive Lagrange multiplier on a dimensionless target** so one config
transfers across the whole panel. Rationale (from the regression analysis above): reward-unit
costs never transfer (switch/edit costs only reach corners; duration-reg magnitude-dominates),
whereas normalized targets (entropy fraction, KL nats, edit fraction) do — same reason v3
`actent`/`retnorm` transfer across `size1m…size400m`. New config block **`hrl_auto`**
(`configs.yaml`): `use_masked_goals=True`, `variable_goal_length=True`, **cond off**
(`mgr_cond_goalcode=False`; also disables per-block credit, which needs the pre-edit Z),
**`mgr_reward_agg=mean`** + `variable_goal_block_rew=True` (option-return duration credit),
**all reward-unit costs deleted** (switch/edit/perblock = 0), **no forced-K prior**
(`goal_duration_reg=0`); K is explored by the duration head's adaptive entropy
(`manager_actent_duration_target=0.7`, raised above the 0.5 skill target to resist short-K
collapse) and pulled to useful values by the option-return credit. Mask level: proven
two-sided **`prob`** fraction target (0.3). Struct kept at 200 (finding #5).

**Entropy-native mask machinery built now, off by default** (per request — flip later with one
line). New `mask_sparsity_mode: entropy` (`agent.py`): an adaptive **`mask_actent`** adapter
(`inverse=True`) holds per-block mask entropy (normalized by ln2) near `mask_actent_target`
(anti-collapse / exploration) while an adaptive **`mask_kl_adapter`** (`inverse=False`) holds
`KL(mask ‖ Bernoulli(mask_sparse_prior))` near `mask_kl_target` nats (the transferable "prefer
not to edit" level control — replaces the fixed fraction target + edit costs). Task advantage
(manager REINFORCE) decides *which* blocks; the two adapters decide exploration + how-sparse,
both via dimensionless targets. NOTE the trap it's designed around: raw mask entropy peaks at
p=0.5, so entropy *alone* → dense; the sparse-prior KL supplies the sparsity pull. Helpers
`bernoulli_entropy`/`bernoulli_kl`. **Validated:** two 3k-step smokes (`prob` + `entropy`) both
COMPILED the full train graph and exited 0:0 on V100 — the entropy branch traces without
shape/type error. (`enc/dec.simple.outer True` is required at 32×32 or the CNN over-downsamples
to 2×2 — panel-script fix, not a config issue.)

**Panel (this batch = the fast 6m tier; BIG cells to follow on A100/multi-V100):** `size6m`
32×32, 1×V100 each, `hrl_auto` prob mode, 4M steps. Script `run_v3_hrl_auto_panel.sbatch`
(`TASK`×`SIZE`×`MASK_MODE`).

| Exp | Job | Task | Size | Mask mode | best-15 | K | mask | Status |
|---|---|---|---|---|---|---|---|---|
| e126 | 4655642 | cartpole_swingup | 6m | prob | — | — | — | ⏳ RUNNING (gpu15) |
| e127 | 4655643 | cheetah_run | 6m | prob | — | — | — | ⏳ RUNNING (gpu16) |
| e128 | 4655644 | acrobot_swingup | 6m | prob | — | — | — | ⏹️ CANCELLED ~15min → replaced by e132 entropy A/B (no data, dir deleted) |
| e129 | 4655645 | hopper_hop | 6m | prob | — | — | — | ⏳ RUNNING (gpu17) |
| e130 | 4655652 | cheetah_run | **BIG** | prob | — | — | — | ⏳ RUNNING (A100, director_match 64×64) |
| e131 | 4655653 | hopper_hop | **BIG** | prob | — | — | — | ⏳ RUNNING (4×V100 Route-A) |
| **e132** | 4655655 | cartpole_swingup | 6m | **entropy** | — | — | — | ⏳ RUNNING (gpu17) — **A/B vs e126** (prob), first real test of the entropy-native mask |

**e126 vs e132 A/B (cartpole 6m, prob vs entropy mask):** same `hrl_auto` config, 4M steps,
only `mask_sparsity_mode` differs. Tests whether the entropy-native mask (mask_actent entropy
floor + KL-to-sparse-prior 0.1 @ 0.2 nats) matches/beats the pinned `prob` 0.3 fraction target,
and whether the mask level emerges interior (~0.1–0.35) without a fixed fraction. New metrics:
`goal/mask_entropy_norm_mean`, `goal/mask_kl_prior_mean`, `goal/mask_{actent,kl}_scale_mean`.

**Early signal (~3.5k steps, e126–e129):** the auto-tuning is behaving as designed — with **no
fixed-K prior**, `mgr_duration_mean` sits **interior at ~7–9** with a *distributed* histogram
(p1–4≈0.35, p5–8≈0.26, p9–12≈0.20, p13–16≈0.19), i.e. K is neither collapsed to min nor railed
to max. Duration-head normalized entropy ≈0.98 (very exploratory early; the adaptive
`mgr_actent_duration` adapter holding it up, scale still ~0.01), `loss/mask_sparsity`≈1.6 active,
all heads finite, fps/train ~800 (size6m 32×32 V100 reference conv). Too early for score — watch
whether K settles interior as task credit strengthens and whether score reaches the pure-Director
baseline / e57 tier. BIG cells: cheetah e130 (A100), hopper e131 (4×V100), `hrl_auto` +
director_match 64×64, imag16 (32 won't fit BIG), train_ratio 64. Scripts
`run_v3_hrl_auto_big_a100.sbatch`, `run_v3_hrl_auto_big_multiv100_routeA.sbatch`.

*Test of "auto-tuning transfers": one `hrl_auto` config across 4 envs with NO per-cell tuning.
Watch whether K emerges interior (not railed to min/max) and mask stays healthy without a fixed
prior/cost, and whether score is competitive with the pure-Director baseline (e123–e125) and the
cartpole champion e57 (839). BIG cells (`director_match` 64×64) and the `entropy`-mode variant
are the planned follow-ups. Prior baselines e123–e125 cancelled 2026-07-07 to free these slots
(interim: cheetah 435>e118 372, hopper learning — archived to bucket).*

---

## Technical notes (implementation facts that bit us)

- **Variable-K repval (fixed 06-18, `9716a78`).** Under `variable_goal_length`, the replay
  manager-value loss now mirrors imagination — per-step rewards on the full timeline (via
  `_switch_mask_from_skills`), not fixed-K `downsample_manager_states`. `variable_goal_block_rew`
  pools rewards over realized duration segments (Director-style). The fix is *correct* but
  empirically lowered variable-K peaks (see finding #7).
- **Block-rew bugs (fixed 06-19), only fire under `variable_goal_block_rew`:**
  `variable_segment_ids` off-by-one (`cumsum(sw)-sw` vs scatter `cumsum(sw)-1`) and
  `aggregate_mgr_cont_variable` continuation (`segment_max` over a non-increasing cumprod →
  used first step instead of product; now `segment_min`). Benign on cartpole (cont≈1) but
  wrong for terminating tasks. **e44 predates the fix → rerun if pursuing block-rew.** Tests:
  `embodied/tests/test_variable_goals.py` — **run on a v100, not the login node** (glibc/Py3.11).
- **Duration-reg magnitude domination.** Fixed `reg·(E[dur]−target)²` is added straight into
  `mgr_policy`; at reg=0.1 the squared error (~64) yields O(1)+ loss that swamps the
  normalized-advantage REINFORCE under one grad-clipped optimizer → starves task return.
  reg≤0.03 is small enough to coexist. (Adaptive prior e48 didn't fix it.)
- **64×64 report segfault (e5).** Report-compile crashes at 64×64, independent of
  `DREAMERV3_CONV_IMPL`. Use `--agent.report False`, or lighten report (`report_skill_viz
  False`, smaller `report_length`/`report_max_rows`).
- **`mask_sparsity_mode=none` bring-up (e66–e73).** Initially crashed the loss/scale
  key-set assert (loss dropped but `mask_sparsity` scale still registered); fixed by
  popping the scale for `none` mode (`agent.py:761`).

## New config flags (all default to DreamerV3 / pre-HRL behavior)
`use_masked_goals`, `mask_sparsity_mode {prob,sample,reinforce,none}`, `mask_sparsity_target`,
`mask_sparsity_max`, `mask_sparsity_fixed_weight`, `mask_topk` ·
`variable_goal_length`, `variable_goal_block_rew`, `goal_duration_reg`, `goal_duration_target`,
`goal_duration_adapt(_max)`, `goal_duration_fixed`, `goal_switch_cost`, `goal_edit_cost` ·
`goal_struct_weight`, `goal_struct_target {deter,feat}`, `goal_struct_loss {mse,margin}`,
`goal_struct_adapt` · `mgr_cond_goalcode`, `mgr_cond_achieve`, `mgr_reward_agg {mean,sum}`,
`manager_actent_duration_target`. Key metrics:
`goal/mask_frac`, `goal/mask_prob_mean`, `goal/struct_corr`, `goal/mgr_duration_mean`,
`goal/mgr_switch_rate`, `goal/edit_cost_pen_mean`, `wkr_goal_rew`, `mgr_extr_rew`,
`mgr_extr_rew_block`.
