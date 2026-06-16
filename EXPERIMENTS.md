# Experiment Log

Naming: every run starts with `e{N}_{env}` (e.g. `e5_cartpole`) — used as the SLURM
job name and the run-dir/logdir prefix. Experiment numbers are **global and monotonic**
(never reused). A requeue with a tweak gets a new number; a literal restart keeps it.

Columns: **Exp** · **Job id(s)** · **Investigated hyperparams** · **Hypothesis** ·
**Expected result** · **Status / outcome**. Add the row at launch; fill the outcome later.

Current campaign: "masked / overwrite goals" on Director-style HRL (branch
`feat/hrl-masked-goals`), DMC cartpole swingup, size6m, image 32×32. Control baseline
(HRL no-mask) solves cartpole at ~710.

| Exp | Job id | Investigated hyperparams | Hypothesis | Expected result | Status / outcome |
|---|---|---|---|---|---|
| e1_cartpole | 4633052 | masked goals, soft sparsity `mask_sparsity_target=0.3` | Code-space running-goal overwrite + soft sparsity recovers Director perf and yields sparse edits | Solves cartpole (~710) with ~30% blocks edited/command | **Done (timeout @4M/24h).** Score ~650 (max ~750), manager learns (mgr_extr_rew 0.07→0.29). **But mask collapsed to all-ones** (sparsity adapter pinned at ceiling, couldn't enforce). Diagnostic (`diag_codes.py`): code is dense/distributed, partial edits on-manifold but each block ≈1/8 of goal → sparse edits semantically weak. |
| e2_cartpole | 4633139 → 4633272 → **cancelled** | `mask_topk=3` (hard edit budget, soft penalty off) | Forcing exactly 3 edited blocks/command (vs soft penalty the manager overrode) makes sparsity stick | Cartpole still solved; mgr learns *which* 3 blocks; perf may be throttled by smaller per-command goal moves | **FAILED.** Ran to ~2.95M/4M, mask_frac locked at 0.375, but `episode/score ~0.07` — forced topk budget strangled the policy. Cancelled 06-10 to free a v100 for e4. |
| e3_cartpole | 4633166 → 4633273 → **cancelled @2.57M** | `goal_struct_weight=200` (geometry-preservation goal-AE term; soft sparsity 0.3) | Making code-space cosine_max distances mirror deter distances gives the dense code a smooth metric → predictable partial edits | `goal/struct_corr` climbs above ~0.88 baseline toward 1; cartpole still solves | **WORKED but mask collapsed.** Climbed 240→610 (max 735) by 2.57M — solving cartpole. BUT `train mask_frac=0.99` (mask all-ones) → effectively plain-overwrite+struct, not masking; `struct_corr 0.88→0.91`. Cancelled 06-10 (ckpt ~2.57M, resumable) to free a v100 slot for e6 (K=1). |
| e4_cartpole | 4633343 → **4633384** | `mask_topk=3` + `goal_struct_weight=200` | Forced-sparse budget **on a metric-preserving code** is the cell where sparse masking should actually help | Each of 3 edited blocks moves goal predictably; best chance masking improves over plain overwrite | **Cancelled 06-11 @~3.79M/4M** (ckpt resumable) to free v100s for e8–e11. Outcome: score ~0–11 — the topk budget strangled the policy, same collapse as e2 (confirmed). Resumed ~270k/4M earlier for a machine update. |
| e5_cartpole | 4633344 → **4633385** | Director `size6m_director`, image **64×64**, `--agent.report False` | A 64×64 Director baseline trains on this setup | Trains to 4M steps at 64×64 | **Cancelled 06-11 @~1.79M/4M** (ckpt resumable) to free v100s for e8–e11. Outcome: 64×64 Director baseline trains fine, score ~720 (slow ~20 fps). Open: lighten report to restore eval videos at 64×64. |
| e6_cartpole | 4633345 → **4633386** | e4 params (`mask_topk=3` + `goal_struct_weight=200`) **+ `manager_sample_freq=1` (K=1)** | e2/e4 collapsed because the worker couldn't reach a forced-sparse goal over K=8 steps; re-commanding the goal **every step** (K=1) reduces it to a 1-step goal the worker can track, so forced-sparse masking may finally work | Cartpole solves with genuinely sparse masks (`mask_frac=0.375`), unlike e2/e4 (~0) | **Cancelled 06-11 @~3.42M/4M** (ckpt resumable) to free v100s for e8–e11. Outcome: `mask_frac` held 0.375 (forced) but score only ~100 — K=1 rescued forced-topk from e4's ~0 but still throttled. Script `run_v3_e6_cartpole_topk_struct_k1.sbatch`. |
| e7_cartpole | 4633346 → **4633387** | e6 but **`mask_topk=0`** (hard budget off → soft adaptive sparsity 0.3) + `goal_struct_weight=200` + `manager_sample_freq=1` (K=1) | K=1 control for e6: with soft (not forced) masking at K=1, does the mask stay collapsed-to-all-ones (like e1/e3 at K=8) or does re-commanding every step let soft sparsity bite? | Either solves cartpole with mask collapsed (soft sparsity still loses) **or** holds a sparse mask while solving | **Cancelled 06-11 @~3.45M/4M** (ckpt resumable) to free v100s for e8–e11. Outcome: score ~840 (beats baseline) BUT `mask_frac` 0.51→0.77 (collapsing). **Root cause found** (see note below): the soft penalty was a no-op; e8–e11 are the fix. Script `run_v3_e7_cartpole_struct_k1.sbatch`. |
| e8_cartpole | **4633602** | **B2: `mask_sparsity_mode=prob`** — penalize the differentiable sigmoid probs of the mask head (was: gradient-free hard sample). `mask_sparsity_max` 100→5, struct=200, K=1 | The e1/e3/e7 collapse was because the sparsity penalty had no gradient to the discrete mask head. Penalizing head *probabilities* makes the existing adapter actually pull `mask_frac` to 0.3 | `mask_frac`/`mask_prob_mean` hold near 0.3 (not drifting to 1.0) while cartpole still solves (~710) | **✅ SUCCESS — the campaign's one win. Cancelled 06-12 @3.52M/4M** (converged, ckpt resumable). Score ~800 (> 710 baseline), `mask_frac` settled **sparse ~0.15–0.21** (≈1.5 of 8 blocks/cmd) with **no collapse to dense** — the B2 gradient fix works. Notably `sp_scale≈0`: the one-sided "don't exceed 0.3" adapter never engaged; sparsity came for free from REINFORCE return at K=1, and B2 simply *prevented the runaway-to-100* seen in e7. Caveat: `wkr_goal_rew` (cosine) only ~0.37 — worker tracks goals loosely; manager is still load-bearing (proven by e12/e14 freezing edits → score 0). Script `run_v3_e8_cartpole_maskprob_k1.sbatch`. New metric `goal/mask_prob_mean`. |
| e9_cartpole | **4633603** | B2 (`mask_sparsity_mode=prob`, struct=200) but **K=8** | K=8 control for B2: with the fixed differentiable penalty, does sparsity hold at the original K=8 temporal abstraction (where the no-op penalty collapsed in e1/e3)? | `mask_frac` near 0.3 + cartpole solved at K=8 | **❌ FAILED (over-sparsified to inert). Cancelled 06-12 @3.59M/4M.** B2 at K=8 drove `mask_frac → 0.001` (manager edits **nothing** → goal Z frozen). Worker reaches the frozen goal well (`wkr_goal_rew 0.71`) but it's useless → `mgr_extr_rew 0.01`, **score ~1.0**. Vs e8 (K=1, 0.15): without K=1's per-step REINFORCE return holding edits up, B2's downward push overshoots to zero. Script `run_v3_e9_cartpole_maskprob_k8.sbatch`. |
| e10_cartpole | 4633604 → **4633606** | **C: `mask_sparsity_mode=reinforce`** — subtract `sg(scale)·mask_frac` from the manager extrinsic reward so the mask logp is reinforced toward sparser commands (unbiased discrete estimator). `mask_sparsity_max=1.0`, `init=0.1`, struct=200, K=1 | REINFORCE reward-shaping is the principled fix for a discrete mask; should hold sparsity without the bias of straight-through/prob penalties | `mask_frac` near 0.3 while solving; compare sparsity/score tradeoff vs B2 | **❌ FAILED (mode broken). Cancelled 06-12 @3.54M/4M.** Reward-shaping rails `sp_scale` to its max (1.0) and drives `mgr_extr_rew` **negative (−0.44)**, yet `mask_frac` stays **dense (0.60)** — the penalty wrecks the manager's task signal without reducing density. **Score ~3.** Sign/scale fights the return and loses. Do not pursue as-is. Script `run_v3_e10_cartpole_maskreinf_k1.sbatch`. New metric `goal/mask_sparsity_reward_pen_mean`. |
| e11_cartpole | 4633605 → **4633607** | C (`mask_sparsity_mode=reinforce`, struct=200) but **K=8** | K=8 control for C: does reward-shaped sparsity hold a sparse mask at K=8 while solving? | `mask_frac` near 0.3 + cartpole solved at K=8 | **❌ FAILED (worst case). Cancelled 06-12 @3.56M/4M.** Even worse than e10: `sp_scale` railed to 1.0, `mgr_extr_rew −0.97`, but `mask_frac` pinned **fully dense (0.995)**. **Score ~19.** REINFORCE mode is broken at both K. Script `run_v3_e11_cartpole_maskreinf_k8.sbatch`. |
| e12_cartpole | **4633623** | **B2 with a FIXED multiplier (`mask_sparsity_fixed_weight=10`) at target 0** (constant pressure to edit no blocks) instead of the adaptive 0.3 adapter. `mask_sparsity_mode=prob`, struct=200, K=1 | Floor ablation: with editing actively discouraged toward 0, only edits whose task-return benefit beats the fixed cost survive. Either the manager still solves cartpole with very few essential edits, or it collapses to a frozen goal | `mask_frac`→~0; score reveals whether a near-static goal still solves cartpole (and which blocks REINFORCE refuses to drop) | **❌ FAILED (as designed — clean negative control). Cancelled 06-12 @3.45M/4M.** Weight 10 crushed `mask_frac → 0` (goal Z frozen) → **score 0**. Confirms **a static goal cannot solve cartpole** ⇒ the manager's edits are necessary (worker is *not* ignoring it). Script `run_v3_e12_cartpole_maskprob_fixed0_k1.sbatch`. |
| e13_cartpole | **4633624** | e12 (B2 fixed-multiplier sparsity→0, struct=200) but **K=8** | K=8 control for the drive-to-0 floor: does anything survive at the original K=8 abstraction, and how does score compare to K=1 (e12)? | `mask_frac`→~0; score vs e12 | **❌ FAILED. Cancelled 06-12 @3.47M/4M.** `mask_frac → 0`, **score 0.** Same as e12; nothing survives at K=8 either. Script `run_v3_e13_cartpole_maskprob_fixed0_k8.sbatch`. |
| e14_cartpole | **4633625** | e12 but **milder fixed weight (`mask_sparsity_fixed_weight=1`** vs 10), still target 0, prob mode, struct=200, K=1 | Weight 1 is *below* the ~2.5 the adaptive adapter equilibrated at — a gentle constant push. Does REINFORCE's task return hold the edit fraction up against a mild fixed cost, or does even weak pressure drive it down? | `mask_frac` settles somewhere between e8's ~0.3 and e12's ~0; score vs e8/e12 maps the sparsity↔performance tradeoff | **❌ FAILED. Cancelled 06-12 @3.40M/4M.** Even weight **1 zeroes `mask_frac`** → frozen goal → **score 0**. The fixed-cost floor is far more brittle than expected: there is no gentle regime — any constant push to target 0 collapses edits entirely. Script `run_v3_e14_cartpole_maskprob_fixed0w1_k1.sbatch`. |
| e15_cartpole | **4633626** | e14 (B2 fixed weight=1, target 0, struct=200) but **K=8** | K=8 control for the milder fixed-cost ablation | `mask_frac` mid-range; score vs e14 (K=1) and e9 | **❌ FAILED. Cancelled 06-12 @3.47M/4M.** `mask_frac → 0`, **score 0.** Confirms e14 at K=8. Script `run_v3_e15_cartpole_maskprob_fixed0w1_k8.sbatch`. |
| e16_cartpole | **4633987** | **Manager-input conditioning**: feed the manager the PRE-edit running code Z (`mgr_cond_goalcode`) **+ achievement cosine** (`mgr_cond_achieve`). e8 recipe (B2 prob, struct=200), **K=1, target 0.3**. Fresh. | The manager block-overwrites Z while blind to Z and to whether the worker reached the goal; giving it both should let it edit deliberately + hold reached goals, lifting `wkr_goal_rew` above e8's ~0.37 without collapse | `wkr_goal_rew` > 0.37, `mask_frac` stays >0 (ideally event-triggered), score ≥ ~800; `goal/mgr_cond_achieve_mean` rises | **DONE** (TIMEOUT @3.34M/4M). score **673**, wkr_goal **0.294**, mask_frac 0.256, achieve 0.60. Achievement channel healthy but **did not** lift K=1 reaching — both axes ≤ e17 (Z-only). Conditioning neutral-to-negative here. |
| e17_cartpole | **4633988** | e16 but **Z only** (`mgr_cond_achieve` off). K=1, target 0.3 | Isolates the effect of the achievement signal vs. Z-conditioning alone | vs e16: does achievement add value over just seeing Z? | **DONE** (TIMEOUT @3.67M/4M). score **777** (best K=1, ≈ e8), wkr_goal **0.341** (≈ e8 ~0.35), mask_frac 0.169. Best K=1 arm; achievement (e16) did not beat Z-only. |
| e18_cartpole | **4633989** | e16 (Z + achieve) but **target 0.125** (lower sparsity target) | Lower target gives the manager room to *hold* — pairs with achievement-conditioning to test emergent hold-then-edit (temporal abstraction at K=1) | `mask_frac`→~0.125 but bimodal/event-triggered, not inert; score holds | **DONE** (TIMEOUT @3.34M/4M). score **684**, wkr_goal **0.334**, mask_frac 0.107 (hit target, stable not inert), achieve 0.68. Lower target held cleanly but no reaching/score gain over e17. |
| e19_cartpole | **4633990** | e18 but **Z only**. K=1, target 0.125 | Z-only control at the lower target | vs e18 | **DONE** (TIMEOUT @3.68M/4M). score **624**, wkr_goal **0.313**, mask_frac 0.093. ≈ e18; achievement gave a small edge at this target (e18 > e19). |
| e20_cartpole | **4633991** | e16 (Z + achieve) but **K=8** (temporal-abstraction control). target 0.3 | Does conditioning rescue K=8, where B2 over-sparsified to inert (e9)? Achievement signal may let the manager re-command only when the goal is reached | `mask_frac` >0 (not e9's 0.001), tighter cosine than K=1, score recovers | **DONE** (TIMEOUT @3.39M/4M). score **504**, wkr_goal **0.763** (2× e8!), mask_frac 0.236, achieve **0.88**. K=8 makes goals reachable (tight cosine); conditioning helped here (e20 > e21 on score AND reaching). But score still well below K=1 — reachable ≠ useful goals. |
| e21_cartpole | **4633992** | e20 but **Z only**. K=8, target 0.3 | Z-only control at K=8 | vs e20 | **DONE** (TIMEOUT @3.69M/4M). score **365**, wkr_goal **0.654**, mask_frac 0.260. Conditioning (e20) beat this on both axes — clearest pro-conditioning signal in the grid. |
| e22_cartpole | **4633993** | e20 (Z + achieve) but **target 0.125**. K=8 | K=8 + lower target + full conditioning | vs e20/e18 | **DONE/FAILED** (TIMEOUT @3.42M/4M). score **3** (collapsed), wkr_goal 0.721, mask_frac drifted 0.10→0.07, achieve 0.84. **Only collapse in the grid**: K=8 + tight sparsity + achievement is unstable — manager holds a reachable-but-useless goal, task reward dies. |
| e23_cartpole | **4633994** | e22 but **Z only**. K=8, target 0.125 | Z-only control, K=8, lower target | vs e22 | **DONE** (TIMEOUT @3.71M/4M). score **143**, wkr_goal **0.720**, mask_frac 0.107. Z-only survived where e22 (achieve) collapsed — at K=8/tight-target, achievement conditioning was *destabilizing*. |
| e24_cartpole | **4634685** | **VARIABLE GOAL LENGTH (learned K), PLAIN goals.** Manager emits a duration `p∈{1..16}` (categorical head) and holds each goal `p` steps; full-resolution manager AC with REINFORCE masked to switch steps, worker λ-return reset at switch boundaries. `imag_length=32`, struct=200, no masking. New flag `agent.variable_goal_length`. | The e16–e23 verdict was "the bottleneck is the **K-vs-score tradeoff**, not manager context → try a **learned/adaptive K** (temporal sparsity head)." A learned per-decision duration should let the manager get K=1's task score **and** K=8's worker reaching, instead of trading one for the other. | `mgr_duration_mean` non-degenerate (not pinned 1 or 16); `switch_rate ≈ 1/duration`; score ≥ e17/e8 (~780–800) while `wkr_goal_rew` lifts above K=1's ~0.32. | **LAUNCHED 06-15.** Smoke (debug) validated all 3 paths; `switch_rate≈1/mean_duration` holds. Script `run_v3_e24_cartpole_vargoal_plain.sbatch`. New metrics `goal/mgr_duration_mean`, `goal/mgr_switch_rate`. |
| e25_cartpole | **4634686** | e24 **+ MASKED goals** (the e17 recipe — B2 prob sparsity 0.3, struct=200, `mgr_cond_goalcode`, mask_max=5 — with the fixed K=1 replaced by the learned duration). | Paired with e24 to isolate masking under variable K. If a learned K resolves the tradeoff, masking on top (e17's best K=1 arm) should hold sparse edits **and** lift reaching without the score collapse K=8 caused. | Same as e24 plus `mask_frac` stays sparse (~0.15–0.3, no collapse); beats e17's 777 / 0.341 on at least one axis. | **LAUNCHED 06-15.** Smoke Leg 3 (masked+variable) validated: duration 3.57, switch 0.31, mask_frac 0.49, finite losses. Script `run_v3_e25_cartpole_vargoal_masked.sbatch`. |

### e26–e32: ablation of e25's three scaffolding knobs (06-16)

All seven are the **e25 base** (variable-K + masked goals: `size6m masked_goals variable_goals`, duration 1–16, mgr_actent 0.5, goal-AE β0.25 / sum / lr4e-5) with one or more knobs turned **off**, to find which of e25's supports are load-bearing. Knobs: **K1** = e8-like *no goal conditioning* (`mgr_cond_goalcode False`); **K2** = *no target mask sparsity* (`mask_sparsity_mode sample` — the documented gradient-free no-op, so the adaptive sparsity adapter no longer pulls `mask_frac` to 0.3); **K3** = *no structural goal loss* (`goal_struct_weight 0.0`). e26–e31 on **gpu-v100**, e32 on **gpu-a100**.

| Exp | Job id | Investigated hyperparams | Hypothesis | Expected result | Status / outcome |
|---|---|---|---|---|---|
| e26_cartpole | **4634958** | K1 only: `mgr_cond_goalcode False` (else e25). | Manager-input conditioning (the e16–e23 win) may be redundant once K is learned + masking is on. | If conditioning is non-essential, score/reaching ≈ e25; if it's load-bearing, tracking loosens (lower `wkr_goal_rew`). | **LAUNCHED 06-16, RUNNING** (saion-gpu15). Script `run_v3_e26_cartpole_vargoal_masked_k1.sbatch`. |
| e27_cartpole | **4634959** | K2 only: `mask_sparsity_mode sample` (no sparsity target). | Spatial sparsity pressure is what kept e25's edits minimal; without it `mask_frac` should drift up. | `mask_frac` rises toward ~0.5+; test whether score holds or the unconstrained mask hurts tracking. | **LAUNCHED 06-16, RUNNING** (saion-gpu16). Script `run_v3_e27_cartpole_vargoal_masked_k2.sbatch`. |
| e28_cartpole | **4634960** | K3 only: `goal_struct_weight 0.0` (no struct alignment). | Structural alignment keeps decoded goals meaningful vs the masked code; removing it may degrade goal quality. | Watch goal-AE recon/KL and `wkr_goal_rew`; score may drop if goals become un-trackable. | **LAUNCHED 06-16, RUNNING** (saion-gpu16). Script `run_v3_e28_cartpole_vargoal_masked_k3.sbatch`. |
| e29_cartpole | **4634961** | K1 + K2: no cond + no sparsity target. | Compound: drop both manager context and spatial sparsity. | Degradation beyond e26/e27 alone would show interaction; otherwise additive. | **LAUNCHED 06-16, RUNNING** (saion-gpu16). Script `run_v3_e29_cartpole_vargoal_masked_k1k2.sbatch`. |
| e30_cartpole | **4634962** | K1 + K3: no cond + no struct. | Compound: drop manager context and structural alignment. | vs e26/e28 alone — interaction vs additive. | **LAUNCHED 06-16, RUNNING** (saion-gpu17). Script `run_v3_e30_cartpole_vargoal_masked_k1k3.sbatch`. |
| e31_cartpole | **4634963** | K2 + K3: no sparsity target + no struct. | Compound: unconstrained mask and no structural alignment — goals could degrade fastest. | vs e27/e28 alone — interaction vs additive. | **LAUNCHED 06-16, RUNNING** (saion-gpu17). Script `run_v3_e31_cartpole_vargoal_masked_k2k3.sbatch`. |
| e32_cartpole | **4634964** | K1 + K2 + K3: all off (full strip-down). | variable-K + masked goals with **none** of the e16–e25 scaffolding — lower bound on what masking+learned-K alone deliver. | If score/reaching survive, the scaffolding is optional; if they collapse, it pins how much the supports carried e25. | **LAUNCHED 06-16, RUNNING** on **gpu-a100** (saion-gpu24). Script `run_v3_e32_cartpole_vargoal_masked_k1k2k3.sbatch`. |

## Notes / open threads
- **Root cause of the e1/e3/e7 mask collapse (found 06-11):** the soft sparsity penalty
  was a **no-op on the mask head**. `mask_frac` was computed from the *hard* Bernoulli
  sample (`Binary.sample()` → `jax.random.bernoulli`, no straight-through, cf. `OneHot`
  which has one), so `losses['mask_sparsity'] = sg(scale)·mask_frac` had **zero gradient**
  to the mask logits. The `mask_sparsity` adapter then ratcheted its multiplier to the
  `mask_sparsity_max=100` ceiling without effect (penalty value ~63 ≈ 97% of the manager
  loss stack, yet exerting no force), while the mask drifted dense under the REINFORCE
  return. e8–e11 fix the gradient path two ways: **B2 (`mask_sparsity_mode=prob`)** feeds
  the differentiable sigmoid probs to the adapter; **C (`mask_sparsity_mode=reinforce`)**
  subtracts the edit fraction from the manager reward so the mask logp is reinforced.
  New metrics: `goal/mask_prob_mean` (soft fraction, all modes) and
  `goal/mask_sparsity_reward_pen_mean` (C only). `mask_sparsity_max` lowered 100→5 (B2)
  / →1.0 (C, reward-scale). Legacy no-op kept as `mask_sparsity_mode=sample`.
- **e8–e15 verdict (06-12): the B2 prob fix works, but only at K=1 — 1 win of 8.**
  Final converged numbers (~3.4–3.6M/4M, all cancelled 06-12 and archived):
  | run | mode | K | score | wkr_goal_rew | mgr_extr_rew | mask_frac | sp_scale |
  |---|---|---|---|---|---|---|---|
  | **e8** | prob (B2) | 1 | **~800 ✅** | 0.37 | +0.69 | **0.15–0.21** | 0 |
  | e9 | prob (B2) | 8 | ~1 | 0.71 | +0.01 | 0.001 | 0 |
  | e10 | reinforce | 1 | ~3 | 0.32 | −0.44 | 0.60 | 1.0(max) |
  | e11 | reinforce | 8 | ~19 | 0.43 | −0.97 | 0.995 | 1.0(max) |
  | e12 | fix wt10→0 | 1 | 0 | 0.46 | 0.00 | 0.0 | 10 |
  | e13 | fix wt10→0 | 8 | 0 | 0.71 | 0.00 | 0.0 | 10 |
  | e14 | fix wt1→0 | 1 | 0 | 0.45 | 0.00 | 0.0 | 1 |
  | e15 | fix wt1→0 | 8 | 0 | 0.72 | 0.00 | 0.0 | 1 |
  Takeaways: (1) **B2 (`mask_sparsity_mode=prob`) is the right fix** — kills the e7 runaway-to-100;
  e8 holds a genuinely sparse mask (~0.15) *and* beats baseline (800 > 710). (2) **REINFORCE mode
  (C) is broken** — rails scale to max, drives `mgr_extr_rew` negative, mask stays dense; don't reuse.
  (3) **Driving sparsity to 0 (fixed weight, even =1) always collapses edits → frozen goal → score 0**
  — a clean control proving the manager is load-bearing (worker is *not* solving cartpole on its own).
  (4) **Core unresolved tension: K=1 vs K=8.** K=1 gives a useful *moving* goal (solves task) but loose
  goal-reaching (cosine ~0.37); K=8 gives tight goal-reaching (cosine ~0.7) but the mask collapses to
  inert (e9) or dense+negative (e11) → task fails. Can't yet get tight tracking *and* a useful goal.
  Next: tighten the goal-reach channel on the e8 recipe without collapsing the mask (intermediate
  K=2–4, and/or a worker-side goal-reach bonus); e8 ckpt is kept in bucket as the warm-start point.
- **e16–e23 campaign (06-12): manager-input conditioning.** Root cause of e8's loose tracking,
  found by code audit: the **manager policy conditions only on `feat2tensor(feat)=[deter‖stoch]`**
  at every call site — it block-overwrites the running goal code **Z while blind to Z's contents**,
  and never sees whether the worker reached the previous goal (the worker cosine is computed but
  stays in the worker AC path). Fix (`agent._mgr_input`): config-gated, stop-gradient'd channels
  appended to the manager input — `mgr_cond_goalcode` (pre-edit Z, L*C), `mgr_cond_achieve`
  (scalar `cosine_max(decode(pre-edit Z), deter)`), `mgr_cond_decgoal` (decoded goal, off). All
  default off ⇒ existing runs unchanged. The training re-derivation reconstructs the **pre-edit**
  Z by a one-step time-shift (`pre_code = concat([seed, post_code[:-1]])` at full res, then `[::K]`)
  to avoid feeding the manager the code it just produced (leakage); the exact step-0 seed is
  threaded out of `_imagine_with_manager`. New metric `goal/mgr_cond_achieve_mean`. Grid:
  **{K=1,8} × {target 0.3, 0.125} × {Z, Z+achieve}** = e16–e23, all fresh (no warm-start, since the
  new manager input width changes `manager_pol`'s first layer). Success = `wkr_goal_rew` > e8's 0.37
  without mask collapse; emergent temporal abstraction read from the K=1 arm vs the K=8 control.
  - **VERDICT (06-15, all 8 TIMEOUT @3.3–3.7M/4M, converged not crashed): hypothesis NOT supported.**
    The wiring works — the manager's achievement view (`mgr_cond_achieve_mean`) is healthy 0.60–0.88
    and the mask stayed stable in 7/8. But **manager conditioning did not break the real tradeoff,
    which is governed by K, not by what the manager sees.** Clean finding:
    - **K=1** keeps task score high (e17 Z-only = 777 ≈ e8) **but worker reaching stays ~0.30–0.34**
      (manager churns goals faster than the worker can reach them). Conditioning was neutral-to-
      **negative** here (e16 achieve < e17 Z-only on both axes).
    - **K=8** lifts worker reaching to **0.65–0.76** (≈2× e8) — a stable 8-step goal is reachable —
      **but task score collapses** (504 → 365 → 143 → 3). Reachable ≠ useful: the worker hits the
      goals but the goals aren't good for the task.
    - Achievement conditioning's effect was **mixed and K-dependent**: clearly helped at K=8/t0.3
      (e20 > e21 on score AND reaching) but **caused the only collapse** at K=8/t0.125 (e22 score 3
      vs e23 Z-only 143) — tight sparsity + achievement lets the manager lock onto a reachable-but-
      useless goal. No emergent event-triggered/bimodal mask appeared at K=1 (masks held flat near
      target, not inert but not bursty).
    - **Implication / next direction:** the bottleneck isn't manager context, it's the
      **K-vs-score tradeoff** itself. Worth targeting directly — e.g. a learned/adaptive K (temporal
      sparsity head), intermediate K (2–4), or K=8 with the manager's extrinsic reward shaped so the
      held goal must also serve the task. Best baseline to beat on score remains e17/e8 (~780–800).
    - Runs archived to bucket + deleted from /work on 06-15 per the standing clean rule.
- Decisive follow-up for e3/e4: re-run `diag_codes.py` on their checkpoints — if the
  geometry loss worked, diag-1 code volatility (~2.5 blocks/step) drops and diag-2
  `goal_drift` smooths (partial edits became meaningful).
- e5: 64×64 report-compile segfault is independent of `DREAMERV3_CONV_IMPL`. To get
  report back at 64×64, lighten it rather than disable (e.g. `report_skill_viz False`,
  smaller `report_length`/`report_max_rows`).
