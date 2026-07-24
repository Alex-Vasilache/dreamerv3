# Per-block REINFORCE credit for the edit mask (shaping sparsity through the estimator)

## Goal
Use the manager's **existing REINFORCE estimator** to (a) assign credit to *each edited
block* for its task contribution and (b) let sparsity emerge from that credit — instead of a
sparsity target (e57) or a one-sided cost (e70–e77, corner-seeking). The mask is already a
factored REINFORCE action; we are only changing the *advantage each block sees*, not the
estimator.

## The math (why no new estimator is needed)

REINFORCE / score-function:  ∇_θ E[R] = E[ R · ∇_θ log π_θ(a|s) ].
We never differentiate R or the environment — only `log π`. Code: `-mgr_logpi · sg(adv)`
(agent.py:2840), where `sg(adv)` is R held constant.

The mask is L independent Bernoulli edits, so the manager log-prob **factorizes**:
  log π(a|s) = log π(skill) + Σ_i log π(m_i) + log π(dur),   m_i ~ Bern(σ(ℓ_i)).
That sum is `mgr_logpi = sum(head_logp_time(...))`. Today **every** block's log π(m_i) is
multiplied by the **same scalar** advantage A. That is unbiased but:
  - high variance (block i rides the global outcome, not its own effect), and
  - gives no reason to prefer fewer edits.

**Per-factor baseline (the lever).** For a factored policy you may subtract a separate
baseline b_i per factor *as long as b_i ⊥ a_i*:
  ∇_θ E[R] = E[ Σ_i (R − b_i) ∇_θ log π(m_i) ].
Unbiasedness proof (2 lines): E_{m_i}[ b_i ∇log π(m_i) ] = b_i ∇ Σ_{m_i} π(m_i) =
b_i ∇(1) = 0. So any b_i not depending on the sampled m_i is free variance reduction and,
if chosen as a *counterfactual*, gives per-block credit.

**Counterfactual (COMA) baseline** = marginalize block i, holding the others fixed:
  b_i = Σ_{m_i'∈{0,1}} π_i(m_i') · Q(s, m_{-i}, m_i'),
  A_i = Q(s, m_actual) − b_i.
A_i answers exactly "did editing block i help vs. the policy-average alternative for that
block." That is the per-block credit we want, and it stays inside REINFORCE.

## Split the gradient: cost analytic, return counterfactual

The sparsity **cost** c(m)=λ Σ_i w_i m_i is a *known* function of the actions, so its REINFORCE
contribution has a **closed form** (zero variance):
  ∇_{ℓ_i}(−E[c]) = −λ w_i σ(ℓ_i)(1−σ(ℓ_i)) = ∇_{ℓ_i}(−λ w_i σ(ℓ_i)).
i.e. penalizing the *expected* cost λ w_i σ(ℓ_i) directly (differentiable, the `prob`-mode
penalty path, `_mask_prob`) IS the zero-variance version of REINFORCE-ing the cost. **Lesson
from e10/e11 vs B2:** never push the cost through REINFORCE (adds variance, railed); push only
the part you can't compute — the task return R — through REINFORCE, and shape sparsity with the
analytic expected-cost penalty (optionally state-dependent w_i = achievement, the differentiable
twin of B.3).

So the final manager mask gradient is:
  per block i:  (A_i) ∇log π(m_i)        # counterfactual task credit (REINFORCE)
              − λ w_i ∇σ(ℓ_i)            # analytic sparsity shaping (zero variance)
skill/dur heads keep the scalar advantage A as today.

## What the codebase already gives us (at the imag_loss_mgr site)
- `pre_code` = Z_old (pre-edit running goal), `mgr_skills['skill']` = z_new (emitted code),
  `mgr_skills['mask']` = sampled m, `mgr_skills['goal_code']` = Z_actual = where(m, z_new, Z_old).
  → both single-block counterfactuals are constructible with no re-roll:
    Z|m_i=0 = Z_actual with block i set to Z_old[i];  Z|m_i=1 = Z_actual with block i set to z_new[i].
- `self._mask_prob(mgr_policy['mask'])` = σ(ℓ_i); `goal_reward_cosine_max`, `feat2deter` for w_i.

## The missing piece (the only real implementation cost)
The manager critic `mgr_extr_val(inp_eff)` takes the **WM feature only**, NOT Z
(agent.py:1927, 2045). A counterfactual baseline needs Q(s, Z) evaluable at flipped Z. So we
must add a **goal-conditioned manager Q-head**:
  Q_g(s, Z) := MLPHead over concat[ feat2tensor(s), flatten(Z) ]  (twohot, like mgr_extr_val),
trained on the same λ-returns as `mgr_extr_val`. Evaluating Q_g at the L single-block flips is
L cheap forwards (L is small), no extra imagination rollouts.

## Implementation tiers

### Tier 1 — runnable NOW in the held v100 slot (no new critic): analytic state-dependent shaping
Test the *cheap half* — sparsity shaping via the zero-variance expected-cost penalty with a
**state-dependent** weight (so it's two-sided like B.3, not corner-seeking), no target:
  loss_sparsity = λ · mean_i [ achievement_i · σ(ℓ_i) ]     # differentiable, on _mask_prob
- New flag `mask_sparsity_mode: prob_ach` (or `mask_sparsity_ach_weight`): reuse the existing
  `prob` penalty machinery (agent.py:1975-…) but multiply the per-block prob by `sg(achievement_i)`
  before the mean, and use a small FIXED λ (no Lagrange target).
- Hypothesis: editing a reached block is penalized, an unmet block is free → interior σ(ℓ_i)
  without a target, and (unlike e78 which puts the same cost through the reward/REINFORCE) this
  is the zero-variance analytic form → cleaner, faster.
- This is e78's idea moved from the high-variance reward path to the low-variance gradient path.
- ~30-line change, no new module, AST-checkable; smoke + early poll like e78–e80.

### Tier 2 — the real per-block credit (counterfactual A_i): needs the Q_g head
1. Add `Q_g` head + slow copy; register scales/optimizer next to `mgr_extr_val`.
2. In `imag_loss_mgr`, build the L flipped goal codes (vectorized over blocks), evaluate
   Q_g at Z_actual and at each flip → b_i, then A_i = Q_g(Z_actual) − b_i, normalized by the
   same retnorm scale as the scalar advantage.
3. Replace the mask head’s contribution in `mgr_policy`: instead of `logp(m) · sg(A)`, use
   `Σ_i logp(m_i) · sg(A_i)`; keep scalar A for skill/dur. Add a tiny fixed edit cost (analytic)
   so ties break toward fewer edits; the credit decides *which* blocks survive.
4. Behind a flag `mask_perblock_credit: True`.
- Risk: Q_g(s,Z) must learn the goal→return map for the counterfactual to be meaningful; noisy
  early. The single-step Q_g counterfactual ignores how a block-flip changes the *downstream*
  rollout (true counterfactual would re-roll) — it is an approximation, justified because the
  goal code's main effect is on the immediate worker conditioning captured in Q_g(s,Z).
- Falsifiable diagnostics: per-block |A_i| spread > 0 (credit is discriminating, not uniform);
  mask_frac lands interior with the kept blocks stable; score vs e57 (839).

## Recommendation
Run **Tier 1 now** in the held slot (cheap, directly tests "analytic state-dependent sparsity
shaping beats the reward-cost version e78"). Build **Tier 2** as the principled follow-up only
if Tier 1 shows the analytic path is cleaner but still wants per-block credit to pick the right
blocks. Both keep the estimator we already trust; neither needs re-deriving REINFORCE.

## Do you need to derive anything by hand?
No new estimator. The only hand-math worth doing is the **2-line unbiasedness check** for the
per-factor baseline (above) and, if you want a guarantee of an interior optimum, locating where
the marginal task-return per block crosses λ — an objective-shape check, not a gradient
re-derivation. Autodiff of `-Σ_i logp(m_i)·sg(A_i) − λ Σ_i w_i σ(ℓ_i)` gives the exact gradient.
