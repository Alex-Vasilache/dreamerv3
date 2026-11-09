# Plan: finish TCE-KL, then variable goal length (written 2026-11-06)

Order agreed with the user on 2026-11-06: **TCE comes first.** Variable goal length starts only
once TCE-KL is settled (mechanism behaves as intended across seeds, strength fixed, 1x16 vs 4x12
decided). Until then the GPUs run TCE-KL arms, and this document is where we pick up afterwards.

---

## Part 1. TCE-KL: state on 2026-11-06 and what "settled" means

### Arms

| arm | config | h-numbers | array | status |
|---|---|---|---|---|
| c=1, 1x16 hash | `tce_kl` | h0073-h0078 | 4798833 | seed 0 only, finished; rest cancelled (push off) |
| c=10, 1x16 hash | `tce_kl tce_kl_c10` | h0133-h0162 | 4799105 | **finished 11-08**: pin pad six 3/5, five 3/5 by 0.8M (one seed late 45), DMC ~ US-SOM |
| c=30, 1x16 hash | `tce_kl tce_kl_c30` | h0163-h0192 | 4799106 | **cancelled 11-06 19:10** after seed 0 + part of seed 1: over-pushes |
| c=10, 4x12 hash | `tce_kl4 tce_kl_c10` | h0193-h0222 | 4800784 | **finished 11-10**: pin pad six 1/5, four-clear-task 68.8 vs 1x16 66.6; not better than 1x16, keep 1x16 |
| c=10 + alpha floor 0.04, 1x16 | `tce_kl tce_kl_c10 tce_kl_floor` | h0223-h0252 | 4800914 | **finished 11-09**: floor holds w ~0.2 but no earlier reward; pin pad six 2/5; four-clear-task score 70.8 vs US-SOM 72.8 / sqrt 71.3, kept for the extra seeds (4807344) |

Figures and tables: `collapse_figs/tce_kl/` (see `collapse_figs/INDEX.md`); scripts
`tools/collapse_study/tcekl_report.py` (per seed: score, gap, weight, alpha) and
`compare_matched.py tcekl`.

### What the mechanism should do, and what we check (every supervision tick)

TCE-KL makes entropy + TCE = -alpha * KL(pi || q) with q(g|s) ~ (n(s,g)+1)^-c, so the manager
should (1) prefer rarely-tried goals before reward (gap = E_pi[b] - E_uniform[b] > 0), (2) hand
over to reward once it arrives (pi ~ q * exp(R / alpha): gap drifts back toward 0 on pin pads),
(3) not cost DMC.

| check | metric | intended | seen so far (c=10, 2 seeds) |
|---|---|---|---|
| pushes toward rare goals before reward | `fromto/gap` | > 0 by ~0.3M | yes on 3 of 4 pin pad runs; seed 0 pin pad five ~0 |
| strong enough early | `fromto/weight` before first reward | >= ~0.2 (what the successful 1x16 seeds used) | **0.03-0.08 for the first 0.3-0.5M**: weight = c*alpha/log 8 and alpha starts at 0.005-0.015 |
| reward takes over | gap after reward | back toward 0 | yes (pin pad five/six seed 0) |
| pin pad five | first reward | <= 0.8M | 0.07M, 0.46M (c=1: 0.90M) |
| pin pad six | solved seeds | >= 3/5 | 1/2 so far (seed 1 stuck at 0 with weight 0.06-0.09) |
| DMC | late score vs OG and vs c=1 | no loss | cheetah 478 / hopper 224 vs OG 511 / 333 and c=1 636 / 370 (1 seed each) |

### Decision rules (I act on these without waiting)

- **Early weakness confirmed** (>= 2 of 5 pin pad five/six seeds with weight < 0.1 through 0.5M
  and no reward by 1M): fix by decoupling the early push from the entropy controller's
  start-up. Candidates, in order of preference:
  1. floor the alpha that TCE uses: alpha' = c * max(alpha, alpha_floor) / log 8, alpha_floor set
     so alpha' >= ~0.2 at c=10 (alpha_floor ~ 0.04); the entropy term itself is untouched;
  2. warm-start the manager entropy controller's alpha at ~0.05 instead of its current init.
  New h-numbers, EXPERIMENTS.md entry, commit + push, requeue.
- **DMC cost confirmed** (c=10 below OG on cheetah or hopper over >= 3 seeds, p < 0.1): the
  push never turns off on dense-reward tasks (gap stays +0.3 on cheetah). Fix candidate: let the
  bonus fade with total count (the TCE-grow idea, half-life growing with t) so q -> uniform once
  everything has been tried; test on top of whichever of the fixes above survives.
- **Settled** = c=10 (with or without a fix) solves pin pad five 5/5 by 0.8M, pin pad six >= 3/5,
  DMC within noise of OG / the 1x16 adaptive arm, and gap/weight behave as in the table above on
  every seed. Then pick 1x16 vs 4x12 from h0133-h0162 vs h0193-h0222 and freeze TCE.

---

## Part 2. Variable goal length: where we left off

### History (corrected)

- **Jun-Jul: masked goals** (manager edits a running goal code; separate mask head, then a
  joint 9-way head with an abstain class). Abandoned 07-15 for **implicit sparsity** (manager
  regenerates the whole code, reuse is measured and encouraged).
- **Jul: learned duration head (var-K)**, HiPPO-style, with priors (fixed / adaptive / Lagrangian)
  on the mean duration, block pooling, timed-goal countdown (HiTS). Every arm was dead or below
  Director (duration drifts to ~2.5 or rails to max; hopper 0 everywhere).
- **07-27 to 07-30 audit** found the variable-length path itself was wrong, independent of the
  duration idea: manager losses averaged over the padded decision buffer (8x too weak), return
  /advantage normalisers fed padded copies, a non-trainable last switch counted, the worker
  trained with a different credit rule than Director, the pinned duration head still trained
  through three channels, the replay window lacked its final bootstrap anchor, plus three
  smaller duration bugs (countdown budget, relabel off-by-one, Lagrangian bias). All fixed with
  tests. The post-fix runs e372-e389 (single seeds, pinned and free durations) were **stopped on
  07-30 without a verdict**. So: **it is still open whether variable length itself is bad or the
  bugs were.** Nothing has re-tested it since.
- **Oct: goal-space masking in the goal autoencoder** (`goal_mask8`, e1600-e1659, 6 x 5): the
  goal code also decodes a keep/ignore flag per RSSM block, the GAE trains on random masks with
  reconstruction on kept blocks only, the worker is scored on kept blocks only. Result: helps
  pin pad four only (p = 0.016 / 0.008), pin pad six worse (2/5 -> 0/5 on OG, 3/5 -> 1/5 on
  SOM), cheetah -60 on both bases. The manager does use partial goals (keeps 40-60% of blocks on
  pin pads/hopper, 85-95% on cheetah).
- I found no variable-length (duration) runs in Aug-Oct in EXPERIMENTS.md or git; if there were
  some under another name, add them here.

### What we will test (after TCE), all on the best US-SOM + TCE-KL base

Two ways to make the length variable, tested separately:

**A. Pinned control first (must pass).** The variable-length machinery with the length pinned at
K = 8 must reproduce the base arm (same tasks, same seeds, within noise). If it does not, the
machinery is still buggy and nothing below is interpretable.

**B. Length from reaching the goal (no duration head).** The worker holds the goal until it is
reached or K_max steps pass (with a minimum hold K_min); the manager then decides again. No
duration head, so no REINFORCE noise on durations; length comes from the world, not from a
learned choice. This is the HAC (Levy et al. 2019) style of termination.

**C. Learned duration head, re-tested on the fixed code.** The July design (duration head +
Lagrangian prior on the mean) with all fixes in, so we finally get a fair verdict on it.

Later, only if B or C works: let the manager say *when* explicitly (HiTS timed subgoals,
`worker_timed_goals` countdown already exists), and combine with the GAE masking (a masked
block does not count toward "reached").

### How the critics and returns must handle variable length (semi-MDP rules)

With variable holds the manager lives in a semi-MDP (Sutton, Precup & Singh 1999). Most of this
exists in the code already and was what the July bugs broke:

1. **Manager reward per decision = discounted sum over the hold**, sum_i gamma^i r_i, not the mean
   (`mgr_reward_agg: sum`). The mean favours short holds. Note: `sum` was listed as a dead end
   (e66, e181), but both runs were before the reduction bugs were fixed, so it was never fairly
   tested.
2. **Bootstrap with gamma^tau**, tau = realised hold length; lambda-returns are taken over decisions
   with per-decision discount gamma^tau_k.
3. **Every manager reduction is normalised by the number of real decisions**, never the padded
   buffer width (losses, return/advantage normalisers, entropy controllers).
4. **A hold cut off by the imagination horizon is truncated, not terminated**: bootstrap from it,
   never treat its end as a real switch (`goal_duration_truncated_policy: drop_duration`).
5. **The worker's return resets at each switch**, same credit rule as Director's windowed one.
6. **TCE** counts one (start cell, goal cell) pair per decision; nothing else changes. With
   reach-termination the "8 steps back" start offsets become "since the last decision".

### When is a goal "reached"? (the threshold question)

A fixed cosine threshold is task-specific: the worker's goal reward is Director's cosine_max
between the current state and the decoded goal, and its typical value at the end of an 8-step
hold differs a lot between tasks (and drifts as the world model trains). Options, in order:

1. **Code match (preferred).** Reached when the goal autoencoder's code of the current state
   equals the goal code in all kept blocks, with a SOM tolerance of +-1 class on the line. Codes
   are categorical and the SOM orders them, so this means the same thing on every task and
   needs no scale. It also composes with masking (ignored blocks do not count).
2. **Adaptive percentile.** Reached when cosine_max exceeds a running per-task quantile (e.g.
   the 75th percentile) of the cosine_max reached at the end of fixed-K holds; one quantile for
   all tasks, the threshold itself adapts.
3. **Learned reachability** (a head predicting "will be reached within k steps"); more moving
   parts, only if 1 and 2 fail.

Whatever we use is logged (`goal/reach_rate`, mean hold, hold histogram) so we can see if it is
degenerate (always reached at K_min, or never before K_max).

### THICK and other regularisation against degenerate lengths

THICK (Gumbsch et al., ICLR 2024) puts a sparsity penalty (L0-style, on gated updates of a
slow context latent) into the **world model**, so the context changes only at real events; the
high level then acts at those change points and predicts the time to the next one. Uses for us:

- **As a termination signal (option D, later):** switch when the slow context changes. Needs a
  gated context added to the RSSM, i.e. a world-model change; too large to start with. A cheap
  stand-in is "the current state's goal code changed in >= k blocks" from the GAE we already have.
- **As a regulariser:** the general lesson, also in Schiewer & Subramoney (2024), is that
  unregularised variable length collapses to 1-step or whole-episode holds. For B this is
  handled by K_min / K_max and the reach rule; for C by the Lagrangian on |E[dur] - target| we
  already have. A one-sided per-switch cost (`goal_switch_cost`) is a known dead end: it rails
  the length to the max.

### How we run and watch these

- Every check uses the full panel (6 tasks x 5 seeds), seed-major, as for TCE.
- Watched like TCE, with early-stop rules so bad arms do not burn full seeds:
  - mean hold length collapses to K_min or rails to K_max for > 0.5M steps -> stop;
  - `mgr_extr_adv` ~ 0 for > 0.5M while the base's is not -> stop;
  - after seed 0 (all 6 tasks at >= 1M): any task more than 30% below the base at matched steps,
    or pin pad five with no reward by 1M -> stop the arm, diagnose, fix, requeue with new numbers.
- The pinned control (A) gates B and C: if A fails, fix the machinery before running B/C.

### References

- Sutton, Precup, Singh 1999, Between MDPs and semi-MDPs (option returns, gamma^tau bootstrap).
- Levy et al. 2019, Hierarchical Actor-Critic (terminate on achievement, max H attempts).
- Guertler et al. 2021, HiTS: Hierarchical RL with Timed Subgoals, https://arxiv.org/pdf/2112.03100
- Gumbsch et al. 2024, THICK, https://iclr.cc/virtual/2024/poster/18558
- Schiewer & Subramoney 2024, Exploring the limits of hierarchical world models, https://arxiv.org/html/2406.00483v1
- Overview: Discovering Temporal Structure (2025), https://arxiv.org/pdf/2506.14045
