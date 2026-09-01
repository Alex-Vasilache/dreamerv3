# JAX Director verification log

Tracks which parts of the JAX HRL implementation have been checked against the
original TensorFlow Director (`code/director/embodied/agents/director/`), how,
and what is left. Every row is either backed by a test that runs under pytest,
or marked as not yet checked. "Matches" never means "read the code and it looked
the same" -- it means a test asserts it.

Motivating question: five identical hopper_hop Director-arm seeds (e391, e395,
e399, e403, e407) finish at 307, 220, 146, 41, 59. The spread is far wider than
the published Director seed bands, and `train/mgr_extr_rew` climbs monotonically
in every seed while `episode/score` collapses in three of them.

Test files added (counts are collected tests, i.e. after parametrization):

Phase 1 — the variance investigation:

- `embodied/tests/test_replay_staleness.py` (15)
- `embodied/tests/test_determinism.py` (7)
- `embodied/tests/test_lambda_return.py` (19)

Phase 2 — the full-codebase audit:

- `embodied/tests/test_tensors_vs_director.py` (72) — fixed-K block pooling
  against a numpy transcription of Director's `abstract_traj`
- `embodied/tests/test_losses_gradflow.py` (34) — stop-gradient routing in
  `imag_loss`, `repl_loss`, `imag_loss_wkr`, `imag_loss_mgr`
- `embodied/tests/test_heads_helpers.py` (35) — shifts, smoothing kernels,
  entropy slicing, REINFORCE head selection
- `embodied/tests/test_rssm.py` (27) — world-model carry, reset, KL routing
- `embodied/tests/test_outs_twohot.py` (46) — the reward/critic head numerics
- `embodied/tests/test_opt_and_norm.py` (28) — AGC and the running normalizers
- `embodied/tests/test_nets_and_video.py` (36) — `Norm` and the video helpers
- `embodied/tests/test_remaining_helpers.py` (30) — everything else with no
  coverage anywhere
- `embodied/tests/conftest.py` — restores global JAX config between tests (see
  BUG-3 in `AUDIT_FINDINGS.md`)

Run them with:

```bash
REPO=$PWD TESTS="embodied/tests/test_replay_staleness.py embodied/tests/test_determinism.py embodied/tests/test_lambda_return.py" \
  sbatch -p gpu-v100 --gres=gpu:v100:1 sbatch/run_pytest_cpu.sbatch
```

(`-p gpu-v100` because the DMC env test needs an EGL context; on `intel` that
one test skips itself.)

For a single file, `./sbatch/audit_test.sh <path>` submits and blocks until the
summary is available — about a minute per cycle.

`test_parallel.py` and `test_train.py` do not collect: they import `zerofun`,
which the codebase replaced with `portal` and which is not installed. That is
pre-existing and unrelated to this work.

## Status key

- `OK` -- a test asserts the property and passes.
- `DIFF` -- checked, and it differs from TF Director on purpose. The reason is
  recorded; the difference is a candidate cause of the variance.
- `BUG` -- checked, differs, and the difference is not intentional.
- `TODO` -- not yet checked.

## Headline findings

### 1. The replay buffer never evicts anything (`BUG` in effect)

`replay.size: 5e6` against `run.steps: 4e6`. Capacity is never reached, so
`Replay._insert` never calls `_remove`. Confirmed directly in the run logs:
`replay/items` grows linearly to 3,999,440 and equals the step count.

`5e6` is the upstream DreamerV3 default, inherited unchanged. DreamerV3's own
runs are much longer than 4M steps, so there it does evict. At our run length it
does not. TF Director uses `replay_size: 1e6`.

Consequences, all test-backed:

- Sampling is `fracs: {uniform: 1.0}`, so the training distribution is uniform
  over the entire run history. Mean age of a sampled item is ~N/2, i.e. **2M
  steps** at the end of a 4M-step run. With Director's 1e6 cap it would be 500k.
- The `online` queue supplies fresh sequences at one per `batch_length` steps
  per worker: 8 workers / 64 length = 8 per 64 env steps, against 1 sequence
  drawn per env step (`samples/insert == 1.00`, measured in e403). So
  **12.5% of each batch is fresh and 87.5% is uniform over all history.**
- Item 0 -- collected by the untrained policy -- is still sampleable at step 4M.

This is the leading explanation for both symptoms. See finding 2.

### 2. `train/mgr_extr_rew` measures the replay history, not the current policy

The metric is `dec_mean(mgr_extr_rew[:, 1:])` over imagined rollouts whose start
states come from replay. Given finding 1, those starts are uniform over all
history, so the metric is a cumulative average by construction.

Measured on the five seeds (correlation of `train/mgr_extr_rew` against two
candidate references):

| seed | r vs recent-100-episode mean | r vs cumulative mean |
|---|---|---|
| e391 | 0.938 | 0.996 |
| e395 | 0.830 | 0.992 |
| e399 | 0.883 | 0.994 |
| e403 | 0.440 | 0.971 |
| e407 | 0.443 | 0.922 |

Every seed tracks the cumulative mean better, and the gap is widest in exactly
the two seeds whose score collapses. So "high extrinsic reward with a bad score"
is not a reward-head miscalibration -- the metric simply cannot fall when the
current policy degrades, because most of what it averages over is old.

**`train/mgr_extr_rew` should not be read as a progress signal.** Use
`episode/score` on a sliding window.

### 3. Runs are not reproducible from a seed, and neither is TF Director

| Source | Status | Note |
|---|---|---|
| DMC env RNG | not seeded | `suite.load(domain, task)` with no `task_kwargs={'random': ...}`. Only `dmlab` sets `use_seed`. Verified empirically: two freshly built `DMC('hopper_hop')` envs produce different initial states. **TF Director has the identical gap.** |
| replay sampler | seeded, but always 0 | `make_replay` never forwards `config.seed`, so `selectors.Uniform(0)` in every run. All five "seeds" share one sampling order. |
| report / save / log schedules | wall-clock | `LocalClock` uses `time.time()` and ignores its `step` argument. `report` draws from the *same* `replay.sampler` as training, so a report firing at a wall-clock-dependent moment shifts the training RNG stream. |
| `should_train` | step-based | `elements.when.Ratio` -- reproducible. |
| driver / env ordering | deterministic | Lock-step: pipes are received in fixed index order and callbacks run `for i in range(length)`. Parallel envs do **not** introduce ordering noise. |
| XLA | nondeterministic | `jax.deterministic: False`. |

So the seed-to-seed spread we measured is **seed variance plus uncontrolled
run-to-run variance**, and we have no measurement separating the two. Repeating
one seed twice would give different numbers.

To make a run reproducible, all of these are needed:
`task_kwargs={'random': seed}` in `embodied/envs/dmc.py`; `seed=config.seed` in
`make_replay`'s kwargs; step-based schedules for report/save; and
`jax.deterministic: True`.

### 4. Ruled out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| `episode/score` computed differently from TF | ruled out | Both sum episode rewards. TF casts to float64 first; over 1000 steps that is a ~1e-4 relative difference. |
| `lambda_return` differs from Director's `gve` | ruled out | 9 differential tests against a verbatim numpy transcription of `VFunction.target(impl='gve')`, across lambda in {0, 0.5, 0.95, 1} and gamma in {0.99, 0.997}, including a mid-trajectory terminal. Exact to 1e-5. |
| Gradient clipping is throttling learning | ruled out | Actor-critic `grad_rms/param_rms` is 0.003-0.009 after 0.5M against `agc: 0.3` -- two orders of magnitude below the threshold. It binds only in the first few thousand steps (0.34 at 10k, and 14.7 for the world model). |
| `horizon: 333` vs Director's `0.99` drives return variance | mostly ruled out | `return_lambda: 0.95` caps the effective horizon at ~20 steps, well short of both. Over the real `imag_length: 16` window the two discounts differ by **4.5%**. Over 64 steps at lambda=1 the gap is only 1.23x, not the 3.3x the nominal horizons suggest. |

## Config comparison: what "director_match" actually matches

`director_match` sets network *sizes* only (deter 1024, units 512, layers 4,
depth 64). Everything else stays at DreamerV3 defaults. The arm is "DreamerV3
with Director's HRL and Director's layer sizes", not a port of TF Director.

| Knob | TF Director | Ours | Status |
|---|---|---|---|
| replay size | 1e6 | 5e6 (never evicts at 4e6 steps) | **BUG in effect** |
| optimizer | adam lr 1e-4, eps 1e-6, wd 1e-2 (kernel) | lr 4e-5, eps 1e-20, wd 0 | DIFF; lr+wd fixed by the `director_optmatch` block (2026-09-01), `eps` still DIFF |
| grad clip | global norm 100 | adaptive `agc` 0.3 | DIFF, not binding |
| discount | 0.99 | horizon 333 (~0.997) | DIFF, 4.5% effect |
| return norm | `std`, decay .999, max 1e2 | `perc` 5/95, limit 1.0 | DIFF, TODO |
| adv norm | `mean_std`, decay .99, max 1e8 -- on **both** worker and manager | one `config.advnorm` key feeds both `mgr_advnorm` and `wkr_goal_advnorm` (`agent.py:384-385`); `none` by default, `meanstd` under `director_stable` | **Now matches Director in impl**, but the two actors **cannot be set independently**. Turning it on raised `wkr_goal_adv_mag` 0.037-0.043 -> 0.699-0.700 (~17x) with `wkr_goal_adv_std` unchanged. Split the key to ablate them separately. |
| reward head | 4x512, symlog **mse** | 1x1024, symexp **twohot 255 bins** | DIFF, TODO |
| activation / norm | elu / layer | silu / rms | DIFF |
| precision | fp16 | bfloat16 | DIFF |
| replay ratio | 64 | 64 | OK |
| imag horizon | 16 | 16 | OK |
| skill_shape | [8, 8] | [8, 8] | OK |
| skill duration | 8 | 8 | OK |
| manager_rews | extr 1.0, expl 0.1 | extr 1.0, expl 0.1 | OK |
| worker_report_horizon | 64 | 32 | DIFF |
| goal_reward | `cosine_max` | `cosine_max` | OK (`test_tensors_vs_director.py`) |
| goal VAE KL weight | `rec + kl`, no beta | `rec + 0.25 * kl` | **DIFF, binding** -- see *Goal autoencoder* below |
| goal AE nets | 4x512, elu/layer | 4x512, silu/rms | OK on shape (`director_match`); act/norm DIFF |
| deter (goal AE target dim) | 1024 | 1024 | OK |

## Component checklist

### Metrics and logging

| Item | Status | Evidence |
|---|---|---|
| `episode/score` = sum of episode rewards | OK | Read against TF; equivalent. |
| `episode/length` | DIFF (cosmetic) | TF reports `len(reward) - 1`; ours counts every transition. |
| `train/mgr_extr_rew` semantics | OK | Finding 2 -- it is a cumulative-history average. |

### Determinism

| Item | Status | Evidence |
|---|---|---|
| DMC env seeding | BUG (shared with TF) | `test_determinism.py::TestEnvSeeding` |
| replay sampler seeding | BUG | `TestSamplerSeeding` |
| schedule determinism | BUG | `TestScheduleSeeding`, `TestScheduleDeterminism` |
| driver ordering | OK | Lock-step, read and confirmed. |
| XLA determinism | DIFF | `TestJaxDeterminism` |
| seed reaches the agent | OK | `TestWhatSeedDoesControl` |

### Replay

| Item | Status | Evidence |
|---|---|---|
| eviction at capacity | OK | `TestCapacityEviction` |
| sample age distribution | OK | `TestSampleAge` |
| online fraction | OK | `TestOnlineFraction` |

### HRL core mechanics

| Item | Status | Evidence |
|---|---|---|
| `lambda_return` vs TF `gve` | OK | `test_lambda_return.py::TestMatchesDirector` |
| lambda/gamma limit cases | OK | `TestLimitCases`, `TestHorizonSensitivity` |
| `last`-flag trajectory boundaries | OK | `TestTrajectoryBoundaries` (ours only; TF has no equivalent) |
| Worker goal reward (`cosine_max`) | OK | `test_tensors_vs_director.py::TestGoalRewardMatchesDirector` -- matches Director's formula exactly, including the `max(norm)` magnitude penalty. |
| Worker credit windows vs Director `split_traj` | OK | `test_split_traj_equivalence.py` (45 tests). Our boundary-masked `lambda_return` equals Director's per-window return elementwise across T/k combinations and lambda in {0, 0.5, 0.95, 1}; credit provably does not cross a goal boundary. |
| Worker reward composition | OK | Worker receives only `wkr_goal_rew`; no extrinsic or exploration term, matching Director's `worker_rews: {extr: 0, expl: 0, goal: 1}`. |
| `Consec` training-batch slicing | OK | `test_streams_consec.py` (23 tests). Windows tile the source with no gap and no overlap; the prefix repeats the previous window's tail. Previously untested despite every batch passing through it. |
| Env wrapper chain | OK (equivalent) | `NormalizeAction` is functionally identical to Director's (same finite-bound mask, same affine map). We add `ClipAction` and `UnifyDtypes`; Director adds `ExpandScalars` and applies `TimeLimit` itself. For DMC with `repeat: 1` the resulting task is the same. |
| Manager entropy controller | OK (not a variance source) | Measured across baseline seeds: normalized entropy holds 0.50-0.56 against the 0.5 target and the multiplier stays in a narrow band. Never saturated. |
| Worker advantage scale | OK (not a variance source) **in the plain baseline only** | `wkr_goal_adv_mag` is 0.017-0.026 across all five baseline seeds at both 1M and 4M, so the missing `advnorm` is not producing cross-seed scale drift *there*. **Superseded for any `director_stable` run (2026-09-01):** that block sets `advnorm.impl: meanstd`, which the worker shares, and `wkr_goal_adv_mag` becomes 0.699-0.700 -- ~17x the baseline. The "missing advnorm" is no longer missing, and the worker's step size is now a live variable. |
| Manager extrinsic reward block aggregation | TODO | |
| Manager block/skill boundary alignment | TODO | |
| Skill duration / `split_traj` equivalence | TODO | |
| Manager REINFORCE gradient path | TODO | |
| Actor entropy (`actent`) normalization | TODO | |
| Return / advantage normalization | TODO | The `perc` vs `std` difference is unexamined. |

### Goal autoencoder

Audited 2026-09-01, line by line against
`code/director/embodied/agents/director/hierarchy.py` (`train_vae_replay`,
`train_vae_imag`, `elbo_reward`) and their `configs.yaml`. **Before this the
component had no row anywhere in this document, not even a TODO.** These rows
are `read the code on both sides and compared`, not test-backed; the status key's
stricter sense of "OK" does not apply until tests exist (see next actions).

| Item | Status | Evidence |
|---|---|---|
| Code shape (`skill_shape`) | OK | `[8, 8]` both. |
| Encoder inputs | OK | Director's `goal_encoder.inputs: [goal]` -- `context` is passed to the MLP but not listed, so it is ignored. Ours encodes `deter` alone. |
| Decoder inputs | OK | Director's `goal_decoder.inputs: [skill]`, same reasoning. Ours decodes the skill code alone. |
| Reconstruction target | OK | `deter`, `deter: 1024` on both sides. |
| Reconstruction loss | OK | Director `dist: mse` -> `MSEDist(out, len(shape), 'sum')`, summed over the 1024 deter dims. Ours `MSEDist('sum')` with `goal_rec_loss_agg: sum`. Same reduction over the same dimension count. |
| KL prior and reduction | OK | Uniform one-hot prior, `Independent` over the L=8 block axis, so KL sums over blocks on both sides. |
| KL controller | OK | Director `encdec_kl: {impl: mult, target: 10.0, min: 1e-5, max: 1.0}`; ours `goal_kl_impl/target/min/max` identical. **Trap:** Director's config reads `scale: 0.0`, which looks like the multiplier starts at zero. It does not -- `AutoAdapt.__init__` ignores `scale` unless `impl == 'fixed'` and always initialises to `tf.ones`. Our `goal_kl_init: 1.0` is correct. |
| **KL weight (`goal_autoencoder_beta`)** | **DIFF, binding** | Director: `loss = (rec + kl).mean()`, no coefficient, in both `train_vae_replay` and `train_vae_imag`. Ours: `rec + 0.25 * kl_adapted`. Verified absent in Director exhaustively -- `0.25` appears nowhere in their `configs.yaml`; `encdec_kl` occurs exactly once in the codebase (the AutoAdapt construction) and `goal_kl` only as a boolean; no task preset overrides either; their `Optimizer.__call__` carries only the fp16 grad scale, divided back out. **Binding, not nominal:** `goal/kl_adapt_scale_mean` is pinned at its 1.0 ceiling from ~1M onward with `goal/kl_raw_mean` 11.9 against target 10, i.e. the controller is saturated asking for more pressure while we apply a quarter of it. Ours introduced at 1.0 (`e1ab1ed`, 2026-05-15), tuned to 0.25 in `713d15b` (2026-06-02). Restored by the `director_vaebeta` block. |
| Training data | OK | Director `vae_replay: True`, `vae_imag: False` -- replay only. Ours trains from `repfeat` (replay), matching. |
| Manager exploration reward | OK | Director `expl_rew: adver` -> `elbo_reward` with `adver_impl: squared` -> `((dec.mode() - feat) ** 2).mean(-1)[1:]`, mean over dims from a *sampled* code. Ours `_mgr_expl_reward` computes the same. |
| Goal AE optimizer | DIFF (lr), fixed | Director `encdec_opt`: lr 1e-4, wd 1e-2 kernel-only, eps 1e-6, clip 100. Ours was lr 4e-5, wd 0; `director_stable` set wd 2.5e-2 (matching the decoupled shrinkage `wd*lr = 1e-6`), and `director_optmatch` (2026-09-01) now sets lr 1e-4 + wd 1e-2 exactly. `eps` and the `agc`-vs-`clip` difference remain. |
| `manager_delta` | OK | `False` both -- the decoded goal is absolute, not a residual on the current state. |
| Manager entropy target | OK | Director's `mconfig` sets `actent.target = manager_actent = 0.5`, normalized, matching our `manager_actent_target: 0.5`. |
| Goal AE numerics vs Director | TODO | No test asserts encoder/decoder/KL numerics against a transcription of Director's, the way `test_tensors_vs_director.py` does for block pooling. |
### World model

| Item | Status | Evidence |
|---|---|---|
| RSSM prior/post shapes and dtypes | TODO | |
| KL balancing | TODO | |
| Reward head output distribution (twohot vs mse) | TODO | |
| Continue head / discount | TODO | |

### Numerics

| Item | Status | Evidence |
|---|---|---|
| bfloat16 vs fp32 accumulation in losses | TODO | |
| Per-loss-term gradient magnitudes | partial | `train/opt/*_grad_rms` inspected for clipping only. |
| NaN/inf guards | TODO | |

## Suggested next actions

Ordered by expected effect on the variance, largest first.

1. **Set `replay.size` to 1e6** for 4M-step runs, matching Director. This is a
   one-line config change and is the only finding so far that plausibly
   accounts for the size of the spread.
2. **Stop using `train/mgr_extr_rew` as a progress metric**; it cannot fall.
3. **Seed the DMC env** (`task_kwargs={'random': seed}`) and **forward
   `config.seed` into `make_replay`**, so that repeating a seed is meaningful
   and seed variance can actually be measured.
4. Re-measure seed spread with 1 and 2 in place before touching lr, batch size
   or gradient steps -- none of those have evidence against them yet, and
   gradient clipping already has evidence *for* being fine.

Added 2026-09-01, from the goal-VAE audit and the e718-e721 null (see
EXPERIMENTS.md for the measurements behind each):

5. **`goal_autoencoder_beta` 0.25 -> 1.0** (`director_vaebeta`). The only
   *active* goal-VAE difference from Director: the KL controller is saturated at
   its ceiling and we apply a quarter of what it asks for.
6. **`director_optmatch`** (added 2026-09-01): lr 4e-5 -> 1e-4 and wd -> 1e-2,
   matching Director exactly. Layered after `director_stable` this holds the
   decoupled shrinkage constant at `wd*lr = 1e-6` and changes only the lr.
7. **Log `success_manager` and a time-to-reach within the K-block.** Director
   logs `success_manager` (fraction of rollouts whose final goal reward > 0.7)
   and we log no equivalent, so the "worker arrives early and idles for the rest
   of the block" hypothesis -- which the e718-e721 collapse signature points at
   -- currently cannot be tested at all.
8. **Test the goal AE numerics against Director**, the way
   `test_tensors_vs_director.py` covers block pooling. Today's audit is a code
   read on both sides, not an assertion.
