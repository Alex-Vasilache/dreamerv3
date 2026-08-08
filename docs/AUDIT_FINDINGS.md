# Overnight JAX audit — findings log

Running log. Each entry: what was checked, what was found, what was done.
Companion to `docs/VERIFICATION.md` (coverage matrix).

Worktree: `code/dreamerv3_audit`, branch `audit/jax-verification`, based on
`feat/goal-ae-somvae-lipvq` @ ddccdd8. Kept separate from `dreamerv3_somvae`
because six live jobs (e486-e490, e494) run `main.py` from that tree and
self-requeue near their 48h walltime, re-reading the source.

## Bugs found and fixed

### BUG-1 `align_skill_events` broadcasts along the wrong axis (`hrl/heads.py`)

```python
elif e.ndim + 1 == ref.ndim and e.shape == ref.shape[:e.ndim]:
  events[k] = jnp.broadcast_to(e[:, None], ref.shape)   # was
  events[k] = jnp.broadcast_to(e[..., None], ref.shape) # now
```

The branch exists to add the head's missing trailing class axis to a sampled
event. `e[:, None]` inserts the axis at position 1, giving `(B, 1, T)`, which
cannot broadcast to `(B, T, C)` — it raises for the normal case, and when
`T == C` it broadcasts *silently and wrongly*, spreading the time axis across
the class axis.

Impact: **latent, not active.** Every `OneHot` head's `sample()` already returns
a one-hot with `pred()`'s shape, so the first branch always wins and this one is
dead. It would fire the moment a head returned indices instead of one-hots.

Tests: `test_heads_helpers.py::TestAlignSkillEvents` (both the shape and the
`T == C` silent-corruption case).

### BUG-2 `goal/rec_std` is identically 0.0 in every VQ arm (`agent.py`)

```python
goal_rec_loss = vq_mets['vq/rec_q'] + vq_mets.get('vq/rec_e', ...)   # was
...
'goal/rec_std': goal_rec_loss.std(),
```

`vq_goal_loss` returns `'vq/rec_q': rec_q.mean()` — already a scalar. So
`goal_rec_loss` was a scalar and `.std()` of a scalar is 0.0.

**Confirmed in production data**, not just by reading: `train/goal/rec_std` is
exactly `0.0` in every logged row of e486 (744 rows), e488 (640) and e490 (600).
The Director arm computes the same metric from a real `(B, T)` tensor and
reports a genuine spread, so `goal/rec_std` was silently incomparable between
the Director baseline and every quantized arm.

Impact: **reporting only, no effect on training.** `goal_base_loss` — the value
that actually enters `losses['goal_autoencoder']` — was always correct.
`goal/rec_mean` was also correct (a scalar's mean is itself).

Fix: `vq_goal_loss` now also returns the unreduced per-element reconstruction as
`vq/_rec_bt`; `agent.py` pops it before logging and uses it for both
`goal/rec_mean` and `goal/rec_std`. Keys whose name starts with `_` after the
prefix are internal unreduced tensors by convention.

Tests: `test_goal_ae.py::TestReconstructionSpreadMetric` (5 tests), plus the
existing metric-scalar check tightened to assert every logged metric is 0-d.

Confirmed fixed end to end (smoke job 4676699, 300 steps, debug scale): the VQ
arms now report `rec_std` in all 20 logged rows (`som` 0.1757,
`som_orig_line` 0.1721) where it was `0.0` before, and the Director arm's value
is unchanged at 0.0965.

### BUG-3 The full test suite could never run in one process (test harness)

`embodied/jax/internal.py::setup` sets process-global JAX config, including
`jax_transfer_guard='disallow'`. Any test that constructs an `embodied.jax.Agent`
runs it, and every later test in the same pytest process then fails on
`jnp.asarray(<numpy array>)` with

```
XlaRuntimeError: INVALID_ARGUMENT: Disallowed host-to-device transfer
```

Every test file passes on its own, which is why this went unnoticed. Running the
whole suite in one process gave **502 failed / 396 passed** — every failure an
ordering artefact, not a real defect.

Impact: **test infrastructure only.** No effect on training. But it means the
suite's green status has only ever been established file by file, and any CI
that ran `pytest embodied/tests/` would have reported a wall of false failures.

Fix: `embodied/tests/conftest.py` — an autouse fixture that snapshots the six
`jax.config` keys `setup` writes and restores them after each test. `setup`'s
behaviour is left alone; the guard is correct for real runs.

### BUG-4 `Replay(capacity=1)` crashes (`embodied/core/selectors.py`)

`Uniform.__delitem__` asserted `2 <= len(self)`. Eviction has to empty the
selector before re-filling it, so any replay with `capacity <= 1` died on its
second insert with `AssertionError: 1`.

The assertion was simply too strict: the swap-with-last removal below it is
correct at `len == 1` (index 0, `keys.pop()` empties the list, `index !=
len(self.keys)` is then False so no swap runs, leaving `keys == []` and
`indices == {}`). Verified directly before changing it.

Impact: **none in production** — we run `capacity = 5e6`. It did block 10
parametrizations of `test_replay.py`.

Fix: `assert 1 <= len(self)`. **This is the only change to `embodied/`**, which
is otherwise byte-identical to the upstream `code/dreamerv3` tree — worth
knowing if you ever re-sync from upstream.

### BUG-5 Test rot: 53 replay tests had not run since an API change

`Replay.dataset(batch)` was replaced by `Replay.sample(batch, mode)` (the old
method survives commented out at the bottom of `embodied/core/replay.py`).
`test_replay.py` was never ported, so all 53 of its tests failed with
`AttributeError: 'Replay' object has no attribute 'dataset'`.

So the replay buffer — the component this whole investigation turned out to
hinge on — had no working test coverage at all.

Fix: a `make_dataset(replay, batch)` shim over `sample`, plus the `capacity=1`
fix above. `test_replay.py` now runs **61 passed, 2 skipped**.

The 2 skips are real upstream edge cases, both documented in the test body and
both unreachable in production:

1. `test_restore_noclear[5-25-2]` — `Replay.load()` into a **non-empty** buffer
   can leave an item whose sequence spans more chunks than were restored;
   `_getseq` walks `chunk.succ` and raises `KeyError`. Needs
   `length > 2 * chunksize`. Production loads into a fresh replay at checkpoint
   restore, and length 64 against chunksize 1024 never spans 3 chunks.
2. `test_threading` — a race in `Replay.load()`: it `setdefault`s a `refs` entry
   only for the chunks it loaded, then iterates **all** of `self.chunks` doing
   `self.refs[uuid] += delta`, so a concurrently added chunk gives
   `KeyError: <id>` (a different id each run). Production calls `load()` once at
   checkpoint restore, before `driver.reset()` and before any adder thread runs.

### BUG-6 Three more stale tests, all asserting removed behaviour

- `test_hrl_package.py::test_no_unlisted_submodules` — `SUBMODULES` was never
  updated when `goal_ae`, `lipschitz` and `vq` were added. The test was doing
  its job; the list was fixed.
- `test_driver.py::test_env_reset` / `test_unexpected_reset` — asserted
  `seq['reset']`, but `Driver._step` builds the transition before putting
  `reset` into `self.acts` for the next env call. The vector they checked is
  identical to `is_last`, asserted on the preceding line. Rewritten to assert
  the current contract.
- `test_goal_ae_agent.py::...agree_on_the_entropy_scale` — asserted the ring
  head and the Director head start at similar normalized entropy. Commit
  `4c0e3cb` ("Make the ring head's scale an actual temperature") deliberately
  broke that: `scale_init` is now a temperature chosen so the head starts at or
  **below** the 0.5 entropy target, and `ManagerRingHead`'s docstring quantifies
  it ("normalized entropy is 0.16 on a random codebook"). Measured 0.144.
  Rewritten to assert the documented requirement, plus a new test checking the
  invariant the old assertion was really guarding — that both heads normalize by
  `blocks * log(classes)`. They do; there is no L-sized bug.

### Note: stale pytest cache can silently run old test code

On this Lustre filesystem, pytest's assertion-rewritten `__pycache__` entries
can survive an edit, so a run executes the **previous** version of a test file.
Symptom: pytest prints `>   ???` for the failing line and a line number that
does not match the source. This cost one confusing cycle here.
`sbatch/run_pytest_cpu.sbatch` now clears `__pycache__` and `.pytest_cache`
before every run.

## End-to-end validation of the fixes

Smoke job 4676699 ran four arms (`director`, `som`, `som_orig_line`,
`som_orig_lipvq_line`) for 300 steps each through the real env + replay + driver
loop at debug scale, on the audit tree with all three fixes applied.
**SMOKE RESULT: PASS** — every arm reached 300 steps, wrote metrics, and had no
non-finite value in any logged key.

## Suite status

| | tests | passed | failed | skipped |
|---|---|---|---|---|
| Branch HEAD `ddccdd8` (ordering fix only) | 585 | 528 | **57** | 0 |
| After this audit | 947 | **945** | **0** | 2 |

362 tests added. The 57 baseline failures are gone: 53 were the unported replay
suite (BUG-5), 2 the driver `reset` assertions, 1 the stale submodule list, 1
the ring-head entropy assertion (all BUG-6). The 2 skips are the documented,
production-unreachable upstream edge cases listed under BUG-5.

Failure sets were compared directly between a pristine baseline worktree
(`code/dreamerv3_baseline`, `ddccdd8` + `conftest.py` only) and this tree:
**no test that passed at baseline fails here.**

`test_parallel.py` and `test_train.py` are excluded from all counts: they import
`zerofun`, which the codebase replaced with `portal` and which is not installed.
Pre-existing and unrelated.

## Verified correct (no change needed)

| Component | Evidence |
|---|---|
| `lambda_return` vs Director `gve` | 19 tests, exact to 1e-5 across lambda/gamma, terminals included |
| Fixed-K block pooling vs Director `abstract_traj` | 72 tests: pooled reward, pooled continuation, decision-state indices, sparse scatter, gradients |
| Actor-critic stop-gradient routing | 34 tests: policy loss never reaches a critic, value loss never reaches the policy, actions/rewards/weights/bootstraps all constant where intended |
| `hrl/heads.py` pure helpers | 35 tests: shifts, ring/line smoothing kernels, entropy slicing, REINFORCE head selection |

## Notable non-bugs (deliberate differences)

- Our worker actor is pure REINFORCE (`sg` on the advantage), so the goal reward
  is a constant to it. TF Director used `actor_grad_cont: backprop` and let the
  reward's gradient reach the imagined states. Different algorithm, not an error.
- `repl_loss` uses `disc = 1 - 1/horizon` regardless of `contdisc`, while
  `imag_loss` uses `disc = 1` under `contdisc: True`. Identical in the upstream
  `code/dreamerv3` tree, so this is upstream DreamerV3 design.
