"""Replay-buffer behaviour that decides what a training batch contains.

These tests exist because five identical hopper_hop Director-arm seeds finished
at 307, 220, 146, 41 and 59, and `train/mgr_extr_rew` rose monotonically in all
five while `episode/score` collapsed in three. Both are consistent with the
training distribution being dominated by old data, so the properties that
control that are pinned here rather than left to config reading.

Reference point: TF Director (`code/director/embodied/agents/director/
configs.yaml`) uses `replay_size: 1e6`. Our runs use `replay.size: 5e6` with
`run.steps: 4e6`.
"""

import time

import numpy as np
import pytest

import embodied


def make_step(value=0.0):
  return {
      'image': np.zeros((4, 4, 3), np.uint8),
      'reward': np.float32(value),
      'is_first': np.bool_(False),
      'is_last': np.bool_(False),
      'is_terminal': np.bool_(False),
  }


def fill(replay, steps, workers=1):
  for i in range(steps):
    for w in range(workers):
      replay.add(make_step(float(i)), worker=w)


class TestCapacityEviction:
  """`capacity` is the only thing that bounds how old a sample can be."""

  def test_nothing_is_evicted_when_capacity_exceeds_total_inserts(self):
    # The production setting: capacity 5e6, run length 4e6. Scaled down by
    # 1000x, the ratio is what matters.
    replay = embodied.replay.Replay(length=8, capacity=5000)
    fill(replay, 4000)
    # Every item ever inserted is still sampleable.
    assert len(replay) == 4000 - 8 + 1
    assert replay.stats()['items'] == len(replay)

  def test_eviction_happens_once_capacity_is_the_binding_constraint(self):
    # Director's ratio: capacity 1e6 against a 4e6-step run.
    replay = embodied.replay.Replay(length=8, capacity=1000)
    fill(replay, 4000)
    assert len(replay) == 1000

  def test_oldest_sampleable_item_is_bounded_by_capacity_not_by_run_length(self):
    small = embodied.replay.Replay(length=8, capacity=1000)
    fill(small, 4000)
    # `fifo` holds insertion order, so its head is the oldest survivor.
    oldest_itemid = small.fifo[0]
    assert oldest_itemid == small.itemid - 1000

    large = embodied.replay.Replay(length=8, capacity=5000)
    fill(large, 4000)
    assert large.fifo[0] == 0, 'item 0 -- the random policy -- is still in the buffer'


class TestSampleAge:
  """How stale the average training sample is."""

  @staticmethod
  def mean_age(replay, draws, rng):
    """Mean gap between the newest item and a uniformly drawn one."""
    newest = replay.itemid - 1
    ids = list(replay.items.keys())
    return float(np.mean([newest - ids[rng.integers(len(ids))] for _ in range(draws)]))

  def test_uniform_sampling_over_a_never_evicting_buffer_is_half_the_run_old(self):
    rng = np.random.default_rng(0)
    replay = embodied.replay.Replay(length=8, capacity=5000)
    fill(replay, 4000)
    age = self.mean_age(replay, 2000, rng)
    # Uniform over [0, N) has mean N/2. At 4M production steps this is ~2M.
    assert 0.4 * 4000 < age < 0.6 * 4000

  def test_capacity_bound_buffer_is_far_fresher(self):
    rng = np.random.default_rng(0)
    replay = embodied.replay.Replay(length=8, capacity=1000)
    fill(replay, 4000)
    age = self.mean_age(replay, 2000, rng)
    # Uniform over the surviving window: mean = capacity/2.
    assert 0.4 * 1000 < age < 0.6 * 1000

  def test_the_two_regimes_differ_by_the_capacity_ratio(self):
    rng = np.random.default_rng(0)
    a = embodied.replay.Replay(length=8, capacity=5000)
    b = embodied.replay.Replay(length=8, capacity=1000)
    fill(a, 4000)
    fill(b, 4000)
    # 4x staler on average, which is the 4e6/1e6 production ratio.
    assert self.mean_age(a, 2000, rng) > 3.0 * self.mean_age(b, 2000, rng)


class TestOnlineFraction:
  """`online=True` injects fresh sequences, but only a fixed small share."""

  def test_online_queue_gains_one_entry_per_length_steps_per_worker(self):
    replay = embodied.replay.Replay(length=64, capacity=10 ** 6, online=True)
    fill(replay, 640, workers=1)
    # Inserts only start once the stream holds `length` steps, so the count is
    # steps/length minus the first partial window.
    assert len(replay.queue) == 640 // 64 - 1

  def test_online_share_of_a_batch_is_bounded_by_length_over_sample_rate(self):
    # Production: 8 workers each queue 1 per 64 steps -> 8 per 64 env steps,
    # while the learner draws 1 sequence per env step (samples/insert == 1.00,
    # measured in e403's metrics.jsonl). So at most 8/64 = 12.5% is fresh.
    workers, length = 8, 64
    env_steps = 64
    queued = workers * env_steps // length
    drawn = env_steps
    assert queued / drawn == pytest.approx(0.125)


class TestSamplerSeeding:
  """The replay sampler's RNG is not derived from `--seed`."""

  def test_uniform_selector_defaults_to_seed_zero(self):
    got = embodied.replay.Replay(length=8).sampler.rng.integers(0, 10 ** 6, 8)
    want = np.random.default_rng(0).integers(0, 10 ** 6, 8)
    np.testing.assert_array_equal(got, want)

  def test_two_replays_built_the_default_way_draw_the_same_order(self):
    a = embodied.replay.Replay(length=4, capacity=1000)
    b = embodied.replay.Replay(length=4, capacity=1000)
    fill(a, 200)
    fill(b, 200)
    assert [a.sampler() for _ in range(50)] == [b.sampler() for _ in range(50)]

  def test_an_explicit_seed_does_change_the_draw_order(self):
    a = embodied.replay.Replay(length=4, capacity=1000, seed=0)
    b = embodied.replay.Replay(length=4, capacity=1000, seed=1)
    fill(a, 200)
    fill(b, 200)
    assert [a.sampler() for _ in range(50)] != [b.sampler() for _ in range(50)]

  def test_make_replay_does_not_forward_the_config_seed(self):
    """Regression guard: if `seed=` is ever wired through, update this."""
    import inspect

    from dreamerv3 import main
    src = inspect.getsource(main.make_replay)
    kwargs_block = src.split('kwargs = dict(')[1].split(')')[0]
    assert 'seed' not in kwargs_block, (
        'make_replay now forwards a seed; the sampler is no longer shared '
        'across seeds and these tests need revisiting')


class TestScheduleDeterminism:
  """`report`/`save`/`log` fire on wall-clock, not on step count."""

  def test_localclock_ignores_the_step_argument_entirely(self):
    clock = embodied.LocalClock(3600)
    clock(step=0)  # first call only anchors `prev`; returns `first` (False)
    # A billion-step jump does not fire it, because steps are not consulted.
    assert not clock(step=10 ** 9)

  def test_localclock_fires_on_elapsed_time_with_no_step_change(self):
    clock = embodied.LocalClock(0.05)
    clock(step=0)
    assert not clock(step=0)
    time.sleep(0.06)
    assert clock(step=0), 'fired purely because wall time passed'

  def test_report_and_train_share_one_sampler_so_report_shifts_training(self):
    """The mechanism that breaks same-seed reproducibility.

    `make_stream(replay, 'report')` and `make_stream(replay, 'train')` both
    call `replay.sample(...)`, which draws from `self.sampler`. Report fires on
    a wall-clock schedule, so two runs of the same seed on differently loaded
    nodes consume the shared RNG at different step counts and diverge.
    """
    replay = embodied.replay.Replay(length=4, capacity=1000)
    fill(replay, 200)
    baseline = [replay.sampler() for _ in range(20)]

    other = embodied.replay.Replay(length=4, capacity=1000)
    fill(other, 200)
    # Simulate one report batch landing before the same 20 training draws.
    [other.sampler() for _ in range(16)]
    perturbed = [other.sampler() for _ in range(20)]

    assert baseline != perturbed
