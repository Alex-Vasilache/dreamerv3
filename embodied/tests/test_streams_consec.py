"""`embodied/core/streams.py` -- the slicer every training batch passes through.

`Consec` turns a replay sample into the consecutive training windows the agent
sees, carrying `replay_context` steps of prefix so the RSSM can warm its carry.
It had no dedicated test anywhere, yet a wrong offset here silently changes what
every gradient step is computed on: windows would overlap (double-counting
transitions) or leave gaps (dropping them).

Production settings: `length = batch_length = 64`, `consec = consec_train = 1`,
`prefix = replay_context = 1`.
"""

import numpy as np
import pytest

from embodied.core import streams


def source(total, batch=2, key='is_first'):
  """A stream whose values encode their own absolute index, so slices are checkable."""
  arange = np.tile(np.arange(total, dtype=np.int32), (batch, 1))
  while True:
    yield {key: np.zeros((batch, total), bool), 'idx': arange.copy()}


class ListSource:
  """Minimal stream over a fixed list of batches, with the save/load API."""

  def __init__(self, batches):
    self.batches = list(batches)
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):
    out = self.batches[self.i % len(self.batches)]
    self.i += 1
    return out

  def save(self):
    return {'i': self.i}

  def load(self, data):
    self.i = data['i']


def make(total, length, consec, prefix, batch=2, **kw):
  gen = source(total, batch)
  batches = [next(gen) for _ in range(8)]
  st = streams.Consec(ListSource(batches), length, consec, prefix, **kw)
  return iter(st)


class TestProductionShape:
  """length 64, consec 1, prefix 1 -- what the agent actually receives."""

  def test_chunk_is_length_plus_prefix_steps(self):
    it = make(total=65, length=64, consec=1, prefix=1)
    chunk = next(it)
    assert chunk['idx'].shape == (2, 65)

  def test_chunk_starts_at_the_context_step(self):
    it = make(total=65, length=64, consec=1, prefix=1)
    chunk = next(it)
    np.testing.assert_array_equal(chunk['idx'][0], np.arange(65))

  def test_a_consec_index_is_attached(self):
    it = make(total=65, length=64, consec=1, prefix=1)
    chunk = next(it)
    assert 'consec' in chunk
    assert chunk['consec'].shape == chunk['is_first'].shape
    assert (chunk['consec'] == 0).all()


class TestWindowTiling:
  """With consec > 1 the windows must tile the source exactly."""

  def test_successive_windows_advance_by_length(self):
    it = make(total=3 * 4 + 2, length=4, consec=3, prefix=2)
    starts = [next(it)['idx'][0, 0] for _ in range(3)]
    assert starts == [0, 4, 8]

  def test_training_steps_tile_without_gap_or_overlap(self):
    """Dropping the prefix, the remaining steps must be exactly contiguous."""
    length, consec, prefix = 4, 3, 2
    it = make(total=consec * length + prefix, length=length, consec=consec,
              prefix=prefix)
    seen = []
    for _ in range(consec):
      chunk = next(it)
      seen.extend(chunk['idx'][0, prefix:].tolist())
    assert seen == sorted(seen), 'windows are out of order'
    assert len(seen) == len(set(seen)), 'a transition was used twice'
    assert seen == list(range(prefix, consec * length + prefix))

  def test_the_prefix_repeats_the_tail_of_the_previous_window(self):
    """The context steps are the previous window's last steps, not new data."""
    length, consec, prefix = 4, 3, 2
    it = make(total=consec * length + prefix, length=length, consec=consec,
              prefix=prefix)
    a = next(it)['idx'][0]
    b = next(it)['idx'][0]
    np.testing.assert_array_equal(b[:prefix], a[length:length + prefix])

  def test_consec_index_counts_up_then_wraps(self):
    it = make(total=3 * 4 + 2, length=4, consec=3, prefix=2)
    idxs = [int(next(it)['consec'][0, 0]) for _ in range(7)]
    assert idxs == [0, 1, 2, 0, 1, 2, 0]

  def test_a_new_source_batch_is_pulled_only_when_the_cycle_restarts(self):
    length, consec, prefix = 4, 3, 2
    gen = source(consec * length + prefix, 2)
    batches = [next(gen) for _ in range(4)]
    for n, b in enumerate(batches):
      # Full width: Consec slices every key on the time axis, so a (B, 1) key
      # would come back empty for any window past the first.
      b['tag'] = np.full(b['idx'].shape, n, np.int32)
    src = ListSource(batches)
    it = iter(streams.Consec(src, length, consec, prefix))
    tags = [int(next(it)['tag'][0, 0]) for _ in range(6)]
    assert tags == [0, 0, 0, 1, 1, 1]


class TestPrefixZero:

  def test_without_a_prefix_the_window_is_exactly_length(self):
    it = make(total=8, length=4, consec=2, prefix=0)
    a, b = next(it), next(it)
    assert a['idx'].shape == (2, 4)
    np.testing.assert_array_equal(a['idx'][0], [0, 1, 2, 3])
    np.testing.assert_array_equal(b['idx'][0], [4, 5, 6, 7])


class TestStrictness:
  """`strict` pins the source length so a mis-sized replay sample is caught."""

  def test_strict_rejects_a_source_longer_than_needed(self):
    it = make(total=100, length=4, consec=3, prefix=2, strict=True)
    with pytest.raises(AssertionError):
      next(it)

  def test_non_strict_accepts_a_longer_source(self):
    it = make(total=100, length=4, consec=3, prefix=2, strict=False)
    chunk = next(it)
    assert chunk['idx'].shape == (2, 6)

  def test_a_source_that_is_too_short_is_always_rejected(self):
    for strict in (True, False):
      it = make(total=5, length=4, consec=3, prefix=2, strict=strict)
      with pytest.raises(AssertionError):
        next(it)

  def test_the_exact_production_size_passes_strict(self):
    it = make(total=65, length=64, consec=1, prefix=1, strict=True)
    assert next(it)['idx'].shape == (2, 65)


class TestDtypesAndKeys:

  def test_every_source_key_survives_the_slice(self):
    it = make(total=65, length=64, consec=1, prefix=1)
    chunk = next(it)
    assert {'is_first', 'idx', 'consec'} <= set(chunk)

  def test_dtypes_are_preserved(self):
    it = make(total=65, length=64, consec=1, prefix=1)
    chunk = next(it)
    assert chunk['is_first'].dtype == bool
    assert chunk['idx'].dtype == np.int32
    assert chunk['consec'].dtype == np.int32

  def test_the_batch_axis_is_untouched(self):
    it = make(total=65, length=64, consec=1, prefix=1, batch=5)
    assert next(it)['idx'].shape[0] == 5

  def test_contiguous_flag_produces_contiguous_arrays(self):
    it = make(total=65, length=64, consec=1, prefix=1, contiguous=True)
    chunk = next(it)
    for k, v in chunk.items():
      assert v.flags['C_CONTIGUOUS'], k


class TestSaveLoad:
  """Checkpoint restore must resume mid-cycle, not silently restart it."""

  def test_the_cycle_position_is_saved(self):
    length, consec, prefix = 4, 3, 2
    it = make(total=consec * length + prefix, length=length, consec=consec,
              prefix=prefix)
    next(it); next(it)
    assert it.save()['index'] == 2

  def test_loading_mid_cycle_cannot_resume_and_raises(self):
    """`load` restores `index` but not `current`, so a mid-cycle resume breaks.

    Latent, and doubly unreachable in production: the train loop checkpoints
    only step/agent/replay (`embodied/run/train.py`), never the stream, so
    `load` is not called at all; and with `consec_train = 1` a saved index is
    always >= consec, which makes `__next__` reset to 0 and pull a fresh batch
    before touching `current`.

    Pinned as a raise rather than silently "fixed": restoring `current` would
    need the exact replay sample back, and resetting the index instead would
    quietly discard part of a cycle. If `consec_train` is ever raised above 1
    AND the stream is added to the checkpoint, this needs a real answer.
    """
    length, consec, prefix = 4, 3, 2
    it = make(total=consec * length + prefix, length=length, consec=consec,
              prefix=prefix)
    next(it); next(it)
    data = it.save()
    it2 = make(total=consec * length + prefix, length=length, consec=consec,
               prefix=prefix)
    it2.load(data)
    with pytest.raises(AttributeError):
      next(it2)

  def test_loading_at_a_cycle_boundary_does_resume_cleanly(self):
    """index == consec resets to 0 and pulls a fresh batch -- the consec=1 case."""
    it = make(total=65, length=64, consec=1, prefix=1)
    next(it)
    data = it.save()
    it2 = make(total=65, length=64, consec=1, prefix=1)
    it2.load(data)
    assert next(it2)['idx'].shape == (2, 65)

  def test_saving_at_a_cycle_boundary_restores_to_the_boundary(self):
    length, consec, prefix = 4, 3, 2
    it = make(total=consec * length + prefix, length=length, consec=consec,
              prefix=prefix)
    for _ in range(3):
      next(it)
    assert it.save()['index'] == 3


class TestStateless:

  def test_it_calls_the_function_once_per_item(self):
    calls = []
    def fn():
      calls.append(1)
      return {'x': np.zeros((1, 2))}
    st = iter(streams.Stateless(fn))
    next(st); next(st)
    assert len(calls) == 2

  def test_arguments_are_forwarded(self):
    seen = []
    def fn(a, b=None):
      seen.append((a, b))
      return {'x': np.zeros((1, 2))}
    st = iter(streams.Stateless(fn, 7, b=9))
    next(st)
    assert seen == [(7, 9)]
