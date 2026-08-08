"""`embodied/core/chunk.py` -- the on-disk storage every transition passes through.

A `Chunk` is a fixed-size block of consecutive transitions; `Replay` stitches
sequences out of them and follows `succ` when a sequence spans a boundary. Only
incidentally exercised by `test_replay.py`, so its own invariants -- that a
slice returns the transitions actually written, that save/load round-trips
exactly, and that the successor link survives -- were never asserted directly.

A silent error here corrupts training data rather than crashing: the agent
would train on the wrong transitions with no visible symptom.
"""

import numpy as np
import pytest

import elements
from embodied.core import chunk as chunklib


def step(i):
  return {
      'reward': np.float32(i),
      'action': np.full((3,), i, np.float32),
      'is_first': np.bool_(i == 0),
  }


def filled(size, n=None):
  c = chunklib.Chunk(size)
  for i in range(size if n is None else n):
    c.append(step(i))
  return c


class TestAppend:

  def test_length_tracks_the_number_of_appends(self):
    c = chunklib.Chunk(8)
    assert c.length == 0
    for i in range(5):
      c.append(step(i))
    assert c.length == 5

  def test_appending_past_the_size_is_rejected(self):
    c = filled(4)
    with pytest.raises(AssertionError):
      c.append(step(99))

  def test_buffers_are_allocated_at_full_size_on_first_append(self):
    c = chunklib.Chunk(8)
    c.append(step(0))
    assert c.data['action'].shape == (8, 3)
    assert c.data['reward'].shape == (8,)

  def test_dtypes_follow_the_first_step(self):
    c = filled(4)
    assert c.data['reward'].dtype == np.float32
    assert c.data['action'].dtype == np.float32
    assert c.data['is_first'].dtype == bool

  def test_values_land_at_their_own_index(self):
    c = filled(6)
    np.testing.assert_array_equal(c.data['reward'][:6], np.arange(6))

  def test_only_the_first_step_is_flagged_is_first(self):
    c = filled(6)
    assert c.data['is_first'][0]
    assert not c.data['is_first'][1:6].any()


class TestSlice:

  def test_a_slice_returns_the_transitions_written(self):
    c = filled(10)
    out = c.slice(3, 4)
    np.testing.assert_array_equal(out['reward'], [3, 4, 5, 6])
    np.testing.assert_array_equal(out['action'][:, 0], [3, 4, 5, 6])

  def test_a_full_length_slice_is_allowed(self):
    c = filled(10)
    assert c.slice(0, 10)['reward'].shape == (10,)

  def test_slicing_past_the_written_length_is_rejected(self):
    c = filled(10, n=6)
    with pytest.raises(AssertionError):
      c.slice(4, 4)

  def test_a_negative_index_is_rejected(self):
    c = filled(10)
    with pytest.raises(AssertionError):
      c.slice(-1, 2)

  def test_an_empty_slice_is_allowed(self):
    c = filled(10)
    assert c.slice(3, 0)['reward'].shape == (0,)

  def test_every_key_is_present_in_the_slice(self):
    c = filled(10)
    assert set(c.slice(0, 2)) == {'reward', 'action', 'is_first'}


class TestUpdate:
  """`Replay.update` writes learner outputs (e.g. priorities) back into chunks."""

  def test_update_overwrites_the_named_range(self):
    c = filled(10)
    c.update(2, 3, {'reward': np.array([90, 91, 92], np.float32)})
    np.testing.assert_array_equal(c.data['reward'][:6], [0, 1, 90, 91, 92, 5])

  def test_update_past_the_written_length_is_rejected(self):
    c = filled(10, n=5)
    with pytest.raises(AssertionError):
      c.update(3, 4, {'reward': np.zeros(4, np.float32)})

  def test_a_negative_index_is_rejected(self):
    c = filled(10)
    with pytest.raises(AssertionError):
      c.update(-1, 2, {'reward': np.zeros(2, np.float32)})

  def test_updating_a_subset_of_keys_leaves_the_others_alone(self):
    c = filled(10)
    before = c.data['action'].copy()
    c.update(0, 3, {'reward': np.zeros(3, np.float32)})
    np.testing.assert_array_equal(c.data['action'], before)


class TestIdentityAndSuccessor:

  def test_each_chunk_gets_a_distinct_uuid(self):
    a, b = chunklib.Chunk(4), chunklib.Chunk(4)
    assert a.uuid != b.uuid

  def test_the_successor_starts_unset(self):
    assert chunklib.Chunk(4).succ == elements.UUID(0)

  def test_the_filename_encodes_time_uuid_succ_and_length(self):
    c = filled(6)
    parts = c.filename.split('.')[0].split('-')
    assert len(parts) == 4
    assert int(parts[3]) == 6

  def test_chunks_sort_by_time_then_uuid(self):
    a, b = chunklib.Chunk(4), chunklib.Chunk(4)
    assert (a < b) != (b < a)


class TestSaveLoad:

  def test_round_trip_preserves_every_value(self, tmpdir):
    c = filled(10, n=7)
    c.save(tmpdir)
    path = elements.Path(str(tmpdir)) / c.filename
    back = chunklib.Chunk.load(path)
    np.testing.assert_array_equal(back.data['reward'], c.data['reward'][:7])
    np.testing.assert_array_equal(back.data['action'], c.data['action'][:7])

  def test_round_trip_preserves_dtypes(self, tmpdir):
    c = filled(6)
    c.save(tmpdir)
    back = chunklib.Chunk.load(elements.Path(str(tmpdir)) / c.filename)
    assert back.data['reward'].dtype == np.float32
    assert back.data['is_first'].dtype == bool

  def test_only_the_written_prefix_is_saved(self, tmpdir):
    c = filled(16, n=5)
    c.save(tmpdir)
    back = chunklib.Chunk.load(elements.Path(str(tmpdir)) / c.filename)
    assert back.length == 5
    assert back.data['reward'].shape == (5,)

  def test_the_uuid_and_successor_survive(self, tmpdir):
    c = filled(6)
    other = chunklib.Chunk(6)
    c.succ = other.uuid
    c.save(tmpdir)
    back = chunklib.Chunk.load(elements.Path(str(tmpdir)) / c.filename)
    assert back.uuid == c.uuid
    assert back.succ == c.succ

  def test_saving_twice_is_rejected(self, tmpdir):
    c = filled(4)
    c.save(tmpdir)
    with pytest.raises(AssertionError):
      c.save(tmpdir)

  def test_a_corrupt_file_raises_by_default(self, tmpdir):
    c = filled(4)
    c.save(tmpdir)
    path = elements.Path(str(tmpdir)) / c.filename
    path.write(b'not an npz', mode='wb')
    with pytest.raises(Exception):
      chunklib.Chunk.load(path)

  def test_a_corrupt_file_can_be_skipped(self, tmpdir):
    c = filled(4)
    c.save(tmpdir)
    path = elements.Path(str(tmpdir)) / c.filename
    path.write(b'not an npz', mode='wb')
    assert chunklib.Chunk.load(path, error='none') is None


class TestSequenceStitching:
  """The property `Replay._getseq` relies on when a sequence spans chunks."""

  def test_two_linked_chunks_reconstruct_the_original_stream(self):
    a, b = chunklib.Chunk(4), chunklib.Chunk(4)
    for i in range(4):
      a.append(step(i))
    for i in range(4, 8):
      b.append(step(i))
    a.succ = b.uuid
    joined = np.concatenate([a.slice(2, 2)['reward'], b.slice(0, 2)['reward']])
    np.testing.assert_array_equal(joined, [2, 3, 4, 5])

  def test_nbytes_reports_allocated_memory_not_the_written_prefix(self):
    """`replay/ram_gb` is built from this, and the buffer really is allocated.

    A chunk allocates all `size` rows on its first append, so a partially
    filled chunk occupies exactly as much RAM as a full one. Reporting the
    written prefix instead would understate the buffer's true footprint --
    which is the number that matters for staying inside the node's memory.
    """
    small = filled(64, n=4)
    big = filled(64, n=32)
    assert small.nbytes == big.nbytes
    assert chunklib.Chunk(64).nbytes == 0, 'nothing allocated before first append'
    assert filled(128, n=1).nbytes == 2 * big.nbytes
