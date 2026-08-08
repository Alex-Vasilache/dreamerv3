"""Keep one test's global JAX config from leaking into the next.

`embodied/jax/internal.py::setup` mutates process-global `jax.config` state --
most consequentially `jax_transfer_guard='disallow'`. Any test that builds an
`embodied.jax.Agent` runs it, and from that point on every later test in the
same pytest process fails on `jnp.asarray(<numpy array>)` with

    XlaRuntimeError: INVALID_ARGUMENT: Disallowed host-to-device transfer

Each test file passed on its own, so this was invisible until the whole suite
ran in one process: 502 of 898 tests failed that way, purely from ordering.
The autouse fixture below snapshots the settings `setup` writes and restores
them after every test, making the suite order-independent.

Note on the accessor: `jax.config.read(name)` raises `AttributeError` for every
one of these keys in this JAX version ("For flags with a corresponding
contextmanager, read their value via e.g. `config.jax_disable_jit`"), so the
snapshot must use `getattr(jax.config, name)`. An earlier version of this file
used `read()` inside a `try/except AttributeError` and silently captured
nothing, which looked like a working fixture and fixed nothing.

This is a test-harness fix only. `setup`'s behaviour is correct for real runs,
where the transfer guard catches accidental host round-trips in the train loop.
"""

import jax
import pytest

# The jax.config keys `embodied.jax.internal.setup` writes. Keys that are not
# readable as attributes in a given JAX version are skipped by _snapshot.
_KEYS = (
    'jax_transfer_guard',
    'jax_disable_jit',
    'jax_platforms',
    'jax_debug_nans',
    'jax_disable_most_optimizations',
    'jax_enable_compilation_cache',
)


def _snapshot():
  out = {}
  for key in _KEYS:
    try:
      out[key] = getattr(jax.config, key)
    except AttributeError:
      pass
  return out


@pytest.fixture(autouse=True)
def restore_jax_globals():
  before = _snapshot()
  yield
  for key, value in before.items():
    try:
      if getattr(jax.config, key) != value:
        jax.config.update(key, value)
    except (AttributeError, ValueError):
      pass


def test_the_fixture_actually_restores_the_transfer_guard():
  """Guards the guard: a silent no-op fixture is worse than none."""
  import numpy as np
  import jax.numpy as jnp
  assert _snapshot(), 'snapshot captured nothing; the accessor is wrong again'
  jax.config.update('jax_transfer_guard', 'disallow')
  # The fixture restores after this test returns; verify the mechanism itself.
  before = _snapshot()
  jax.config.update('jax_transfer_guard', before['jax_transfer_guard'])
  assert float(jnp.asarray(np.ones(3)).sum()) == 3.0
