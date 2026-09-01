"""The count tables under multi-device sharding.

Every other test in test_mgr_novel.py runs on one device, which matches how the
arm actually runs on a100 (1 GPU per job, nothing sharded). It does NOT match
v100/p100, where a job takes 4 GPUs: `partition_rules` is unset so params default
to P() (replicated) while the batch is sharded across the mesh.

That combination is the risky one, because the count tables are nj.Variables
updated by ACCUMULATION rather than by gradients:

    add = deposit(ids).sum(0) * weight
    select_tab <- select_tab + add

`.sum(0)` reduces over the sharded batch axis while the result has to land
replicated, so XLA must insert a cross-device all-reduce. If it kept each
device's local partial sum instead, select_mass would be 1/4 of the truth; if it
added the full sum once per device, 4x. Nothing would crash either way -- the
bonus 1/sqrt(n+1) would just be quietly wrong, in the exact quantity the arm is
studying. select_mass must equal decisions * weight, and that invariant has
already caught two real accounting bugs.

These need 4 devices. JAX fixes the device count at first use, so run this file
on its own:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python -m pytest embodied/tests/test_mgr_novel_sharded.py -v

Under a full-suite run another module will usually have initialised JAX with one
device first, and these skip rather than pass vacuously.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from dreamerv3.hrl.explore import CodeCounts

L = 8
NDEV = 4

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < NDEV,
    reason=(f'needs {NDEV} devices; run this file alone with '
            f'XLA_FLAGS=--xla_force_host_platform_device_count={NDEV}'))


def _build(**kw):
  kw.setdefault('decay', 1.0)
  kw.setdefault('weight', 1.0)
  cc = CodeCounts(name='cc', **kw)
  params = nj.init(lambda i: cc.update(i))(
      {}, jnp.zeros((1, 1, L), jnp.int32), seed=0)
  return cc, params


def _mass(cc, params, ids, weight):
  """select_mass after one update, run under the given input shardings."""
  def fn(i):
    cc.update_selected(i, weight)
    return cc.select_tab.read().sum()
  _, out = jax.jit(nj.pure(fn))(params, ids)   # nj.pure -> (newparams, out)
  return float(out)


class TestShardedAccumulation:

  @pytest.mark.parametrize('batch', [8, 16])
  def test_sharded_batch_gives_the_same_mass_as_one_device(self, batch):
    cc, params = _build()
    ids = jnp.asarray(
        np.random.default_rng(0).integers(0, 8, (batch, 2, L)), jnp.int32)
    expect = batch * 2 * 1.0        # one decision deposits exactly 1.0

    single = _mass(cc, params, ids, 1.0)

    mesh = Mesh(np.array(jax.devices()[:NDEV]), ('d',))
    sparams = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P())), params)
    sids = jax.device_put(ids, NamedSharding(mesh, P('d')))
    multi = _mass(cc, sparams, sids, 1.0)

    assert single == pytest.approx(expect, rel=1e-5)
    # The failure this exists for: 1/NDEV means local partial sums were kept,
    # NDEV means the full sum was applied once per device.
    assert multi == pytest.approx(expect, rel=1e-5), (
        f'sharded mass {multi} vs {expect} ({multi / expect:.2f}x)')

  def test_train_ratio_weight_still_reads_in_env_steps_when_sharded(self):
    """The real config: weight = 1/train_ratio, so mass counts env steps."""
    cc, params = _build(weight=1 / 64.)
    ids = jnp.asarray(
        np.random.default_rng(1).integers(0, 8, (64, 1, L)), jnp.int32)
    mesh = Mesh(np.array(jax.devices()[:NDEV]), ('d',))
    sparams = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P())), params)
    sids = jax.device_put(ids, NamedSharding(mesh, P('d')))
    assert _mass(cc, sparams, sids, 1 / 64.) == pytest.approx(1.0, rel=1e-5)

  def test_bonus_reads_the_same_table_on_every_shard(self):
    """ucb_bonus gathers from the replicated table; a stale shard would differ."""
    cc, params = _build()
    ids = jnp.asarray(
        np.random.default_rng(2).integers(0, 8, (NDEV * 2, 1, L)), jnp.int32)

    def fn(i):
      cc.update_selected(i, 1.0)
      return cc.ucb_bonus(i.reshape((-1, L)))

    mesh = Mesh(np.array(jax.devices()[:NDEV]), ('d',))
    sparams = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P())), params)
    _, b_single = jax.jit(nj.pure(fn))(params, ids)
    _, b_multi = jax.jit(nj.pure(fn))(
        sparams, jax.device_put(ids, NamedSharding(mesh, P('d'))))
    np.testing.assert_allclose(
        np.asarray(b_single), np.asarray(b_multi), rtol=1e-5)
