#!/usr/bin/env python3
"""What does a code look like as a goal, and what does moving it a given
index distance do?

Draws uniform hard codes, then for each builds four partners at index distances
that are fractions of what that code can actually reach:

    d = dmax(c),  3/4 dmax,  1/2 dmax,  1/4 dmax

``dmax(c) = sum_l max(c_l, C-1-c_l)`` is a property of the CODE, not a
constant -- over uniform codes it averages 44 of the 56 the axis admits, and
only a code with every block at an extreme reaches 56. Using fractions of the
code's own dmax therefore compares like with like across base codes.

Each row of the output is one base code and its four partners:

    col 1     the base code, decoded to a goal and rendered to pixels
    cols 2-5  partners at 1/4, 1/2, 3/4 and the full dmax away

Under each panel is the realized index distance and the cosine_max between the
partner's goal and the base's, which is the quantity the worker is rewarded on.

  python -u render_code_pairs.py --run_dir /work/.../e592_... --out out/e592
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'experiments/goal_geometry'))
sys.path.insert(0, str(root / 'experiments/goal_code_struct_corr'))

import jax
import jax.numpy as jnp
import numpy as np
import ninjax as nj

from code_sim_to_state_sim import cosine_max, onehot_of
from diag_goal_geometry import build, make_fns
from dreamerv3.hrl import explore

FRACS = (0.25, 0.5, 0.75, 1.0)


def partners_at(ids, fracs, classes, seed=0):
  """For each base code, one partner at each fraction of its own dmax."""
  key = jax.random.PRNGKey(seed)
  _, _, cap, dmax = explore.capacities(jnp.asarray(ids), classes)
  out, dists = [], []
  for i, f in enumerate(fracs):
    d = jnp.maximum(jnp.round(f * dmax.astype(jnp.float32)).astype(jnp.int32), 1)
    mag = explore.spread(jax.random.fold_in(key, i), cap, d, classes)
    down, up = jnp.asarray(ids), (classes - 1) - jnp.asarray(ids)
    can_up, can_dn = mag <= up, mag <= down
    coin = jax.random.uniform(
        jax.random.fold_in(key, 100 + i), mag.shape) < 0.5
    go_up = jnp.where(can_up & can_dn, coin, can_up)
    part = jnp.clip(jnp.asarray(ids) + jnp.where(go_up, mag, -mag),
                    0, classes - 1)
    out.append(np.asarray(part))
    dists.append(np.asarray(jnp.abs(part - jnp.asarray(ids)).sum(-1)))
  return out, dists, np.asarray(dmax)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--ckpt', default=None)
  p.add_argument('--out', required=True)
  p.add_argument('--rows', type=int, default=6)
  p.add_argument('--seed', type=int, default=0)
  a = p.parse_args()

  config, agent = build(a.run_dir, a.ckpt)
  impl, _, decode = make_fns(config, agent)
  L, C = [int(x) for x in config.agent.skill_shape]
  name = pathlib.Path(str(a.run_dir).rstrip('/')).name
  print(f'=== {name} | impl={impl} L={L} C={C} | task={config.task}')

  rng = np.random.default_rng(a.seed)
  base = rng.integers(0, C, size=(a.rows, L))
  parts, dists, dmax = partners_at(base, FRACS, C, a.seed)
  print('dmax per base code:', ' '.join(str(int(v)) for v in dmax))

  # decode every code to a goal deter, then render the deters to pixels
  allids = np.concatenate([base] + parts, 0)
  deters = np.asarray(decode(onehot_of(allids, C)), np.float32)

  model = agent.model
  imgkeys = list(model.dec.imgkeys)
  assert imgkeys, 'this world model has no image decoder'

  def render(det):
    carry = model.dec.initial(len(det))
    feat = dict(deter=jnp.asarray(det),
                stoch=jnp.zeros((len(det), *model.dyn.entry_space['stoch'].shape),
                                jnp.float32))
    reset = jnp.ones((len(det),), bool)
    return model._decode_images_uint8(carry, feat, reset)

  imgs = {k: np.asarray(v) for k, v in
          nj.pure(render)(agent.params, deters)[1].items()}
  key = imgkeys[0]
  frames = imgs[key]
  n = a.rows
  grid = [frames[:n]] + [frames[(i + 1) * n:(i + 2) * n]
                         for i in range(len(FRACS))]

  sim = [cosine_max(deters[:n], deters[(i + 1) * n:(i + 2) * n])
         for i in range(len(FRACS))]
  print('%8s %10s %10s' % ('fraction', 'mean d', 'mean cos_max'))
  for i, f in enumerate(FRACS):
    print('%8.2f %10.1f %10.3f' % (f, dists[i].mean(), sim[i].mean()))

  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      str(out) + '.npz', base=base, parts=np.stack(parts), dists=np.stack(dists),
      dmax=dmax, frames=np.stack(grid), sim=np.stack(sim), fracs=np.array(FRACS),
      task=str(config.task), run=name, imgkey=key)
  print('wrote', str(out) + '.npz')


if __name__ == '__main__':
  main()
