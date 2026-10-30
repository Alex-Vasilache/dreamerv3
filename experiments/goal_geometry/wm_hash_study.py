#!/usr/bin/env python3
"""Which world-model hash should key TCE's start side, and how stable is it?

TCE counts (start cell -> goal cell). Today the start cell is the goal code of
the state coarsened to 4 bins per block (4^8 rows). The alternative is to hash
the world model's own state. A hash only helps if it (a) separates states that
matter and (b) keeps meaning the same thing while the world model trains,
because counts made under an old world model are read under a new one.

Method: roll out the final checkpoint's policy in real envs and record the
observations and the previous-action carry. Replay exactly those sequences
through every 0.5M milestone checkpoint (same observations, same actions), so
each candidate hash is computed for the SAME real states under each world model.

Candidates (all 16 bits = 65536 rows, the size of today's start side):
  code4      today's start cell (goal code, 4 bins per block)
  simhash_d  sign of a fixed random projection of the deter state (centred)
  simhash_s  sign of a fixed random projection of the posterior one-hot stoch
  stochmode  posterior class of the first 16 stoch groups, top bit only

Reported per checkpoint: cells used, effective cells exp(H), share of steps
staying in the same cell (temporal coherence), pin pad prefix purity; and per
pair of consecutive checkpoints: share of states keeping their cell
(stability), and the count-weighted version (share of the state distribution
whose old count lands in the same cell).

  python -u wm_hash_study.py --run_dir <run> --out <npz>
"""
import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from diag_goal_geometry import build
from prefix_check import prefix

BITS = 16


def rollout(config, agent, envs, steps):
  """Run the policy; return per-step obs, prevact carry (pytree) and images."""
  import dreamerv3.main as m
  import jax
  B = envs
  env = [m.make_env(config, i) for i in range(B)]
  acts = {k: np.zeros((B,) + v.shape, v.dtype) for k, v in env[0].act_space.items()}
  acts['reset'] = np.ones(B, bool)
  carry = agent.init_policy(B)
  obs_seq, prev_seq = [], []
  for t in range(steps):
    obs = [e.step({k: v[i] for k, v in acts.items()}) for i, e in enumerate(env)]
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    pobs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    pobs = {k: pobs[k] for k in agent.obs_space}
    prev_seq.append(jax.tree.map(lambda x: np.asarray(x), carry[3]))
    obs_seq.append(pobs)
    carry, act, _ = agent.policy(carry, pobs, mode='eval')
    acts = {**act, 'reset': obs['is_last'].copy()}
  for e in env:
    e.close()
  return obs_seq, prev_seq


def replay(agent, obs_seq, prev_seq):
  """Same observations and previous actions through this checkpoint."""
  import jax
  B = len(obs_seq[0]['is_first'])
  carry = agent.init_policy(B)
  deter, stoch, code = [], [], []
  for obs, prev in zip(obs_seq, prev_seq):
    carry = list(carry)
    carry[3] = jax.tree.map(lambda x: x, prev)
    carry = tuple(carry)
    carry, _, outs = agent.policy(carry, obs, mode='eval')
    deter.append(np.asarray(outs['log/struct_diag_deter'], np.float32))
    stoch.append(np.asarray(outs['log/struct_diag_stoch']))
    code.append(np.asarray(outs['log/fromto_code']))
  # (T, B, ...) -> (B, T, ...)
  return [np.swapaxes(np.stack(x), 0, 1) for x in (deter, stoch, code)]


def keys(deter, stoch, code, proj_d, proj_s, classes):
  B, T = code.shape[:2]
  bits = 2 ** np.arange(BITS)
  d = deter.reshape(B * T, -1)
  d = d - d.mean(0, keepdims=True)
  s1h = np.eye(classes)[stoch.reshape(B * T, -1)].reshape(B * T, -1)
  s1h = s1h - s1h.mean(0, keepdims=True)
  out = {
      'code4': (np.minimum(code.reshape(B * T, -1) * 4 // classes, 3)
                * 4 ** np.arange(code.shape[-1])).sum(-1),
      'simhash_d': ((d @ proj_d) > 0) @ bits,
      'simhash_s': ((s1h @ proj_s) > 0) @ bits,
      'stochmode': ((stoch.reshape(B * T, -1)[:, :BITS] >= classes // 2) @ bits),
  }
  return {k: v.reshape(B, T) for k, v in out.items()}


def stats(k, pre=None):
  flat = k.reshape(-1)
  cnt = collections.Counter(flat.tolist())
  p = np.array(list(cnt.values()), np.float64) / len(flat)
  r = dict(cells=len(cnt), eff=float(np.exp(-(p * np.log(p)).sum())),
           coherence=float((k[:, 1:] == k[:, :-1]).mean()))
  if pre is not None:
    by = collections.defaultdict(collections.Counter)
    for kk, pp in zip(flat.tolist(), pre):
      by[kk][pp] += 1
    r['purity'] = sum(c.most_common(1)[0][1] for c in by.values()) / len(flat)
  return r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--envs', type=int, default=8)
  ap.add_argument('--steps', type=int, default=1500)
  ap.add_argument('--out', required=True)
  a = ap.parse_args()
  import jax
  run = pathlib.Path(a.run_dir)
  ms = sorted((run / 'logdir' / 'ckpt_milestones').iterdir())
  config, agent = build(str(run), ckpt_path=str(ms[-1]))
  print('rollout with', ms[-1].name, flush=True)
  obs_seq, prev_seq = rollout(config, agent, a.envs, a.steps)
  pinpad = 'pinpad' in str(config.task)
  pre = None
  if pinpad:
    imgs = np.swapaxes(np.stack([o['image'] for o in obs_seq]), 0, 1)
    pre = [prefix(im, 2) for im in imgs.reshape((-1,) + imgs.shape[2:])]
  classes = int(config.agent.dyn.rssm.classes)
  rng = np.random.default_rng(0)
  proj_d = proj_s = None
  per, raw = {}, {}
  for path in ms:
    _, agent = build(str(run), ckpt_path=str(path))
    deter, stoch, code = replay(agent, obs_seq, prev_seq)
    if proj_d is None:
      proj_d = rng.normal(size=(deter.shape[-1], BITS))
      proj_s = rng.normal(size=(stoch.shape[-1] * classes, BITS))
    per[path.name] = keys(deter, stoch, code, proj_d, proj_s, classes)
    raw[path.name] = (deter.astype(np.float16), stoch.astype(np.int8), code.astype(np.int8))
    for name, k in per[path.name].items():
      print(f'{int(path.name) / 1e6:.1f}M {name:10s}', stats(k, pre), flush=True)
  # Noise floor: the last checkpoint replayed again (stoch is sampled).
  deter, stoch, code = replay(agent, obs_seq, prev_seq)
  raw['repeat'] = (deter.astype(np.float16), stoch.astype(np.int8), code.astype(np.int8))
  names = list(per)
  for a0, a1 in zip(names[:-1], names[1:]):
    for name in per[a0]:
      k0, k1 = per[a0][name], per[a1][name]
      same = float((k0 == k1).mean())
      print(f'{int(a0) / 1e6:.1f}->{int(a1) / 1e6:.1f}M {name:10s} same cell {same:.3f}', flush=True)
  feats = {f'{n}/{c}': v for n, (d, s_, c_) in raw.items() for c, v in (('deter', d), ('stoch', s_), ('code', c_))}
  extra = {'is_first': np.swapaxes(np.stack([o['is_first'] for o in obs_seq]), 0, 1)}
  # Observation-space cells (Go-Explore style, no drift): 16x16 grey images.
  img = np.swapaxes(np.stack([o['image'] for o in obs_seq]), 0, 1).astype(np.float32)
  B_, T_, H_, W_ = img.shape[:4]
  grey = img.mean(-1).reshape(B_, T_, 16, H_ // 16, 16, W_ // 16).mean((3, 5))
  extra['img16'] = grey.astype(np.uint8)
  if pre is not None:
    extra['prefix2'] = np.array(['-'.join(p) for p in pre])
  np.savez_compressed(a.out, **feats, **extra)


if __name__ == '__main__':
  main()
