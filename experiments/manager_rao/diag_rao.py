"""Rao's quadratic entropy of the manager policy, on a trained checkpoint.

Question: the manager's per-block categorical is held at ~0.5 normalized entropy
by ``manager_actent_target``, but entropy is blind to WHICH classes carry the
mass. On a SOM *line* the classes are ordered, so "0.5 entropy spread over three
adjacent classes" and "0.5 entropy split between the two ends" are very
different policies that the entropy controller cannot tell apart. Rao's
quadratic entropy

    Q(p) = sum_{a,b} p_a p_b d(a, b)

does tell them apart: it is the expected distance between two classes drawn
independently from p. With an ordered distance d it is large only when the mass
sits on classes that are FAR APART along the line.

Whole-code vs per-block. The manager emits L independent categoricals, so the
distribution over whole codes factorizes, p(z) = prod_l p_l(z_l). If the
distance between two whole codes is additive over blocks, d(z, z') =
sum_l d_l(z_l, z'_l), then

    Q(p) = E[d(z, z')] = sum_l E[d_l(z_l, z'_l)] = sum_l Q(p_l)

exactly. Whole-code Rao IS the sum of per-block Raos -- the C^L = 8^8 = 16.7M
squared pair sum never has to be formed. Everything below is therefore computed
per block (C x C = 64 pairs) and reported as the per-block mean, which is the
whole-code value up to the constant factor L.

Distance. Line topology, class a at position x_a = a / (C - 1) in [0, 1], and
d(a, b) = (x_a - x_b)^2. Then Q = 2 * Var(x), and the reachable range is

    Q = 0                      all mass on one class
    Q = 0.2143 (C=8)           uniform over the line
    Q = 0.5                    half the mass at each end (the maximum)

so all self-Rao numbers below are reported normalized by 0.5, i.e. 1.0 means
"half at each end". Note the maximum is a two-atom distribution whose
normalized entropy is log2/log8 = 0.333 -- BELOW the 0.5 entropy target -- so
the entropy controller already forbids the degenerate corner.

What is measured, at manager decision steps only (``mgr_diag_switch``):

  ent_norm       per-block normalized entropy, H / log C. Sanity check against
                 the run's own ``manager_actent_target`` (0.5).
  rao_self       Rao of the manager's own distribution at one decision. "Given
                 the entropy it is holding, how far apart are the classes it is
                 spreading over?"
  rao_ceiling    the largest Rao attainable at the entropy this policy actually
                 has, found by a small numerical maximization. rao_self /
                 rao_ceiling is the honest headroom: how much of the available
                 spread the manager is leaving on the table.
  rao_cross      Rao between CONSECUTIVE decisions, sum_{a,b} p_t(a) p_{t+1}(b)
                 d(a,b): how far the goal moves along the line per switch. This
                 is the ordered generalization of the ``goal_soft_reuse``
                 overlap term, which is exactly this quantity with the
                 order-blind distance d(a,b) = 1[a != b].
  overlap        that existing order-blind quantity, sum_c p_t(c) p_{t+1}(c),
                 for direct comparison.
  step_idx       mean |index change| between consecutive sampled codes, in
                 classes. Interpretable directly: "the manager moves this many
                 steps along an 8-class line per decision".
  rao_marginal   Rao of the POOLED distribution of sampled classes over the
                 whole rollout: does the manager use the whole line across
                 decisions even if each single decision is narrow?
  r_idx_goal     Pearson r between code index distance and decoded goal
                 distance over pairs of decisions. Tests the premise that with
                 SOM-line + LiP, index distance predicts goal distance.

Usage (see run_diag_rao.sbatch):
  python -u diag_rao.py --run_dir /work/.../e534_... --out results/e534.npz
"""
import argparse
import json
import pathlib
import sys

# Two levels below the repo root that holds the `dreamerv3` and `embodied`
# packages (neither is pip-installed -- see source_dreamerv3_env.sh).
root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from functools import partial as bind

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main as m


def load_config(run_dir, ckpt_path=None):
  """Rebuild the architecture exactly as trained, from the run's own dump.

  Same reasoning as experiments/goal_code_struct_corr/diag_goal_struct_corr.py:
  the saved config is a full resolved dump, so layering it onto today's
  defaults would silently keep differently-shaped current defaults and break
  checkpoint restore. Overrides go on the flat dict to bypass Config.update's
  "key must already exist" check, so a new diagnostic flag can be set on a run
  that predates it.
  """
  cfgpath = elements.Path(run_dir) / 'logdir' / 'config.yaml'
  if not cfgpath.exists():
    cfgpath = elements.Path(run_dir) / 'config.yaml'
  saved = yaml.YAML(typ='safe').load(cfgpath.read())
  flat = elements.Config(saved).flat
  flat.update({
      'logdir': str(elements.Path(run_dir) / 'logdir'),
      'random_agent': False,
      'agent.policy_mgr_diag': True,
      'agent.policy_struct_diag': False,
      'agent.policy_goal_image': False,
      'agent.report_mask_viz': False,
      'jax.platform': 'cuda',
      'jax.prealloc': False,
      'jax.precompile': False,
      'jax.policy_devices': [0],
      'jax.train_devices': [0],
  })
  return elements.Config(flat)


# --- Rao's quadratic entropy -------------------------------------------------

def line_distance(C):
  """Squared distance between classes on a unit-length open line."""
  x = np.arange(C, dtype=np.float64) / max(C - 1, 1)
  return (x[:, None] - x[None, :]) ** 2


def rao(p, D, q=None):
  """sum_{a,b} p_a q_b d(a,b), batched over leading dims. q defaults to p."""
  q = p if q is None else q
  return np.einsum('...a,ab,...b->...', p, D, q)


def rao_ceiling_at_entropy(h_norm, D, C, iters=4000, seed=0):
  """Max normalized Rao attainable at a given normalized entropy.

  There is no closed form, so this is a small projected search: start from the
  symmetric two-ends-plus-middle family (which is optimal by symmetry for a
  squared distance on a line) and bisect its free mass to hit the entropy,
  then polish with random perturbations that keep the entropy within tolerance.
  Only used to report headroom, so approximate is fine.
  """
  target = h_norm * np.log(C)

  def ent(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum()

  # Symmetric family: mass q at each end, the rest spread over the interior.
  def family(q):
    p = np.zeros(C)
    p[0] = p[-1] = q
    rest = max(1.0 - 2 * q, 0.0)
    if C > 2:
      p[1:-1] = rest / (C - 2)
    return p

  lo, hi = 1.0 / C, 0.5
  for _ in range(200):
    mid = 0.5 * (lo + hi)
    if ent(family(mid)) > target:
      lo = mid
    else:
      hi = mid
  best = family(0.5 * (lo + hi))
  best_q = rao(best, D)
  rng = np.random.default_rng(seed)
  for _ in range(iters):
    cand = np.clip(best + rng.normal(0, 0.02, C), 1e-12, None)
    cand /= cand.sum()
    if abs(ent(cand) - target) > 0.01:
      continue
    val = rao(cand, D)
    if val > best_q:
      best, best_q = cand, val
  return best_q


def summarize(probs, ids, goals, C, max_pairs=2000):
  """All Rao diagnostics for one run's collected decisions.

  probs: (N, L, C) manager distribution at each decision
  ids:   (N, L)    sampled class index in force after each decision
  goals: (N, G)    decoded goal deter vector for each decision
  """
  N, L, _ = probs.shape
  D = line_distance(C)
  qmax = D.max()  # 1.0 on the unit line; self-Rao maxes at qmax/2

  ent = -(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1))).sum(-1)
  ent_norm = ent / np.log(C)                                    # (N, L)

  # Both normalized by the SAME constant (qmax/2 = the self-Rao maximum), so
  # they are directly comparable: if consecutive decisions were independent
  # draws from one stationary distribution, cross would equal self. cross <
  # self means the manager holds its position on the line more than chance
  # (reuse); cross > self means it systematically jumps further than chance.
  self_q = rao(probs, D) / (qmax / 2)                           # (N, L)
  cross_q = rao(probs[:-1], D, probs[1:]) / (qmax / 2)          # (N-1, L)
  overlap = (probs[:-1] * probs[1:]).sum(-1)                    # (N-1, L)
  step_idx = np.abs(np.diff(ids.astype(np.float64), axis=0))    # (N-1, L)
  changed = (np.diff(ids, axis=0) != 0).astype(np.float64)      # (N-1, L)

  # Marginal over the whole rollout, per block, then Rao of that.
  marg = np.stack([
      np.bincount(ids[:, l].astype(int), minlength=C) / N for l in range(L)])
  marg_q = rao(marg, D) / (qmax / 2)                            # (L,)
  marg_ent = -(np.clip(marg, 1e-12, 1) * np.log(np.clip(marg, 1e-12, 1))
               ).sum(-1) / np.log(C)

  ceiling = rao_ceiling_at_entropy(
      float(ent_norm.mean()), D, C) / (qmax / 2)

  # Index distance vs decoded goal distance, over a subsample of decision pairs.
  rng = np.random.default_rng(0)
  k = min(max_pairs, N)
  sel = rng.choice(N, size=k, replace=False)
  # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, NOT the (k, k, dim) difference
  # tensor: at k=2000 and dim=1024 that broadcast is 32 GB and OOMs the node.
  def pdist(v):
    sq = (v * v).sum(-1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * (v @ v.T), 0.0))

  di = pdist(ids[sel].astype(np.float64) / max(C - 1, 1))       # (k, k)
  dg = pdist(goals[sel])                                        # (k, k)
  off = ~np.eye(k, dtype=bool)
  a, b = di[off], dg[off]
  ac, bc = a - a.mean(), b - b.mean()
  r_idx_goal = float((ac * bc).sum() / (
      np.sqrt((ac * ac).sum()) * np.sqrt((bc * bc).sum()) + 1e-12))

  return dict(
      n_decisions=int(N), L=int(L), C=int(C),
      ent_norm=float(ent_norm.mean()), ent_norm_std=float(ent_norm.std()),
      rao_self=float(self_q.mean()), rao_self_std=float(self_q.std()),
      rao_ceiling=float(ceiling),
      rao_headroom=float(self_q.mean() / max(ceiling, 1e-12)),
      rao_cross=float(cross_q.mean()),
      overlap=float(overlap.mean()),
      step_idx=float(step_idx.mean()),
      changed_frac=float(changed.mean()),
      rao_marginal=float(marg_q.mean()),
      ent_marginal=float(marg_ent.mean()),
      r_idx_goal=r_idx_goal,
      per_block_rao_self=self_q.mean(0).tolist(),
      per_block_ent=ent_norm.mean(0).tolist(),
      per_block_marginal=marg.tolist(),
  )


# --- Rollout -----------------------------------------------------------------

def run(run_dir, out_path, n_envs=8, n_decisions=4000, ckpt_path=None):
  config = load_config(run_dir)
  L, C = [int(x) for x in config.agent.skill_shape]
  K = max(1, int(config.agent.manager_sample_freq))
  print(f'run: {elements.Path(run_dir).name}')
  print(f'task={config.task} skill_shape=({L}, {C}) manager_sample_freq={K} '
        f'actent_target={config.agent.manager_actent_target}')

  # ``Driver(steps=)`` counts transitions pooled across workers, not per
  # worker, and one decision lands every K transitions, so the pooled budget is
  # n_decisions * K regardless of n_envs (which buys wall time, not samples).
  steps = n_decisions * K + n_envs * K
  print(f'driving {n_envs} envs for {steps} steps -> ~{n_decisions} decisions')

  agent = m.make_agent(config)
  cp = elements.Checkpoint(directory=elements.Path(config.logdir) / 'ckpt')
  cp.agent = agent
  if ckpt_path:
    cp.load(path=elements.Path(ckpt_path), keys=['agent'])
    print('loaded checkpoint from snapshot', ckpt_path)
  else:
    cp.load(keys=['agent'])
    print('loaded checkpoint from', elements.Path(config.logdir) / 'ckpt')

  env_fns = [bind(m.make_env, config, i) for i in range(n_envs)]
  driver = embodied.Driver(env_fns, parallel=(n_envs > 1))
  driver.reset(agent.init_policy)

  probs_by_env = [[] for _ in range(n_envs)]
  ids_by_env = [[] for _ in range(n_envs)]
  goals_by_env = [[] for _ in range(n_envs)]

  def collect(tran, worker):
    if 'log/mgr_diag_probs' not in tran:
      return
    # Decision steps only: on held steps the manager head still runs, but the
    # code in force did not change, so those samples are duplicates of the
    # decision that produced them.
    if float(np.asarray(tran['log/mgr_diag_switch'])) <= 0:
      return
    probs_by_env[worker].append(
        np.asarray(tran['log/mgr_diag_probs'], np.float64).copy())
    ids_by_env[worker].append(
        np.asarray(tran['log/mgr_diag_ids'], np.float64).copy())
    goals_by_env[worker].append(
        np.asarray(tran['log/mgr_diag_goal'], np.float64).copy())

  driver.on_step(collect)
  driver(agent.policy, steps=steps)
  driver.close()

  probs = np.concatenate(
      [np.stack(p, 0) for p in probs_by_env if p], 0).reshape(-1, L, C)
  ids = np.concatenate([np.stack(p, 0) for p in ids_by_env if p], 0)
  goals = np.concatenate([np.stack(p, 0) for p in goals_by_env if p], 0)
  print(f'collected {probs.shape[0]} decisions')

  res = summarize(probs, ids, goals, C)
  res['run'] = elements.Path(run_dir).name
  res['task'] = str(config.task)
  res['ckpt'] = str(ckpt_path or 'final')

  print()
  print(f'  entropy (normalized)        {res["ent_norm"]:.3f} '
        f'(target {float(config.agent.manager_actent_target):.2f})')
  print(f'  Rao self, normalized        {res["rao_self"]:.3f}')
  print(f'  Rao ceiling at that entropy {res["rao_ceiling"]:.3f}')
  print(f'  -> fraction of ceiling      {res["rao_headroom"]:.3f}')
  print(f'  Rao cross (consecutive)     {res["rao_cross"]:.3f}')
  print(f'  overlap (order-blind)       {res["overlap"]:.3f}')
  print(f'  mean |index step| (classes) {res["step_idx"]:.3f}')
  print(f'  blocks changed per decision {res["changed_frac"]:.3f}')
  print(f'  Rao of pooled marginal      {res["rao_marginal"]:.3f}')
  print(f'  r(index dist, goal dist)    {res["r_idx_goal"]:.3f}')

  if out_path:
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out), probs=probs.astype(np.float32),
        ids=ids.astype(np.int8), goals=goals.astype(np.float32),
        summary=json.dumps(res))
    out.with_suffix('.json').write_text(json.dumps(res, indent=2))
    print('wrote', out)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--out', default='')
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--n_decisions', type=int, default=4000)
  ap.add_argument('--ckpt_path', default='')
  args = ap.parse_args()
  run(args.run_dir, args.out, args.n_envs, args.n_decisions,
      args.ckpt_path or None)


if __name__ == '__main__':
  main()
