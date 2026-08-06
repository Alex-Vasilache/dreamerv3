"""Goal-code <-> goal-space geometry correlation on live Director rollouts.

Answers the paper's open question ("goals are generated in a latent space that
does not correlate nearby latent goal codes with nearby goals" -- introduction.tex,
Sec. "Efficiency: Goal Code Reuse"): for a trained checkpoint, does a small change
in the RSSM ``deter`` state (the space goals live in) correspond to a small change
in the goal-VAE's discrete code, and vice versa?

Method: drive the loaded policy live across ``n_envs`` parallel env workers
(``agent.policy_struct_diag: True``, see ``dreamerv3/agent.py`` policy() and
``configs.yaml``), which emits, at every real env step, (1) the current state's
``deter`` vector and (2) that state's soft goal-code (``goal_enc(deter)``
softmax probs) -- a pure forward pass through the already-trained encoder, no
gradients, no effect on the policy actually taken. Parallel workers both
speed up collection (one batched policy call yields ``n_envs`` samples instead
of one) and give samples that are trivially decorrelated across workers,
on top of a per-worker ``stride`` thinning consecutive same-trajectory steps.

The collected pool is split into ``n_batches`` disjoint batches of ``n_states``
each (default: batch_size x batch_length from the run's own training config --
the same cardinality the goal-VAE is actually trained on per gradient step).
For each batch we compute the exact diagnostic used inside training
(``dreamerv3/agent.py`` ``goal_struct_weight`` block): the pairwise
``cosine_max`` Gram matrix in deter-space vs. code-space, and their Pearson
correlation over off-diagonal pairs -- both for the soft (softmax) code and
the hard (argmax) code the manager actually samples via REINFORCE. We report
the mean +/- std of that correlation across the ``n_batches`` batches, i.e.
the batch-to-batch variability of the metric, not just a single point estimate.

Usage (needs a live env -- MUJOCO_GL=egl, GPU node; see run_diag_goal_struct_corr.sbatch
for the full production invocation across all three baselines):
  python -u diag_goal_struct_corr.py --run_dir /bucket/.../e124_..._dir \
      --n_batches 10 --out results/e124_hopper.npz
"""
import argparse
import pathlib
import sys

# This file lives at code/dreamerv3/experiments/goal_code_struct_corr/, two
# levels below the repo root that holds the `dreamerv3` and `embodied`
# packages (neither is pip-installed -- see source_dreamerv3_env.sh).
root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from functools import partial as bind

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main as m

# Fixed bin edges for the goal-space-similarity -> goal-code-similarity trend
# curve (shared with make_figure.py's axis range). Fixed, not per-checkpoint
# min/max, so that per-seed trend curves land on identical x positions and
# can be averaged (mean +/- std across seeds) point-for-point.
TREND_XMIN, TREND_XMAX, TREND_NBINS = -0.2, 1.05, 22
# Cap on states used for the trend-curve pairwise Gram matrix: this is an
# O(n^2) computation, so we subsample the pooled collection rather than using
# all n_batches * n_states states -- 4096 is already ~8M pairs, far more than
# needed to fill 22 bins smoothly, and keeps this step fast (<1s).
TREND_MAX_N = 4096


def load_config(run_dir):
  cfgpath = elements.Path(run_dir) / 'logdir' / 'config.yaml'
  if not cfgpath.exists():
    cfgpath = elements.Path(run_dir) / 'config.yaml'
  saved = yaml.YAML(typ='safe').load(cfgpath.read())
  # ``saved`` is a full resolved dump (every key, not a diff against defaults) --
  # rebuild the architecture exactly as trained, rather than layering onto
  # *today's* configs.yaml defaults. Schema drift since the run means some keys
  # would silently keep a different-shaped current default and break checkpoint
  # restore (seen in practice: 'con'/'dec' shape mismatch). Add overrides
  # directly on the flat dict (bypassing Config.update's strict "key must
  # already exist" check) so brand-new keys like our ``policy_struct_diag``
  # diagnostic flag can be added even though the run predates them.
  flat = elements.Config(saved).flat
  flat.update({
      'logdir': str(elements.Path(run_dir) / 'logdir'),
      'random_agent': False,
      'agent.policy_struct_diag': True,
      'agent.policy_goal_image': False,
      'agent.report_mask_viz': False,
      'jax.platform': 'cuda',
      'jax.prealloc': False,
      # We only ever call agent.policy() (small batch); skip precompiling the
      # full train-batch/report graphs, which OOM a single V100 for nothing.
      'jax.precompile': False,
      # Some baselines (e.g. e124) trained multi-GPU (train_devices: [0,1,2,3]);
      # this job only requests one GPU, so pin both device lists to it.
      'jax.policy_devices': [0],
      'jax.train_devices': [0],
  })
  return elements.Config(flat)


def pairwise_cosmax(x):
  norm = np.linalg.norm(x, axis=-1) + 1e-12
  dot = x @ x.T
  nm = np.maximum(norm[:, None], norm[None, :])
  return dot / (nm * nm)


def pearson_offdiag(a, b):
  n = a.shape[0]
  offdiag = ~np.eye(n, dtype=bool)
  af, bf = a[offdiag], b[offdiag]
  ac, bc = af - af.mean(), bf - bf.mean()
  return float((ac * bc).sum() / (
      np.sqrt((ac * ac).sum()) * np.sqrt((bc * bc).sum()) + 1e-12))


def fixed_bin_trend(sd, sh, xmin=TREND_XMIN, xmax=TREND_XMAX, nbins=TREND_NBINS):
  """Binned mean of goal-code similarity (sh) vs goal-space similarity (sd)
  on FIXED bin edges (not sd.min()/sd.max()), so curves from different
  seeds/checkpoints share x-positions and can be averaged directly."""
  edges = np.linspace(xmin, xmax, nbins + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])
  bin_idx = np.clip(np.digitize(sd, edges) - 1, 0, nbins - 1)
  means = np.full(nbins, np.nan)
  counts = np.zeros(nbins, dtype=np.int64)
  for b in range(nbins):
    m = bin_idx == b
    counts[b] = m.sum()
    if counts[b] >= 10:
      means[b] = sh[m].mean()
  return centers, means, counts


def batch_corr(deters, probs, L, C):
  """Both correlations for one batch of (deter, soft-code) pairs."""
  n = deters.shape[0]
  sd = pairwise_cosmax(deters)
  sz = pairwise_cosmax(probs)
  corr_soft = pearson_offdiag(sd, sz)
  ids = probs.reshape(n, L, C).argmax(-1)
  hamming = (ids[:, None, :] != ids[None, :, :]).mean(-1)
  sh = 1.0 - hamming
  corr_hard = pearson_offdiag(sd, sh)
  return corr_soft, corr_hard, sd, sz, sh


def run(run_dir, stride, out_path, seed, n_states=None, n_batches=10,
        n_envs=8, steps=None, ckpt_path=None, train_step=None):
  config = load_config(run_dir)
  print('logdir:', config.logdir, '| task:', config.task,
        '| skill_shape:', config.agent.skill_shape)
  # Default sample count = one typical training batch's worth of states
  # (batch_size sequences x batch_length steps), the same cardinality the
  # world model/goal-VAE is actually trained on per gradient step -- a
  # principled size rather than an arbitrary round number.
  if n_states is None:
    n_states = int(config.batch_size) * int(config.batch_length)
    print(f'n_states defaulted to batch_size({config.batch_size}) x '
          f'batch_length({config.batch_length}) = {n_states}')

  total_needed = n_batches * n_states
  # Per-worker raw env-steps so that, after thinning by ``stride`` within each
  # worker's own trajectory (real decorrelation -- workers are pooled only
  # *after* striding, so this isn't diluted by interleaving across workers),
  # the n_envs workers together yield >= total_needed usable states.
  per_env_usable = -(-total_needed // n_envs)  # ceil
  steps_per_env = stride * per_env_usable
  if steps is None:
    steps = n_envs * steps_per_env
  print(f'n_batches={n_batches} n_states/batch={n_states} -> '
        f'total_needed={total_needed} | n_envs={n_envs} stride={stride} '
        f'steps_per_env={steps_per_env} | total driver steps={steps}')

  agent = m.make_agent(config)
  ckptdir = elements.Path(config.logdir) / 'ckpt'
  cp = elements.Checkpoint(directory=ckptdir)
  cp.agent = agent
  if ckpt_path:
    # Explicit snapshot override -- used to read a mid-training checkpoint
    # copied aside by watch_and_diag_running.py before elements.Checkpoint's
    # keep=1 rotation deleted it (the live run's ckptdir/latest has since
    # moved on). Architecture/config is still read from run_dir's own
    # logdir/config.yaml above (unchanged across training), only the weights
    # come from this specific snapshot.
    cp.load(path=elements.Path(ckpt_path), keys=['agent'])
    print('Loaded checkpoint from snapshot', ckpt_path)
  else:
    cp.load(keys=['agent'])
    print('Loaded checkpoint from', ckptdir)

  # n_envs parallel workers: one batched policy call now yields n_envs
  # diagnostic samples per step instead of one, so wall-clock for the same
  # amount of data drops by roughly n_envs (subprocess workers, CPU-bound
  # mujoco stepping runs concurrently with the single GPU policy call).
  env_fns = [bind(m.make_env, config, i) for i in range(n_envs)]
  driver = embodied.Driver(env_fns, parallel=(n_envs > 1))
  driver.reset(agent.init_policy)

  deters_by_env = [[] for _ in range(n_envs)]
  probs_by_env = [[] for _ in range(n_envs)]

  def collect(tran, worker):
    if 'log/struct_diag_deter' in tran:
      deters_by_env[worker].append(np.asarray(tran['log/struct_diag_deter']).copy())
      probs_by_env[worker].append(np.asarray(tran['log/struct_diag_probs']).copy())

  driver.on_step(collect)
  driver(agent.policy, steps=steps)
  driver.close()

  deters = np.concatenate(
      [np.stack(d, 0)[::stride] for d in deters_by_env if d], axis=0)
  probs = np.concatenate(
      [np.stack(p, 0)[::stride] for p in probs_by_env if p], axis=0)
  N, D = deters.shape
  L, C = config.agent.skill_shape
  print(f'collected N={N} usable states across {n_envs} workers '
        f'(target {total_needed}), D={D}, L={L}, C={C}')
  if N < total_needed:
    print(f'WARNING: only collected N={N}, fewer than requested '
          f'total_needed={total_needed} -- increase --steps or lower --stride')

  rng = np.random.default_rng(seed)
  n_batches_actual = min(n_batches, N // n_states)
  perm = rng.permutation(N)[:n_batches_actual * n_states]
  batch_idx = perm.reshape(n_batches_actual, n_states)

  corr_soft_list, corr_hard_list = [], []
  sd_plot = sz_plot = sh_plot = None
  for b in range(n_batches_actual):
    idx = batch_idx[b]
    cs, ch, sd, sz, sh = batch_corr(deters[idx], probs[idx], L, C)
    corr_soft_list.append(cs)
    corr_hard_list.append(ch)
    if b == 0:
      sd_plot, sz_plot, sh_plot = sd, sz, sh  # first batch only, for the scatter figure

  corr_soft_arr = np.array(corr_soft_list)
  corr_hard_arr = np.array(corr_hard_list)
  print(f'\n=== {config.task}: {n_batches_actual} batches of N={n_states} ===')
  print(f'Pearson r(deter, SOFT code): mean={corr_soft_arr.mean():.4f} '
        f'std={corr_soft_arr.std():.4f}  [{", ".join(f"{v:.3f}" for v in corr_soft_arr)}]')
  print(f'Pearson r(deter, HARD code): mean={corr_hard_arr.mean():.4f} '
        f'std={corr_hard_arr.std():.4f}  [{", ".join(f"{v:.3f}" for v in corr_hard_arr)}]')

  # Fixed-bin trend curve (goal-code sim vs goal-space sim), on a pooled
  # subsample bigger than one batch for a smoother per-seed curve, using
  # GLOBAL fixed bin edges so this checkpoint's curve is directly averageable
  # with other seeds' curves at the figure stage (same sampling methodology
  # -- same seed, same subsample cap -- applied identically to every seed).
  trend_n = min(N, TREND_MAX_N)
  trend_idx = rng.permutation(N)[:trend_n]
  _, _, sd_trend, _, sh_trend = batch_corr(deters[trend_idx], probs[trend_idx], L, C)
  trend_x, trend_y, trend_counts = fixed_bin_trend(sd_trend, sh_trend)

  if out_path:
    iu = np.triu_indices(n_states, k=1)
    np.savez(
        out_path, task=str(config.task), N=n_states, n_batches=n_batches_actual,
        L=L, C=C,
        # step in training the *checkpoint* was taken at (approximate: the
        # metrics.jsonl step read by the caller at snapshot time), not when
        # this diagnostic was run -- -1 if unknown (final/only checkpoint).
        train_step=int(train_step) if train_step is not None else -1,
        corr_soft_mean=float(corr_soft_arr.mean()), corr_soft_std=float(corr_soft_arr.std()),
        corr_hard_mean=float(corr_hard_arr.mean()), corr_hard_std=float(corr_hard_arr.std()),
        corr_soft_per_batch=corr_soft_arr, corr_hard_per_batch=corr_hard_arr,
        # Backward-compatible single-batch fields, for the plotting script.
        corr_soft=float(corr_soft_arr.mean()), corr_hard=float(corr_hard_arr.mean()),
        sd_pairs=sd_plot[iu].astype(np.float32),
        sz_pairs=sz_plot[iu].astype(np.float32),
        sh_pairs=sh_plot[iu].astype(np.float32),
        trend_x=trend_x.astype(np.float32), trend_y=trend_y.astype(np.float32),
        trend_counts=trend_counts)
    print('Saved to', out_path)
  return dict(task=str(config.task), n_batches=n_batches_actual,
              corr_soft_mean=float(corr_soft_arr.mean()), corr_soft_std=float(corr_soft_arr.std()),
              corr_hard_mean=float(corr_hard_arr.mean()), corr_hard_std=float(corr_hard_arr.std()))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--steps', type=int, default=None,
                   help='total driver steps (across all workers); auto-computed if omitted')
  ap.add_argument('--stride', type=int, default=3)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default=None)
  ap.add_argument('--n_states', type=int, default=None,
                   help='states per batch; defaults to batch_size x batch_length from the run config')
  ap.add_argument('--n_batches', type=int, default=10)
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--ckpt_path', default=None,
                   help='explicit checkpoint folder to load instead of ckpt/latest -- '
                        'for reading a mid-training snapshot copied aside from a still-'
                        'running run (see watch_and_diag_running.py)')
  ap.add_argument('--train_step', type=int, default=None,
                   help='training step the --ckpt_path snapshot was taken at (recorded '
                        'in the output npz; informational only)')
  args = ap.parse_args()
  run(args.run_dir, args.stride, args.out, args.seed, args.n_states,
      args.n_batches, args.n_envs, args.steps, args.ckpt_path, args.train_step)


if __name__ == '__main__':
  main()
