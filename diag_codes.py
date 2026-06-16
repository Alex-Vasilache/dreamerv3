"""Offline goal-code diagnostics on a trained checkpoint.

Loads the agent + checkpoint from a run dir, pulls a few report batches from the
existing replay, and runs Agent._code_diag (via report) to answer:

  (1) Are the goal codes "very distinct" along trajectories? -> per-block change
      rate between consecutive steps and between manager commands (K apart).
  (2) Does a partial block-overwrite stay on the code manifold? -> round-trip
      block mismatch and goal drift as we overwrite k random blocks with blocks
      from another real state.

Usage (inside an sbatch with the GPU env):
  python -u diag_codes.py --run_dir /work/.../<run> [--batches 8]
"""
import argparse
import os
import pathlib
import sys

folder = pathlib.Path(__file__).parent / 'dreamerv3'
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main as m


def load_config(run_dir):
  cfgpath = elements.Path(run_dir) / 'logdir' / 'config.yaml'
  if not cfgpath.exists():
    cfgpath = elements.Path(run_dir) / 'config.yaml'
  saved = yaml.YAML(typ='safe').load(cfgpath.read())
  # Start from the *current* defaults (so new keys like report_code_diag exist),
  # then overlay the run's resolved values to match the checkpoint's architecture.
  defaults = yaml.YAML(typ='safe').load(
      (folder / 'configs.yaml').read_text())['defaults']
  config = elements.Config(defaults).update(saved)
  # Point at the existing run; enable the diagnostic; silence the heavy report panels.
  config = config.update({
      'logdir': str(elements.Path(run_dir) / 'logdir'),
      'random_agent': False,
      'agent.report_code_diag': True,
      'agent.report_gradnorms': False,
      'agent.report_mask_viz': False,
      'agent.report_skill_viz': False,
      'agent.report_goal_enc_viz': False,
      'agent.report_vec_viz': False,
      'agent.report_mgr_recon': False,
  })
  return config


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--batches', type=int, default=8)
  ap.add_argument('--max_items', type=int, default=300000)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  config = load_config(args.run_dir)
  print('logdir:', config.logdir)
  print('use_masked_goals:', config.agent.use_masked_goals,
        '| skill_shape:', config.agent.skill_shape,
        '| manager_sample_freq:', config.agent.manager_sample_freq)

  agent = m.make_agent(config)

  # Load only the agent weights from the latest checkpoint.
  ckptdir = elements.Path(config.logdir) / 'ckpt'
  cp = elements.Checkpoint(directory=ckptdir)
  cp.agent = agent
  cp.load(keys=['agent'])
  print('Loaded checkpoint from', ckptdir)

  replay = m.make_replay(config, 'replay')
  replay.load(amount=args.max_items)
  print('Replay size:', len(replay))

  stream = iter(agent.stream(m.make_stream(config, replay, 'report')))
  carry = agent.init_report(config.batch_size)

  real_ids = []
  rt_real, mism, drift, K = [], [], [], None
  for i in range(args.batches):
    carry, mets = agent.report(carry, next(stream))
    real_ids.append(np.asarray(mets['codediag/real_ids']))      # (B,T,L)
    rt_real.append(float(mets['codediag/rt_mismatch_real']))
    mism.append(np.asarray(mets['codediag/rt_mismatch_by_k']))  # (L,)
    drift.append(np.asarray(mets['codediag/goal_drift_by_k']))  # (L,)
    K = int(mets['codediag/manager_freq'])
    print(f'  batch {i+1}/{args.batches} done')

  ids = np.concatenate(real_ids, 0)        # (B*, T, L)
  L = ids.shape[-1]
  rt_real = float(np.mean(rt_real))
  mism = np.mean(np.stack(mism), 0)        # (L,)
  drift = np.mean(np.stack(drift), 0)      # (L,)

  # --- Diagnostic 1: code volatility along real trajectories ---
  consec = (ids[:, 1:] != ids[:, :-1]).mean()                 # per-block flip rate t->t+1
  consec_perblock = (ids[:, 1:] != ids[:, :-1]).mean((0, 1))  # (L,)
  if ids.shape[1] > K:
    kstep = (ids[:, K:] != ids[:, :-K]).mean()                # t -> t+K (one manager cmd)
  else:
    kstep = float('nan')
  # how many distinct codes actually occur (out of C^L) vs how many states
  flatids = ids.reshape(-1, L)
  uniq = len(np.unique(flatids, axis=0))

  print('\n================ DIAG 1: code volatility ================')
  print(f'samples (states): {flatids.shape[0]}   blocks L={L}   manager K={K}')
  print(f'distinct codes observed: {uniq}  ({100*uniq/flatids.shape[0]:.1f}% of states)')
  print(f'per-block change rate  t -> t+1   : {consec:.3f}'
        f'   (~{consec*L:.2f} of {L} blocks change each step)')
  print(f'per-block change rate  t -> t+{K} : {kstep:.3f}'
        f'   (~{kstep*L:.2f} of {L} blocks change per manager command)')
  print('per-block flip rate by block:',
        ' '.join(f'{x:.2f}' for x in consec_perblock))

  print('\n================ DIAG 2: on-manifold splice test ================')
  print(f'baseline round-trip mismatch on REAL codes (k=0): {rt_real:.3f}'
        '   (encode->decode->encode self-consistency; ~0 = clean AE)')
  print(' k = blocks overwritten from another real state')
  print(f'{"k":>3} | {"rt_mismatch":>11} | {"goal_drift":>10}')
  print('-' * 32)
  for k in range(1, L + 1):
    print(f'{k:>3} | {mism[k-1]:>11.3f} | {drift[k-1]:>10.3f}')
  print('\nReading: if rt_mismatch stays ~baseline as k grows, partial overwrites '
        'are on-manifold (sparse edits viable). If it jumps with small k, splices '
        'are off-manifold -> the manager must overwrite ~all blocks. goal_drift '
        'shows how much a k-block edit actually moves the decoded goal.')

  if args.out:
    np.savez(args.out, ids=ids, rt_real=rt_real, mism=mism, drift=drift,
             consec_perblock=consec_perblock, K=K)
    print('\nSaved arrays to', args.out)


if __name__ == '__main__':
  main()
