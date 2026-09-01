"""How much does the UCB bonus actually change the manager's goal choice?

The one number that decides whether ``mgr_ucb_c`` is set sensibly. Neither
failure mode is visible in the training scalars:

  * too weak  -- the arm silently becomes the no-count control and the runs
                 answer nothing;
  * too strong -- goal choice ignores the manager and is effectively random.

Reads the selection table straight out of a checkpoint, so it needs no GPU and
does not touch the running jobs. The manager's policy is modelled from its
measured entropy rather than evaluated: at the 50% target each block is one
dominant class plus a near-uniform tail, which is enough to reproduce the
log-probability spread that the bonus has to compete against.

    python tools/ucb_influence.py <run_dir> [--step 500000] [--c 20] [--K 8]
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np
from scipy.optimize import brentq

BLOCKS, CLASSES, BINS = 8, 8, 4


def load_select_tab(run_dir, step=None):
  """Pull ``code_counts/select_tab`` out of a milestone (or the live) ckpt."""
  base = pathlib.Path(run_dir) / 'logdir'
  if step == 'live':
    # The live checkpoint is rewritten by the trainer, so a read can race a
    # write. Fine for a spot check, never for a recorded result -- milestones
    # are immutable once written and are what any reported number should use.
    p = base / 'ckpt' / 'latest' / 'agent.pkl'
    if not p.exists():
      cand = sorted(d for d in (base / 'ckpt').glob('*/agent.pkl'))
      if not cand:
        raise SystemExit(f'no checkpoint under {base}/ckpt')
      p = cand[-1]
  elif step:
    p = base / 'ckpt_milestones' / f'{int(step):012d}' / 'agent.pkl'
  else:
    miles = sorted((base / 'ckpt_milestones').glob('*/agent.pkl'))
    if not miles:
      raise SystemExit(
          f'no milestone checkpoints under {base} yet (first lands at 500k) -- '
          f'pass --step live for a spot check against the running checkpoint')
    p = miles[-1]
  if not p.exists():
    raise SystemExit(f'missing {p}')
  with open(p, 'rb') as f:
    blob = pickle.load(f)
  # The agent pickle is a nested dict of param groups; find the table by name
  # rather than assuming a layout, which differs between versions.
  found = {}

  def walk(node, path=''):
    if isinstance(node, dict):
      for k, v in node.items():
        walk(v, f'{path}/{k}' if path else str(k))
    elif hasattr(node, 'shape') and 'select_tab' in path:
      found[path] = np.asarray(node)

  walk(blob)
  # Optimizer slots mirror some params (e.g. params/opt/.../goal_dec/codebook),
  # so match the real table exactly rather than by substring, or this silently
  # reports on the wrong array.
  exact = {k: v for k, v in found.items()
           if k.endswith('code_counts/select_tab/value') and '/opt/' not in k}
  if not exact:
    raise SystemExit(
        f'no code_counts/select_tab in {p} -- was the run launched with '
        f'mgr_ucb_c > 0? (the reward-only arm never fills it). '
        f'saw: {sorted(found)[:4]}')
  if len(exact) > 1:
    raise SystemExit(f'ambiguous select_tab entries: {sorted(exact)}')
  tab = list(exact.values())[0]
  if tab.shape != (BINS ** BLOCKS,):
    raise SystemExit(f'unexpected select_tab shape {tab.shape}, '
                     f'expected {(BINS ** BLOCKS,)} -- grid changed?')
  return tab, p


def block_probs(entropy_nats):
  """Per-block categorical matching the measured entropy: one dominant class."""
  h = entropy_nats / BLOCKS

  def ent(q):
    r = (1 - q) / (CLASSES - 1)
    return -q * np.log(q) - (CLASSES - 1) * r * np.log(r)

  q = brentq(lambda q: ent(q) - h, 1.0 / CLASSES + 1e-9, 1 - 1e-9)
  return np.array([q] + [(1 - q) / (CLASSES - 1)] * (CLASSES - 1))


def cell_of(codes):
  """Joint fine-grid cell index, matching CodeCounts._key."""
  q = np.minimum((codes * BINS) // CLASSES, BINS - 1)
  place = BINS ** np.arange(BLOCKS)
  return (q * place).sum(-1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--step', default=None,
                  help='milestone step, or "live" for the running checkpoint')
  ap.add_argument('--c', type=float, default=20.0)
  ap.add_argument('--K', type=int, default=8)
  ap.add_argument('--entropy', type=float, default=8.7,
                  help='manager entropy in nats (measured: 8.7 at the 50% target)')
  ap.add_argument('--trials', type=int, default=20000)
  a = ap.parse_args()

  step = a.step if a.step in (None, 'live') else int(a.step)
  tab, path = load_select_tab(a.run_dir, step)
  p = block_probs(a.entropy)
  rng = np.random.RandomState(0)

  # Candidate CELLS are drawn from the table itself, not from an i.i.d. block
  # model. The table is the empirical histogram of the manager's own samples, so
  # it already encodes where the real state-conditioned policy lands. Drawing
  # blocks independently instead spreads candidates far wider than the real
  # policy does: they fall in empty cells, every bonus reads ~1.0, and the flip
  # rate comes out insensitive to c (measured 19.1% at both c=1 and c=20, which
  # is the signature of this mistake rather than a property of the system).
  #
  # Log-probabilities still come from the entropy model, and are paired with the
  # cells independently. That is an approximation -- within one state the code a
  # sample lands on and its log-prob are correlated -- but it reproduces the
  # quantity that matters here, the spread the bonus must overcome.
  w = tab.astype(np.float64)
  if w.sum() <= 0:
    raise SystemExit('selection table is empty -- run has not trained yet')
  # Keep a tail of never-selected cells: the policy does occasionally propose
  # something new, and those are exactly the candidates the bonus should favour.
  unseen = (w <= 0).sum()
  tail = min(0.05, unseen / len(w))
  probs = (1 - tail) * (w / w.sum())
  probs[w <= 0] += tail / max(unseen, 1)
  cells = rng.choice(len(w), size=(a.trials, a.K), p=probs / probs.sum())
  draws = rng.choice(CLASSES, size=(a.trials, a.K, BLOCKS), p=p)
  logp = np.log(p)[draws].sum(-1)
  bonus = 1.0 / np.sqrt(tab[cells] + 1.0)

  base = logp.argmax(1)
  tilted = (logp + a.c * bonus).argmax(1)
  flip = (base != tilted).mean()

  srt = np.sort(logp, 1)
  gap = (srt[:, -1] - srt[:, -2]).mean()
  spread = (bonus.max(1) - bonus.min(1)).mean()

  occupied = int((tab > 1.0).sum())
  eff = tab.sum() ** 2 / max((tab ** 2).sum(), 1e-9)

  print(f'checkpoint      {path}')
  print(f'selection table mass={tab.sum():,.0f}  occupied={occupied:,} / {BINS**BLOCKS:,}'
        f'  effective={eff:,.0f}')
  print(f'manager         entropy={a.entropy} nats, top1-top2 logp gap={gap:.2f} nats')
  print(f'bonus           mean={bonus.mean():.3f}  spread across {a.K} candidates={spread:.3f}')
  print(f'tilt            c*spread={a.c * spread:.2f} nats  (vs {gap:.2f} nat gap)')
  print()
  print(f'>>> FLIP RATE   {100 * flip:.1f}%  of decisions changed by the bonus')
  verdict = ('INERT -- c far too low, this is the no-count control' if flip < 0.05 else
             'weak -- c on the low side' if flip < 0.20 else
             'balanced' if flip < 0.60 else
             'strong -- novelty usually wins' if flip < 0.85 else
             'DOMINATING -- the manager is being ignored, c far too high')
  print(f'>>> VERDICT     {verdict}')
  print()
  print('c that would give a ~40% flip rate:')
  for c in (1, 2, 5, 10, 20, 40, 80):
    f = (logp.argmax(1) != (logp + c * bonus).argmax(1)).mean()
    print(f'   c={c:3d} -> {100 * f:5.1f}%')


if __name__ == '__main__':
  sys.exit(main())
