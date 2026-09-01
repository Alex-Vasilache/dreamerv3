"""How far apart are consecutive class indices, inside and outside the range?

The grid draws every column the same width, which quietly suggests that stepping
a block from 3 to 4 moves the goal as far as stepping it from 6 to 7, or from -1
to -2. It does not. This measures the actual step sizes, in two spaces:

  codebook  ||e_{j+1} - e_j||     -- ours only; Director has no embedding
  goal      ||g(t+1) - g(t)||     -- both; g is goal_dec's deter output, the
                                     vector the worker is actually rewarded for
                                     reaching, so it is the common ground

Extrapolation uses a clamped segment (see render_core.class_weights): below 0 it
repeats the 0->1 step forever, above C-1 the (C-2)->(C-1) step. So the
extrapolated classes ARE evenly spaced among themselves -- but at the end
segment's size, which is generally not the typical interior step, and is
different on the two sides.

    python tools/goal_demo/spacing.py [task ...]
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)

import render_core as rc  # noqa: E402
import runs as runsmod  # noqa: E402

EXTRA = 4
BASE = 3       # the code the demo opens on
SEEDS = 8      # random base codes to average the goal-space steps over


def codebook_gaps(renderer):
  """``(L, C-1)`` distances between consecutive codebook entries."""
  table = renderer.codebook()
  if table is None:
    return None
  return np.linalg.norm(table[:, 1:] - table[:, :-1], axis=-1)


def goal_gaps(renderer, seed=0):
  """``(L, T-1)`` distance in deter space between consecutive class indices.

  Averaged over random settings of the other blocks, since the decoder is not
  additive and one block's step size depends a little on the rest of the code.
  """
  L, C = renderer.blocks, renderer.classes
  ts = np.arange(-EXTRA, C + EXTRA, dtype=np.float32)
  rng = np.random.RandomState(seed)
  bases = np.concatenate([
      np.full((1, L), BASE, np.float32),
      rng.randint(0, C, (SEEDS - 1, L)).astype(np.float32)])
  out = np.zeros((L, len(ts) - 1))
  for base in bases:
    for block in range(L):
      codes = np.repeat(base[None], len(ts), 0)
      codes[:, block] = ts
      _, goals = renderer.render(codes)
      out[block] += np.linalg.norm(np.diff(goals, axis=0), axis=-1)
  return out / len(bases), ts


def report(task):
  print(f'\n{runsmod.RUNS[task]["label"]}')
  for arm in runsmod.ARMS:
    r = rc.GoalRenderer(task, arm, compute_dtype='float32')
    C = r.classes
    print(f'  {runsmod.ARM_LABELS[arm]}  ({r.info["exp"]})')

    cb = codebook_gaps(r)
    if cb is None:
      print('    codebook step   : no embedding; one-hot inputs are all '
            f'sqrt(2) apart, extrapolated ones too')
    else:
      inner = cb.mean(0)
      print('    codebook step   : ' +
            ' '.join(f'{v:.2f}' for v in inner) +
            f'   (mean over blocks, class j->j+1)')
      print(f'    largest / smallest interior step: '
            f'{cb.max(1).mean() / cb.min(1).mean():.1f}x')
      # How much the fitted-line rule differs from just repeating the end
      # segment: the step it takes, and the direction it takes it in. Both
      # entries are (outward end segment, outward fitted direction, end gap).
      table = r.codebook()
      sides = [
          ('below 0', table[:, 0] - table[:, 1], -r.line_dirs, cb[:, 0]),
          (f'above {C - 1}', table[:, -1] - table[:, -2], r.line_dirs, cb[:, -1]),
      ]
      for side, seg, new_dir, end_gap in sides:
        unit = seg / (np.linalg.norm(seg, axis=-1, keepdims=True) + 1e-9)
        cos = np.clip((unit * new_dir).sum(-1), -1, 1)
        print(f'    vs the old end-segment rule, {side}: step '
              f'{r.line_steps.mean() / end_gap.mean():.2f}x longer, direction '
              f'{np.degrees(np.arccos(cos)).mean():.1f} deg away')

    gaps, ts = goal_gaps(r)
    mid = gaps[:, EXTRA:EXTRA + C - 1].mean()          # steps inside 0..C-1
    left = gaps[:, :EXTRA].mean()                      # steps below 0
    right = gaps[:, EXTRA + C - 1:].mean()             # steps above C-1
    lo = gaps[:, EXTRA:EXTRA + C - 1].min()
    hi = gaps[:, EXTRA:EXTRA + C - 1].max()
    print(f'    goal step       : inside {mid:.2f} '
          f'(smallest {lo:.2f}, largest {hi:.2f}, {hi / lo:.1f}x spread)')
    print(f'                      below 0 {left:.2f}   above {C - 1} '
          f'{right:.2f}   ({left / mid:.2f}x and {right / mid:.2f}x '
          'of the interior step)')
    # Within each side, extrapolated steps are constant in the decoder's INPUT
    # space -- the weights move along one fixed direction at a fixed rate -- so
    # any unevenness here is the decoder MLP being nonlinear.
    lvar = gaps[:, :EXTRA].std(1).mean() / (left + 1e-9)
    rvar = gaps[:, EXTRA + C - 1:].std(1).mean() / (right + 1e-9)
    print(f'                      evenness of the extrapolated steps: '
          f'{lvar:.1%} / {rvar:.1%} relative spread (0% = perfectly even)')

    # Goal movement per unit of input movement, inside vs outside. This
    # separates "the end segments are short" from "the decoder flattens out
    # there": the first is already divided out here, so a drop is the second.
    if cb is None:
      step_in = np.full((r.blocks, C - 1), np.sqrt(2.0))
      step_lo = np.full((r.blocks, EXTRA), np.sqrt(2.0))
      step_hi = np.full((r.blocks, EXTRA), np.sqrt(2.0))
    else:
      # Extrapolation now moves one MEAN segment per class along the fitted
      # line, so that -- not either end gap -- is the input step out there.
      step_in = cb
      step_lo = np.repeat(r.line_steps[:, None], EXTRA, 1)
      step_hi = step_lo
    gain_in = (gaps[:, EXTRA:EXTRA + C - 1] / step_in).mean()
    gain_lo = (gaps[:, :EXTRA] / step_lo).mean()
    gain_hi = (gaps[:, EXTRA + C - 1:] / step_hi).mean()
    print(f'    decoder gain    : inside {gain_in:.2f} goal per unit input, '
          f'below 0 {gain_lo:.2f} ({gain_lo / gain_in:.2f}x), '
          f'above {C - 1} {gain_hi:.2f} ({gain_hi / gain_in:.2f}x)')


def main():
  tasks = sys.argv[1:] or runsmod.TASKS
  for task in tasks:
    report(task)
  print()


if __name__ == '__main__':
  main()
