"""Snapshots of the UCB-in-imagination count table (ImagCounts) from milestone
checkpoints, reduced to what the fill-up figure needs (the raw table is 16.7M
floats per checkpoint). Run with dreamerv3_env (checkpoints are numpy-2 pickles):

  python tools/collapse_study/ucb_fill_extract.py <run_dir> [<run_dir> ...]

Writes /work/.../collapse_figs/ucb_fill/<exp>.npz with, per milestone:
  bonus_map   (512, 512) mean bonus 1/sqrt(n+1) of the 8^8 goal grid laid out as
              blocks 0-3 (rows) x blocks 4-7 (cols), each lexicographic, 8x8-pooled
              (arm A; for arm B the goal grid after summing over start states)
  pair01      (G, G) share of count mass over blocks 0 and 1 (others summed out)
  frac_hi     fraction of cells with bonus > 0.5 (n < 3), > 0.25 (n < 15)
  mass        total decayed count mass
  bands       share of cells well tried (bonus < 0.25) / partly / near untried (> 0.5)
  block_marg  (8, G) per block, share of count mass on each class
"""
import glob, os, pickle, sys
import numpy as np

OUT = '/work/DoyaU/vasilache/work/collapse_figs/ucb_fill'


def goal_table(tab, cfg_goal_bins, state_bins):
  if state_bins:
    t = tab.reshape((state_bins ** 8, cfg_goal_bins ** 8)).sum(0)
  else:
    t = tab
  return t.reshape((cfg_goal_bins,) * 8)


for run in sys.argv[1:]:
  exp = os.path.basename(run).split('_')[0]
  b = 'ucb_imag_b' in run
  gb, sb = (4, 2) if b else (8, 0)
  res = {}
  for ck in sorted(glob.glob(os.path.join(run, 'logdir', 'ckpt_milestones', '0*'))):
    step = int(os.path.basename(ck))
    tab = np.asarray(pickle.load(open(os.path.join(ck, 'agent.pkl'), 'rb'))
                     ['params']['imag_counts/tab/value'], np.float64)
    g = goal_table(tab, gb, sb)
    bonus = 1.0 / np.sqrt(g + 1.0)
    side = gb ** 4
    m = bonus.reshape(side, side)
    f = max(1, side // 512)
    m = m.reshape(side // f, f, side // f, f).mean((1, 3))
    pair = g.sum(axis=(2, 3, 4, 5, 6, 7)); pair = pair / pair.sum()
    full = 1.0 / np.sqrt(tab + 1.0)
    res[f'{step}_bonus_map'] = m.astype(np.float32)
    res[f'{step}_pair01'] = pair.astype(np.float32)
    res[f'{step}_frac_hi'] = np.array([(full > 0.5).mean(), (full > 0.25).mean()])
    res[f'{step}_mass'] = np.array(tab.sum())
    # How much has been tried: share of cells by bonus band (well tried / partly
    # tried / near untried), over the full table (state x goal for arm B).
    res[f'{step}_bands'] = np.array([(full < 0.25).mean(),
                                     ((full >= 0.25) & (full <= 0.5)).mean(),
                                     (full > 0.5).mean()])
    # Where the tries go: per block, share of count mass on each class
    # (the other seven blocks summed out). Rows = blocks, cols = classes; for
    # arm B the goal grid has 4 bins, reported per bin.
    res[f'{step}_block_marg'] = np.stack([
        g.sum(axis=tuple(a for a in range(8) if a != b)) / g.sum()
        for b in range(8)]).astype(np.float32)
    print(exp, step, 'mass %.3g' % tab.sum(), 'frac>0.5 %.3f >0.25 %.3f' % tuple(res[f'{step}_frac_hi']))
  os.makedirs(OUT, exist_ok=True)
  np.savez_compressed(os.path.join(OUT, f'{exp}.npz'), **res)
