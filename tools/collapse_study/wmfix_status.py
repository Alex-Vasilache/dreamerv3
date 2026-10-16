"""World-model drift-fix study status: score and key metrics per 100k-step bin.

Usage: python tools/collapse_study/wmfix_status.py [exp ...]   (default e1068-e1076)
"""
import glob, json, os, sys
import numpy as np

W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
BW = 1e5
KEYS = [('train/wkr_goal_rew', 'goal'), ('train/mgr_ent_norm_skill_mean', 'mgr_ent')]


def find(e):
  for root in (W, B):
    ds = glob.glob(f'{root}/{e}_*')
    if ds:
      return ds[0]


def bins(pairs):
  a = np.array(pairs, float)
  if not len(a):
    return []
  idx = (a[:, 0] // BW).astype(int)
  return [a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(idx.max() + 1)]


def main(exps):
  for e in exps:
    d = find(e)
    if not d:
      print(e, 'not found')
      continue
    arm = os.path.basename(d).split('director_og-')[1].split('_s')[0]
    S = [json.loads(l) for l in open(f'{d}/logdir/scores.jsonl')] if os.path.exists(f'{d}/logdir/scores.jsonl') else []
    M = [json.loads(l) for l in open(f'{d}/logdir/metrics.jsonl')]
    print(f'{e} {arm:20s} step {M[-1]["step"]:>8}  episodes {len(S)}')
    print('   score  ' + ' '.join(f'{v:5.0f}' for v in bins([(x['step'], x['episode/score']) for x in S])))
    for k, n in KEYS:
      v = bins([(x['step'], x[k]) for x in M if k in x])
      print(f'   {n:7s}' + ' '.join(f'{x:5.2f}' for x in v))


if __name__ == '__main__':
  main(sys.argv[1:] or [f'e{i}' for i in range(1068, 1077)])
