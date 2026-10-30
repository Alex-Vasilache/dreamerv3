"""TCE (e6200, half-life 0.5M, adaptive w) vs TCE-anneal (e7000, no decay, w=2): exploration metrics over training.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/tce_vs_anneal_diag.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, scores  # noqa: E402

PATS = {'TCE': 'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*',
        'anneal': 'e70[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-fromto_anneal-pubmatch_s*'}
KEYS = ['fromto/weight', 'fromto/novelty', 'fromto/novelty_uniform', 'fromto/lean', 'fromto/mass',
        'fromto/row_mass_median', 'mgr_ent_norm_skill_mean', 'mgr_actent_skill_scale_mean', 'mgr_adv_mag', 'mgr_extr_ret', 'wkr_goal_ret_rate', 'goal/entropy_mean']
PTS = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.75]


def load(d):
  rows = []
  for line in open(f'{d}/logdir/metrics.jsonl'):
    if 'fromto/weight' in line:
      r = json.loads(line)
      rows.append([r['step']] + [r.get(('train/' + k if '/' not in k or k.startswith('fromto') else 'train/' + k), np.nan) for k in KEYS])
  return np.array(rows)


def at(a, col, p, bw=1.25e5):
  m = np.abs(a[:, 0] - p * 1e6) < bw
  return a[m, col].mean() if m.any() else np.nan


tasks = sys.argv[1:] or [t for t, *_ in TASKS]
for t in tasks:
  print(f'\n=== {t}   (columns: steps {PTS} M)')
  for arm, p in PATS.items():
    ds = runs(p.format(t=t))
    data = [(os.path.basename(d)[:5], load(d), scores(d)) for d in ds]
    for name, a, s in data:
      sc = [np.mean(s[np.abs(s[:, 0] - q * 1e6) < 1.25e5, 1]) if len(s) else np.nan for q in PTS]
      print(f'{arm:7s} {name} score      ' + ' '.join(f'{v:7.1f}' for v in sc))
    for j, k in enumerate(KEYS, 1):
      v = np.array([[at(a, j, q) for q in PTS] for _, a, _ in data])
      with np.errstate(all='ignore'):
        m = np.nanmean(v, 0)
      print(f'{arm:7s} mean  {k[:22]:22s} ' + ' '.join(f'{x:7.3g}' for x in m))
