"""Summary table of every TCE variant run 2026-10-28..11-03 (finished runs only).

Per arm and task: mean +- sd of each seed's 3-4M mean (runs that reached 3.98M),
pin pad six seeds solved (sustained-50 crossing), normalised overall score, and
the stratified overall difference vs TCE (V+US-SOM+TCE, e6200) with a
permutation p over the tasks both arms have.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/tce_weekend_summary.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, scores, cross50, OUT  # noqa: E402
from compare_arms import MAXRET, overall  # noqa: E402

B = 'director-director_og-wkr_ent_fixed-wkr_v3'
ARMS = [
    ('OG', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
    ('V + SOM', f'e20[0-5]?_{{t}}_{B}-goal_som_line-pubmatch_s*'),
    ('V + US-SOM', f'e20[0-5]?_{{t}}_{B}-goal_som_line-goal_even-pubmatch_s*'),
    ('TCE (US-SOM, 0.5M half-life, adaptive w)', f'e62[0-2]?_{{t}}_{B}-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
    ('TCE-anneal (no decay, w=2)', f'e70[0-2]?_{{t}}_{B}-goal_som_line-goal_even-fromto_dir-fromto_anneal-pubmatch_s*'),
    ('TCE-grow (h=t/2, w=2)', f'e80[0-2]?_{{t}}_{B}-goal_som_line-goal_even-fromto_dir-tce_grow-pubmatch_s*'),
    ('TCE on plain SOM', f'e86[0-2]?_{{t}}_{B}-goal_som_line-fromto_dir-pubmatch_s*'),
    ('TCE, world-model hash start', f'e95[0-2]?_{{t}}_{B}-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
    ('TCE, 4x12-bit world-model hash, w=2', f'e1[09]0[0-3]?_{{t}}_{B}-goal_som_line-goal_even-fromto_dir-tce_hash4-pubmatch_s*'),
    ('TCE w=2, Director expl reward off', f'e96[3-5]?_{{t}}_{B}-goal_som_line-goal_even-fromto_dir-no_gae_expl-tce_wfix2-pubmatch_s*'),
]
REF = 'TCE (US-SOM, 0.5M half-life, adaptive w)'


def late(a):
  m = (a[:, 0] >= 3e6) & (a[:, 0] < 4e6)
  return a[m, 1].mean() if m.sum() > 3 else np.nan


def main():
  vals, six = {}, {}
  for name, pat in ARMS:
    vals[name] = {}
    for t, _, _ in TASKS:
      seeds = [scores(d) for d in runs(pat.format(t=t))]
      seeds = [a for a in seeds if len(a) and a[:, 0].max() >= 3.98e6]
      v = [late(a) for a in seeds]
      vals[name][t] = [x for x in v if not np.isnan(x)]
      if t == 'pinpad_six':
        six[name] = (sum(cross50(a) is not None for a in seeds), len(seeds))
  titles = {t: title for t, title, _ in TASKS}
  head = '| arm | ' + ' | '.join(titles[t] for t, *_ in TASKS) + ' | pin pad six solved | overall | vs TCE |'
  lines = [head, '|' + '---|' * (len(TASKS) + 4)]
  for name, _ in ARMS:
    cells = []
    for t, *_ in TASKS:
      v = vals[name][t]
      cells.append(f'{np.mean(v):.0f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.0f} (n={len(v)})' if v else '-')
    have = [t for t, *_ in TASKS if vals[name][t]]
    ov = np.mean([np.mean(vals[name][t]) / MAXRET[t] for t in have]) if have else np.nan
    ov_s = f'{100 * ov:.0f}%' + ('' if len(have) == len(TASKS) else f' ({len(have)} tasks)')
    common = [t for t in have if vals[REF][t] and len(vals[name][t]) > 1]
    if name == REF or not common:
      vs = '-'
    else:
      d, ci, p = overall({t: vals[REF][t] for t in common}, {t: vals[name][t] for t in common}, n_perm=50000)
      vs = f'{100 * d:+.1f} pts (p={p:.2f})' + ('' if len(common) == len(TASKS) else f', {len(common)} tasks')
    s6 = f'{six[name][0]}/{six[name][1]}'
    lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {s6} | {ov_s} | {vs} |')
  table = '\n'.join(lines)
  out = f'{OUT}/tce_weekend_summary.md'
  open(out, 'w').write('Late 3-4M mean per seed (finished runs only), mean ± sd over seeds; overall = mean over tasks of % of max achievable; vs TCE = stratified difference in overall points, permutation p.\n\n' + table + '\n')
  print(out); print(table)


if __name__ == '__main__':
  main()
