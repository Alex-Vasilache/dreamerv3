"""now vs now+v3 on the matched protocol: the base choice for the goal-AE arms.

  now      director director_og pubmatch      e1230-e1289 (Director rows)
  now+v3   director mgr_expl_perc002 pubmatch e1300-e1329

Per task and seed: late = mean episode score over 3-4M; auc = mean of the
100k-binned curve over 0-4M; drop = best 500k-window mean minus late (a
collapse shows up as a large drop). Group difference: Welch t and an exact
two-sided permutation p (all C(n1+n2, n1) relabelings; 252 for 5 v 5).
Reads /work first, then the bucket, so partial results work. Writes
collapse_figs/now_vs_v3.{png,pdf} in the house style.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/compare_v3.py
"""
import glob, itertools, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
OUT = f'{W}/collapse_figs/now_vs_v3.png'
TASKS = [('pinpad_four', 'Pin Pad Four'), ('pinpad_five', 'Pin Pad Five'), ('pinpad_six', 'Pin Pad Six'),
         ('dmc_cartpole_swingup', 'Cartpole Swingup'), ('dmc_cheetah_run', 'Cheetah Run'), ('dmc_hopper_hop', 'Hopper Hop')]
ARMS = [('now', 'Director, ours (now)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('v3', 'Director, ours (now+v3)', '#009E73', 'e13[0-2]?_{t}_director-mgr_expl_perc002-pubmatch_s*')]
BW, XMAX, FULL = 1e5, 4e6, 3.98e6


def runs(pat):
  got = {}
  for root in (B, W):   # /work wins for a run still training
    for d in glob.glob(f'{root}/{pat}'):
      got[os.path.basename(d)] = d
  return sorted(got.values())


def scores(d):
  f = f'{d}/logdir/scores.jsonl'
  if not os.path.exists(f):
    return np.zeros((0, 2))
  rows = []
  for line in open(f):
    try:
      r = json.loads(line)
    except ValueError:
      continue
    if r.get('episode/score') is not None:
      rows.append((r['step'], r['episode/score']))
  return np.array(rows) if rows else np.zeros((0, 2))


def binned(a):
  edges = np.arange(0, XMAX + BW, BW)
  idx = np.digitize(a[:, 0], edges) - 1
  return np.array([a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)])


def welch(x, y):
  x, y = np.asarray(x, float), np.asarray(y, float)
  vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
  if vx + vy == 0:
    return 0.0, 1.0
  t = (y.mean() - x.mean()) / math.sqrt(vx + vy)
  df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
  # two-sided p from the t distribution by numerical integration of the pdf
  c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
  s = np.linspace(abs(t), abs(t) + 200, 200001)
  tail = np.trapz(c * (1 + s ** 2 / df) ** (-(df + 1) / 2), s)
  return t, min(1.0, 2 * tail)


def perm_p(x, y):
  allv = np.array(list(x) + list(y), float); n = len(x)
  obs = abs(np.mean(y) - np.mean(x)); hits = tot = 0
  for idx in itertools.combinations(range(len(allv)), n):
    m = np.zeros(len(allv), bool); m[list(idx)] = True
    hits += abs(allv[~m].mean() - allv[m].mean()) >= obs - 1e-9; tot += 1
  return hits / tot


fig, axes = plt.subplots(2, 3, figsize=(13, 8.2)); axes = axes.ravel()
x = (np.arange(0, XMAX, BW) + BW / 2) / 1e6
print(f'{"task":17s} {"metric":5s} | {"now":>14s} | {"now+v3":>14s} | {"diff":>7s}  t      p_welch p_perm  n')
summary = []
for ax, (task, title) in zip(axes, TASKS):
  stats = {}
  for key, label, color, pat in ARMS:
    curves, late, auc, drop, done = [], [], [], [], 0
    for d in runs(pat.format(t=task)):
      a = scores(d)
      if not len(a):
        continue
      y = binned(a); curves.append(y)
      if a[:, 0].max() >= FULL:
        done += 1
        late.append(a[(a[:, 0] >= 3e6), 1].mean())
        auc.append(np.nanmean(y))
        sm = np.convolve(np.nan_to_num(y), np.ones(5) / 5, 'valid')
        drop.append(max(0.0, sm.max() - late[-1]))
    stats[key] = dict(late=late, auc=auc, drop=drop)
    if curves:
      st = np.stack(curves)
      with np.errstate(all='ignore'):
        n_ok = (~np.isnan(st)).sum(0)
        m = np.where(n_ok >= 2, np.nanmean(st, 0), np.nan)
        sd = np.where(n_ok >= 2, np.nanstd(st, 0, ddof=1), np.nan)
      ax.fill_between(x, m - sd, m + sd, color=color, lw=0, alpha=0.15)
      ax.plot(x, m, color=color, lw=2.4, label=f'{label}')
  for metric in ('late', 'auc', 'drop'):
    a, b = stats['now'][metric], stats['v3'][metric]
    if len(a) >= 2 and len(b) >= 2:
      t, pw = welch(a, b); pp = perm_p(a, b)
      print(f'{task:17s} {metric:5s} | {np.mean(a):6.0f} ± {np.std(a, ddof=1):5.0f} | {np.mean(b):6.0f} ± {np.std(b, ddof=1):5.0f} | {np.mean(b) - np.mean(a):+7.0f}  {t:+5.2f}  {pw:6.3f}  {pp:6.3f}  {len(a)}v{len(b)}')
      if metric == 'late':
        summary.append((task, np.mean(a), np.mean(b), pw, pp))
    else:
      print(f'{task:17s} {metric:5s} | n={len(a)} v n={len(b)} finished seeds: not enough yet')
  ax.set_title(title, fontsize=17); ax.set_xlim(0, 4); ax.set_ylim(bottom=0)
  ax.tick_params(labelsize=13, direction='out', length=3)
  ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
  for sp in ax.spines.values():
    sp.set_color('#333333'); sp.set_linewidth(0.8)
for ax in (axes[0], axes[3]):
  ax.set_ylabel('episode return', fontsize=15)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.075, 'environment steps (M)', ha='center', fontsize=15)
fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=2.0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor='white'); fig.savefig(OUT.replace('.png', '.pdf'))
if summary:
  # task-normalized late score: divide by the larger of the two arm means per task
  rel = [(b - a) / max(a, b, 1e-9) for _, a, b, _, _ in summary]
  print('mean normalized late difference (v3 - now)/max: %+.3f over %d tasks' % (np.mean(rel), len(rel)))
print(OUT)
