"""now against other arms on the matched protocol, figure + markdown table.

  SET=v3   now vs now+v3                   -> now_vs_v3.{png,pdf}
  SET=gae  now vs SOM / LiP / SOM+LiP      -> now_vs_goal_ae.{png,pdf}

Per task and seed: late = mean raw episode score over 3-4M; auc = mean of the
100k-binned curve over 0-4M. Difference vs now: Welch t p and exact two-sided
permutation p (252 relabelings for 5 v 5). Same statistics as
code/dreamerv3/tools/collapse_study/compare_v3.py.
"""
import glob, itertools, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
TASKS = [('pinpad_four', 'Pin Pad Four'), ('pinpad_five', 'Pin Pad Five'), ('pinpad_six', 'Pin Pad Six'),
         ('dmc_cartpole_swingup', 'Cartpole Swingup'), ('dmc_cheetah_run', 'Cheetah Run'), ('dmc_hopper_hop', 'Hopper Hop')]
NOW = ('now', 'Director, ours (now)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*')
SETS = {
    'v3': ('now_vs_v3', [
        ('now+v3', 'Director, ours (now+v3)', '#009E73', 'e13[0-2]?_{t}_director-mgr_expl_perc002-pubmatch_s*')]),
    'gae': ('now_vs_goal_ae', [
        ('SOM', 'now + SOM', '#0072B2', 'e1[34]??_{t}_director-director_og-goal_som_line-pubmatch_s*'),
        ('LiP', 'now + LiP', '#009E73', 'e1[34]??_{t}_director-director_og-goal_lipvq_prod-lip_relu-goal_vq_f32-pubmatch_s*'),
        ('SOM+LiP', 'now + SOM+LiP', '#CC79A7',
         'e1[34]??_{t}_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s*')]),
    'dv3': ('now_vs_dreamerv3', [
        ('DreamerV3', 'DreamerV3, ours', '#0072B2', 'e12[3-8]?_{t}_dreamerv3-pubmatch_s*')]),
}
BW, XMAX, FULL = 1e5, 4e6, 3.98e6
# Maximum achievable return per task. DMC: 1000. Pin pad (2000-step episodes,
# +10 per completed sequence, 10-step countdown, random respawn): optimal
# shortest-path policy from embodied/envs/pinpad.py by BFS, 2026-10-15.
MAXRET = {'pinpad_four': 411, 'pinpad_five': 352, 'pinpad_six': 292,
          'dmc_cartpole_swingup': 1000, 'dmc_cheetah_run': 1000, 'dmc_hopper_hop': 1000}


def overall(a_by_task, b_by_task, n_perm=200000, seed=0):
  """Mean over tasks of (b - a) / max achievable, stratified permutation p
  (arm labels shuffled within each task) and stratified bootstrap 95% CI."""
  rng = np.random.default_rng(seed)
  def stat(pairs):
    return np.mean([(np.mean(b) - np.mean(a)) / MAXRET[t] for t, a, b in pairs])
  pairs = [(t, np.asarray(a_by_task[t], float), np.asarray(b_by_task[t], float)) for t in a_by_task]
  obs = stat(pairs)
  diffs = np.zeros(n_perm); boots = np.zeros(n_perm // 10)
  for t, a, b in pairs:
    allv = np.concatenate([a, b]); n = len(a)
    idx = np.argsort(rng.random((n_perm, len(allv))), 1)
    sh = allv[idx]
    diffs += (sh[:, n:].mean(1) - sh[:, :n].mean(1)) / MAXRET[t]
    ra = a[rng.integers(0, len(a), (len(boots), len(a)))].mean(1)
    rb = b[rng.integers(0, len(b), (len(boots), len(b)))].mean(1)
    boots += (rb - ra) / MAXRET[t]
  diffs /= len(pairs); boots /= len(pairs)
  p = (np.sum(np.abs(diffs) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
  return obs, np.percentile(boots, [2.5, 97.5]), p



def runs(pat):
  got = {}
  for root in (B, W):
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
    return 1.0
  t = (y.mean() - x.mean()) / math.sqrt(vx + vy)
  df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
  c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
  s = np.linspace(abs(t), abs(t) + 200, 200001)
  return min(1.0, 2 * np.trapz(c * (1 + s ** 2 / df) ** (-(df + 1) / 2), s))


def perm_p(x, y):
  allv = np.array(list(x) + list(y), float); n = len(x)
  obs = abs(np.mean(y) - np.mean(x)); hits = tot = 0
  for idx in itertools.combinations(range(len(allv)), n):
    m = np.zeros(len(allv), bool); m[list(idx)] = True
    hits += abs(allv[~m].mean() - allv[m].mean()) >= obs - 1e-9; tot += 1
  return hits / tot


name, others = SETS[os.environ.get('SET', 'v3')]
arms = [NOW] + others
out = f'{W}/collapse_figs/{name}.png'
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2)); axes = axes.ravel()
x = (np.arange(0, XMAX, BW) + BW / 2) / 1e6
stats = {}
for ax, (task, title) in zip(axes, TASKS):
  hs = []
  for key, label, color, pat in arms:
    curves, late, auc = [], [], []
    for d in runs(pat.format(t=task)):
      a = scores(d)
      if not len(a):
        continue
      y = binned(a); curves.append(y)
      if a[:, 0].max() >= FULL:
        late.append(a[a[:, 0] >= 3e6, 1].mean()); auc.append(np.nanmean(y))
    stats[task, key] = dict(late=late, auc=auc)
    if curves:
      st = np.stack(curves)
      with np.errstate(all='ignore'):
        ok = (~np.isnan(st)).sum(0) >= 2
        m = np.where(ok, np.nanmean(st, 0), np.nan); sd = np.where(ok, np.nanstd(st, 0, ddof=1), np.nan)
      ax.fill_between(x, m - sd, m + sd, color=color, lw=0, alpha=0.15)
      hs.append((key, ax.plot(x, m, color=color, lw=2.4, label=label)[0]))
  ax.set_title(title, fontsize=17); ax.set_xlim(0, 4); ax.set_ylim(bottom=0)
  ax.tick_params(labelsize=13, direction='out', length=3)
  ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
  for sp in ax.spines.values():
    sp.set_color('#333333'); sp.set_linewidth(0.8)
for ax in (axes[0], axes[3]):
  ax.set_ylabel('episode return', fontsize=15)
# Legend: each arm's overall late score (% of max achievable, mean of 6 tasks)
# under its name, with the change relative to now and the stratified perm p.
h, l = axes[0].get_legend_handles_labels()
base = {t: stats[t, 'now']['late'] for t, _ in TASKS}
nb = np.mean([np.mean(base[t]) / MAXRET[t] for t in base])
lab = []
for (key, *_), name in zip(arms, l):
  if key == 'now':
    lab.append('%s\n%.0f%%' % (name, 100 * nb)); continue
  arm = {t: stats[t, key]['late'] for t, _ in TASKS}
  na = np.mean([np.mean(arm[t]) / MAXRET[t] for t in arm])
  _, _, pv = overall(base, arm)
  lab.append('%s\n%.0f%% (%+.0f%%, p=%.2g)' % (name, 100 * na, 100 * (na - nb) / nb, pv))
fig.legend(h, lab, loc='lower center', ncol=len(arms), fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.105, 'environment steps (M)', ha='center', fontsize=15)
fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=2.0)
fig.savefig(out, dpi=200, facecolor='white'); fig.savefig(out.replace('.png', '.pdf'))
print(out)

f = lambda v: '%.0f ± %.0f' % (np.mean(v), np.std(v, ddof=1))
# Percent table: late score as % of max achievable, change relative to now, perm p.
print('\n| Task | now | ' + ' | '.join(k for k, *_ in others) + ' |')
print('|---' * (2 + len(others)) + '|')
for task, title in TASKS:
  a = stats[task, 'now']['late']; cells = [title, '%.0f%%' % (100 * np.mean(a) / MAXRET[task])]
  for k, *_ in others:
    b = stats[task, k]['late']
    ch = '%+.0f%%' % (100 * (np.mean(b) - np.mean(a)) / np.mean(a)) if np.mean(a) > 1 else 'n/a'
    cells.append('%.0f%% (%s, p=%.2g)' % (100 * np.mean(b) / MAXRET[task], ch, perm_p(a, b)))
  print('| ' + ' | '.join(cells) + ' |')
for metric in ('late', 'auc'):
  print(f'\n{metric} ({"mean score 3-4M" if metric == "late" else "mean of whole curve 0-4M"})\n')
  print('| Task | now | ' + ' | '.join(f'{k} | Δ | p (Welch / perm)' for k, *_ in others) + ' |')
  print('|---' * (2 + 3 * len(others)) + '|')
  norm = {k: [] for k, *_ in others}
  for task, title in TASKS:
    a = stats[task, 'now'][metric]; cells = [title, f(a)]
    for k, *_ in others:
      b = stats[task, k][metric]
      d = np.mean(b) - np.mean(a)
      pct = '%+.0f%%' % (100 * d / np.mean(a)) if np.mean(a) > 1 else 'n/a'
      cells += [f(b) + f' ({len(b)})', '%+.0f (%s)' % (d, pct), '%.3f / %.3f' % (welch(a, b), perm_p(a, b))]
      norm[k].append(d / max(np.mean(a), np.mean(b), 1e-9))
    print('| ' + ' | '.join(cells) + ' |')
  base = {t: stats[t, 'now'][metric] for t, _ in TASKS}
  nb = np.mean([np.mean(base[t]) / MAXRET[t] for t in base])
  print('\noverall, normalized by max achievable (DMC 1000, pin pad 411/352/292), mean of 6 tasks:')
  print('  now %.3f' % nb)
  for k, *_ in others:
    arm = {t: stats[t, k][metric] for t, _ in TASKS}
    d, ci, pv = overall(base, arm)
    print('  %s %.3f  diff %+.3f  95%% CI [%+.3f, %+.3f]  p %.4f' % (
        k, np.mean([np.mean(arm[t]) / MAXRET[t] for t in arm]), d, ci[0], ci[1], pv))
