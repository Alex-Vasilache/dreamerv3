"""Return curves + late-score table for the goal-AE add-ons on `OG`.

  ucb   OG (e1230-e1289) | OG+SOM (e1330-e1419) | OG+SOM+UCB (e1430-e1459)
  mask  OG | OG+mask (e1600-e1629) | OG+SOM | OG+SOM+mask (e1630-e1659)

Per task and seed: late = mean episode score over 3-4M, only for runs that
reached 3.98M (an unfinished run is drawn but not scored). Pin pads also give
the sustained-50 crossing (first 50k bin from which the next 4 bins all average
>= 50): seeds crossing and the median step. p = exact two-sided permutation
test of each arm against `OG`. Also draws the arm's own diagnostics over
training (UCB: count/tilt metrics; mask: what the manager keeps).

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/compare_late.py ucb
  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/compare_late.py mask

Writes collapse_figs/{set}_returns.{png,pdf}, {set}_diag.{png,pdf}, {set}_table.md.
"""
import glob, itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
OUT = f'{W}/collapse_figs'
TASKS = [('pinpad_four', 'Pin Pad Four', '#000000'), ('pinpad_five', 'Pin Pad Five', '#E69F00'),
         ('pinpad_six', 'Pin Pad Six', '#56B4E9'), ('dmc_cartpole_swingup', 'Cartpole Swingup', '#009E73'),
         ('dmc_cheetah_run', 'Cheetah Run', '#0072B2'), ('dmc_hopper_hop', 'Hopper Hop', '#CC79A7')]
NOW = ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*')
SOM = ('som', 'OG + SOM', '#CC79A7', 'e1[34]??_{t}_director-director_og-goal_som_line-pubmatch_s*')
SETS = {
    'ucb': [NOW, SOM,
            ('ucb', 'OG + SOM + UCB', '#0072B2', 'e14[3-5]?_{t}_director-director_og-goal_som_line-mgr_ucb-pubmatch_s*')],
    'wkrent': [NOW, ('wkrent', 'OG + fixed worker entropy', '#0072B2',
                     'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*')],
    'wkrv3': [NOW, ('wkrent', 'OG + fixed worker entropy', '#E69F00'.replace('E69F00', '999999'),
                    'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*'),
              ('wkrR', '+ worker v3 return norm', '#0072B2',
               'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3_retnorm-pubmatch_s*'),
              ('wkrV', '+ worker v3 (all)', '#009E73',
               'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*')],
    'ucbimag': [NOW, SOM,
                ('ucb', 'OG + SOM + UCB (env)', '#000000', 'e14[3-5]?_{t}_director-director_og-goal_som_line-mgr_ucb-pubmatch_s*'),
                ('imagA', 'OG + SOM + UCB imag A', '#0072B2', 'e18[2-4]?_{t}_director-director_og-goal_som_line-ucb_imag_a-ucb_imag_c3-pubmatch_s*'),
                ('imagB', 'OG + SOM + UCB imag B', '#56B4E9', 'e18[5-7]?_{t}_director-director_og-goal_som_line-ucb_imag_b-ucb_imag_c3-pubmatch_s*')],
    'mask': [NOW, ('mask', 'OG + mask', '#000000', 'e16[0-2]?_{t}_director-director_og-goal_mask8-pubmatch_s*'),
             SOM, ('sommask', 'OG + SOM + mask', '#56B4E9', 'e16[3-5]?_{t}_director-director_og-goal_som_line-goal_mask8-pubmatch_s*')],
}
DIAG = {
    'wkrent': [None, [('train/wkr_actent_action_mean', 'worker normalized entropy'),
                      ('train/wkr_goal_rew', 'worker goal reward'),
                      ('train/mgr_actent_skill_mean', 'manager normalized entropy'),
                      ('train/wkr_ent/action', 'worker entropy (nats)')]],
    'ucbimag': [None, [('train/imag_ucb/novelty_gain', 'novelty gain (picked / mean candidate)'),
                       ('train/imag_ucb/logp_shift', 'log-prob shift of picked goal (nats)'),
                       ('train/imag_ucb/pick_bonus', 'UCB bonus of picked goal'),
                       ('train/mgr_actent_skill_mean', 'manager normalized entropy')]],
    'ucb': ['ucb', [('train/novel/ucb_logp_shift', 'log-prob shift of picked goal (nats)'),
                    ('train/novel/ucb_novelty_gain', 'novelty gain (picked / mean candidate)'),
                    ('train/novel/select_eff_cells', 'effective cells selected (of 65,536)'),
                    ('train/novel/select_bonus_mean', 'mean UCB bonus of candidates')]],
    'mask': [None, [('train/goal/mask_keep_mgr', 'fraction of blocks kept'),
                    ('train/goal/mask_full_mgr', 'fraction of goals with all blocks kept'),
                    ('train/goal/mask_acc', 'mask head accuracy'),
                    ('train/wkr_goal_rew', 'worker goal reward')]],
}
BW, XMAX, FULL = 1e5, 4e6, 3.98e6


def runs(pat):
  got = {}
  for root in (B, W):   # /work wins for a run still training
    for d in glob.glob(f'{root}/{pat}'):
      got[os.path.basename(d)] = d
  return sorted(got.values())


def scores(d):
  rows = []
  if not os.path.exists(f'{d}/logdir/scores.jsonl'):
    return np.zeros((0, 2))
  for line in open(f'{d}/logdir/scores.jsonl'):
    try:
      r = json.loads(line)
    except ValueError:
      continue
    if r.get('episode/score') is not None:
      rows.append((r['step'], r['episode/score']))
  return np.array(rows) if rows else np.zeros((0, 2))


def binned(a, bw=BW):
  idx = (a[:, 0] // bw).astype(int)
  return np.array([a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(int(XMAX // bw))])


def cross50(a):
  b = binned(a, 5e4)
  return next((i * 5e4 for i in range(len(b) - 3) if np.all(b[i:i + 4] >= 50)), None)


def perm_p(x, y):
  allv = np.array(list(x) + list(y), float); n = len(x)
  obs = abs(np.mean(y) - np.mean(x)); hits = tot = 0
  for idx in itertools.combinations(range(len(allv)), n):
    m = np.zeros(len(allv), bool); m[list(idx)] = True
    hits += abs(allv[~m].mean() - allv[m].mean()) >= obs - 1e-9; tot += 1
  return hits / tot


def style(ax, title):
  ax.set_title(title, fontsize=17)
  ax.tick_params(labelsize=13, direction='out', length=3)
  ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
  for sp in ax.spines.values():
    sp.set_color('#333333'); sp.set_linewidth(0.8)


def finish(fig, axes, xlabel, ylabel, ncol, out):
  for ax in axes[:, 0]:
    ax.set_ylabel(ylabel, fontsize=15)
  h, l = axes[0, 0].get_legend_handles_labels()
  rows = -(-len(l) // ncol)
  bottom = 0.05 + 0.04 * rows
  fig.legend(h, l, loc='lower center', ncol=ncol, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.0))
  fig.text(0.5, bottom - 0.03, xlabel, ha='center', fontsize=15)
  fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2.0)
  fig.savefig(out + '.png', dpi=200, facecolor='white'); fig.savefig(out + '.pdf')
  print(out + '.png')


def metric_curve(d, key):
  xs, ys = [], []
  if not os.path.exists(f'{d}/logdir/metrics.jsonl'):
    return None
  for line in open(f'{d}/logdir/metrics.jsonl'):
    if key not in line:
      continue
    r = json.loads(line)
    if r.get(key) is not None:
      xs.append(r['step']); ys.append(r[key])
  a = np.array([xs, ys]).T if xs else np.zeros((0, 2))
  return binned(a, 2e5) if len(a) else None


def main(which):
  arms = SETS[which]
  os.makedirs(OUT, exist_ok=True)
  fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
  x = (np.arange(0, XMAX, BW) + BW / 2) / 1e6
  lines = ['| task | ' + ' | '.join(a[1] for a in arms) + ' |', '|---' * (len(arms) + 1) + '|']
  for ax, (task, title, _) in zip(axes.ravel(), TASKS):
    late = {}
    cells = []
    for key, label, color, pat in arms:
      curves, lt, cr, steps = [], [], [], []
      for d in runs(pat.format(t=task)):
        a = scores(d)
        if not len(a):
          continue
        curves.append(binned(a)); steps.append(a[:, 0].max())
        if a[:, 0].max() >= FULL:
          lt.append(a[a[:, 0] >= 3e6, 1].mean()); cr.append(cross50(a))
      late[key] = lt
      if curves:
        st = np.stack(curves)
        with np.errstate(all='ignore'):
          n_ok = (~np.isnan(st)).sum(0)
          m = np.where(n_ok >= 2, np.nanmean(st, 0), np.nan)
          sd = np.where(n_ok >= 2, np.nanstd(st, 0, ddof=1), np.nan)
        ax.fill_between(x, m - sd, m + sd, color=color, lw=0, alpha=0.15)
        ax.plot(x, m, color=color, lw=2.4, label=label)
      cell = f'{np.mean(lt):.0f} ± {np.std(lt, ddof=1) if len(lt) > 1 else 0:.0f}' if lt else '-'
      if task.startswith('pinpad') and lt:
        ok = [c for c in cr if c is not None]
        cell += f', {len(ok)}/{len(lt)}' + (f' @{np.median(ok) / 1e6:.2f}M' if ok else '')
      if key != 'og' and len(lt) >= 2 and len(late['og']) >= 2:
        cell += f' (p={perm_p(late["og"], lt):.2f})'
      if len(lt) < len(steps):
        cell += f' [n={len(lt)}, {len(steps) - len(lt)} running]'
      cells.append(cell)
    lines.append(f'| {title} | ' + ' | '.join(cells) + ' |')
    ax.set_xlim(0, 4); ax.set_ylim(bottom=0)
    style(ax, title)
  finish(fig, axes, 'environment steps (M)', 'episode return', min(len(arms), 3), f'{OUT}/{which}_returns')
  table = '\n'.join(lines)
  open(f'{OUT}/{which}_table.md', 'w').write(table + '\n')
  print(table)

  # Diagnostics: one panel per metric, one line per task (mean over seeds), for
  # each arm that logs them.
  only, metrics = DIAG[which]
  diag_arms = [a for a in arms if (a[0] == only if only else a[0] in (
      ('imagA', 'imagB') if which == 'ucbimag' else ('wkrent',) if which == 'wkrent'
      else ('mask', 'sommask')))]
  fig, axes = plt.subplots(2, 2, figsize=(13, 8.6 if len(diag_arms) == 1 else 11), sharex=True)
  xd = (np.arange(0, XMAX, 2e5) + 1e5) / 1e6
  for ax, (mkey, mtitle) in zip(axes.ravel(), metrics):
    for key, label, _, pat in diag_arms:
      ls = '-' if key in ('ucb', 'mask', 'imagA') else ':'
      for task, title, c in TASKS:
        cs = [metric_curve(d, mkey) for d in runs(pat.format(t=task))]
        cs = [v for v in cs if v is not None]
        if cs:
          with np.errstate(all='ignore'):
            m = np.nanmean(np.stack(cs), 0)
          ax.plot(xd, m, color=c, lw=2.4, ls=ls,
                  label=title if len(diag_arms) == 1 else f'{title}, {label}')
    ax.set_xlim(0, 4)
    style(ax, mtitle)
  finish(fig, axes, 'environment steps (M)', '', 3 if len(diag_arms) == 1 else 2, f'{OUT}/{which}_diag')


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'ucb')
