"""World-model drift-fix figures (e1068-e1094).

wmfix_pinpad4.png  pinpad_four: score and worker goal reward per arm vs the control.
wmfix_transfer.png wm_norepgrad on the other benchmark tasks vs director_og baselines.
Reads runs from /work first, then the bucket, so it works before and after archiving.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
OUT = '/work/DoyaU/vasilache/work/collapse_figs'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
BLUE, GRAY = '#2a78d6', '#9a998f'


def find(e):
  for root in (W, B):
    ds = sorted(glob.glob(f'{root}/{e}_*'))
    if ds:
      return ds[0]


def series(d, key):
  if key == 'score':
    f = f'{d}/logdir/scores.jsonl'
    rows = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    return [(r['step'], r['episode/score']) for r in rows]
  out = []
  for l in open(f'{d}/logdir/metrics.jsonl'):
    r = json.loads(l)
    if key in r:
      out.append((r['step'], r[key]))
  return out


def binned(pairs, xmax, bw):
  edges = np.arange(0, xmax + bw, bw)
  if not pairs:
    return edges[:-1] + bw / 2, np.full(len(edges) - 1, np.nan)
  a = np.array(pairs, float)
  idx = np.digitize(a[:, 0], edges) - 1
  return edges[:-1] + bw / 2, np.array(
      [a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)])


def mean_where2(curves):
  st = np.stack(curves)
  ok = (~np.isnan(st)).sum(0) >= 2
  with np.errstate(all='ignore'):
    return np.where(ok, np.nanmean(st, 0), np.nan)


def style(ax):
  ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)


def draw(ax, exps, key, xmax, bw, color, label, single=True):
  curves = []
  for e in exps:
    d = find(e)
    if not d:
      continue
    x, y = binned(series(d, key), xmax, bw)
    curves.append(y)
    if single:
      ax.plot(x / 1e6, y, color=color, lw=1, alpha=0.4)
  if len(curves) >= 2:
    ax.plot(x / 1e6, mean_where2(curves), color=color, lw=2.4, label=f'{label} (mean of {len(curves)})')
  return len(curves)


plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

# --- pinpad_four arms -------------------------------------------------------
CTRL = ['e1057', 'e1058', 'e1059']
ARMS = [('Real rewards in replay value', ['e1068', 'e1069', 'e1070']),
        ('No critic gradient into world model', ['e1071', 'e1072', 'e1073']),
        ('Batch length 64, lr / 4 (old rewards)', ['e1074', 'e1075', 'e1076']),
        ('No critic gradient + batch length 64', ['e1077', 'e1078', 'e1079'])]
XMAX, BW = 4e6, 5e4
fig, axes = plt.subplots(2, 4, figsize=(18, 7.4), sharex=True, sharey='row')
for j, (title, exps) in enumerate(ARMS):
  for i, (key, ylabel, ylim) in enumerate([('score', 'episode score', (0, 320)),
                                           ('train/wkr_goal_rew', 'worker goal reward', (0, 0.8))]):
    ax = axes[i, j]; style(ax)
    draw(ax, CTRL, key, XMAX, BW, GRAY, 'control, 1.2M', single=False)
    draw(ax, exps, key, XMAX, BW, BLUE, 'this arm')
    ax.set_ylim(*ylim); ax.set_xlim(0, XMAX / 1e6)
    if i == 0:
      ax.set_title(title, fontsize=11, color=INK, loc='left', fontweight='bold')
      ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    else:
      ax.set_xlabel('env steps (M)')
    if j == 0:
      ax.set_ylabel(ylabel)
fig.suptitle('Fixing the pinpad collapse: which world-model change holds the sequence',
             x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
fig.text(0.01, 0.935, 'pinpad_four, director_og, size6m, 3 seeds per arm, 50k-step bins. Gray: '
         'control e1057-59 (old code, ran to 1.2M); "no critic gradient" continued to 4M, the others stop at 2M. Thin lines: single seeds. Bottom row: the '
         'goal-reward jump that comes with collapse.', fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.92))
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/wmfix_pinpad4.png', dpi=130, facecolor='white')
print(f'{OUT}/wmfix_pinpad4.png')

# --- transfer ---------------------------------------------------------------
TASKS = [('pinpad_five', 'Pin Pad Five', 'e1080'), ('pinpad_six', 'Pin Pad Six', 'e1083'),
         ('dmc_cartpole_swingup', 'Cartpole swingup', 'e1086'),
         ('dmc_cheetah_run', 'Cheetah run', 'e1089'), ('dmc_hopper_hop', 'Hopper hop', 'e1092')]
XMAX, BW = 4e6, 1e5
fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
for ax, (task, title, first) in zip(axes, TASKS):
  style(ax)
  base = [os.path.basename(d).split('_')[0] for d in
          sorted(glob.glob(f'{B}/e9[1-3][0-9]_{task}_director-director_og_s*'))]
  n0 = int(first[1:])
  draw(ax, base, 'score', XMAX, BW, GRAY, 'director_og baseline')
  draw(ax, [f'e{n0 + k}' for k in range(3)], 'score', XMAX, BW, BLUE, 'no critic gradient into WM')
  ax.set_title(title, fontsize=11, color=INK, loc='left', fontweight='bold')
  ax.set_xlabel('env steps (M)'); ax.set_xlim(0, XMAX / 1e6); ax.set_ylim(bottom=0)
axes[0].set_ylabel('episode score (100k-step bins)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.995, 0.97))
fig.suptitle('The world-model fix on the other benchmark tasks', x=0.01, ha='left',
             fontsize=13, color=INK, fontweight='bold')
fig.text(0.01, 0.905, 'director_og, size6m, 3 seeds each, 4M steps. Gray: benchmark baselines '
         '(e919-e936). Blue: same config with agent.hrl_repval_grad False and real replay rewards.',
         fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(f'{OUT}/wmfix_transfer.png', dpi=130, facecolor='white')
print(f'{OUT}/wmfix_transfer.png')
