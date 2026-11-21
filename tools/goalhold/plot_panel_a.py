"""Goal-hold panel array A (4831340): reach cap 8 vs done cap 32, 6 tasks x 5 seeds."""
import glob, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3/'
ds = glob.glob(B + 'h05[4-9][0-9]_*j4831340_*') + glob.glob(B + 'h060[01]_*j4831340_*') \
   + [g for h in ('h0447', 'h0435', 'h0453') for g in (glob.glob(B + h + '_*') or glob.glob('/work/DoyaU/vasilache/work/' + h + '_*'))[:1]]
TASKS = ['pinpad_four', 'pinpad_five', 'pinpad_six', 'dmc_cartpole_swingup', 'dmc_cheetah_run', 'dmc_hopper_hop']
ARMS = {'c8': ('Reach 0.8, cap 8', '#0072B2'), 'd32': ('Worker done, cap 32', '#E69F00')}
bins = np.arange(0, 4.0e6 + 1, 1e5)
data = {}
for d in ds:
  task = d.split('/')[-1].split('_director')[0][6:]
  arm = 'c8' if 'cap8' in d else 'd32'
  S = [(r['step'], r['episode/score']) for r in map(json.loads, open(d + '/logdir/scores.jsonl')) if 'episode/score' in r]
  t, s = np.array(S).T
  idx = np.digitize(t, bins) - 1
  curve = np.array([s[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)])
  late = s[t >= 3.5e6].mean()
  ok = np.nan_to_num(curve, nan=-1) >= 50
  cross = next((bins[i] for i in range(len(ok)) if ok[i:].all() and ok[i]), np.nan)
  data.setdefault((task, arm), []).append((curve, late, cross))
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15, 'legend.fontsize': 14, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
fig, axs = plt.subplots(2, 3, figsize=(17, 9))
x = (bins[:-1] + 5e4) / 1e6
for ax, task in zip(axs.flat, TASKS):
  for arm, (lab, col) in ARMS.items():
    C = np.array([c for c, _, _ in data.get((task, arm), [])])
    if not len(C): continue
    m = np.nanmean(C, 0); se = np.nanstd(C, 0) / np.sqrt(len(C))
    ax.plot(x, m, color=col, lw=2.2, label=f'{lab} (n={len(C)})'); ax.fill_between(x, m - se, m + se, color=col, alpha=0.2)
  ax.set_title(task.replace('dmc_', '').replace('_', ' ')); ax.set_xlabel('env steps (M)'); ax.grid(alpha=0.3)
for a in axs[:, 0]: a.set_ylabel('episode score')
axs[0, 0].legend(loc='upper left')
fig.tight_layout()
out = '/work/DoyaU/vasilache/work/collapse_figs/goalhold_panel_a'
fig.savefig(out + '.png', dpi=200); fig.savefig(out + '.pdf')
print('task | reach cap 8: late mean ± se [sustained-50 M] | done cap 32')
for task in TASKS:
  row = [task.replace('dmc_', '')]
  for arm in ARMS:
    v = data.get((task, arm), []); L = np.array([l for _, l, _ in v]); X = np.array([c for _, _, c in v]) / 1e6
    cr = '' if 'pinpad' not in task else f" [{np.sum(~np.isnan(X))}/{len(X)} sustain, med {np.nanmedian(X) if np.any(~np.isnan(X)) else float('nan'):.2f}M]"
    row.append(f'{L.mean():.0f} ± {L.std() / np.sqrt(len(L)):.0f}{cr}')
  print(' | '.join(row))
print(out + '.png')
