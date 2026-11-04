"""Why did pin pad five seed 0 of the 4x12-bit hash (e19101) fail? Compare its
training metrics with the pin pad five seeds that solved (same arm, single hash, TCE).

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/pp5_failure_diag.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import runs, scores, OUT  # noqa: E402

B = 'director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir'
GROUPS = [
    ('4x12 hash, adaptive w: seed 0 (failed)', '#D55E00', '-', 3.2, f'e19101_pinpad_five_{B}-tce_hash4a-pubmatch_s*'),
    ('4x12 hash, adaptive w: other seeds', '#000000', '-', 1.6, f'e191[0-3]?_pinpad_five_{B}-tce_hash4a-pubmatch_s[1-9]*'),
    ('1x16 hash, adaptive w (5 seeds)', '#009E73', '--', 1.6, f'e95[0-2]?_pinpad_five_{B}-tce_hash-pubmatch_s*'),
    ('TCE codebook (5 seeds)', '#0072B2', ':', 1.6, f'e62[0-2]?_pinpad_five_{B}-pubmatch_s*'),
]
KEYS = {
    'weight': 'train/fromto/weight', 'lean': 'train/fromto/lean', 'novelty': 'train/fromto/novelty',
    'nov_uniform': 'train/fromto/novelty_uniform', 'mgr_ent': 'train/mgr_ent_norm_skill_mean',
    'ent_scale': 'train/mgr_actent_skill_scale_mean', 'extr_ret': 'train/mgr_extr_ret',
    'expl_rew': 'train/mgr_expl_rew', 'adv_mag': 'train/mgr_adv_mag', 'extr_adv': 'train/mgr_extr_adv',
    'wkr_rate': 'train/wkr_goal_ret_rate', 'goal_ent': 'train/goal/entropy_mean',
    'row_mass': 'train/fromto/row_mass_median',
}
BW = 1e5
EDGES = np.arange(0, 4e6 + BW, BW)
X = (EDGES[:-1] + BW / 2) / 1e6


def load(d):
  rows = [json.loads(l) for l in open(f'{d}/logdir/metrics.jsonl') if 'fromto/weight' in l]
  st = np.array([r['step'] for r in rows])
  idx = np.digitize(st, EDGES) - 1
  out = {}
  for k, key in KEYS.items():
    v = np.array([r.get(key, np.nan) for r in rows], float)
    out[k] = np.array([np.nanmean(v[idx == b]) if (idx == b).any() else np.nan for b in range(len(X))])
  s = scores(d)
  si = np.digitize(s[:, 0], EDGES) - 1
  out['score'] = np.array([s[si == b, 1].mean() if (si == b).any() else np.nan for b in range(len(X))])
  out['first_reward'] = s[s[:, 1] > 0, 0].min() / 1e6 if (s[:, 1] > 0).any() else np.nan
  srt = s[np.argsort(s[:, 0])]
  hit = [srt[i, 0] for i in range(len(srt)) if srt[i:i + 20, 1].mean() >= 150 and i + 20 <= len(srt)]
  out['reach150'] = hit[0] / 1e6 if hit else np.nan
  return out


data = {name: [load(d) for d in runs(p)] for name, *_, p in GROUPS}
at = lambda a, m: a[int(m * 10) - 1]
print('| run | first reward (M) | score>=150 sustained (M) | w @0.3/0.6/1/2/3M | lean @0.3/0.6/1M | mgr entropy @1M | extr ret @1M/2M | expl rew @0.3/1M |')
print('|---|---|---|---|---|---|---|---|')
for name, *_ in GROUPS:
  for i, o in enumerate(data[name]):
    w = '/'.join(f'{at(o["weight"], m):.2g}' for m in (0.3, 0.6, 1, 2, 3))
    ln = '/'.join(f'{at(o["lean"], m):.3f}' for m in (0.3, 0.6, 1))
    print(f'| {name.split(":")[0]} #{i} {name.split(":")[-1].strip()[:14]} | {o["first_reward"]:.2f} | {o["reach150"]:.2f} | {w} | {ln} | '
          f'{at(o["mgr_ent"], 1):.3f} | {at(o["extr_ret"], 1):.3g}/{at(o["extr_ret"], 2):.3g} | {at(o["expl_rew"], .3):.3g}/{at(o["expl_rew"], 1):.3g} |')

PANELS = [('score', 'episode return', False), ('weight', 'TCE weight w (log)', True), ('lean', 'lean', False),
          ('mgr_ent', 'manager entropy (normalised)', False), ('extr_ret', 'manager extrinsic return (imagined)', False),
          ('expl_rew', 'Director exploration reward', True)]
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
for ax, (k, lab, logy) in zip(axes.ravel(), PANELS):
  for name, col, ls, lw, _ in GROUPS:
    for j, o in enumerate(data[name]):
      ax.plot(X, o[k], color=col, ls=ls, lw=lw, alpha=1.0 if lw > 2 else 0.55, label=name if j == 0 else None)
  ax.set_title(lab, fontsize=16, loc='left')
  if logy:
    ax.set_yscale('log')
  if k == 'lean':
    ax.axhline(1.1, color='#8a8880', lw=1, ls='--'); ax.axhline(1 / 1.1, color='#8a8880', lw=1, ls='--')
  ax.set_xlim(0, 4); ax.tick_params(labelsize=13); ax.grid(True, color='#d5d5d2', lw=0.7)
  for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
for ax in axes[1]:
  ax.set_xlabel('environment steps (M)', fontsize=15)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=13.5, frameon=False)
fig.text(0.5, 0.105, 'Pin pad five. Dashed grey lines on the lean panel: the controller\'s dead band (weight moves only outside 0.909-1.1).', ha='center', fontsize=13)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(f'{OUT}/pp5_failure.png', dpi=200, facecolor='white'); fig.savefig(f'{OUT}/pp5_failure.pdf')
print(f'{OUT}/pp5_failure.png')
