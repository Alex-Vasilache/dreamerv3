"""Everything the figures need, as one JSON."""
import numpy as np, json, sys, glob, os

L, C, BINS = 8, 8, 4
d = np.load(sys.argv[1])
out = {}

runs = ['e713_cartpole_s1','e714_cheetah_s0','e715_cheetah_s1','e716_hopper_s0','e717_hopper_s1']
steps = sorted({int(k.split('|')[1]) for k in d.files})
out['runs'], out['steps'] = runs, steps

# --- what the tables look like, at 4M, for one representative run each ---
def blocks_marg(t, bins):
  a = t.reshape((bins,) * L)
  return [a.sum(tuple(i for i in range(L) if i != l)).tolist() for l in range(L)]

tabs = {}
for r in ['e714_cheetah_s0', 'e713_cartpole_s1']:
  for name, bins in (('select_tab', 4), ('fine_tab', 4), ('coarse_tab', 2)):
    t = d[f'{r}|4000000|{name}'].astype(np.float64)
    a = t.reshape((bins,) * L)
    tabs[f'{r}|{name}'] = dict(
        bins=bins, total=float(t.sum()),
        marg=blocks_marg(t, bins),
        pair01=a.sum(tuple(range(2, L))).tolist(),      # blocks 0 x 1 heatmap
        sorted_frac=(np.sort(t)[::-1] / t.sum()).cumsum()[::max(1, len(t)//400)].tolist(),
        sorted_mass=np.sort(t)[::-1][::max(1, len(t)//400)].tolist(),
        nz=float((t > 0).mean()), eff=float(t.sum()**2/(t**2).sum()))
out['tables'] = tabs

# --- mass / bonus distribution of select_tab over training ---
prof = {}
for r in runs:
  prof[r] = {}
  for s in steps:
    t = d[f'{r}|{s}|select_tab'].astype(np.float64)
    srt = np.sort(t)[::-1]
    prof[r][str(s)] = dict(
        q=[float(x) for x in np.percentile(t, [1,5,10,25,50,75,90,95,99])],
        bonus_q=[float(1/np.sqrt(x+1)) for x in np.percentile(t, [99,95,90,75,50,25,10,5,1])],
        rank_mass=srt[::max(1, len(srt)//200)].tolist(),
        nz=float((t>0).mean()), gt1=float((t>1).mean()), gt10=float((t>10).mean()),
        eff=float(t.sum()**2/(t**2).sum()), mass=float(t.sum()))
out['profile'] = prof

# --- measured gain from the archived metrics ---
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
DIRS = {
 'e713_cartpole_s1':'e713_dmc_cartpole_swingup_som_lipvq_line_relu_ucb_s1_BIG_j4702050',
 'e714_cheetah_s0':'e714_dmc_cheetah_run_som_lipvq_line_relu_ucb_s0_BIG_j4702051',
 'e715_cheetah_s1':'e715_dmc_cheetah_run_som_lipvq_line_relu_ucb_s1_BIG_j4702052',
 'e716_hopper_s0':'e716_dmc_hopper_hop_som_lipvq_line_relu_ucb_s0_BIG_j4702053',
 'e717_hopper_s1':'e717_dmc_hopper_hop_som_lipvq_line_relu_ucb_s1_BIG_j4702054'}
KG, KB, KS, KE = ('train/novel/ucb_novelty_gain','train/novel/select_bonus_mean',
                  'train/novel/ucb_logp_shift','train/novel/select_eff_cells')
meas = {}
for r, dd in DIRS.items():
  rows = []
  for line in open(f'{B}/{dd}/logdir/metrics.jsonl'):
    j = json.loads(line)
    if KG in j: rows.append((j['step'], j[KG], j.get(KB), j.get(KS), j.get(KE)))
  b = {}
  for st, g, bo, sh, ec in rows: b.setdefault(int(st//500000), []).append((g, bo, sh, ec))
  meas[r] = dict(
      bins=[(i+0.5)*0.5 for i in sorted(b)],
      gain=[float(np.mean([x[0] for x in b[i]])) for i in sorted(b)],
      gain_se=[float(np.std([x[0] for x in b[i]])/len(b[i])**.5) for i in sorted(b)],
      bonus=[float(np.mean([x[1] for x in b[i] if x[1] is not None])) for i in sorted(b)],
      shift=[float(np.mean([x[2] for x in b[i] if x[2] is not None])) for i in sorted(b)],
      eff=[float(np.mean([x[3] for x in b[i] if x[3] is not None])) for i in sorted(b)])
out['measured'] = meas

for f, key in ((sys.argv[2], 'frozen'), (sys.argv[3], 'counterfactual')):
  out[key] = json.load(open(f))
json.dump(out, open(sys.argv[4], 'w'))
print('wrote', sys.argv[4], os.path.getsize(sys.argv[4])//1024, 'KB')
