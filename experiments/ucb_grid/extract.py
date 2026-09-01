"""Pull the three count tables out of every milestone checkpoint into one npz.

Runs under dreamerv3_env (numpy 2.x) because the pickles were written there.
Output is tiny: 65536+65536+256 floats per milestone.
"""
import pickle, sys, time, pathlib
import numpy as np

B = pathlib.Path('/bucket/DoyaU/vasilache/bucket/results/dreamerv3')
RUNS = {
    'e713_cartpole_s1': 'e713_dmc_cartpole_swingup_som_lipvq_line_relu_ucb_s1_BIG_j4702050',
    'e714_cheetah_s0':  'e714_dmc_cheetah_run_som_lipvq_line_relu_ucb_s0_BIG_j4702051',
    'e715_cheetah_s1':  'e715_dmc_cheetah_run_som_lipvq_line_relu_ucb_s1_BIG_j4702052',
    'e716_hopper_s0':   'e716_dmc_hopper_hop_som_lipvq_line_relu_ucb_s0_BIG_j4702053',
    'e717_hopper_s1':   'e717_dmc_hopper_hop_som_lipvq_line_relu_ucb_s1_BIG_j4702054',
}
WANT = ('select_tab', 'fine_tab', 'coarse_tab')

def tables(path):
  with open(path, 'rb') as f:
    blob = pickle.load(f)
  found = {}
  def walk(n, p=''):
    if isinstance(n, dict):
      for k, v in n.items(): walk(v, f'{p}/{k}' if p else str(k))
    elif hasattr(n, 'shape') and '/opt/' not in p:
      for w in WANT:
        if p.endswith(f'code_counts/{w}/value'):
          found[w] = np.asarray(n, np.float64)
  walk(blob)
  return found

out = {}
for tag, run in RUNS.items():
  for mile in sorted((B / run / 'logdir' / 'ckpt_milestones').glob('*')):
    step = int(mile.name)
    t = time.time()
    try:
      f = tables(mile / 'agent.pkl')
    except Exception as e:
      print(f'{tag} {step}: FAILED {e}', flush=True); continue
    for w, v in f.items():
      out[f'{tag}|{step}|{w}'] = v.astype(np.float32)
    got = {w: (float(f[w].sum()) if w in f else None) for w in WANT}
    print(f'{tag} {step:>9d}  {time.time()-t:5.1f}s  '
          f'select={got["select_tab"]:.0f} fine={got["fine_tab"]:.1f} '
          f'coarse={got["coarse_tab"]:.1f}', flush=True)
np.savez_compressed(sys.argv[1], **out)
print('wrote', sys.argv[1], len(out), 'arrays')
