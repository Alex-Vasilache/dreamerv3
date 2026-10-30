"""Average the agent params of several milestones of one run (SWA), as a pseudo run dir for evalpol / from_sim.
  python swa.py RUN OUT_RUN STEP [STEP...]   -> OUT_RUN/logdir/{config.yaml, ckpt_milestones/<last step>/}"""
import os, pickle, shutil, sys
import numpy as np
run, out, steps = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
cks = [pickle.load(open(f'{run}/logdir/ckpt_milestones/{s:012d}/agent.pkl', 'rb')) for s in steps]


def avg(xs):
  if isinstance(xs[0], dict):
    return {k: avg([x[k] for x in xs]) for k in xs[0]}
  a = np.asarray(xs[0])
  if np.issubdtype(a.dtype, np.floating):
    return np.mean([np.asarray(x, np.float32) for x in xs], 0).astype(a.dtype)
  return xs[-1]


new = dict(cks[-1])
# Flat keys; optimizer state ('opt/...') and counters are kept from the last checkpoint.
new['params'] = {k: (v if k.startswith('opt/') else avg([c['params'][k] for c in cks]))
                 for k, v in cks[-1]['params'].items()}
d = f'{out}/logdir/ckpt_milestones/{steps[-1]:012d}'
os.makedirs(d, exist_ok=True)
pickle.dump(new, open(f'{d}/agent.pkl', 'wb'))
shutil.copy(f'{run}/logdir/ckpt_milestones/{steps[-1]:012d}/step.pkl', d)
open(f'{d}/done', 'w').close()
shutil.copy(f'{run}/logdir/config.yaml', f'{out}/logdir/config.yaml')
print('wrote', d)
