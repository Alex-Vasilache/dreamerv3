"""Health check for the UCB-in-imagination batch (e1760-e1819, arms A and B).

One line per run dir: env step, steps/hour since the previous check, late score,
the imagination-tilt diagnostics, and red flags.

  pick/cand   mean UCB bonus of the picked goal / of the 8 candidates. Both 1.0
              while the table is empty (tilt inert), then fall as counts fill.
  gain        pick/cand. 1.0 = inert; >1 = the tilt is choosing rarer goals.
  shift       log pi(picked) - mean log pi(candidates), nats, <= 0. How hard the
              tilt bends the manager away from its own preference.
  eff         effective cells of the count table (participation ratio).
  ent         manager normalized entropy (controller target 0.5).

Flags: NaN/inf, traceback/OOM in the SLURM .err, tilt inert after 300k steps
(gain < 1.01), tilt too strong (shift < -2 nats), manager entropy off target,
worker goal reward dead.

  python tools/watch_pubucbimag.py                    # the batch
  python tools/watch_pubucbimag.py '/work/.../e176*'  # a subset
"""
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watch_pubmask import bad, err_flag, tail_rows  # noqa: E402

WORK = '/work/DoyaU/vasilache/work'
STATE = os.path.join(WORK, 'slurm_logs', 'watch_pubucbimag_state.json')
KEYS = {
    'pick': 'train/imag_ucb/pick_bonus',
    'cand': 'train/imag_ucb/cand_bonus',
    'gain': 'train/imag_ucb/novelty_gain',
    'shift': 'train/imag_ucb/logp_shift',
    'eff': 'train/imag_ucb/eff_cells',
    'ent': 'train/mgr_actent_skill_mean',
    'wrew': 'train/wkr_goal_rew',
}


def main():
  pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
      WORK, 'e1[78][0-9][0-9]_*ucb_imag_*')
  prev = {}
  if os.path.exists(STATE):
    try:
      prev = json.load(open(STATE))
    except ValueError:
      prev = {}
  now = time.time()
  state = {}
  for d in sorted(glob.glob(pattern)):
    name = os.path.basename(d)
    exp = name.split('_')[0]
    arm = 'B' if 'ucb_imag_b' in name else 'A'
    task = re.sub(r'^e\d+_|_director.*$', '', name)
    seed = re.search(r'_s(\d+)_j', name)
    mrows = tail_rows(os.path.join(d, 'logdir', 'metrics.jsonl'))
    srows = tail_rows(os.path.join(d, 'logdir', 'scores.jsonl'), 50)
    step = max([r.get('step', 0) for r in mrows + srows] or [0])
    rate = ''
    if name in prev and now > prev[name]['t'] + 600:
      dh = (now - prev[name]['t']) / 3600
      rate = f'{(step - prev[name]["step"]) / dh / 1e3:.0f}k/h'
    state[name] = {'t': now, 'step': step} if name not in prev or rate else prev[name]
    vals = {}
    for k, key in KEYS.items():
      xs = [r[key] for r in mrows if key in r]
      vals[k] = xs[-1] if xs else None
    score = sum(r['episode/score'] for r in srows) / len(srows) if srows else float('nan')
    flags = []
    if any(bad(v) for r in mrows for k, v in r.items() if k.startswith('train/')):
      flags.append('NaN/inf')
    e = err_flag(d)
    if e:
      flags.append(e)
    if vals['gain'] is not None and step > 300_000 and vals['gain'] < 1.01:
      flags.append('tilt inert')
    if vals['shift'] is not None and vals['shift'] < -2.0:
      flags.append('tilt too strong')
    if vals['ent'] is not None and step > 200_000 and abs(vals['ent'] - 0.5) > 0.1:
      flags.append('mgr entropy off target')
    if vals['wrew'] is not None and step > 500_000 and vals['wrew'] < 0.05:
      flags.append('worker goal reward dead')
    f = lambda v, w=5: ' ' * (w - 1) + '-' if v is None else f'{v:{w}.2f}'
    print(f'{exp} {arm} s{seed.group(1) if seed else "?"} {task:21s} '
          f'{step / 1e6:5.2f}M {rate:>7s} score {score:6.1f} | '
          f'pick{f(vals["pick"])} cand{f(vals["cand"])} gain{f(vals["gain"])} '
          f'shift{f(vals["shift"])} eff{f(vals["eff"] and vals["eff"] / 1e3, 7)}k '
          f'ent{f(vals["ent"])} wrew{f(vals["wrew"])}'
          + (f'  !! {", ".join(flags)}' if flags else ''))
  if len(sys.argv) == 1:
    json.dump(state, open(STATE, 'w'))


if __name__ == '__main__':
  main()
