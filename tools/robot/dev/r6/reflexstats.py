"""Per-run stats of the catch reflex rounds (PF, RF: sim_phone runs with
pushes and walls).

  python reflexstats.py OUT.json ARMS.txt [run root ...]

ARMS: lines "<exp> <arm> <seed>"; run roots default to /work (Saion); on
IAS pass ~/robot_runs. Per run, from the recorded experience
(online_shared/experience): per minute, time upright (|tilt| < 3 deg),
bumper crossings (tilt crossing +6.5 or -5 deg, as emostats.py) and impact
pain (the 'cost' channel, summed); from emotion.csv (full, or subsampled
with env.robot.sim_phone_log_every, which keeps every push, contact and
correction row): pushes and falls per push (a crossing within 1 s of a push
on an upright body), bumper contacts and their peak true force (N; a contact
= consecutive steps with force > 0), and the reflex: the share of steps it
changed the command and the mean |a' - a| over those steps.

Summary per run: t90 (first minute upright >= 0.9), and means over minutes
11 to the end of crossings/min, contacts/min, peak force, impact pain/min,
falls per push, reflex rate and |a' - a|; and the phone loop's mean rate.
"""

import csv
import glob
import json
import os
import sys

import numpy as np

HZ = 65
MIN = 60 * HZ
LATE = 10   # minutes skipped for the 'late' means


def find(n, roots):
  for root in roots:
    dirs = sorted(glob.glob(f'{root}/e{n}_*/logdir'))
    if dirs:
      return dirs[0]
  return None


def run(d):
  L = []
  for f in sorted(glob.glob(d + '/online_shared/experience/*.npz')):
    try:
      x = np.load(f)
      if 'drive' in x.files:
        L.append({k: x[k] for k in ('orientation', 'cost')})
    except Exception:  # noqa: BLE001 -- a chunk being written
      pass
  if not L:
    return None
  tilt = np.degrees(np.concatenate([x['orientation'] for x in L])[:, 0] * 0.18)
  cost = np.concatenate([x['cost'] for x in L])
  cross = np.zeros(len(tilt), bool)
  cross[1:] = ((tilt[1:] > 6.5) & (tilt[:-1] <= 6.5)) | (
      (tilt[1:] < -5) & (tilt[:-1] >= -5))
  pushes, fires, contacts = [], [], []
  last_push, cur = -10**9, None
  try:
    rows = csv.DictReader(open(d + '/emotion.csv'))
    for r in rows:
      try:
        k = int(r['step'])
      except (TypeError, ValueError):
        continue
      push = float(r.get('push') or 0)
      if push and k - last_push > 10:
        pushes.append(k)
      if push:
        last_push = k
      ref = float(r.get('reflex') or 0)
      if ref:
        fires.append((k, abs(ref)))
      force = float(r.get('force') or 0)
      if force > 0:
        if cur is not None and k == cur[1] + 1:
          cur = (cur[0], k, max(cur[2], force))
        else:
          if cur is not None:
            contacts.append(cur)
          cur = (k, k, force)
    if cur is not None:
      contacts.append(cur)
  except OSError:
    pass
  fires = np.array(fires) if fires else np.zeros((0, 2))
  out = []
  for i in range(len(tilt) // MIN):
    a, b = i * MIN, (i + 1) * MIN
    ps = [p for p in pushes if a <= p < b and p < len(tilt) and abs(tilt[p]) < 3]
    fell = [bool(cross[p:p + HZ].any()) for p in ps]
    fi = fires[(fires[:, 0] >= a) & (fires[:, 0] < b)] if len(fires) else fires
    peaks = [c[2] for c in contacts if a <= c[0] < b]
    out.append(dict(
        up=float(np.mean(np.abs(tilt[a:b]) < 3)), cross=int(cross[a:b].sum()),
        pain=float(cost[a:b].sum()), pushes=len(ps),
        falls=int(sum(fell)), contacts=len(peaks),
        force=float(np.mean(peaks)) if peaks else float('nan'),
        fire=float(len(fi) / MIN), delta=float(fi[:, 1].mean()) if len(fi) else 0.0,
        delta_sum=float(fi[:, 1].sum()) if len(fi) else 0.0, nfire=int(len(fi))))
  return out


def fps(d):
  """The phone loop's mean rate (fps/policy, Hz; 65 = real time)."""
  v = []
  try:
    for line in open(d + '/metrics.jsonl'):
      if '"fps/policy"' in line:
        v.append(json.loads(line)['fps/policy'])
  except OSError:
    pass
  return float(np.mean(v)) if v else float('nan')


def summary(mins):
  late = mins[LATE:]
  t90 = next((i + 1 for i, m in enumerate(mins) if m['up'] >= 0.9), float('nan'))
  pushes = sum(m['pushes'] for m in late)
  nfire = sum(m['nfire'] for m in late)
  return dict(
      minutes=len(mins), t90=t90,
      cross=float(np.mean([m['cross'] for m in late])),
      contacts=float(np.mean([m['contacts'] for m in late])),
      force=float(np.nanmean([m['force'] for m in late])),
      pain=float(np.mean([m['pain'] for m in late])),
      fall_push=float(sum(m['falls'] for m in late) / pushes) if pushes else float('nan'),
      fire=float(np.mean([m['fire'] for m in late])),
      delta=float(sum(m['delta_sum'] for m in late) / nfire) if nfire else 0.0)


KEYS = ('t90', 'cross', 'contacts', 'force', 'pain', 'fall_push', 'fire', 'delta', 'fps')


if __name__ == '__main__':
  roots = sys.argv[3:] or ['/work/DoyaU/vasilache/work']
  roots = [os.path.expanduser(r) for r in roots]
  res = {}
  for line in open(sys.argv[2]):
    if not line.strip() or line.startswith('#'):
      continue
    n, arm, seed = line.split()[:3]
    d = find(n.lstrip('e'), roots)
    mins = run(d) if d else None
    if mins and len(mins) > LATE:
      res.setdefault(arm, {})[n] = dict(seed=seed, minutes=mins,
                                        summary=dict(summary(mins), fps=fps(d)))
  json.dump(res, open(sys.argv[1], 'w'))
  for arm, runs in res.items():
    S = [r['summary'] for r in runs.values()]
    cells = ' '.join(f'{k} {np.nanmean([s[k] for s in S]):.3g}+-{np.nanstd([s[k] for s in S]):.2g}'
                     for k in KEYS)
    print(f'{arm:14s} n={len(S)} mins={min(s["minutes"] for s in S)} {cells}')
