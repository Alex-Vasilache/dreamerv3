"""A face trace (JSON) from a sim run's emotion.csv, for replaying on the face page.

  python tools/robot/face/trace_from_csv.py LOGDIR/emotion.csv OUT.json \
      [--start 300] [--seconds 120]

Open it with face.html?trace=OUT.json (served next to the page). The CSV has
the policy's and the applied command for the left wheel only; the trace uses
it for both. It has no reward column: the trace leaves reward out.
"""
import argparse
import csv
import json


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('csv')
  ap.add_argument('out')
  ap.add_argument('--start', type=float, default=300.0, help='s into the run')
  ap.add_argument('--seconds', type=float, default=120.0)
  ap.add_argument('--hz', type=float, default=10.0)
  ap.add_argument('--bumpers', default='-0.087,0.113', help='back,front tilt (rad)')
  ap.add_argument('--zone', type=float, default=0.035)
  args = ap.parse_args()
  rows = list(csv.DictReader(open(args.csv)))
  t0 = float(rows[0]['t']) + args.start
  rows = [r for r in rows if t0 <= float(r['t']) < t0 + args.seconds]
  if not rows:
    raise SystemExit('no rows in that window')
  keys = dict(valence='valence', arousal='arousal', dominance='dominance',
              pain='pain', fear='fear', tilt='theta', act_l='act0',
              act_r='act0', chosen_l='chosen0', chosen_r='chosen0')
  out = {k: [] for k in keys}
  out['t'] = []
  # Resampled to a fixed rate, each sample the last row at or before it.
  n, i = int(args.seconds * args.hz), 0
  for s in range(n):
    t = t0 + s / args.hz
    while i + 1 < len(rows) and float(rows[i + 1]['t']) <= t:
      i += 1
    out['t'].append(round(s / args.hz, 2))
    for k, c in keys.items():
      out[k].append(round(float(rows[i].get(c) or 0.0), 4))
  lo, hi = (float(x) for x in args.bumpers.split(','))
  out.update(hz=args.hz, bumper_back=lo, bumper_front=hi, pain_zone=args.zone)
  with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
  print(f'{args.out}: {n} samples, {args.seconds:.0f} s')


if __name__ == '__main__':
  main()
