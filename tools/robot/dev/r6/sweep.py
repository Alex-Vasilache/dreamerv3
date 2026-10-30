"""Closed-loop sysid: every real-robot recording's policy, replayed in sim variants.
  python sweep.py NAME CODE_DIR CHUNK variants.txt   (variants.txt: 'name opts' per line)
Each policy runs every variant with its recording's stick (command runs) and floor;
a job takes CHUNK variants. Results: res/NAME/<policy>_<k>.json; score with loss.py."""
import os, subprocess, sys
W = '/work/DoyaU/vasilache/work'; R = f'{W}/sysid/recordings'
POL = {  # policy: (run dir, recording, floor, replay stick)
  'e2994': ('e2994_simrobot_balance65_flat_20261028_122654_KBOCsr', 'sim4_carpet_balance_20261028_142612', 'carpet', False),
  'e2608': ('e2608_simrobot_balance65_flat_20261026_195516_7NZ3fE', 'sim_carpet_balance_20261028_142731', 'carpet', False),
  'e2995': ('e2995_simrobot_balance65_flat_20261028_122654_3DLEyN', 'sim4_table_balance_20261028_141429', 'table', False),
  'e2503': ('e2503_simrobot_balance65_flat_20261026_183321_VPH0Al', 'sim_table_balance_20261028_141929', 'table', False),
  'e2992': ('e2992_simrobot_command65_flat_20261028_122654_u3v0qf', 'sim4_carpet_command_20261028_142846', 'carpet', True),
  'e6031': ('e6031_simrobot_command65_flat_20261028_162028_S0b7Oj', 'r5_robot_20261030/sim5_carpet_command_20261030_130541', 'carpet', True),
  'e6033': ('e6033_simrobot_command65_flat_20261028_162028_u7ZNUS', 'r5_robot_20261030/sim5_table_command_20261030_130843', 'table', True),
}
# Measured bumper contact angles (Alex, 2026-10-30), rad from theta_zero.
BUMP = {'carpet': 'bumper_hi=0.1361,bumper_lo=-0.1082', 'table': 'bumper_hi=0.161,bumper_lo=-0.120'}
BASE = 'sim_rand=,sim_eq_every=0,eq=0.2,sim_stick=0.2,sim_inertia=1,sim_backlash=0.0,sim_tilt_lag=0.0'


def main():
  name, code, chunk, vfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
  pols = sys.argv[5].split(',') if len(sys.argv) > 5 else list(POL)
  var = [l.split(None, 1) for l in open(vfile) if l.strip() and not l.startswith('#')]
  os.makedirs(f'{W}/sysid/r6/res/{name}', exist_ok=True)
  for p in pols:
    run, rec, floor, cmd = POL[p]
    for k in range(0, len(var), chunk):
      specs = []
      for v in var[k:k + chunk]:
        opts = v[1].strip() if len(v) > 1 else ''
        # 'C:key=val' / 'T:key=val' apply on carpet / table only.
        parts = [o for o in opts.split(',') if o]
        parts = [o[2:] if o[:2] in ('C:', 'T:') else o for o in parts
                 if o[:2] not in ('C:', 'T:') or o[0] == floor[0].upper()]
        opts = ','.join(parts)
        fl = 'surface=' + floor + ',' + BUMP[floor]
        specs.append(f'{v[0]}:{BASE},{fl}' + (',' + opts if opts else ''))
      env = dict(os.environ, CODE=code, CMDS=f'{R}/{rec}.csv' if cmd else '')
      out = subprocess.run(['sbatch', '--parsable', '-J', f'r6s_{name}_{p}', '-x', 'saion-gpu[01,02,04,06,07-10]',
                            f'{W}/sysid/r6/cl.sh', f'{W}/{run}', f'res/{name}/{p}_{k}.json', str(MIN), 'eval'] + specs,
                           env=env, capture_output=True, text=True)
      print(p, k, out.stdout.strip(), out.stderr.strip()[:200])


MIN = float(os.environ.get('MINUTES', 3.5))
if __name__ == '__main__':
  main()
