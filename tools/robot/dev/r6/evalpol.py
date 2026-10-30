"""Evaluate trained command policies closed loop in the fitted sim and a robustness set,
replaying a real recording's stick (hard forward/back).
  python evalpol.py TAG CODE FLOOR RUN_DIR [RUN_DIR...]   (env MINUTES, CKPT)"""
import os, subprocess, sys
W = '/work/DoyaU/vasilache/work'; R = f'{W}/sysid/recordings/r5_robot_20261030'
STICK = {'carpet': f'{R}/sim5_carpet_command_20261030_130541.csv', 'table': f'{R}/sim5_table_command_20261030_130843.csv'}
BUMP = {'carpet': 'bumper_hi=0.1361,bumper_lo=-0.1082', 'table': 'bumper_hi=0.161,bumper_lo=-0.120'}
BASE = 'sim_rand=,sim_eq_every=0,eq=0.2,sim_stick=0.3,sim_gear_k=1,sim_inertia=2,sim_backlash=0.015,sim_tilt_lag=0.0'
SET = {
  'carpet': ['fit:C,sim_scrape=20,rstatic=7', 'clean:C,sim_scrape=0', 'scr40:C,sim_scrape=40', 'rs10:C,sim_scrape=20,rstatic=10',
             'i3bl025:C,sim_scrape=20,rstatic=7,sim_inertia=3,sim_backlash=0.025', 'eqm1:C,sim_scrape=20,rstatic=7,eq=-0.8', 'eqp1:C,sim_scrape=20,rstatic=7,eq=1.2',
             'k3:C,sim_scrape=20,rstatic=7,sim_gear_k=3', 'noise2:C,sim_scrape=20,rstatic=7,noise=2', 'delay1:C,sim_scrape=20,rstatic=7,delay=1'],
  'table': ['fit:C', 'i15bl01:C,sim_inertia=1.5,sim_backlash=0.01', 'i3bl025:C,sim_inertia=3,sim_backlash=0.025', 'eqm1:C,eq=-0.8',
            'eqp1:C,eq=1.2', 'k3:C,sim_gear_k=3', 'st15:C,sim_stick=0.15', 'noise2:C,noise=2', 'delay1:C,delay=1', 'carpet:surface=carpet,sim_scrape=20,rstatic=7,' + BUMP['carpet']],
}


def main():
  tag, code, floor = sys.argv[1:4]
  os.makedirs(f'{W}/sysid/r6/ev/{tag}', exist_ok=True)
  for run in sys.argv[4:]:
    exp = os.path.basename(run.rstrip('/')).split('_')[0]
    if os.environ.get('CKPT'):
      exp += '@' + os.environ['CKPT']
    specs = []
    for v in SET[floor]:
      name, opts = v.split(':', 1)
      base = f'{BASE},surface={floor},{BUMP[floor]}'
      specs.append(f'{name}:{base},' + opts.replace('C,', '').replace('C', ''))
    for k in range(0, len(specs), 5):
      env = dict(os.environ, CODE=code, CMDS=STICK[floor])
      out = subprocess.run(['sbatch', '--parsable', '-J', f'r6e_{tag}_{exp}', '-x', 'saion-gpu[01,02,04,06,07-10]',
                            f'{W}/sysid/r6/cl.sh', run, f'ev/{tag}/{exp}_{k}.json', os.environ.get('MINUTES', '3.5'), 'eval']
                           + specs[k:k + 5], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      print(exp, k, out.stdout.strip(), out.stderr.strip()[:200])


if __name__ == '__main__':
  main()
