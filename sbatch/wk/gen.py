"""Params files for the weekend rounds of 2026-11-07/09 (docs/robot/WEEKEND_1107.md).

  python sbatch/wk/gen.py <round> <first exp number> [out dir]

Writes one line per run in the submit_sim_robot.sh format
(<exp> flat <task> 65 <gpu> flags...); the IAS launcher reads the same lines.
Each round is a list of (arm, flags) crossed with seeds.
"""
import sys

# The Germany lab floor in the fitted sim (sim table schedule plus per-minute
# randomisation of the drive train), the phone's own loop, and the live
# demo's settings (e40006, docs/robot/LIVE_20261106.md): policy published
# every sync, drive home capped at 10 s, phone policy age ~1.3 s on IAS.
FLOOR = ('--env.robot.sim_schedule table --env.robot.sim_inertia 2 '
         '--env.robot.sim_backlash 0.015 --env.robot.sim_stick 0.35 '
         '--env.robot.sim_gear_k 1 --env.robot.sim_rand_every 60 '
         '--env.robot.sim_scrape 20 --env.robot.sim_rand '
         'inertia:1.5:2.5,backlash:0.01:0.02,stick:0.2:0.45,gear_k:0.8:2,'
         'rstatic:5:9,scrape:0:40,stall:0.85:1.1,tilt_lag:0:0.015')
CARPET = FLOOR.replace('--env.robot.sim_schedule table', '--env.robot.sim_surface carpet')
PHONE = ('--online_sync_every 4 --online_publish_every 1 --env.robot.sim_phone True '
         '--env.robot.sim_phone_delay 1.3 --env.robot.weights_emotion False')
LIVE_BAL = ('--env.robot.home_radius 3045 --env.robot.home_speed 1.0 '
            '--env.robot.home_ramp 1200 --env.robot.home_max_s 10 '
            '--env.robot.balance_bumper 1.0 --env.robot.wheel_penalty 0.5 '
            '--env.robot.pos_tau 1.0 --env.robot.pos_penalty 1.0 '
            '--env.robot.pos_range 3000 --env.robot.drive_cap0 0')
SCHED = '--online_ratio_switch 19500 --online_ratio_late 128'
MIN = 3900  # actor steps per minute at 65 Hz

SEEDS4 = [811, 812, 813, 814]
SEEDS8 = SEEDS4 + [815, 816, 817, 818]

# Round W1 (balance, speed-up screen): one factor at a time against the base,
# at the slowest (P100, uncapped ~280) and fastest (IAS A5000/3090, ~570)
# learner. The base's drift control is filled in from rounds IA1/IA2.
W1_BASE = None  # set by round(): IA1/IA2 winner flags
W1_ARMS = [
    ('base', ''),
    ('b16', '--batch_size 16'),
    ('b64', '--batch_size 64'),
    ('l32', '--batch_length 32'),
    ('lr2', '--agent.opt.lr 6e-4 --agent.ac_opt.lr 6e-4'),
    ('aclr2', '--agent.ac_opt.lr 6e-4'),
    ('small', '--.*\\.units 128 --agent.dyn.rssm.deter 256 --agent.dyn.rssm.hidden 128'),
    ('big', '--.*\\.units 512 --agent.dyn.rssm.deter 1024 --agent.dyn.rssm.hidden 512'),
    ('h1', '--env.robot.horizon_s 1.0'),
    ('h3', '--env.robot.horizon_s 3.0'),
    ('ent', '--agent.imag_loss.actent 1e-3'),
    ('fear', '--env.robot.cost_obs True --agent.fear.weight 1.0'),
    ('smooth', '--env.robot.action_rate_quad 0.5'),
    ('still', '--env.robot.wheel_penalty 1.0 --env.robot.pos_penalty 2.0'),
]


def bal_line(exp, gpu, flags, seed, minutes=30, floor=FLOOR):
  return (f'{exp} flat balance 65 {gpu} {floor} {PHONE} {LIVE_BAL} '
          f'--run.steps {minutes * MIN} {flags} --seed {seed}').replace('  ', ' ')


# Command following (robot_command block: bumper_track reward, relabel and
# mirror 0.5) on the same floor and phone loop, 60 min.
K1_ARMS = [
    ('base', ''),
    ('acwd', '--agent.ac_opt.wd 0.1 --agent.ac_opt.wdinit True'),
    ('sched', SCHED),
    ('fine', '--env.robot.command_sigma_fine 0.15'),
    ('pzero', '--env.robot.command_p_zero 0.5'),
    ('slow', '--env.robot.command_speed 1200 --env.robot.command_turn 1500'),
    ('smooth', '--env.robot.action_rate_quad 0.25 --agent.imag_loss.smooth 0.15'),
    ('b16', '--batch_size 16'),
]


def cmd_line(exp, gpu, flags, seed, minutes=60, floor=FLOOR):
  return (f'{exp} flat command 65 {gpu} {floor} {PHONE} '
          f'--run.steps {minutes * MIN} {flags} --seed {seed}').replace('  ', ' ')


def main():
  rnd, first = sys.argv[1], int(sys.argv[2])
  out = sys.argv[3] if len(sys.argv) > 3 else 'sbatch/wk'
  lines, n = [], first
  if rnd == 'k1':
    for seed in SEEDS4:
      for arm, flags in K1_ARMS:
        lines.append(f'# {arm}'); lines.append(cmd_line(f'e{n}', 'v100', flags, seed)); n += 1
  elif rnd.startswith('w1'):
    base = sys.argv[4] if len(sys.argv) > 4 else ''
    gpu = {'w1p': 'p100', 'w1i': 'any'}[rnd]
    for seed in SEEDS4:
      for arm, flags in W1_ARMS:
        lines.append(f'# {arm}'); lines.append(bal_line(f'e{n}', gpu, f'{base} {flags}', seed)); n += 1
  else:
    raise SystemExit(f'unknown round {rnd}')
  path = f'{out}/{rnd}.txt'
  open(path, 'w').write('\n'.join(lines) + '\n')
  print(path, n - first, 'runs, last', n - 1)


if __name__ == '__main__':
  main()
