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
CARPET = FLOOR.replace('--env.robot.sim_schedule table', '--env.robot.sim_schedule carpet')
PHONE = ('--online_sync_every 4 --online_publish_every 1 --env.robot.sim_phone True '
         '--env.robot.sim_phone_delay 1.3 --env.robot.weights_emotion True')
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
# IA1 (11-06): actor-critic shrink-perturb / L2-to-init did not stop the drift
# at V100/A5000 learner speeds, so the base keeps the live ratio schedule.
W1_BASE = SCHED
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
    ('sw3', '--online_ratio_switch 11700'),
    ('acwd', '--agent.ac_opt.wd 0.1 --agent.ac_opt.wdinit True'),
]


# The recipe of rounds CR1/CR2 and IA1/IA2 (e30114's sim settings): drive home
# at half speed with a 3 s ramp, no cap; policy published every second sync;
# phone policy age 3 s.
OLD_PHONE = ('--online_sync_every 4 --env.robot.sim_phone True '
             '--env.robot.weights_emotion True')
OLD_BAL = ('--env.robot.home_radius 3045 --env.robot.home_speed 0.5 '
           '--env.robot.home_ramp 3000 --env.robot.balance_bumper 1.0 '
           '--env.robot.wheel_penalty 0.5 --env.robot.pos_tau 1.0 '
           '--env.robot.pos_penalty 1.0 --env.robot.pos_range 3000 '
           '--env.robot.drive_cap0 0')
GATE = '--online_ratio_switch_reward 0.4'
HOME_LIVE = '--env.robot.home_speed 1.0 --env.robot.home_ramp 1200 --env.robot.home_max_s 10'

# Round W2 (Sat 11-07): W1's base (live settings) learnt far worse than the
# CR2 recipe (P100: two of four seeds on a bumper for 10-20 min). Which live
# setting does it, and does gating the ratio switch on reward fix the slow GPU?
W2_ARMS = [
    ('old', 'old', SCHED),
    ('live', 'live', SCHED),
    ('old_home', 'old', f'{SCHED} {HOME_LIVE}'),
    ('old_pub1', 'old', f'{SCHED} --online_publish_every 1'),
    ('old_delay', 'old', f'{SCHED} --env.robot.sim_phone_delay 1.3'),
    ('live_nosched', 'live', ''),
    ('old_gate', 'old', f'{SCHED} {GATE}'),
    ('live_gate', 'live', f'{SCHED} {GATE}'),
    ('old_gate_b16', 'old', f'{SCHED} {GATE} --batch_size 16'),
    ('live_gate_b16', 'live', f'{SCHED} {GATE} --batch_size 16'),
]


# Round W3 (Sat 11-07): the bumper trap. A seed resting on the front bumper
# at 8.1 deg sits just under the penalty zone (9.2 - 1.0 deg) and earns +0.08;
# up to half the seeds of W2's arms lay there for 10+ min. Widen the margin.
MARGIN = '--env.robot.bumper_margin 0.035'
W3_ARMS = [
    ('old', 'old', SCHED),
    ('old_m35', 'old', f'{SCHED} {MARGIN}'),
    ('old_m50', 'old', f'{SCHED} --env.robot.bumper_margin 0.05'),
    ('old_m35_b16', 'old', f'{SCHED} {MARGIN} --batch_size 16'),
    ('old_m35_gate', 'old', f'{SCHED} {MARGIN} {GATE}'),
    ('live_m35', 'live', f'{SCHED} {MARGIN}'),
]


# Round W4 (Sat 11-07): the bumper trap fixed (margin 0.05: no stuck seed in
# 16; batch 16 fastest), re-screen on the fixed base, n = 8, P100 and IAS:
# what still speeds learning, which live setting hurts, and the late ratio.
WBASE = f"{SCHED} --env.robot.bumper_margin 0.05 --batch_size 16"
W4_ARMS = [
    ('B', 'old', WBASE),
    ('fear', 'old', f'{WBASE} --env.robot.cost_obs True --agent.fear.weight 1.0'),
    ('lr2', 'old', f'{WBASE} --agent.opt.lr 6e-4 --agent.ac_opt.lr 6e-4'),
    ('ent', 'old', f'{WBASE} --agent.imag_loss.actent 1e-3'),
    ('home', 'old', f'{WBASE} {HOME_LIVE}'),
    ('pub1', 'old', f'{WBASE} --online_publish_every 1'),
    ('delay', 'old', f'{WBASE} --env.robot.sim_phone_delay 1.3'),
    ('nosched', 'old', '--env.robot.bumper_margin 0.05 --batch_size 16'),
    ('late96', 'old', f'{WBASE} --online_ratio_late 96'),
    ('late192', 'old', f'{WBASE} --online_ratio_late 192'),
    ('acwd', 'old', f'{WBASE} --agent.ac_opt.wd 0.1 --agent.ac_opt.wdinit True'),
    ('l32', 'old', f'{WBASE} --batch_length 32'),
]


# Round W5 (Sat 11-07): W4 base B (CR2 recipe + schedule + margin 2.9 deg +
# batch 16) is the best yet; the fear critic cuts late crossings 9.6 -> 5.8
# but moves more. Combine, and retry stillness now that the bumper trap is
# shut; plus the live recipe with only the two fixes, as it could run today.
FEAR = '--env.robot.cost_obs True --agent.fear.weight 1.0'
W5_ARMS = [
    ('B', 'old', WBASE),
    ('fear', 'old', f'{WBASE} {FEAR}'),
    ('still', 'old', f'{WBASE} --env.robot.wheel_penalty 1.0 --env.robot.pos_penalty 2.0'),
    ('w075', 'old', f'{WBASE} --env.robot.wheel_penalty 0.75'),
    ('fear_w075', 'old', f'{WBASE} {FEAR} --env.robot.wheel_penalty 0.75'),
    ('fear_still', 'old', f'{WBASE} {FEAR} --env.robot.wheel_penalty 1.0 --env.robot.pos_penalty 2.0'),
    ('fear_ent', 'old', f'{WBASE} {FEAR} --agent.imag_loss.actent 1e-3'),
    ('live_fix', 'live', WBASE),
]


def bal_line(exp, gpu, flags, seed, minutes=30, floor=FLOOR, base='live'):
  phone, bal = (PHONE, LIVE_BAL) if base == 'live' else (OLD_PHONE, OLD_BAL)
  return (f'{exp} flat balance 65 {gpu} {floor} {phone} {bal} '
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


# Round K2 (Sat 11-07): K1 winners combined on the smoothing arm (forward
# tracking 0.7 in 11 min, 5.7 crossings/min against 15), plus a narrower
# tracking kernel for standing still, and the full combination at P100 speed.
SMOOTH = K1_ARMS[6][1]
ACWD = K1_ARMS[1][1]
K2_ARMS = [
    ('s', SMOOTH),
    ('s_acwd', f'{SMOOTH} {ACWD}'),
    ('s_b16', f'{SMOOTH} --batch_size 16'),
    ('s_acwd_b16', f'{SMOOTH} {ACWD} --batch_size 16'),
    ('s_sig25', f'{SMOOTH} --env.robot.command_sigma 0.25'),
    ('s_acwd_b16_sig25', f'{SMOOTH} {ACWD} --batch_size 16 --env.robot.command_sigma 0.25'),
    ('s_acwd_b16_p100', f'{SMOOTH} {ACWD} --batch_size 16 --online_learner_fps 17500'),
    ('s_p100', f'{SMOOTH} --online_learner_fps 17500'),
]


# Round K3 (Sat 11-07): the K2 pick (smoothing + L2-to-init + batch 16:
# forward 0.7 in 10.0 min, 4.9 crossings/min, same at P100 speed) with a
# command curriculum (command_grow_s) against the 30-60 falls/min of the
# first minutes, longer relabelling, shorter holds.
KPICK = f'{SMOOTH} {ACWD} --batch_size 16'
K3_ARMS = [
    ('pick', KPICK),
    ('grow300', f'{KPICK} --env.robot.command_grow_s 300'),
    ('grow600', f'{KPICK} --env.robot.command_grow_s 600'),
    ('grow300_p100', f'{KPICK} --env.robot.command_grow_s 300 --online_learner_fps 17500'),
    ('relabel9', f'{KPICK} --cmd_relabel 0.9'),
    ('hold24', f'{KPICK} --env.robot.command_hold_min 2.0 --env.robot.command_hold_max 4.0'),
]


# Round K4 (Sat 11-07): K3 found no gain from a command curriculum, relabel
# 0.9 or shorter holds; the pick holds over 8 seeds. Now its robustness: 90
# min (late collapse?), carpet (held out), pushes + walls, the live phone
# settings, and a wider bumper margin.
K4_ARMS = [
    ('pick90', KPICK, 90, 'table'),
    ('carpet', KPICK, 60, 'carpet'),
    ('push', f'{KPICK} --env.robot.sim_push 5:0.05:0.15 --env.robot.sim_walls 0.5', 60, 'table'),
    ('live', f'{KPICK} --online_publish_every 1 --env.robot.sim_phone_delay 1.3', 60, 'table'),
    ('m35', f'{KPICK} --env.robot.bumper_margin 0.035', 60, 'table'),
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
  elif rnd == 'k2':
    for seed in SEEDS4:
      for arm, flags in K2_ARMS:
        lines.append(f'# {arm}'); lines.append(cmd_line(f'e{n}', 'v100', flags, seed)); n += 1
  elif rnd == 'k3':
    for seed in SEEDS4:
      for arm, flags in K3_ARMS:
        lines.append(f'# {arm}'); lines.append(cmd_line(f'e{n}', 'v100', flags, seed)); n += 1
    for seed in SEEDS8[4:]:
      lines.append('# pick'); lines.append(cmd_line(f'e{n}', 'v100', KPICK, seed)); n += 1
  elif rnd == 'k4':
    for seed in SEEDS4:
      for arm, flags, minutes, floor in K4_ARMS:
        lines.append(f'# {arm}')
        lines.append(cmd_line(f'e{n}', 'v100', flags, seed, minutes=minutes,
                              floor=CARPET if floor == 'carpet' else FLOOR)); n += 1
  elif rnd == 'smoke':
    for arm, flags in W1_ARMS:
      if arm in ('base', 'big', 'small', 'fear', 'b64', 'acwd'):
        lines.append(bal_line(f'e{n}', 'v100', f'{W1_BASE} {flags}', 1, minutes=4)); n += 1
    lines.append(cmd_line(f'e{n}', 'v100', K1_ARMS[2][1] + ' ' + K1_ARMS[3][1], 1, minutes=4)); n += 1
  elif rnd in ('w2p', 'w2i'):
    gpu = {'w2p': 'p100', 'w2i': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, base, flags in W2_ARMS:
        lines.append(f'# {arm}'); lines.append(bal_line(f'e{n}', gpu, flags, seed, base=base)); n += 1
  elif rnd in ('w3v', 'w3i', 'w3p'):
    gpu = {'w3v': 'v100', 'w3i': 'any', 'w3p': 'p100'}[rnd]
    for seed in SEEDS8:
      for arm, base, flags in W3_ARMS:
        lines.append(f'# {arm}'); lines.append(bal_line(f'e{n}', gpu, flags, seed, base=base)); n += 1
  elif rnd in ('w4p', 'w4i'):
    gpu = {'w4p': 'p100', 'w4i': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, base, flags in W4_ARMS:
        lines.append(f'# {arm}'); lines.append(bal_line(f'e{n}', gpu, flags, seed, base=base)); n += 1
  elif rnd in ('w5v', 'w5i'):
    gpu = {'w5v': 'v100', 'w5i': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, base, flags in W5_ARMS:
        lines.append(f'# {arm}'); lines.append(bal_line(f'e{n}', gpu, flags, seed, base=base)); n += 1
  elif rnd == 'smoke2':
    lines.append(cmd_line(f'e{n}', 'p100', SMOOTH + ' --env.robot.command_grow_s 120', 1, minutes=4)); n += 1
  elif rnd.startswith('w1'):
    base = W1_BASE
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
