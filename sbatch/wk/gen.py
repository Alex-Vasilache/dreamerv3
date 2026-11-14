"""Params files for the weekend rounds of 2026-11-07/09 (docs/robot/WEEKEND_1107.md).

  python sbatch/wk/gen.py <round> <first exp number> [out dir]

Writes one line per run in the submit_sim_robot.sh format
(<exp> flat <task> 65 <gpu> flags...); the IAS launcher reads the same lines.
Each round is a list of (arm, flags) crossed with seeds.
"""
import os
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


# Round BF (Sat 11-07 evening, final): W5 V100 (n = 7 of 8 in): fear critic +
# stillness (wheel 1.0, pos 2.0) cuts late crossings to 2.6 / min (B 12.5,
# live recipe with both fixes 17.3) at B's start (3.1 min) and less movement.
# 60 min, n = 12, against today's live recipe (e40006: live + schedule), B and
# B + fear; holdouts on carpet and with pushes + walls (n = 4).
STILL = '--env.robot.wheel_penalty 1.0 --env.robot.pos_penalty 2.0'
SEEDS12 = SEEDS8 + [819, 820, 821, 822]
PUSH = '--env.robot.sim_push 5:0.05:0.15 --env.robot.sim_walls 0.5'
BF_ARMS = [
    ('live', 'live', SCHED),
    ('B', 'old', WBASE),
    ('fear', 'old', f'{WBASE} {FEAR}'),
    ('fear_still', 'old', f'{WBASE} {FEAR} {STILL}'),
]
BF_HOLD = [
    ('fs_carpet', 'old', f'{WBASE} {FEAR} {STILL}', 'carpet'),
    ('fs_push', 'old', f'{WBASE} {FEAR} {STILL} {PUSH}', 'table'),
]


# Round Z (Mon 11-09, sample-efficiency study, docs/robot/PLAN.md): which
# component makes learning slow and drift late -- the world model or the
# actor-critic? Balance pick (BF fear_still).
# Z1 (late drift, Saion V100, 60 min): the pick without the ratio schedule
# (uncapped learner, drifts), with it, and uncapped with the world model or
# the actor-critic frozen from minute 5 (actor step 19500).
# Z2 (slow start, IAS, 30 min): the pick from scratch, and started from the
# final weights of a BF pick run (e43159, V100, min 55) for the world model
# only, the actor-critic only, or both.
PICK = f'{WBASE} {FEAR} {STILL}'
PICK_UNC = f'{PICK.replace(SCHED, "")} '.strip()
Z1_ARMS = [
    ('unc', PICK_UNC),
    ('sched', PICK),
    ('unc_fwm', f'{PICK_UNC} --online_freeze_model_at 19500'),
    ('unc_fac', f'{PICK_UNC} --online_freeze_ac_at 19500'),
]
# Absolute path of the source checkpoint on the cluster that runs Z2 (Z_SRC).
SRC = os.environ.get('Z_SRC', 'ckpt_src/e43159_214500')
# No quotes: the launchers expand the flags unquoted, so quotes would stay in
# the regex. These contain no spaces or glob characters.
WM_RE = '^(dyn|enc|dec|rew|con|cst)/'
AC_RE = '^(pol|val|slowval|cval|slowcval|retnorm|cretnorm)(/|_count)'
ALL_RE = '^(dyn|enc|dec|rew|con|cst|pol|val|slowval|cval|slowcval|retnorm|cretnorm)(/|_count)'
LOAD = f'--run.from_checkpoint {SRC} --run.from_checkpoint_regex'
Z2_ARMS = [
    ('fresh', PICK),
    ('wm', f'{PICK} {LOAD} {WM_RE}'),
    ('ac', f'{PICK} {LOAD} {AC_RE}'),
    ('both', f'{PICK} {LOAD} {ALL_RE}'),
]
SEEDS_Z = [831, 832, 833, 834, 835, 836, 837, 838]


# Round Y (step 2, Saion V100, 60 min, n = 8, after Z1 so Z1's unc/sched arms
# are its controls): world-model dropout (rssm.drop) at the uncapped ratio,
# where the pick drifts late, and once with the schedule.
DROP = '--agent.dyn.rssm.drop'
Y_ARMS = [
    ('unc_d05', f'{PICK_UNC} {DROP} 0.05'),
    ('unc_d10', f'{PICK_UNC} {DROP} 0.1'),
    ('unc_d20', f'{PICK_UNC} {DROP} 0.2'),
    ('sched_d10', f'{PICK} {DROP} 0.1'),
]
# Round X (step 3, IAS, 60 min, n = 8): observational dropout (rssm.obsdrop),
# with its own controls on IAS's GPUs.
ODROP = '--agent.dyn.rssm.obsdrop'
X_ARMS = [
    ('unc', PICK_UNC),
    ('sched', PICK),
    ('unc_o10', f'{PICK_UNC} {ODROP} 0.1'),
    ('unc_o30', f'{PICK_UNC} {ODROP} 0.3'),
    ('sched_o30', f'{PICK} {ODROP} 0.3'),
]

# Round V (step 4, IAS, 60 min, n = 8, controls = X's unc/sched): N
# imagination rollouts per start state (agent.imag_repeat).
REP = '--agent.imag_repeat'
V_ARMS = [
    ('unc_r2', f'{PICK_UNC} {REP} 2'),
    ('unc_r4', f'{PICK_UNC} {REP} 4'),
    ('sched_r2', f'{PICK} {REP} 2'),
    ('sched_r4', f'{PICK} {REP} 4'),
]

# Round U (Mon 11-09, Saion P100, 60 min, n = 8): DreamerV4's "one rollout
# per context" (arXiv 2509.24527 §3.3): imagination starts only from the last
# state of each replayed sequence (agent.imag_last 1) instead of every step
# (16 x 64 = 1024 starts), alone (16 starts, far cheaper updates), with 4x the
# sequences (batch 64), and imag_last 8 in between; on the pick (schedule on).
LAST = '--agent.imag_last'
U_ARMS = [
    ('sched', PICK),
    ('k1', f'{PICK} {LAST} 1'),
    ('k1_b64', f'{PICK} {LAST} 1 --batch_size 64'),
    ('k8', f'{PICK} {LAST} 8'),
]


# Round M (step 6, 11-10): combine the parts that helped. Nothing tested
# shortened the first minutes; three gave a calmer late phase: the ratio
# schedule (in the pick), world-model dropout 0.05 (Y) and 2 rollouts per
# start (V). 2 x 2 factorial on top of the pick, n = 12, at three GPU speeds.
M_ARMS = [
    ('pick', PICK),
    ('d05', f'{PICK} {DROP} 0.05'),
    ('r2', f'{PICK} {REP} 2'),
    ('d05_r2', f'{PICK} {DROP} 0.05 {REP} 2'),
]

# Round N (11-10 evening): the late drift without the hand-set ratio
# schedule. Alex: the schedule is handcrafted, it has to work without it.
# The two regularisers that calmed the uncapped late phase (world-model
# dropout 0.05, Y; N rollouts per start, V) together and uncapped, at the
# fastest learner (IAS), against uncapped and scheduled controls; n = 8.
N_ARMS = [
    ('unc', PICK_UNC),
    ('sched', PICK),
    ('unc_d05_r2', f'{PICK_UNC} {DROP} 0.05 {REP} 2'),
    ('unc_d05_r4', f'{PICK_UNC} {DROP} 0.05 {REP} 4'),
]

# Round P (11-10 night): the learner sets its own train ratio from learning
# progress on held-out recent experience (online_progress, replay.holdout),
# no switch time, no target ratio. Against uncapped and the hand-set
# schedule, and with world-model dropout 0.05 (M's step 6b pick); n = 8 at
# V100, P100 and IAS.
PROG = '--replay.holdout 0.1 --online_progress.every 20'
P_ARMS = [
    ('unc', PICK_UNC),
    ('sched', PICK),
    ('prog', f'{PICK_UNC} {PROG}'),
    ('prog_d05', f'{PICK_UNC} {PROG} {DROP} 0.05'),
]

# Round Q (11-11 morning): IAS P put uncapped, scheduled and progress-controlled
# at the same late level (2.2-2.3), unlike N (2.6 vs 1.7). Confirm N's
# uncapped dropout 0.05 + 2 rollouts against both controls on fresh seeds,
# IAS, n = 8.
SEEDS_Q = [821, 822, 823, 824, 825, 826, 827, 828]
Q_ARMS = [N_ARMS[0], N_ARMS[1], N_ARMS[2]]

# Round R (11-11): C (uncapped + dropout 0.05 + 2 rollouts) on the slower
# Saion GPUs, same seeds as P's uncapped and scheduled arms there, n = 8.
R_ARMS = [N_ARMS[2]]

# Round S2 (11-11): uncapped world-model dropout 0.05 alone. R showed the
# second rollout costs speed on slow GPUs (P100 t90 3.8 vs 3.1); M had dropout
# free of speed cost (with the schedule). imag_repeat 1 explicit: the robot
# default is 2 since 11-11. Seeds 811-818 pair with P/R (Saion) and N (IAS).
S2_ARMS = [('unc_d05', f'{PICK_UNC} {DROP} 0.05 {REP} 1')]

# Round PF (11-12, pain and fear; docs/robot/EMOTION.md target design): on
# S2's pick (uncapped, dropout 0.05, the fear critic) with pushes and walls,
# (a) today's pain (bumper closeness + reflex override) and runaway brake,
# (b) impact pain (|change of tilt rate| per step) for the threat critic,
# no brake, (c) = (b) + the catch reflex in the phone loop. 60 min, n = 8.
# Logging only (no effect on learning): the phone loop feels every step and
# records every arousal variant, pleasure's parts and the empowerment terms
# at 16 and 32 steps, for comparing the emotion variants afterwards.
PF_LOG = '--env.robot.sim_phone_feel_every 1 --env.robot.empower_extra 32'
PF_BASE = f'{S2_ARMS[0][1]} {PUSH} {PF_LOG}'
PF_IMPACT = '--env.robot.pain_mode impact --env.robot.runaway_s 0'
# catch_tmax 0.2: in the 4-min smokes (e44501-e44502) the threat critic sat
# at 0.08-0.11 (p99 0.14) and rose to 0.2-0.26 around impacts.
PF_REFLEX = '--env.robot.catch_reflex True --env.robot.catch_tmax 0.2'
PF_ARMS = [
    ('a_bumper_brake', PF_BASE),
    ('b_impact', f'{PF_BASE} {PF_IMPACT}'),
    ('c_impact_reflex', f'{PF_BASE} {PF_IMPACT} {PF_REFLEX}'),
]


# Round RF (11-12, catch reflex): PF c's threat critic falls from ~0.15 to
# ~0.001 (median) within 10 min, so the absolute catch_tmax 0.2 fires on
# 20-30% of steps in minutes 2-3 and never after minute 7; and T(a) held 16
# steps predicts a crossing worse than 4 steps (offline AUC 0.62-0.65 vs
# 0.76). On PF c's setup: a limit relative to the run (mean + 3 SD of log
# T(a) over 10 s: the linear z of the first smokes was blown up for minutes
# by the untrained critic's spikes), and a small search over the drive
# (a + 0, +-0.25, +-0.5, +-1)
# at 4, 8 and 16 steps (evaluated every 1, 2, 4 steps: ~4 ms each), against
# the original slope step capped at 0.5. The CSV keeps every 16th row plus
# every push, contact and correction (sim_phone_log_every 16). The phone
# loop's BLAS is capped at one thread (sim_phone_threads 1): with the default
# the RF smokes (e44600-e44603) ran the loop at 25-50 Hz instead of 65, so b
# and c are rerun here with the same cap.
RF_LOG = '--env.robot.sim_phone_log_every 16 --env.robot.sim_phone_threads 1'
RF_COMMON = (f'{PF_BASE} {PF_IMPACT} {RF_LOG} --env.robot.catch_reflex True '
             '--env.robot.catch_z 3 --env.robot.catch_zlog True')
RF_ARMS = [
    ('b', f'{PF_BASE} {PF_IMPACT} {RF_LOG}'),
    ('c', f'{PF_BASE} {PF_IMPACT} {RF_LOG} {PF_REFLEX}'),
    ('g4z', f'{RF_COMMON} --env.robot.catch_mode slope --env.robot.catch_dmax 0.5'),
    ('s4z', f'{RF_COMMON} --env.robot.catch_mode search'),
    ('s8z', f'{RF_COMMON} --env.robot.catch_mode search --env.robot.catch_steps 8 --env.robot.catch_every 2'),
    ('s16z', f'{RF_COMMON} --env.robot.catch_mode search --env.robot.catch_steps 16 --env.robot.catch_every 4'),
]


# Round RF2 (11-12 evening): in RF the log-z limit of g4z and s4z stuck at
# the 1e-4 floor (the critic sat at or below it through the 10 s warm-up, SD
# 0, and the capped update kept it there), so both fired on 57-91% of steps:
# an always-on threat-minimising filter, not a reflex. s4z that way had the
# fewest crossings and the least pain. Fixed (SD floor 0.1 in log space) and
# with an absolute floor catch_tmin 0.02; plus the always-on search on
# purpose (catch_tmax 1e-4, no z) to confirm what RF's s4z measured.
RF2_COMMON = f'{PF_BASE} {PF_IMPACT} {RF_LOG} --env.robot.catch_reflex True'
RF2_Z = '--env.robot.catch_z 3 --env.robot.catch_zlog True --env.robot.catch_tmin 0.02'
RF2_ARMS = [
    ('s4z2', f'{RF2_COMMON} {RF2_Z} --env.robot.catch_mode search'),
    ('g4z2', f'{RF2_COMMON} {RF2_Z} --env.robot.catch_mode slope --env.robot.catch_dmax 0.5'),
    ('s4on', f'{RF2_COMMON} --env.robot.catch_mode search --env.robot.catch_tmax 1e-4'),
]


# Round EM (11-14, emotion definitions; TASKS.md step 3): one sim round with
# the current best definitions on PF's setup (pushes, walls, threat critic
# on cost_obs, 60 min, seeds 811-818, Saion V100). Arousal = Pearce-Hall
# associability (the 11-14 bake-off winner, eta 0.07 per step, z over 60 s;
# now the default, set here explicitly); dominance refreshed ~1 s
# (empower_s 0.3: the 32-step rollout with its 16 ms gaps takes ~0.7 s);
# the CSV keeps the light columns every step and the rest every 16th row.
# a: impact pain at the sim calibration (iota 0.5/1.5, PF3), no brake, the
# catch reflex s4z2 (RF2's pick); b: bumper pain, no brake, no reflex (PF2 d).
EM_LOG = ('--env.robot.sim_phone_threads 1 --env.robot.sim_phone_heavy_every 16 '
          '--env.robot.empower_s 0.3 --env.robot.arousal_mode pearce_hall '
          '--env.robot.arousal_eta 0.07 --env.robot.arousal_z_s 60')
EM_ARMS = [
    ('em_a', f'{PF_BASE} {EM_LOG} {PF_IMPACT} --env.robot.pain_iota0 0.5 '
             f'--env.robot.pain_iota1 1.5 --env.robot.catch_reflex True {RF2_Z} '
             '--env.robot.catch_mode search'),
    ('em_b', f'{PF_BASE} {EM_LOG} --env.robot.pain_mode bumper --env.robot.runaway_s 0'),
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


# Round KF (Sun 11-08, final): the command pick against the robot_command
# base, with the bumper margin of the balance pick (2.9 deg) and 2.0 deg,
# 60 min, n = 8, at each learner speed (P100, V100, IAS A5000/3090).
KF_ARMS = [
    ('base', ''),
    ('pick', KPICK),
    ('pick_m35', f'{KPICK} {MARGIN}'),
    ('pick_m50', f'{KPICK} --env.robot.bumper_margin 0.05'),
]


def cmd_line(exp, gpu, flags, seed, minutes=60, floor=FLOOR):
  return (f'{exp} flat command 65 {gpu} {floor} {PHONE} '
          f'--run.steps {minutes * MIN} {flags} --seed {seed}').replace('  ', ' ')


# Round S (Sun 11-08): the two picks under live conditions and crossed.
# Balance pick (fear + stillness) with the live policy age (1.3 s) and with
# the full live settings; command pick with the fear critic, on the table
# floor and on carpet (where the pick never reached 0.7 in K4).
S_ARMS = [
    ('bal_fs_delay', 'balance', f'{WBASE} {FEAR} {STILL} --env.robot.sim_phone_delay 1.3', 'old', 'table'),
    ('bal_fs_live', 'balance', f'{WBASE} {FEAR} {STILL}', 'live', 'table'),
    ('cmd_fear', 'command', f'{KPICK} {FEAR}', None, 'table'),
    ('cmd_fear_carpet', 'command', f'{KPICK} {FEAR}', None, 'carpet'),
]


# Round C (Sun 11-08): commands never learn on CARPET (K4, S: forward corr
# 0.0, 15-20 falls/min) and balance stands still there (21 mm/s). CARPET
# carries FLOOR's carpet-only scrape (sim_scrape 20, drawn 0-40 per minute)
# on top of the fitted carpet. Is the holdout too harsh, or carpet commands
# broken? The pick on carpet without scrape, on the plain fitted carpet, and
# the robot_command base on the plain carpet (e2234: 0.7 in ~18 min).
CARPET_NS = CARPET.replace('--env.robot.sim_scrape 20', '--env.robot.sim_scrape 0').replace(',scrape:0:40', '')
CARPET_PLAIN = '--env.robot.sim_schedule carpet'
C_ARMS = [
    ('cmd_pick_noscrape', 'command', KPICK, CARPET_NS),
    ('cmd_pick_plain', 'command', KPICK, CARPET_PLAIN),
    ('cmd_base_plain', 'command', '', CARPET_PLAIN),
    ('bal_fs_noscrape', 'balance', f'{WBASE} {FEAR} {STILL}', CARPET_NS),
]


# Round L (Sun 11-08): late collapse beyond the 60 min of BF/KF, on the
# fastest learner (IAS A5000/3090): both picks for 120 min, n = 8.
L_ARMS = [
    ('bal_fs_120', 'balance', f'{WBASE} {FEAR} {STILL}'),
    ('cmd_pick_120', 'command', KPICK),
]


# Round D (Sun 11-08): C showed CARPET (table-fitted drive randomisation on
# carpet) is what breaks commands; on the plain fitted carpet both base and
# pick track (~25 min) but fall ~50/min. Last screens: the stillness dose of
# the balance pick (V100), and three changes against the carpet falls (P100).
D_ARMS_V = [
    ('fs_w1p1', 'balance', f'{WBASE} {FEAR} --env.robot.wheel_penalty 1.0 --env.robot.pos_penalty 1.0', FLOOR),
    ('fs_w15p3', 'balance', f'{WBASE} {FEAR} --env.robot.wheel_penalty 1.5 --env.robot.pos_penalty 3.0', FLOOR),
    ('fs_p2', 'balance', f'{WBASE} {FEAR} --env.robot.pos_penalty 2.0', FLOOR),
]
D_ARMS_P = [
    ('cpl_slow', 'command', f'{KPICK} --env.robot.command_speed 1200 --env.robot.command_turn 1500', CARPET_PLAIN),
    ('cpl_h3', 'command', f'{KPICK} --env.robot.horizon_s 3.0', CARPET_PLAIN),
    ('cpl_bp4', 'command', f'{KPICK} --env.robot.bumper_penalty 4.0', CARPET_PLAIN),
]


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
  elif rnd in ('kfp', 'kfv', 'kfi'):
    gpu = {'kfp': 'p100', 'kfv': 'v100', 'kfi': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, flags in KF_ARMS:
        lines.append(f'# {arm}'); lines.append(cmd_line(f'e{n}', gpu, flags, seed)); n += 1
  elif rnd in ('sp', 'sv', 'si'):
    gpu = {'sp': 'p100', 'sv': 'v100', 'si': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, task, flags, base, floor in S_ARMS:
        fl = CARPET if floor == 'carpet' else FLOOR
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, floor=fl, base=base)
                     if task == 'balance' else cmd_line(f'e{n}', gpu, flags, seed, floor=fl)); n += 1
  elif rnd in ('cp', 'cv'):
    gpu = {'cp': 'p100', 'cv': 'v100'}[rnd]
    seeds = SEEDS4 if rnd == 'cv' else SEEDS8[4:]
    for seed in seeds:
      for arm, task, flags, fl in C_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, floor=fl, base='old')
                     if task == 'balance' else cmd_line(f'e{n}', gpu, flags, seed, floor=fl)); n += 1
  elif rnd == 'li':
    for seed in SEEDS8:
      for arm, task, flags in L_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', 'any', flags, seed, minutes=120, base='old')
                     if task == 'balance' else cmd_line(f'e{n}', 'any', flags, seed, minutes=120)); n += 1
  elif rnd in ('dv', 'dp'):
    gpu, arms = {'dv': ('v100', D_ARMS_V), 'dp': ('p100', D_ARMS_P)}[rnd]
    for seed in SEEDS8:
      for arm, task, flags, fl in arms:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, floor=fl, base='old')
                     if task == 'balance' else cmd_line(f'e{n}', gpu, flags, seed, floor=fl)); n += 1
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
  elif rnd in ('bfp', 'bfv', 'bfi'):
    gpu = {'bfp': 'p100', 'bfv': 'v100', 'bfi': 'any'}[rnd]
    for seed in SEEDS12:
      for arm, base, flags in BF_ARMS:
        lines.append(f'# {arm}'); lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base=base)); n += 1
    for seed in SEEDS4:
      for arm, base, flags, floor in BF_HOLD:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base=base,
                              floor=CARPET if floor == 'carpet' else FLOOR)); n += 1
  elif rnd in ('y', 'x', 'v', 'u', 'ysmoke', 'vsmoke'):
    arms, gpu = {'y': (Y_ARMS, 'v100'), 'x': (X_ARMS, 'any'), 'v': (V_ARMS, 'any'),
                 'u': (U_ARMS, 'p100'),
                 'ysmoke': ([Y_ARMS[2], X_ARMS[3]], 'any'),
                 'vsmoke': ([V_ARMS[1]], 'any')}[rnd]
    seeds, minutes = (SEEDS_Z, 60) if 'smoke' not in rnd else ([1], 5)
    for seed in seeds:
      for arm, flags in arms:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=minutes, base='old')); n += 1
  elif rnd in ('pv', 'pp', 'pi'):
    gpu = {'pv': 'v100', 'pp': 'p100', 'pi': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, flags in P_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base='old')); n += 1
  elif rnd in ('rv', 'rp'):
    gpu = {'rv': 'v100', 'rp': 'p100'}[rnd]
    for seed in SEEDS8:
      for arm, flags in R_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base='old')); n += 1
  elif rnd in ('sv2', 'sp2', 'si2'):
    gpu = {'sv2': 'v100', 'sp2': 'p100', 'si2': 'any'}[rnd]
    for seed in SEEDS8:
      for arm, flags in S2_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base='old')); n += 1
  elif rnd == 'qi':
    for seed in SEEDS_Q:
      for arm, flags in Q_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', 'any', flags, seed, minutes=60, base='old')); n += 1
  elif rnd == 'ni':
    for seed in SEEDS8:
      for arm, flags in N_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', 'any', flags, seed, minutes=60, base='old')); n += 1
  elif rnd in ('mv', 'mp', 'mi'):
    gpu = {'mv': 'v100', 'mp': 'p100', 'mi': 'any'}[rnd]
    for seed in SEEDS12:
      for arm, flags in M_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base='old')); n += 1
  elif rnd in ('z1', 'z2', 'zsmoke'):
    arms, gpu, minutes = {'z1': (Z1_ARMS, 'v100', 60), 'z2': (Z2_ARMS, 'any', 30),
                          'zsmoke': (Z1_ARMS[2:] + Z2_ARMS[1:], 'any', 4)}[rnd]
    seeds = SEEDS_Z if rnd != 'zsmoke' else [1]
    for seed in seeds:
      for arm, flags in arms:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=minutes, base='old')); n += 1
  elif rnd == 'smoke2':
    lines.append(cmd_line(f'e{n}', 'p100', SMOOTH + ' --env.robot.command_grow_s 120', 1, minutes=4)); n += 1
  elif rnd in ('pfsmoke', 'pfsmoke2', 'pfv', 'pfi'):
    gpu = {'pfsmoke': 'v100', 'pfsmoke2': 'v100', 'pfv': 'v100', 'pfi': 'any'}[rnd]
    # Seeds split by cluster, all three arms of a seed on the same GPU type:
    # 811-814 Saion V100, 815-818 IAS.
    seeds, minutes = {'pfsmoke': ([1], 4), 'pfsmoke2': ([1], 4), 'pfv': (SEEDS4, 60),
                      'pfi': (SEEDS8[4:], 60)}[rnd]
    arms = PF_ARMS[2:] if rnd == 'pfsmoke2' else PF_ARMS
    for seed in seeds:
      for arm, flags in arms:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=minutes, base='old')); n += 1
  elif rnd in ('rfsmoke', 'rfsmoke2', 'rfv', 'rfi'):
    # As PF: seeds 811-814 Saion V100, 815-818 IAS (pairs with PF b and c).
    gpu = {'rfsmoke': 'p100', 'rfsmoke2': 'p100', 'rfv': 'v100', 'rfi': 'any'}[rnd]
    seeds, minutes = {'rfsmoke': ([1], 8), 'rfsmoke2': ([1], 8), 'rfv': (SEEDS4, 60),
                      'rfi': (SEEDS8[4:], 60)}[rnd]
    arms = RF_ARMS[2:] if rnd == 'rfsmoke' else RF_ARMS
    for seed in seeds:
      for arm, flags in arms:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=minutes, base='old')); n += 1
  elif rnd in ('rf2v', 'rf2i'):
    gpu, seeds = {'rf2v': ('v100', SEEDS4), 'rf2i': ('any', SEEDS8[4:])}[rnd]
    for seed in seeds:
      for arm, flags in RF2_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', gpu, flags, seed, minutes=60, base='old')); n += 1
  elif rnd in ('emsmoke', 'emv'):
    seeds, minutes = {'emsmoke': ([1], 4), 'emv': (SEEDS8, 60)}[rnd]
    for seed in seeds:
      for arm, flags in EM_ARMS:
        lines.append(f'# {arm}')
        lines.append(bal_line(f'e{n}', 'v100', flags, seed, minutes=minutes, base='old')); n += 1
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
