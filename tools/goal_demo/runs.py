"""Which checkpoint each panel of the goal-code demo shows.

Three arms on the matched `OG` protocol (6 tasks x 5 seeds each):

  og       director director_og pubmatch                          e1230-e1289
  som      ... goal_som_line ...                                  e1330-e1419
  somlip   ... goal_som_lipvq_line_prod lip_relu goal_vq_f32 ...  e1330-e1419
  sommask  ... goal_som_line goal_mask8 ...                       e1630-e1659

One seed per arm per environment: the best one over the 300k env steps ending
at the 4M checkpoint the demo loads. ``pick_runs.py`` recomputes the choice
(``--emit`` prints a fresh ``RUNS`` block) and flags stale pins.
"""
import os

WORK = '/work/DoyaU/vasilache/work'
BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
STEP = '000004000000'

ARMS = ('og', 'som', 'somlip', 'sommask')
ARM_LABELS = {'og': 'Director (OG)', 'som': 'OG + SOM-line',
              'somlip': 'OG + SOM-line + LiP',
              'sommask': 'OG + SOM-line + mask'}
# Same colours as the comparison plots (collapse_study/compare_late.py).
ARM_COLORS = {'og': '#E69F00', 'som': '#CC79A7', 'somlip': '#0072B2',
              'sommask': '#56B4E9'}
PATTERNS = {
    'og': 'e12[3-8]?_{t}_director-director_og-pubmatch_s*',
    'som': 'e1[34]??_{t}_director-director_og-goal_som_line-pubmatch_s*',
    'somlip': 'e1[34]??_{t}_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s*',
    'sommask': 'e16[3-5]?_{t}_director-director_og-goal_som_line-goal_mask8-pubmatch_s*',
}
TASK_LABELS = {
    'pinpad_four': 'Pin Pad Four', 'pinpad_five': 'Pin Pad Five',
    'pinpad_six': 'Pin Pad Six', 'dmc_cartpole_swingup': 'Cartpole Swingup',
    'dmc_cheetah_run': 'Cheetah Run', 'dmc_hopper_hop': 'Hopper Hop',
}

# task -> arm -> (run dir basename, score over 3.7M-4.0M). From pick_runs.py --emit.
RUNS = {
    'pinpad_four': {
        'og': ('e1254_pinpad_four_director-director_og-pubmatch_s2_j4745769_24', 278.2),
        'som': ('e1366_pinpad_four_director-director_og-goal_som_line-pubmatch_s2_j4751562_36', 284.8),
        'somlip': ('e1378_pinpad_four_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s2_j4751562_48', 281.8),
        'sommask': ('e1630_pinpad_four_director-director_og-goal_som_line-goal_mask8-pubmatch_s0_j4763638_30', 306.2),
    },
    'pinpad_five': {
        'og': ('e1255_pinpad_five_director-director_og-pubmatch_s2_j4745769_25', 225.9),
        'som': ('e1367_pinpad_five_director-director_og-goal_som_line-pubmatch_s2_j4751562_37', 226.3),
        'somlip': ('e1415_pinpad_five_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s4_j4751562_85', 223.7),
        'sommask': ('e1631_pinpad_five_director-director_og-goal_som_line-goal_mask8-pubmatch_s0_j4763638_31', 224.5),
    },
    'pinpad_six': {
        'og': ('e1256_pinpad_six_director-director_og-pubmatch_s2_j4745769_26', 176.3),
        'som': ('e1386_pinpad_six_director-director_og-goal_som_line-pubmatch_s3_j4751562_56', 174.3),
        'somlip': ('e1416_pinpad_six_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s4_j4751562_86', 164.7),
        'sommask': ('e1644_pinpad_six_director-director_og-goal_som_line-goal_mask8-pubmatch_s2_j4763638_44', 171.8),
    },
    'dmc_cartpole_swingup': {
        'og': ('e1245_dmc_cartpole_swingup_director-director_og-pubmatch_s1_j4745769_15', 753.4),
        'som': ('e1351_dmc_cartpole_swingup_director-director_og-goal_som_line-pubmatch_s1_j4751562_21', 801.5),
        'somlip': ('e1399_dmc_cartpole_swingup_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s3_j4751562_69', 836.6),
        'sommask': ('e1657_dmc_cartpole_swingup_director-director_og-goal_som_line-goal_mask8-pubmatch_s4_j4763638_57', 717.9),
    },
    'dmc_cheetah_run': {
        'og': ('e1246_dmc_cheetah_run_director-director_og-pubmatch_s1_j4745769_16', 532.2),
        'som': ('e1370_dmc_cheetah_run_director-director_og-goal_som_line-pubmatch_s2_j4751562_40', 625.4),
        'somlip': ('e1382_dmc_cheetah_run_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s2_j4751562_52', 625.2),
        'sommask': ('e1646_dmc_cheetah_run_director-director_og-goal_som_line-goal_mask8-pubmatch_s2_j4763638_46', 575.5),
    },
    'dmc_hopper_hop': {
        'og': ('e1259_dmc_hopper_hop_director-director_og-pubmatch_s2_j4745769_29', 446.3),
        'som': ('e1389_dmc_hopper_hop_director-director_og-goal_som_line-pubmatch_s3_j4751562_59', 428.7),
        'somlip': ('e1419_dmc_hopper_hop_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s4_j4751562_89', 420.8),
        'sommask': ('e1659_dmc_hopper_hop_director-director_og-goal_som_line-goal_mask8-pubmatch_s4_j4763638_59', 409.4),
    },
}

TASKS = [t for t in TASK_LABELS if t in RUNS]


def run_dir(task, arm):
  name = RUNS[task][arm][0]
  for root in (WORK, BUCKET):
    path = os.path.join(root, name)
    if os.path.isdir(path):
      return path
  raise FileNotFoundError(name)


def ckpt_path(task, arm, step=STEP):
  return os.path.join(run_dir(task, arm), 'logdir', 'ckpt_milestones', step,
                      'agent.pkl')


def config_path(task, arm):
  return os.path.join(run_dir(task, arm), 'logdir', 'config.yaml')


def info(task, arm):
  name, score = RUNS[task][arm]
  seed = int(name.split('_s')[-1].split('_')[0])
  return dict(run=name, exp=name.split('_')[0], seed=seed, score=score,
              label=ARM_LABELS[arm], task_label=TASK_LABELS[task])
