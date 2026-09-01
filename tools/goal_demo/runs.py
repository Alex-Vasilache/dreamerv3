"""Which checkpoint each panel of the goal-code demo shows.

One environment, two arms, one seed each: the better-scoring seed of the pair we
ran, measured over the 300k env steps that end at the 3M checkpoint the demo
loads. Scores here are recorded for display; ``pick_runs.py`` recomputes them
from ``metrics.jsonl`` and will complain if a seed choice has gone stale.
"""
import os

WORK = '/work/DoyaU/vasilache/work'
STEP = '000003000000'

# task -> arm -> (run dir basename, experiment tag, seed, score at 3M)
# Scores are mean episode return over env steps 2.7M-3.0M.
RUNS = {
    'dmc_cartpole_swingup': {
        'label': 'Cartpole Swingup',
        'director': ('e505_dmc_cartpole_swingup_director_s3_BIG_j4676886',
                     'e505', 3, 800.2),
        'ours': ('e592_dmc_cartpole_swingup_som_lipvq_line_relu_gaussian_s0'
                 '_BIG_j4697649', 'e592', 0, 817.0),
    },
    'dmc_hopper_stand': {
        'label': 'Hopper Stand',
        'director': ('e509_dmc_hopper_stand_director_s3_BIG_j4676890',
                     'e509', 3, 817.7),
        'ours': ('e597_dmc_hopper_stand_som_lipvq_line_relu_gaussian_s1'
                 '_BIG_j4697654', 'e597', 1, 814.0),
    },
    'dmc_cheetah_run': {
        'label': 'Cheetah Run',
        'director': ('e555_dmc_cheetah_run_director_s1_BIG_j4677823',
                     'e555', 1, 589.6),
        'ours': ('e594_dmc_cheetah_run_som_lipvq_line_relu_gaussian_s0'
                 '_BIG_j4697651', 'e594', 0, 628.9),
    },
    'dmc_hopper_hop': {
        'label': 'Hopper Hop',
        'director': ('e566_dmc_hopper_hop_director_s0_BIG_j4693744',
                     'e566', 0, 332.3),
        'ours': ('e595_dmc_hopper_hop_som_lipvq_line_relu_gaussian_s0'
                 '_BIG_j4697652', 'e595', 0, 275.9),
    },
}

TASKS = list(RUNS.keys())
ARMS = ('director', 'ours')
ARM_LABELS = {'director': 'Director', 'ours': 'Ours (SOM-line + LipVQ)'}


def run_dir(task, arm):
  return os.path.join(WORK, RUNS[task][arm][0])


def ckpt_path(task, arm, step=STEP):
  return os.path.join(run_dir(task, arm), 'logdir', 'ckpt_milestones', step,
                      'agent.pkl')


def config_path(task, arm):
  return os.path.join(run_dir(task, arm), 'logdir', 'config.yaml')


def info(task, arm):
  name, tag, seed, score = RUNS[task][arm]
  return dict(run=name, exp=tag, seed=seed, score=score,
              label=ARM_LABELS[arm], task_label=RUNS[task]['label'])
