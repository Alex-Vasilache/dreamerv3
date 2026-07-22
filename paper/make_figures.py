#!/usr/bin/env python3
"""Regenerate the paper's figures from run logs (scores.jsonl).

Run with rl_env (python 3.7 + matplotlib):
  source /etc/profile.d/modules.sh && module load python/3.7.3
  source /apps/unit/DoyaU/vasilache/apps/rl_env/bin/activate
  python3 make_figures.py

Reads archived runs from the bucket, falling back to /work for runs still
training or paused there (not yet archived); add new curves by extending FIGS.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
WORK = '/work/DoyaU/vasilache/work'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')

# Okabe & Ito (2008) colorblind-safe palette. Used by semantic role so the
# same color means the same thing across figures: BLUE = Director baseline /
# reference, GREEN = primary new mechanism (best-case / free), VERMILLION =
# secondary or combined-constraint variant, ORANGE = tertiary variant.
OI_BLACK = '#000000'
OI_ORANGE = '#E69F00'
OI_SKYBLUE = '#56B4E9'
OI_GREEN = '#009E73'
OI_YELLOW = '#F0E442'
OI_BLUE = '#0072B2'
OI_VERMILLION = '#D55E00'
OI_PURPLE = '#CC79A7'

# fig name -> list of (run dir basename, label, color, linestyle)
FIGS = {
    'hopper_vark': {
        'title': 'Hopper Hop (pixels, 4M steps)',
        'runs': [
            ('e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29',
             'Director baseline, fixed $K{=}8$ (e124)', OI_BLUE, '-'),
            ('e169_dmc_hopper_hop_plain_vark_BIG_20260710_171509_NzeEmw',
             'variable $K$, target 4 (e169)', OI_VERMILLION, '-'),
            ('e173_dmc_hopper_hop_vargoal_masked_durreg_soft4_BIG_20260710_171539_W0Iadt',
             'masked + variable $K$, target 4 (e173)', OI_ORANGE, '--'),
        ],
    },
    'cartpole_components': {
        'title': 'Cartpole Swingup (pixels, 4M steps)',
        'runs': [
            ('e171_dmc_cartpole_swingup_vargoal_masked_durreg_soft4_BIG_20260710_171541_Nsxo4i',
             'combined: mask + var-$K$ + struct (e171)', OI_BLUE, '-'),
            ('e163_dmc_cartpole_swingup_mask_fixedk_BIG_20260710_171339_xsD1bb',
             'mask only, fixed $K{=}8$ (e163)', OI_VERMILLION, '-'),
            ('e167_dmc_cartpole_swingup_plain_vark_BIG_20260710_171509_YrJtUh',
             'var-$K$ only, target 4 (e167)', OI_ORANGE, '--'),
        ],
    },
    # Single-head (joint) masked manager, free (S0R0) and struct+ratchet
    # (S1R1) variants, vs. their matched Director baseline -- one panel per
    # env x scale (F17/Finding 6). 2026-07-15: e212/e213/e214/e217/e218/e221/
    # e222/e226 are still training (read from /work via the BUCKET->WORK
    # fallback in load_scores); e225/e229 (BIG S1R1) were cancelled+archived
    # 07-15, so their curves are final/complete. Re-run this script once the
    # live runs finish and are archived to refresh from final data.
    'cheetah_singlehead_small': {
        'title': 'Cheetah Run, small scale (pixels, 4M steps)',
        'runs': [
            ('e212_dmc_cheetah_run_director_j4662391',
             'Director baseline, fixed $K{=}8$ (e212)', OI_BLUE, '-'),
            ('e214_dmc_cheetah_run_joint_fixedk_j4662393',
             'single-head mask, free (e214)', OI_GREEN, '-'),
            ('e217_dmc_cheetah_run_joint_fixedk_j4662396',
             'single-head mask, struct+ratchet (e217)', OI_VERMILLION, '--'),
        ],
    },
    'cheetah_singlehead_big': {
        'title': 'Cheetah Run, BIG scale (pixels, 4M steps)',
        'runs': [
            ('e191_dmc_cheetah_run_director_baseline_j4660310',
             'Director baseline, fixed $K{=}8$ (e191)', OI_BLUE, '-'),
            ('e222_dmc_cheetah_run_joint_fixedk_BIG_j4662401',
             'single-head mask, free (e222)', OI_GREEN, '-'),
            ('e225_dmc_cheetah_run_joint_fixedk_BIG_j4662404',
             'single-head mask, struct+ratchet (e225)', OI_VERMILLION, '--'),
        ],
    },
    'hopper_singlehead_small': {
        'title': 'Hopper Hop, small scale (pixels, 4M steps)',
        'runs': [
            ('e213_dmc_hopper_hop_director_j4662392',
             'Director baseline, fixed $K{=}8$ (e213)', OI_BLUE, '-'),
            ('e218_dmc_hopper_hop_joint_fixedk_j4662397',
             'single-head mask, free (e218)', OI_GREEN, '-'),
            ('e221_dmc_hopper_hop_joint_fixedk_j4662400',
             'single-head mask, struct+ratchet (e221)', OI_VERMILLION, '--'),
        ],
    },
    'hopper_singlehead_big': {
        'title': 'Hopper Hop, BIG scale (pixels, 4M steps)',
        'runs': [
            ('e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29',
             'Director baseline, fixed $K{=}8$ (e124)', OI_BLUE, '-'),
            ('e226_dmc_hopper_hop_joint_fixedk_BIG_j4662405',
             'single-head mask, free (e226)', OI_GREEN, '-'),
            ('e229_dmc_hopper_hop_joint_fixedk_BIG_j4662408',
             'single-head mask, struct+ratchet (e229)', OI_VERMILLION, '--'),
        ],
    },
    # Mechanism 4 (goal-space reuse, \S sec:goalreuse) at scale: original
    # target 0.95 (e258/e262, F19, both collapsed) vs. the 2026-07-22 retest
    # at target 0.8 (e278/e280, interim -- still training on /work).
    'hopper_reuse_cont': {
        'title': 'Hopper Hop, BIG, goal-space reuse (4M steps)',
        'runs': [
            ('e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29',
             'Director baseline, fixed $K{=}8$ (e124)', OI_BLUE, '-'),
            ('e278_dmc_hopper_hop_director_BIG_j4668382',
             'reuse target 0.8, interim (e278)', OI_GREEN, '-'),
            ('e258_dmc_hopper_hop_director_BIG_j4664904',
             'reuse target 0.95 (e258)', OI_VERMILLION, '--'),
        ],
    },
    'cheetah_reuse_cont': {
        'title': 'Cheetah Run, BIG, goal-space reuse (4M steps)',
        'runs': [
            ('e123_dmc_cheetah_run_director_baseline_20260706_155117_Vk7oa4',
             'Director baseline, fixed $K{=}8$ (e123)', OI_BLUE, '-'),
            ('e280_dmc_cheetah_run_director_BIG_j4668384',
             'reuse target 0.8, interim (e280)', OI_GREEN, '-'),
            ('e262_dmc_cheetah_run_director_BIG_j4664908',
             'reuse target 0.95 (e262)', OI_VERMILLION, '--'),
        ],
    },
    # Mechanism 5 (block-overlap reuse, \S sec:softreuse) at scale: original
    # target 0.7 (e266/e270, F19) vs. the 2026-07-22 retest at target 0.5
    # (e286/e290, interim -- still training on /work). Both legs struct+ratchet.
    'hopper_reuse_block': {
        'title': 'Hopper Hop, BIG, block-overlap reuse (4M steps)',
        'runs': [
            ('e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29',
             'Director baseline, fixed $K{=}8$ (e124)', OI_BLUE, '-'),
            ('e286_dmc_hopper_hop_director_BIG_j4668390',
             'overlap target 0.5, interim (e286)', OI_GREEN, '-'),
            ('e266_dmc_hopper_hop_director_BIG_j4664964',
             'overlap target 0.7 (e266)', OI_VERMILLION, '--'),
        ],
    },
    'cheetah_reuse_block': {
        'title': 'Cheetah Run, BIG, block-overlap reuse (4M steps)',
        'runs': [
            ('e123_dmc_cheetah_run_director_baseline_20260706_155117_Vk7oa4',
             'Director baseline, fixed $K{=}8$ (e123)', OI_BLUE, '-'),
            ('e290_dmc_cheetah_run_director_BIG_j4668394',
             'overlap target 0.5, interim (e290)', OI_GREEN, '-'),
            ('e270_dmc_cheetah_run_director_BIG_j4664966',
             'overlap target 0.7 (e270)', OI_VERMILLION, '--'),
        ],
    },
}

SMOOTH = 20  # trailing episodes


def load_scores(run):
    path = os.path.join(BUCKET, run, 'logdir', 'scores.jsonl')
    if not os.path.isfile(path):
        work_path = os.path.join(WORK, run, 'logdir', 'scores.jsonl')
        if os.path.isfile(work_path):
            path = work_path
    steps, scores = [], []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('episode/score') is not None:
                steps.append(row['step'])
                scores.append(row['episode/score'])
    smoothed = []
    for i in range(len(scores)):
        w = scores[max(0, i - SMOOTH + 1):i + 1]
        smoothed.append(sum(w) / len(w))
    return steps, smoothed


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, spec in FIGS.items():
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for run, label, color, ls in spec['runs']:
            try:
                steps, scores = load_scores(run)
            except IOError as e:
                print('skip %s: %s' % (run, e))
                continue
            ax.plot([s / 1e6 for s in steps], scores,
                    color=color, ls=ls, lw=1.3, label=label)
        ax.set_xlabel('environment steps (millions)')
        ax.set_ylabel('episode return (smoothed)')
        ax.set_title(spec['title'], fontsize=10)
        ax.legend(fontsize=6.5, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        out = os.path.join(OUT, name + '.pdf')
        fig.savefig(out)
        print('wrote', out)


if __name__ == '__main__':
    main()
