#!/usr/bin/env python3
"""Regenerate the paper's figures from run logs (scores.jsonl).

Run with rl_env (python 3.7 + matplotlib):
  source /etc/profile.d/modules.sh && module load python/3.7.3
  source /apps/unit/DoyaU/vasilache/apps/rl_env/bin/activate
  python3 make_figures.py

Reads archived runs from the bucket; add new curves by extending RUNS.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')

# fig name -> list of (run dir basename, label, color, linestyle)
FIGS = {
    'hopper_vark': {
        'title': 'Hopper Hop (pixels, 4M steps)',
        'runs': [
            ('e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29',
             'Director baseline, fixed $K{=}8$ (e124)', '#1a6faf', '-'),
            ('e169_dmc_hopper_hop_plain_vark_BIG_20260710_171509_NzeEmw',
             'variable $K$, target 4 (e169)', '#c23b22', '-'),
            ('e173_dmc_hopper_hop_vargoal_masked_durreg_soft4_BIG_20260710_171539_W0Iadt',
             'masked + variable $K$, target 4 (e173)', '#8c6d31', '--'),
        ],
    },
    'cartpole_components': {
        'title': 'Cartpole Swingup (pixels, 4M steps)',
        'runs': [
            ('e171_dmc_cartpole_swingup_vargoal_masked_durreg_soft4_BIG_20260710_171541_Nsxo4i',
             'combined: mask + var-$K$ + struct (e171)', '#1a6faf', '-'),
            ('e163_dmc_cartpole_swingup_mask_fixedk_BIG_20260710_171339_xsD1bb',
             'mask only, fixed $K{=}8$ (e163)', '#c23b22', '-'),
            ('e167_dmc_cartpole_swingup_plain_vark_BIG_20260710_171509_YrJtUh',
             'var-$K$ only, target 4 (e167)', '#8c6d31', '--'),
        ],
    },
}

SMOOTH = 20  # trailing episodes


def load_scores(run):
    path = os.path.join(BUCKET, run, 'logdir', 'scores.jsonl')
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
