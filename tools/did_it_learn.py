"""Ask whether a robot run actually learned, rather than eyeballing scores.

Episode score on a shaped reward is noisy and drifts with the hardware, so this
reports a lower-variance behavioural metric alongside it and compares the start
of the run against the end. The policy starts effectively random, so the first
quintile is a usable within-run baseline that shares the same rig conditions.

  .venv/bin/python tools/did_it_learn.py ~/logdir/latest [tolerance_deg]
"""
import glob
import sys

import numpy as np

THETA_ZERO_DEG = 2.69


def load(logdir):
    files = sorted(glob.glob(f'{logdir}/online_shared/experience/*.npz'))
    if not files:
        files = sorted(glob.glob(f'{logdir}/replay/*.npz'))
    if not files:
        raise SystemExit(f'no replay chunks under {logdir}')
    o, r, f = [], [], []
    for path in files:
        d = np.load(path)
        o.append(d['orientation']); r.append(d['reward']); f.append(d['is_first'])
    o, r, f = map(np.concatenate, (o, r, f))
    return np.degrees(np.arctan2(o[:, 0], o[:, 1])), r, f, o[:, 2]


def episodes(tilt, reward, isfirst, tol):
    starts = list(np.where(isfirst)[0]) or [0]
    out = []
    for a, b in zip(starts, starts[1:] + [len(tilt)]):
        if b - a < 20:
            continue
        err = np.abs(tilt[a:b] - THETA_ZERO_DEG)
        out.append((reward[a:b].sum(), (err < tol).mean(), err.mean(), b - a))
    return np.array(out)


def compare(name, early, late):
    from scipy import stats as st
    d = late.mean() - early.mean()
    se = np.hypot(early.std() / np.sqrt(len(early)), late.std() / np.sqrt(len(late)))
    try:
        p = st.mannwhitneyu(early, late, alternative='two-sided').pvalue
    except Exception:
        p = float('nan')
    print(f'  {name:<22} first {early.mean():8.3f}   last {late.mean():8.3f}   '
          f'change {d:+8.3f} +/- {se:.3f}   p={p:.4f}'
          f'{"  *" if p < 0.05 else ""}')


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else str(np.datetime64('now'))
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    tilt, reward, isfirst, omega = load(logdir)
    eps = episodes(tilt, reward, isfirst, tol)
    print(f'{logdir}\n{len(tilt)} steps, {len(eps)} episodes, '
          f'tolerance +/-{tol:.1f} deg around upright\n')
    if len(eps) < 10:
        raise SystemExit('need at least 10 episodes to say anything')
    k = max(2, len(eps) // 5)
    print(f'first {k} episodes vs last {k}:')
    for i, name in enumerate(
            ['episode score', 'frac within tol', 'mean |tilt err| deg']):
        compare(name, eps[:k, i], eps[-k:, i])
    print()
    # Persistence baseline: if the tilt is this predictable from its own past,
    # a world model that cannot beat it has learned nothing about the dynamics.
    for h in (5, 10, 25):
        pers = np.abs(tilt[h:] - tilt[:-h]).mean()
        lin = np.abs(tilt[h:] - (tilt[:-h] + np.degrees(omega[:-h]) * h / 48.3)).mean()
        print(f'  tilt {h:2d}-step ahead: persistence err {pers:6.3f} deg   '
              f'linear-extrapolation err {lin:6.3f} deg')
    print('\n  A learned world model should beat both of these on the same')
    print('  horizons; that is the earliest signal, well before scores move.')


if __name__ == '__main__':
    main()
