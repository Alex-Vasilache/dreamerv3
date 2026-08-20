#!/usr/bin/env python3
"""Code similarity -> goal-state similarity, on the code's own 9 bins.

The measurement the paragraph describes: sample ``n_codes`` hard codes, pair
each with a partner at every one of the L+1 = 9 similarities the code admits
(0 blocks different through all 8 different), decode both, and record the
``cosine_max`` similarity of the generated goal states -- the quantity the
low-level policy is actually rewarded on.

Sampling (``--sampling``):

  uniform   (default) ``n_codes`` codes drawn uniformly over the C^L grid --
            every block an independent uniform class. This measures the
            decoder's geometry over the whole code space the manager can
            address, and needs no environment at all.
  encoded   codes obtained by encoding real world-model states from a live
            rollout, i.e. the codes the manager actually emits on-policy. A
            different question: the geometry of the region the encoder uses.

Either way partners are *constructed* at each exact Hamming distance rather
than drawn as random pairs -- random pairs of uniform codes land almost
entirely in the 7- and 8-block bins and would leave the near end of the axis
empty. One partner per base code per bin, so every bin gets exactly
``n_codes`` pairs.

m=0 is a self-pair, so cosine_max is exactly 1 by construction and serves as the
sanity check on the whole path.

  python -u code_sim_to_state_sim.py --run_dir /work/.../e502_... --out out/e502
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

import numpy as np

from diag_goal_geometry import build, make_fns, collect_states


def cosine_max(x, y):
    """cosine_max(x, y) = (x/m).(y/m), m = max(|x|, |y|) -- rowwise."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    nx = np.linalg.norm(x, axis=-1)
    ny = np.linalg.norm(y, axis=-1)
    m = np.maximum(nx, ny) + 1e-12
    return (x * y).sum(-1) / (m * m)


def onehot_of(ids, classes):
    return np.eye(classes, dtype=np.float32)[ids]


def partners(base_ids, m, classes, rng):
    """For each row, move exactly m distinct blocks to a uniformly random OTHER
    class. Vectorised: argsort of a random matrix picks m distinct blocks per
    row, and adding (new >= old) skips the class the block already had. The
    per-pair Python loop this replaces was fine at 1024 pairs a bin and takes
    hours at a million."""
    N, L = base_ids.shape
    sel = np.argsort(rng.random((N, L)), axis=1)[:, :m]
    old = np.take_along_axis(base_ids, sel, axis=1)
    newc = rng.integers(0, classes - 1, size=(N, m))
    newc += (newc >= old)
    out = base_ids.copy()
    np.put_along_axis(out, sel, newc, axis=1)
    return out


def run(run_dir, out_path, n_codes=1024, n_partners=1024, n_envs=4, stride=8,
        seed=0, sampling='uniform', chunk=32768, ckpt=None):
    # ckpt points at a logdir/ckpt_milestones/<step>/ directory, so the same
    # measurement can be taken at 1M/2M/3M/4M rather than only at the end.
    config, agent = build(run_dir, ckpt)
    impl, encode, decode = make_fns(config, agent)
    blocks, classes = [int(x) for x in config.agent.skill_shape]
    name = pathlib.Path(str(run_dir).rstrip('/')).name
    print(f'=== {name} | impl={impl} L={blocks} C={classes} | '
          f'task={config.task} | sampling={sampling}')

    rng = np.random.default_rng(seed)
    if sampling == 'uniform':
        # Uniform over the C^L grid: each block an independent uniform class.
        # No environment needed -- this only touches the decoder.
        ref_ids = rng.integers(0, classes, size=(n_codes, blocks))
        onehot = onehot_of(ref_ids, classes)
    else:
        deters = collect_states(config, agent, n_envs, stride, n_codes)
        _, _, _, onehot, _ = encode(deters[:n_codes])
        onehot = np.asarray(onehot)
        ref_ids = onehot.argmax(-1)
    ref_goal = decode(onehot)
    print(f'{len(onehot)} hard codes, goal dim {ref_goal.shape[-1]}')

    M = len(onehot)
    sim = np.full((M, blocks + 1), np.nan)
    sim[:, 0] = 1.0                     # self-pair, exactly 1 by construction
    # n_partners partners for every base code at every distance. Decoding is
    # chunked: M * n_partners is ~1e6 goals a bin, and one batch of that would
    # be several GB of activations.
    per_chunk = max(1, chunk // n_partners)
    print(f'{M} base codes x {n_partners} partners x {blocks} bins = '
          f'{M * n_partners * blocks:,} decodes, {per_chunk} bases per batch')
    # Spread WITHIN a run, at fixed code distance. The per-base mean hides it:
    # two pairs the same number of blocks apart can land at very different goal
    # similarities, and that variation is the manager's real uncertainty about
    # what an edit will do. Decomposed by the law of total variance into
    # between-base (which code you start from) and within-base (which partner
    # you land on).
    sim_sd = np.full((M, blocks + 1), 0.0)      # per-base std over partners
    pair_q = np.full((blocks + 1, 5), np.nan)   # p5/p25/p50/p75/p95 over pairs
    pair_sd = np.full(blocks + 1, 0.0)
    for m in range(1, blocks + 1):
        acc = np.zeros(M)
        accsd = np.zeros(M)
        allv = np.empty(M * n_partners, np.float64)
        for lo in range(0, M, per_chunk):
            hi = min(lo + per_chunk, M)
            base = np.repeat(ref_ids[lo:hi], n_partners, axis=0)
            pids = partners(base, m, classes, rng)
            assert ((pids != base).sum(1) == m).all()
            goals = decode(onehot_of(pids, classes))
            ref = np.repeat(ref_goal[lo:hi], n_partners, axis=0)
            v = cosine_max(goals, ref).reshape(hi - lo, n_partners)
            acc[lo:hi] = v.mean(-1)
            accsd[lo:hi] = v.std(-1)
            allv[lo * n_partners:hi * n_partners] = v.ravel()
        sim[:, m] = acc
        sim_sd[:, m] = accsd
        pair_sd[m] = allv.std()
        pair_q[m] = np.percentile(allv, [5, 25, 50, 75, 95])
    pair_q[0] = 1.0

    curve = sim.mean(0)
    between = sim.std(0)                       # across base codes
    within = np.sqrt((sim_sd ** 2).mean(0))    # across partners, pooled
    print('  blocks different -> goal-state cosine_max')
    print('   ' + '  '.join(f'{m}:{v:.4f}' for m, v in enumerate(curve)))
    print('  spread over all pairs (std)')
    print('   ' + '  '.join(f'{m}:{v:.4f}' for m, v in enumerate(pair_sd)))
    print('  decomposed: between-base / within-base')
    print('   ' + '  '.join(f'{m}:{b:.3f}/{w:.3f}'
                            for m, (b, w) in enumerate(zip(between, within))))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    step_tag = pathlib.Path(str(ckpt).rstrip('/')).name if ckpt else 'final'
    np.savez_compressed(str(out) + '.npz', sim=sim, curve=curve,
                        blocks=np.arange(blocks + 1), task=str(config.task),
                        run=name, n_codes=M, n_partners=n_partners,
                        sampling=sampling, sim_sd=sim_sd, pair_sd=pair_sd,
                        pair_q=pair_q, between=between, within=within,
                        ckpt_step=step_tag)
    print('wrote', str(out) + '.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_codes', type=int, default=1024)
    p.add_argument('--n_partners', type=int, default=1024)
    p.add_argument('--chunk', type=int, default=32768)
    p.add_argument('--sampling', choices=['uniform', 'encoded'],
                   default='uniform')
    p.add_argument('--n_envs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt', default=None,
                   help='a ckpt_milestones/<step>/ dir; omit for the final ckpt')
    a = p.parse_args()
    run(a.run_dir, a.out, a.n_codes, a.n_partners, a.n_envs, seed=a.seed,
        sampling=a.sampling, chunk=a.chunk, ckpt=a.ckpt)


if __name__ == '__main__':
    main()
