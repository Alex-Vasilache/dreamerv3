#!/usr/bin/env python3
"""How much state space can the goal codes actually address, and how smoothly?

Three questions, one pass over a checkpoint. All of them live in the SAME
1024-dim ``deter`` space, which is what makes them comparable: the goal decoder
emits a deter vector, and the worker's reward is the similarity between that
vector and the deter the environment actually reaches.

  SIZE          Take the whole code grid the manager can address (C^L = 8^8 =
                16.7M codes; we sample it uniformly) and decode it. How large a
                region of deter space does that cloud occupy, next to the cloud
                of states the policy actually visits? Reported as effective
                dimension (participation ratio of the covariance spectrum,
                (sum L)^2 / sum L^2) and as RMS radius about the mean. A code
                space that decodes into a small ball inside a much larger state
                cloud cannot ask for most of the states that exist.

  OVERLAP       Do those decoded goals correspond to states the environment can
                actually be in? Two directions, both as nearest-neighbour
                cosine_max:
                  coverage      real state -> nearest decoded goal. "Can the
                                manager ask for the states we visit?"
                  realizability decoded goal -> nearest real state. "Is what it
                                asks for a state at all?"
                Both are meaningless without a scale, so we also report the
                real-to-real nearest-neighbour similarity: that is how close two
                genuinely different visited states are, and it is the number
                coverage has to be read against.

  SMOOTHNESS    Change ONE block of a code and measure how far the decoded goal
                moves, relative to the full spread of the goal cloud. A smooth
                map moves a little and moves consistently; a jumpy one has a
                heavy tail, so a single manager edit can teleport the target.
                Reported as the relative step (step / RMS pairwise distance
                within the goal cloud) and its dispersion.

  python -u code_space_extent.py --run_dir /work/.../e502_... --out out/e502
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

import numpy as np

from code_sim_to_state_sim import cosine_max, onehot_of
from diag_goal_geometry import build, make_fns, collect_states


def decode_all(decode, ids, classes, chunk):
    out = []
    for lo in range(0, len(ids), chunk):
        out.append(np.asarray(
            decode(onehot_of(ids[lo:lo + chunk], classes)), np.float64))
    return np.concatenate(out, 0)


def spectrum(x):
    """Covariance eigenvalues, effective dimension, RMS radius."""
    x = np.asarray(x, np.float64)
    c = x - x.mean(0, keepdims=True)
    # D = 1024 and N = 4096, so the DxD covariance is the smaller matrix; the
    # NxN Gram would carry the same nonzero spectrum but cost 16x more.
    cov = (c.T @ c) / len(c)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)[::-1]
    tot = ev.sum()
    pr = (tot ** 2) / max((ev ** 2).sum(), 1e-300)   # participation ratio
    return ev, float(pr), float(np.sqrt(tot))


def nn_cosmax(a, b, chunk=512, exclude_self=False):
    """For each row of a, the max cosine_max to any row of b."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    # exclude_self blanks entry (i, lo+i), which is only the diagonal when the
    # two sets are the same array in the same order.
    assert not exclude_self or a.shape == b.shape, (a.shape, b.shape)
    nb = np.linalg.norm(b, axis=-1)
    out = np.empty(len(a))
    for lo in range(0, len(a), chunk):
        blk = a[lo:lo + chunk]
        na = np.linalg.norm(blk, axis=-1)
        dot = blk @ b.T
        m = np.maximum(na[:, None], nb[None, :]) + 1e-12
        s = dot / (m * m)
        if exclude_self:
            for i in range(len(blk)):
                s[i, lo + i] = -np.inf
        out[lo:lo + chunk] = s.max(1)
    return out


def rms_pairwise(x, rng, n=4000):
    """RMS distance between random pairs -- the goal cloud's own length scale."""
    x = np.asarray(x, np.float64)
    i = rng.integers(0, len(x), n)
    j = rng.integers(0, len(x), n)
    keep = i != j
    d = np.linalg.norm(x[i[keep]] - x[j[keep]], axis=-1)
    return float(np.sqrt((d ** 2).mean())), d


def one_block_steps(ids, decoded, decode, classes, rng, chunk):
    """Move exactly one block of each code to another class; how far does the
    decoded goal move?"""
    n, L = ids.shape
    blk = rng.integers(0, L, n)
    old = ids[np.arange(n), blk]
    new = rng.integers(0, classes - 1, n)
    new += (new >= old)
    ids2 = ids.copy()
    ids2[np.arange(n), blk] = new
    dec2 = decode_all(decode, ids2, classes, chunk)
    step = np.linalg.norm(dec2 - decoded, axis=-1)
    return step, blk, old, new, cosine_max(decoded, dec2)


def run(run_dir, out_path, n_codes=4096, n_states=4096, n_envs=4, stride=8,
        chunk=8192, seed=0):
    config, agent = build(run_dir)
    impl, encode, decode = make_fns(config, agent)
    L, C = [int(x) for x in config.agent.skill_shape]
    name = pathlib.Path(str(run_dir).rstrip('/')).name
    rng = np.random.default_rng(seed)
    print(f'=== {name} | impl={impl} L={L} C={C} | task={config.task}')
    print(f'    code grid is {C}^{L} = {C ** L:,} codes; sampling {n_codes}')

    # --- the addressable goal cloud ------------------------------------
    ids = rng.integers(0, C, size=(n_codes, L))
    G = decode_all(decode, ids, C, chunk)

    # --- the states the policy actually visits -------------------------
    R = np.asarray(collect_states(config, agent, n_envs, stride, n_states),
                   np.float64)
    print(f'    decoded goals {G.shape}, real states {R.shape}')

    # --- 1. SIZE --------------------------------------------------------
    evG, prG, radG = spectrum(G)
    evR, prR, radR = spectrum(R)
    # How much of the real states' variation lies in the goal cloud's own
    # principal subspace: if the goals cannot even span the directions the
    # states move along, the manager cannot address them.
    k = max(1, int(prG))
    cG = G - G.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(cG, full_matrices=False)
    basis = Vt[:k]
    cR = R - R.mean(0, keepdims=True)
    proj = (cR @ basis.T)
    frac_R_in_G = float((proj ** 2).sum() / max((cR ** 2).sum(), 1e-300))
    print(f'    SIZE   eff.dim goals {prG:.2f} vs states {prR:.2f}'
          f'   radius {radG:.3f} vs {radR:.3f}  (ratio {radG / radR:.3f})')
    print(f'           real-state variance inside the top-{k} goal subspace: '
          f'{100 * frac_R_in_G:.1f}%')

    # --- 2. OVERLAP -----------------------------------------------------
    cov = nn_cosmax(R, G)                    # real -> nearest goal
    real = nn_cosmax(G, R)                   # goal -> nearest real
    ref = nn_cosmax(R, R, exclude_self=True)  # real -> nearest OTHER real
    print(f'    OVER   coverage(real->goal) {cov.mean():.4f}'
          f'   realizability(goal->real) {real.mean():.4f}'
          f'   reference(real->real) {ref.mean():.4f}')

    # --- 3. SMOOTHNESS ---------------------------------------------------
    rms, pair_d = rms_pairwise(G, rng)
    step, sblk, sold, snew, step_cos = one_block_steps(
        ids, G, decode, C, rng, chunk)
    rel = step / max(rms, 1e-12)
    print(f'    SMOOTH one-block step {step.mean():.4f} vs cloud RMS {rms:.4f}'
          f'   -> relative {rel.mean():.4f}'
          f'   p95/p50 {np.percentile(rel, 95) / max(np.percentile(rel, 50), 1e-12):.2f}')

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out) + '.npz',
        task=str(config.task), run=name, impl=impl, blocks=L, classes=C,
        dim=int(G.shape[-1]), n_codes=n_codes, n_states=len(R),
        # size
        ev_goal=evG[:256], ev_real=evR[:256], eff_dim_goal=prG,
        eff_dim_real=prR, radius_goal=radG, radius_real=radR,
        frac_real_var_in_goal_subspace=frac_R_in_G, subspace_k=k,
        # overlap
        cover=cov, realiz=real, ref_nn=ref,
        # smoothness
        step=step, step_rel=rel, step_cos=step_cos, cloud_rms=rms,
        pair_dist=pair_d[:4000], step_block=sblk, step_old=sold, step_new=snew,
    )
    print('wrote', str(out) + '.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_codes', type=int, default=4096)
    p.add_argument('--n_states', type=int, default=4096)
    p.add_argument('--n_envs', type=int, default=4)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run(a.run_dir, a.out, a.n_codes, a.n_states, a.n_envs, a.stride,
        a.chunk, a.seed)


if __name__ == '__main__':
    main()
