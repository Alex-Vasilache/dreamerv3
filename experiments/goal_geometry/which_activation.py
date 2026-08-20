#!/usr/bin/env python3
"""Which activation did a TRAINED goal autoencoder actually use?

The saved config is evidence, not proof: a config key only matters if the code
reads it. This decides the question against the trained artifact itself.

The trunk is ``layers x [Linear -> Norm -> act]`` followed by an output Linear
(``lipschitz.LipMLP``). With ``norm='none'`` on the LiP arms there is nothing
between the linear map and the activation, so the whole forward pass can be
recomputed by hand from the checkpoint's weights. Do that twice, once assuming
silu and once assuming relu, and compare both against what the real decoder
returns for the same input. Exactly one should match to f32 round-off.

Also reports the consequence: the per-layer product bound
``prod_l softplus(c_l)`` is a Lipschitz bound only when the activations are
1-Lipschitz. relu is; silu's slope peaks at ~1.0998, so with silu the true
constant can exceed the reported product by 1.0998 per activation.

  python -u which_activation.py --run_dir /work/.../e534_...
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

import numpy as np

from diag_goal_geometry import build, make_fns


def silu(x):
    return x / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


ACTS = {'silu': silu, 'relu': relu, 'tanh': np.tanh, 'gelu':
        lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                         (x + 0.044715 * x ** 3)))}
# max |d act / dx|, i.e. the activation's own Lipschitz constant
LIP = {'silu': 1.0998, 'relu': 1.0, 'tanh': 1.0, 'gelu': 1.1289}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--n', type=int, default=64)
    a = p.parse_args()

    config, agent = build(a.run_dir)
    impl, encode, decode = make_fns(config, agent)
    name = pathlib.Path(str(a.run_dir).rstrip('/')).name
    L, C = [int(x) for x in config.agent.skill_shape]
    cfg_act = str(config.agent.goal_vq_dec.act)
    cfg_norm = str(config.agent.goal_vq_dec.norm)
    print(f'=== {name}')
    print(f'    impl={impl}  config says act={cfg_act}  norm={cfg_norm}')
    assert impl == 'vq', 'this check targets the VQ/LiP decoder'
    assert cfg_norm == 'none', (
        f'norm={cfg_norm}: the hand recomputation below assumes nothing sits '
        'between the linear map and the activation')

    # --- pull the decoder trunk's weights out of the checkpoint -------------
    # Adam moments live under opt/state_goal/{1,2}/... with the SAME leaf
    # names as the weights, so an unfiltered match returns three copies of
    # every layer -- the mistake that once turned a 5-layer decoder into a
    # 15-layer one and its bound into 1.3e18.
    P = {k: np.asarray(v, np.float64) for k, v in agent.params.items()
         if 'goal_dec' in k and not k.startswith('opt/') and '/opt/' not in k}
    trunk = sorted({k.rsplit('/', 1)[0] for k in P if '/mlp/' in k},
                   key=lambda s: (len(s), s))
    print(f'    trunk layers found: {len(trunk)}')
    for t in trunk:
        print(f'      {t}  kernel={P[t + "/kernel"].shape}')

    # --- the real decoder's output for a fixed random code ------------------
    rng = np.random.default_rng(0)
    ids = rng.integers(0, C, size=(a.n, L))
    onehot = np.eye(C, dtype=np.float32)[ids]
    real = np.asarray(decode(onehot), np.float64)

    # --- the codebook lookup that feeds the trunk ---------------------------
    cb = [k for k in P if 'codebook' in k
          and k.rsplit('/', 1)[-1] in ('table', 'embed')]
    assert cb, f'no codebook embedding among {sorted(P)}'
    table = P[cb[0]]                       # (blocks, classes, dim)
    x = np.einsum('nlc,lcd->nld', onehot.astype(np.float64), table)
    x = x.reshape(a.n, -1)

    def lip_kernel(base):
        """Apply the row-wise Lipschitz normalization if this layer has a c."""
        k = P[base + '/kernel']
        cs = [n for n in (base + '/c', base + '/lip_c') if n in P]
        if not cs:
            return k
        c = P[cs[0]]
        bound = np.log1p(np.exp(-np.abs(c))) + np.maximum(c, 0.0)  # softplus
        ratio = bound / (np.abs(k).sum(0) + 1e-12)
        return k * np.minimum(1.0, ratio)[None, :]

    def forward(act_fn, cast=lambda h: h):
        h = cast(x)
        for base in trunk:
            h = cast(cast(h) @ cast(lip_kernel(base)))
            if base + '/bias' in P:
                h = cast(h + P[base + '/bias'])
            h = cast(act_fn(h))
        out = [k for k in P if '/out/' in k and k.endswith('kernel')]
        h = cast(cast(h) @ cast(lip_kernel(out[0].rsplit('/', 1)[0])))
        bk = out[0].rsplit('/', 1)[0] + '/bias'
        if bk in P:
            h = cast(h + P[bk])
        return np.asarray(h, np.float64)

    # The real trunk computes in nets.COMPUTE_DTYPE (bfloat16), which carries
    # 8 mantissa bits ~ 4e-3 relative precision per operation. Recomputing in
    # float64 therefore cannot match better than a few percent however correct
    # the activation is, so run it both ways: float64 identifies the activation
    # by a wide margin, bfloat16 confirms the residual was only precision.
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    print()
    print('    hand-recomputed forward pass vs the real decoder:')
    scale = np.abs(real).mean()
    best, bestrel = None, np.inf
    print('      %-6s %12s %12s' % ('act', 'float64', 'bfloat16'))
    for label, fn in ACTS.items():
        try:
            rel = np.abs(forward(fn) - real).max() / max(scale, 1e-12)
            relb = np.abs(forward(fn, cast=lambda h: h.astype(bf16).astype(
                np.float64)) - real).max() / max(scale, 1e-12)
        except Exception as e:                       # shape mismatch etc.
            print(f'      {label:6s} FAILED ({type(e).__name__}: {e})')
            continue
        print('      %-6s %12.3e %12.3e' % (label, rel, relb))
        if relb < bestrel:
            best, bestrel = label, relb

    print()
    if bestrel < 1e-2:
        print(f'    VERDICT: the trained network uses {best.upper()} '
              f'(match {bestrel:.2e}; config said {cfg_act})')
        agree = 'agrees with' if best == cfg_act else 'CONTRADICTS'
        print(f'             this {agree} the saved config')
    else:
        print(f'    INCONCLUSIVE: closest was {best} at {bestrel:.2e}; the '
              'hand recomputation does not reproduce the trunk')
        return 1

    # --- what it costs the Lipschitz claim ---------------------------------
    cs = [k for k in P if k.endswith('/c') or k.endswith('/lip_c')]
    if cs:
        bounds = []
        for k in sorted(cs):
            c = P[k]
            b = np.log1p(np.exp(-np.abs(c))) + np.maximum(c, 0.0)
            bounds.append(float(np.max(b)))
        prod = float(np.prod(bounds))
        n_act = len(trunk)
        infl = LIP[best] ** n_act
        print()
        print(f'    constrained layers: {len(bounds)}, product of bounds = '
              f'{prod:.4g}')
        print(f'    activations in the trunk: {n_act} x {best} '
              f'(Lipschitz {LIP[best]})')
        print(f'    product bound is exact only for a 1-Lipschitz activation.')
        print(f'    true bound <= product x {LIP[best]}^{n_act} = '
              f'{prod * infl:.4g}  ({100 * (infl - 1):.1f}% above the reported '
              f'product)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
