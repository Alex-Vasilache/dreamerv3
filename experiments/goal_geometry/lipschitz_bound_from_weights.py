#!/usr/bin/env python3
"""The analytic Lipschitz bound of a goal autoencoder, straight from its weights.

`diag_goal_geometry.py` measures the *realized* ratio over sampled state pairs,
which is a lower bound on the true constant and is only as good as the sample.
This computes the other side: the bound the LipVQ construction actually
constrains, for arms that have it and for Director, which does not.

For a linear layer with kernel ``(in, out)`` the inf-norm operator norm is
``max_out sum_in |W|``. Composing 1-Lipschitz activations, the network's
inf-norm Lipschitz constant is at most the product of the per-layer norms --
Eq. (lipschitz_constant_product). A LipLinear stores its kernel RAW and
normalizes at call time, so the quantity to compare against Director's plain
kernel is the EFFECTIVE one, ``W * min(1, softplus(c)/absrowsum)``, whose row
sums are ``min(rowsum_i, softplus(c_i))``.

Two things can invalidate the product as a bound, and both are checked and
reported rather than assumed:

  * a non-1-Lipschitz activation. ``silu`` peaks at slope ~1.0998, so the true
    constant can exceed the product by 1.0998 per activation. Reported as a
    corrected figure alongside the raw product.
  * a normalization layer. RMS/LayerNorm is NOT Lipschitz -- its gain goes to
    infinity as the input approaches zero -- so if the trunk has one, no finite
    product of weight norms bounds the network at all. If that is the case for
    an arm, this prints NO-BOUND for it, which is the honest answer.

Runs on CPU. The checkpoints are pickled with numpy 2.x, so it needs
dreamerv3_env, which means a compute node (`intel`), not the login node.

  python3 lipschitz_bound_from_weights.py --runs /work/.../e502_... [...]
"""
import argparse
import collections
import json
import pathlib
import pickle
import re

import numpy as np


def softplus(x):
  return np.logaddexp(x, 0.0)


def load_params(run_dir):
  ck = pathlib.Path(run_dir) / 'logdir' / 'ckpt'
  latest = (ck / 'latest')
  sub = None
  if latest.exists():
    name = latest.read_text().strip()
    if (ck / name).is_dir():
      sub = ck / name
  if sub is None:
    dirs = sorted([d for d in ck.iterdir() if d.is_dir()])
    sub = dirs[-1] if dirs else None
  if sub is None:
    raise FileNotFoundError(f'no checkpoint under {ck}')
  with open(sub / 'agent.pkl', 'rb') as f:
    blob = pickle.load(f)
  # elements.Checkpoint stores {module: {name: array}} or a flat dict; both
  # appear across the runs here, so flatten whatever came back.
  flat = {}

  def walk(prefix, obj):
    if isinstance(obj, dict):
      for k, v in obj.items():
        walk(f'{prefix}/{k}' if prefix else str(k), v)
    elif isinstance(obj, (np.ndarray, list)):
      arr = np.asarray(obj)
      if arr.dtype != object:
        flat[prefix] = arr

  walk('', blob)
  # The checkpoint holds the weights AND the optimizer state -- Adam's two
  # moment buffers are stored under params/opt/state_goal/{1,2}/ with the SAME
  # leaf names, so an unfiltered walk returns each layer three times and the
  # product over "layers" is meaningless (it read 15 layers for a 5-layer
  # decoder). Keep the parameters only.
  flat = {k[len('params/'):]: v for k, v in flat.items()
          if k.startswith('params/') and '/opt/' not in k}
  return flat, str(sub)


LAYER_RE = re.compile(r'^(?P<mod>.*?(?:goal_enc|goal_dec).*?)/(?P<leaf>[^/]+)$')


def layers_of(flat, which):
  """(layer_key -> {'kernel':..., 'c':..., 'norm':bool}) for goal_enc/goal_dec."""
  out = collections.defaultdict(dict)
  for k, v in flat.items():
    if which not in k:
      continue
    base, leaf = k.rsplit('/', 1)
    if leaf in ('kernel', 'c', 'bias', 'scale', 'offset', 'mean', 'var'):
      out[base][leaf] = v
  return out


def config_norms(run_dir):
  """{'goal_enc': 'rms', 'goal_dec': 'rms'} straight from the run's config."""
  import re as _re
  txt = (pathlib.Path(run_dir) / 'logdir' / 'config.yaml').read_text()
  out = {}
  for which in ('goal_enc', 'goal_dec'):
    m = _re.search(which + r':\s*\{([^}]*)\}', txt, _re.S)
    if m:
      n = _re.search(r'norm:\s*([A-Za-z_]+)', m.group(1))
      out[which] = n.group(1) if n else '?'
  return out


def analyse(run_dir):
  flat, ckpt = load_params(run_dir)
  cfg_norm = config_norms(run_dir)
  name = pathlib.Path(run_dir.rstrip('/')).name
  res = {'run': name, 'ckpt': ckpt}
  for which in ('goal_enc', 'goal_dec'):
    layers = layers_of(flat, which)
    kernels = {k: v for k, v in layers.items() if 'kernel' in v}
    if not kernels:
      res[which] = {'error': 'no kernels found'}
      continue
    # Read the norm setting from the run's own config rather than guessing it
    # from parameter names: LipMLP takes norm= from the same config key as the
    # Director trunk, and its norm parameters do NOT reliably appear as a
    # separate module in the checkpoint, so a name-based heuristic reported the
    # LiP arms as unnormalized when they are not.
    norm_like = bool(cfg_norm.get(which)) and cfg_norm[which] != 'none'
    per_layer, lip = [], 1.0
    for k in sorted(kernels):
      W = np.asarray(kernels[k]['kernel'], np.float64)
      if W.ndim != 2:
        continue
      rowsum = np.abs(W).sum(0)              # (out,) -- kernel is (in, out)
      if 'c' in kernels[k]:
        bound = softplus(np.asarray(kernels[k]['c'], np.float64))
        eff = np.minimum(rowsum, np.broadcast_to(bound, rowsum.shape))
        constrained = True
      else:
        eff = rowsum
        constrained = False
      linf = float(eff.max())
      spec = float(np.linalg.svd(W, compute_uv=False)[0])
      per_layer.append({'layer': k.split('/')[-1] or k, 'shape': list(W.shape),
                        'inf_norm': linf, 'spectral': spec,
                        'constrained': constrained,
                        'raw_inf_norm': float(rowsum.max())})
      lip *= linf
    res[which] = {
        'n_layers': len(per_layer),
        'has_norm_layer': bool(norm_like),
        'config_norm': cfg_norm.get(which, '?'),
        'product_inf_norm': lip,
        'product_spectral': float(np.prod([p['spectral'] for p in per_layer])),
        'layers': per_layer,
    }
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs', nargs='+', required=True)
  ap.add_argument('--out', default=None)
  a = ap.parse_args()
  allres = []
  for r in a.runs:
    try:
      allres.append(analyse(r))
    except Exception as e:
      allres.append({'run': pathlib.Path(r.rstrip('/')).name, 'error': repr(e)})
  for res in allres:
    print('=' * 78)
    print(res['run'])
    if 'error' in res:
      print('  ERROR', res['error']); continue
    for which in ('goal_enc', 'goal_dec'):
      d = res.get(which, {})
      if 'error' in d:
        print(f'  {which}: {d["error"]}'); continue
      tag = ('NO-BOUND: norm=%s is not Lipschitz' % d['config_norm']
             if d['has_norm_layer'] else 'valid bound (norm=none)')
      print(f'  {which}: {d["n_layers"]} layers, product of inf-norms = '
            f'{d["product_inf_norm"]:.6g}   [{tag}]')
      print(f'    product of spectral norms = {d["product_spectral"]:.6g}')
      for p in d['layers']:
        c = 'constrained' if p['constrained'] else 'free'
        extra = ('' if not p['constrained']
                 else f'  (raw {p["raw_inf_norm"]:.4g})')
        print(f'      {p["layer"]:<22s} {str(p["shape"]):<14s} '
              f'inf={p["inf_norm"]:.5g}  spec={p["spectral"]:.5g}  {c}{extra}')
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(allres, indent=1))
    print('\nwrote', a.out)


if __name__ == '__main__':
  main()
