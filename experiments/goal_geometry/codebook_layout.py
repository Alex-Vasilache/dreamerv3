#!/usr/bin/env python3
"""Where do a block's C classes sit in space, on each of the four arms?

Reads the goal-decoder weights straight out of ``agent.pkl`` -- no environment,
no forward pass, no GPU -- and writes the per-block class embeddings so
``plot_codebook_layout.py`` can draw them.

Two spaces, because the arms do not share one:

  book   the codebook, ``goal_dec/codebook/table`` of shape (L, C, D). This is
         where the SOM neighborhood term and the commitment term act, so it is
         the space the arms actually rearrange. Director has no codebook: its
         bottleneck is the one-hot itself, so its C classes are the C basis
         vectors of the block -- a regular simplex, every pair at the same
         distance, no ordering. Stored as the identity for Director, which is
         literally what its code is.

  dec    what the decoder receives. Every arm feeds ``goal_dec/mlp/linear0`` a
         flattened per-block vector, so for every arm each (block, class) pair
         has an image in that layer's 512-unit output space:

             Director:  W[l*C + c, :]                (the one-hot picks a row)
             VQ arms:   table[l, c, :] @ W[l*D:(l+1)*D, :]

         This is the ONE space all four arms share, and it is the space the
         rest of the decoder sees. The map from the codebook to it is linear,
         so it preserves collinearity and betweenness -- a codebook on a line
         is still on a line here -- while distorting distances.

For the Lipschitz arms ``W`` is the *effective* kernel, i.e. after the row-wise
normalization ``lip_normalize`` applies at call time; the raw stored kernel is
not what the network computes.

  python -u codebook_layout.py --out layout
"""
import argparse
import glob
import json
import os
import pathlib
import pickle

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = '/work/DoyaU/vasilache/work'

# arm -> (label, the arm's token in the run directory name). Runs are found by
# name rather than by experiment number so adding seeds or environments needs
# no edit here.
ARMS = [
    ('director', 'Director', 'director'),
    ('som_line', 'SOM-line', 'som_line'),
    ('lipvq', 'LiP-VQ', 'lipvq_prod'),
    ('somlip', 'SOM-line + LiP', 'som_lipvq_line_prod'),
    # This arm is not a clean activation ablation: the only runs that exist
    # take the Gaussian manager head as well, so it differs from
    # ``som_lipvq_line_prod`` in two places, not one.
    ('somlip_relu', 'SOM-line + LiP + ReLU', 'som_lipvq_line_relu_gaussian'),
]


def steps(run_dir):
  """Last ``step`` in metrics.jsonl, read from the tail."""
  p = os.path.join(run_dir, 'logdir', 'metrics.jsonl')
  if not os.path.exists(p):
    return 0
  best = 0
  with open(p, 'rb') as f:
    f.seek(0, 2)
    f.seek(max(0, f.tell() - 200_000))
    for line in f.read().decode('utf8', 'ignore').splitlines()[1:]:
      try:
        best = max(best, int(json.loads(line).get('step', 0)))
      except Exception:
        pass
  return best


def find_runs(token, task, min_steps):
  """Every trained seed of one arm on one task, in seed order.

  The arm token follows the task in the directory name, which is what keeps
  ``dmc_cartpole_swingup`` from also matching ``dmc_cartpole_swingup_sparse``
  and ``som_line`` from matching ``som_lipvq_line_prod``.

  ``min_steps`` drops runs that have not finished. Without it a seed that
  started an hour ago contributes its initialization to the figure and there is
  nothing in the output to say so -- on 2026-08-21 the third ReLU seed was at
  70k of 4M steps while its two siblings were at 4M.
  """
  pat = os.path.join(WORK, 'e*_%s_%s_s*_BIG_*' % (task, token))
  hits = sorted(glob.glob(pat),
                key=lambda p: os.path.basename(p).split('_s')[-1])
  out = []
  for h in hits:
    n = steps(h)
    if n >= min_steps:
      out.append(h)
    else:
      print('    skip %s (%.2fM steps)' % (os.path.basename(h), n / 1e6))
  return out


def load_params(run_dir):
  ckpts = sorted(glob.glob(os.path.join(run_dir, 'logdir', 'ckpt', '2*')))
  assert ckpts, run_dir
  with open(os.path.join(ckpts[-1], 'agent.pkl'), 'rb') as f:
    return {k: np.asarray(v) for k, v in pickle.load(f)['params'].items()}


def softplus(x):
  return np.logaddexp(x, 0.0)


def effective_kernel(params, prefix):
  """``kernel``, with the LipVQ row normalization applied when the arm has it.

  ``lip_normalize`` scales column j of the (in, out) kernel by
  ``min(1, softplus(c_j) / sum_i |W_ij|)``. Arms without the constraint store no
  ``c`` and the kernel is used as-is.
  """
  w = params[prefix + '/kernel'].astype(np.float64)
  if prefix + '/c' not in params:
    return w
  c = np.asarray(params[prefix + '/c'], np.float64)
  bound = softplus(c)
  scale = np.minimum(1.0, np.broadcast_to(bound, (w.shape[-1],)) /
                     (np.abs(w).sum(0) + 1e-12))
  return w * scale[None, :]


def embeddings(params, blocks, classes):
  """(book (L,C,D) or None, dec (L,C,units))."""
  w = effective_kernel(params, 'goal_dec/mlp/linear0')
  key = 'goal_dec/codebook/table'
  if key in params:
    book = params[key].astype(np.float64)
    L, C, D = book.shape
    assert (L, C) == (blocks, classes), book.shape
    assert w.shape[0] == L * D, (w.shape, book.shape)
    dec = np.stack([book[l] @ w[l * D:(l + 1) * D, :] for l in range(L)])
  else:
    # Director: the code IS the one-hot, so in code space the classes are the
    # basis vectors and in decoder space they are the matching rows of W.
    L, C = blocks, classes
    assert w.shape[0] == L * C, w.shape
    book = np.broadcast_to(np.eye(C), (L, C, C)).copy()
    dec = np.stack([w[l * C:(l + 1) * C, :] for l in range(L)])
  return book, dec


def line_stats(pts):
  """Line-ness and index-ordering of one block's C class vectors.

  pca1     fraction of the variance on the first principal component. 1.0 is a
           perfectly straight line; C points in general position in >=C-1
           dimensions give 1/(C-1).
  mono     |Spearman| between class index and the PC1 coordinate. 1.0 means the
           classes are laid out along that line in index order (either
           direction), which is what the SOM term is for. A line the indices
           walk in a scrambled order has pca1 = 1 and mono < 1.
  gap_cv   coefficient of variation of the gaps between index-consecutive
           classes along PC1. 0 is even spacing.
  """
  x = pts - pts.mean(0, keepdims=True)
  ev = np.linalg.svd(x, compute_uv=False) ** 2
  pca1 = float(ev[0] / max(ev.sum(), 1e-12))
  u, s, vt = np.linalg.svd(x, full_matrices=False)
  t = x @ vt[0]
  idx = np.arange(len(t))
  rt = np.argsort(np.argsort(t)).astype(float)
  ri = idx.astype(float)
  mono = float(abs(np.corrcoef(ri, rt)[0, 1]))
  order = t[np.argsort(idx)]
  gaps = np.abs(np.diff(order))
  gap_cv = float(gaps.std() / max(gaps.mean(), 1e-12))
  return pca1, mono, gap_cv


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=str(HERE / 'layout'))
  ap.add_argument('--task', default='dmc_cartpole_swingup')
  ap.add_argument('--blocks', type=int, default=8)
  ap.add_argument('--classes', type=int, default=8)
  ap.add_argument('--min-steps', type=int, default=3_900_000,
                  help='skip runs that have not trained this far.')
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)

  for arm, label, token in ARMS:
    books, decs, runs = [], [], []
    for run in find_runs(token, a.task, a.min_steps):
      params = load_params(run)
      book, dec = embeddings(params, a.blocks, a.classes)
      books.append(book)
      decs.append(dec)
      runs.append(os.path.basename(run))
    if not runs:
      print('%-16s no runs for %s' % (label, a.task))
      continue
    books, decs = np.stack(books), np.stack(decs)
    np.savez_compressed(os.path.join(a.out, arm + '.npz'), arm=arm,
                        label=label, task=a.task, runs=np.array(runs),
                        book=books, dec=decs)
    for space, arr in (('book', books), ('dec', decs)):
      st = np.array([[line_stats(arr[s, l]) for l in range(a.blocks)]
                     for s in range(len(runs))])
      m = st.reshape(-1, 3).mean(0)
      print('%-16s %-5s pca1 %.3f   mono %.3f   gap_cv %.3f' %
            (label, space, m[0], m[1], m[2]))
    print('%-16s wrote %s (%d seeds)' % ('', arm + '.npz', len(runs)))


if __name__ == '__main__':
  main()
