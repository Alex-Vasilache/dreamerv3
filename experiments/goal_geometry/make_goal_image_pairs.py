#!/usr/bin/env python3
"""Visual check: does low code similarity really mean different goals, or is
the goal space itself just subtle?

`diag_goal_geometry.py` / `make_figure_pairs.py` answer "does code similarity
correlate with goal-space similarity" with a single scatter + trend line. That
number is compatible with two very different pictures: (a) many dissimilar
codes really do collapse onto near-identical goals (wasted code capacity), or
(b) the code correlates fine, but the goal space itself is visually narrow, so
even a "0.2-similar" pair of codes decodes to two goals that a human would
call similar-looking. A correlation coefficient cannot tell these apart; the
decoded images can.

Two panels, both restricted to one task (default hopper_stand) and one arm at
a time:

  A. HARD code, full range. Bin pairs of collected states by their hard-code
     cosine-max similarity at targets 1.0, 0.9, ..., 0.2 (nearest realized
     level -- hard sim only takes L+1 discrete values). For each bin, decode
     a few sample pairs THROUGH THE CODE (goal_dec(onehot) -> deter -> image)
     and render them side by side. This is literally "two codes this
     dissimilar decode to these two goals" -- if the images still look near-
     identical at sim=0.2, that is code collapse, not subtlety.

  B. SOFT code, narrow range. Restrict to pairs of collected states whose
     GOAL-SPACE (deter) cosine-max similarity already falls in 0.7/0.8/0.9/1.0
     -- i.e. states that were already close to begin with -- and decode
     THROUGH THE SOFT CODE (goal_dec(soft) -> deter -> image, the pre-sample
     continuous code). This tests the "subtle changes" hypothesis directly:
     within a band of genuinely similar states, does the soft code spread out
     smoothly with the real (small) differences, and do the rendered goals
     actually show them?

No decoded-image plotting infrastructure existed in this repo (the other
`goal_geometry` figures are all scatter/heatmap SVGs over similarity scalars,
never actual pixels) so this writes plain PNGs directly -- no PIL/matplotlib
in dreamerv3_env, so a tiny zlib-based PNG encoder and a 3x5 bitmap font are
included below.

  python -u make_goal_image_pairs.py --run_dir /work/.../e506_..._j4676887 \
      --out results_img_pairs/e506_director_hopper.png
"""
import argparse
import pathlib
import struct
import sys
import zlib

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn
from diag_goal_geometry import build, make_fns, collect_states
from diag_goal_struct_corr import pairwise_cosmax

HERE = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------- PNG output

def write_png(path, arr):
  """``arr``: uint8 (H, W, 3). Stdlib-only encoder (no PIL in dreamerv3_env)."""
  arr = np.asarray(arr, np.uint8)
  h, w, _ = arr.shape

  def chunk(tag, data):
    return (struct.pack('>I', len(data)) + tag + data +
            struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

  sig = b'\x89PNG\r\n\x1a\n'
  ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
  raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(h))
  idat = zlib.compress(raw, 6)
  data = sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')
  pathlib.Path(path).write_bytes(data)


# 3x5 bitmap digits, used to stamp similarity values onto the tiles below.
_FONT = {
    '0': '111101101101111', '1': '010110010010111',
    '2': '111001111100111', '3': '111001111001111',
    '4': '101101111001001', '5': '111100111001111',
    '6': '111100111101111', '7': '111001010010010',
    '8': '111101111101111', '9': '111101111001111',
    '.': '000000000000010', '-': '000000111000000',
    ' ': '000000000000000',
}


def stamp_text(canvas, x, y, text, color, scale=2):
  """Draws ``text`` into ``canvas`` (H, W, 3 uint8) top-left at (x, y)."""
  h, w = canvas.shape[:2]
  cx = x
  for ch in text:
    bits = _FONT.get(ch, _FONT[' '])
    for row in range(5):
      for col in range(3):
        if bits[row * 3 + col] == '1':
          y0, x0 = y + row * scale, cx + col * scale
          if 0 <= y0 < h - scale and 0 <= x0 < w - scale:
            canvas[y0:y0 + scale, x0:x0 + scale] = color
    cx += 4 * scale


# ------------------------------------------------------------- model access

def make_image_fn(agent):
  """deter (goal) vector -> uint8 RGB image, through the world model's image
  decoder. Mirrors ``GoalCodeMixin._feat_from_goal`` + ``Agent.dec`` exactly
  (same path training-time goal viz uses), called forward-only via
  ``nj.pure`` with the restored checkpoint params -- no training touched."""
  model = agent.model
  key = model.dec.imgkeys[0]

  def _feat(d):
    d = nn.cast(d)
    logit = model.dyn._prior(d)
    stoch = nn.cast(model.dyn._dist(logit).pred())
    return dict(deter=d, stoch=stoch, logit=logit)

  def _image(d):
    feat = _feat(d)
    reset = jnp.zeros(d.shape[:1], bool)
    _, _, recons = model.dec({}, feat, reset, training=False)
    return (recons[key].pred(),)

  def image(d, chunk=32):
    # The image decoder's reference conv (DREAMERV3_CONV_IMPL=reference, used
    # off the Route-A env this measurement runs in) is a vmap'd sliding-window
    # implementation, not native cuDNN -- its memory cost is much steeper per
    # batch element, and decoding all ~512 collected states at once OOMs a
    # V100. Chunk it; this is a forward pass over a few hundred goals, cheap
    # either way.
    d = np.asarray(d)
    outs = []
    for s in range(0, len(d), chunk):
      out = nj.pure(_image)(agent.params, jnp.asarray(d[s:s + chunk]))[1][0]
      outs.append(np.asarray(out))
    u8 = np.clip(np.concatenate(outs, 0) * 255, 0, 255).astype(np.uint8)
    if u8.shape[-1] == 1:  # grayscale render: tile to RGB for the PNG writer
      u8 = np.repeat(u8, 3, -1)
    return u8

  return image


def block_distance_similarity(ids, topology, classes):
  """Graded hard-code similarity for a codebook with a real neighborhood
  ordering (``som: True``): per block, |id_i - id_j| (mod-wrapped for a ring,
  not for a line) instead of a plain match/no-match. Summed over blocks and
  normalized by the maximum achievable total distance -- L blocks each at the
  farthest-apart pair, ``classes/2`` for a ring (wrap) or ``classes - 1`` for
  a line (no wrap).

  Plain Hamming equality (``1 - mean(id_i != id_j)``, what ``sim_hard`` used
  before this) treats a 1-apart codebook neighbor exactly like the opposite
  end of the line -- both just "different id" -- even though the SOM
  neighborhood loss trained adjacent ids to decode to nearby goals. This is
  the same formula ``diag_goal_geometry.correlations()`` uses for
  ``sim_code['index']``, just written as a sum instead of a mean so the
  intermediate "total blocks apart" (0..L*(classes-1) for a line) is visible."""
  ids = np.asarray(ids)
  blocks = ids.shape[-1]
  diff = np.abs(ids[:, None, :].astype(np.int64) - ids[None, :, :])
  if topology == 'ring':
    diff = np.minimum(diff, classes - diff)
  half = classes / 2.0 if topology == 'ring' else float(classes - 1)
  return 1.0 - diff.sum(-1) / (blocks * half)


# ---------------------------------------------------------------- sampling

def pick_pairs_hard(sim_hard, targets, n_per_bin, rng):
  """For each target similarity, snap to the nearest REALIZED discrete hard-
  sim level (hard sim only takes L+1 values: 0, 1/L, ..., 1) and sample pairs
  at that level. Returns [(target, actual_level, [(i, j), ...]), ...]."""
  n = sim_hard.shape[0]
  levels = np.unique(np.round(sim_hard[np.triu_indices(n, k=1)], 6))
  out = []
  for t in targets:
    level = levels[np.argmin(np.abs(levels - t))]
    iu, ju = np.where(np.triu(np.isclose(sim_hard, level), k=1))
    if len(iu) == 0:
      out.append((t, level, []))
      continue
    sel = rng.choice(len(iu), min(n_per_bin, len(iu)), replace=False)
    out.append((t, float(level), list(zip(iu[sel].tolist(), ju[sel].tolist()))))
  return out


def pick_pairs_binned(sim, targets, n_per_bin, rng, half_width=0.05):
  """Pairs whose ``sim`` falls within +/- half_width of each target (used for
  the goal-space-similarity bins in panel B)."""
  n = sim.shape[0]
  out = []
  for t in targets:
    lo, hi = t - half_width, (t + half_width if t < 1.0 else t + 1e-6)
    iu, ju = np.where(np.triu((sim >= lo) & (sim <= hi), k=1))
    if len(iu) == 0:
      out.append((t, []))
      continue
    sel = rng.choice(len(iu), min(n_per_bin, len(iu)), replace=False)
    out.append((t, list(zip(iu[sel].tolist(), ju[sel].tolist()))))
  return out


# ------------------------------------------------------------------ layout

def build_grid(cols, cell_w, cell_h, margin_top, gap, bg=(30, 30, 30)):
  """``cols``: list of (col_label, tiles), one entry per similarity bin, IN
  THE ORDER THE CALLER WANTS LEFT TO RIGHT (ascending bin value). Lays out a
  top label strip per column + one pair-block per sample stacked vertically
  within that column, each pair-block = img_i ABOVE img_j (small gap between,
  the two states of the pair read top-to-bottom), with the OPPOSING
  similarity (not the one the column axis already encodes) stamped
  underneath."""
  img_gap = max(2, gap // 2)
  n_rows = max((len(tiles) for _, tiles in cols), default=0)
  pair_w = cell_w
  pair_h = 2 * cell_h + img_gap
  row_h = pair_h + 28
  W = len(cols) * (pair_w + gap)
  H = margin_top + n_rows * (row_h + gap)
  canvas = np.full((H, W, 3), bg, np.uint8)
  for c, (col_label, tiles) in enumerate(cols):
    x0 = c * (pair_w + gap)
    stamp_text(canvas, x0 + pair_w // 2 - 12, 4, col_label, (255, 255, 255), 2)
    for r, (img_i, img_j, label) in enumerate(tiles):
      y0 = margin_top + r * (row_h + gap)
      canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img_i
      canvas[y0 + cell_h + img_gap:y0 + pair_h, x0:x0 + cell_w] = img_j
      stamp_text(canvas, x0, y0 + pair_h + 5, label, (200, 200, 200), 2)
  return canvas


# --------------------------------------------------------------------- main

def run(run_dir, out_prefix, ckpt_path=None, n_envs=8, stride=8, n_states=512,
        n_per_bin=4, seed=0):
  config, agent = build(run_dir, ckpt_path)
  impl, encode, decode = make_fns(config, agent)
  image = make_image_fn(agent)
  blocks, classes = [int(x) for x in config.agent.skill_shape]
  name = pathlib.Path(str(run_dir).rstrip('/')).name
  print(f'=== {name} | impl={impl} L={blocks} C={classes} task={config.task}')

  deters = collect_states(config, agent, n_envs, stride, n_states)
  print(f'collected {len(deters)} states, D={deters.shape[-1]}')
  z_e, z_q, ids, onehot, soft = encode(deters)
  n = len(deters)
  sim_goal = pairwise_cosmax(deters)
  sim_soft = pairwise_cosmax(soft.reshape(n, -1))

  # Hard-code similarity: plain Hamming equality (match/no-match per block) is
  # the right metric for Director's argmax labels, which carry no ordering --
  # but for a codebook trained with the SOM neighborhood loss, two ids one
  # apart decode to nearby goals and equality treats that exactly like the
  # opposite end of the line. Use the graded per-block distance there instead
  # (see block_distance_similarity), matching the codebook's own topology.
  som_on = bool(getattr(config.agent.goal_vq, 'som', False)) if impl == 'vq' \
      else False
  topology = str(getattr(config.agent.goal_vq, 'topology', 'ring')) \
      if som_on else 'none'
  if som_on:
    sim_hard = block_distance_similarity(ids, topology, classes)
    hard_metric = f'block-distance ({topology})'
  else:
    sim_hard = 1.0 - (ids[:, None, :] != ids[None, :, :]).mean(-1)
    hard_metric = 'hamming'
  print(f'hard-code metric: {hard_metric}')

  rng = np.random.default_rng(seed)
  UPSAMPLE = 2  # native dmc render is small (e.g. 64x64); upscale for legibility
  MARGIN, GAP = 44, 6

  def upsample(imgs):
    return np.repeat(np.repeat(imgs, UPSAMPLE, 1), UPSAMPLE, 2)

  # --- Panel A: hard code, full range 0.2 .. 1.0 left to right, decode
  # through the code.
  targets_a = [round(0.2 + 0.1 * k, 2) for k in range(9)]
  if som_on:
    # sim_hard is graded/continuous here (steps of 1/(L*(classes-1))); use a
    # tolerance window per target instead of snapping to a handful of levels.
    bins_a = pick_pairs_binned(sim_hard, targets_a, n_per_bin, rng, half_width=0.05)
  else:
    # Plain Hamming sim only takes L+1 discrete levels; snap each target to
    # the nearest one actually realized in the pool.
    bins_a = [(t, pairs) for t, _, pairs in
              pick_pairs_hard(sim_hard, targets_a, n_per_bin, rng)]
  goal_a = upsample(image(decode(onehot)))  # decode every code once, index below
  CH, CW = goal_a.shape[1:3]
  cols_a = []
  for t, pairs in bins_a:
    tiles = []
    for i, j in pairs:
      # Column axis is code similarity; the label is the OPPOSING number
      # (goal-space similarity) only -- the code similarity is what the
      # column position already says.
      tiles.append((goal_a[i], goal_a[j], f'{sim_goal[i, j]:.2f}'))
    cols_a.append((f'{t:.1f}', tiles))
    print(f'  [A hard] target={t:.2f} n_pairs={len(pairs)}')
  canvas_a = build_grid(cols_a, CW, CH, MARGIN, GAP)
  path_a = f'{out_prefix}_hard.png'
  write_png(path_a, canvas_a)
  print(f'wrote {path_a}  ({canvas_a.shape[1]}x{canvas_a.shape[0]})')

  # --- Panel B: soft code, goal-space sim restricted to 0.7/0.8/0.9/1.0.
  targets_b = [0.7, 0.8, 0.9, 1.0]
  bins_b = pick_pairs_binned(sim_goal, targets_b, n_per_bin, rng, half_width=0.05)
  goal_b = upsample(image(decode(soft)))  # soft (pre-sample) code, through goal_dec
  cols_b = []
  for t, pairs in bins_b:
    tiles = []
    for i, j in pairs:
      # Column axis is goal-space similarity; the label is the OPPOSING
      # number (soft-code similarity) only.
      tiles.append((goal_b[i], goal_b[j], f'{sim_soft[i, j]:.2f}'))
    cols_b.append((f'{t:.1f}', tiles))
    print(f'  [B soft] goal-sim~={t:.2f} n_pairs={len(pairs)}')
  canvas_b = build_grid(cols_b, CW, CH, MARGIN, GAP)
  path_b = f'{out_prefix}_soft.png'
  write_png(path_b, canvas_b)
  print(f'wrote {path_b}  ({canvas_b.shape[1]}x{canvas_b.shape[0]})')

  readme = pathlib.Path(f'{out_prefix}_README.txt')
  readme.write_text(
      f'{name}  impl={impl}  task={config.task}  hard_metric={hard_metric}\n\n'
      f'{path_a.rsplit("/", 1)[-1]} -- panel A, HARD code, full range.\n'
      '  One column per target code-similarity, left to right 0.2..1.0. Each\n'
      '  pair-block (stacked within its column) is two goal IMAGES decoded\n'
      '  from two DIFFERENT codes whose hard-code similarity falls in that\n'
      '  column. Label under each pair is the OPPOSING number only -- the\n'
      '  true goal-state similarity (column position already says the code\n'
      '  similarity).\n'
      + ('  Metric: graded per-block index distance on the codebook\'s own\n'
         f'  {topology} topology (see block_distance_similarity) -- a\n'
         '  1-apart neighbor counts as a near-miss, not a full miss, because\n'
         '  the SOM loss trained adjacent ids to decode to nearby goals.\n'
         if som_on else
         '  Metric: plain Hamming equality (same id per block or not) -- the\n'
         '  right metric for Director\'s unordered argmax labels.\n') +
      '  If images still look near-identical in a low column (e.g. 0.2), that\n'
      '  is code collapse (many dissimilar codes -> one goal), not subtlety.\n\n'
      f'{path_b.rsplit("/", 1)[-1]} -- panel B, SOFT code, narrow range.\n'
      '  Pairs are restricted to states whose GOAL-SPACE (deter) cosine-max\n'
      '  similarity is already in the column\'s band (0.7/0.8/0.9/1.0, left to\n'
      '  right), i.e. states that were already close. Each pair-block decodes\n'
      '  the two states\' SOFT (pre-sample) codes to images. Label under each\n'
      '  pair is the OPPOSING number only -- the soft-code similarity (column\n'
      '  position already says the goal-state similarity). If code sim tracks\n'
      '  goal sim smoothly within this narrow high band and the images show\n'
      '  correspondingly small, real visual differences, that supports "codes\n'
      '  track subtle changes" over "codes collapse".\n')
  print(f'wrote {readme}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--out', required=True, help='prefix; writes <out>_hard.png / _soft.png')
  ap.add_argument('--ckpt_path', default=None)
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--stride', type=int, default=8)
  ap.add_argument('--n_states', type=int, default=512)
  ap.add_argument('--n_per_bin', type=int, default=4)
  ap.add_argument('--seed', type=int, default=0)
  a = ap.parse_args()
  run(a.run_dir, a.out, a.ckpt_path, a.n_envs, a.stride, a.n_states,
      a.n_per_bin, a.seed)


if __name__ == '__main__':
  main()
