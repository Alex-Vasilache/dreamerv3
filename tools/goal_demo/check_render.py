"""Checks that the demo shows what the training run would have shown.

The render path itself is not reimplemented -- ``render_core`` calls the real
``goal_dec`` / ``_feat_from_goal`` / ``dec`` modules -- so what is left to
establish is that the two settings the demo changes for speed do not change the
picture, and that the weights it loads are a coherent checkpoint.

Run under ``dreamerv3_env`` on the login node:

    python tools/goal_demo/check_render.py [task]
"""
import sys

import numpy as np

import render_core as rc
import runs as runsmod


def _codes(renderer, n=24, seed=0):
  rng = np.random.RandomState(seed)
  return rng.randint(0, renderer.classes,
                     (n, renderer.blocks)).astype(np.float32)


def _diff(a, b):
  a, b = a.astype(np.int16), b.astype(np.int16)
  return np.abs(a - b).max(), np.abs(a - b).mean()


def test_conv_impls_agree(task, arm):
  """Native conv (what the demo uses) vs the einsum fallback the sbatch env
  forces for the V100 cuDNN bug. Different code, same convolution."""
  import os
  ref = rc.GoalRenderer(task, arm, compute_dtype='float32')
  codes = _codes(ref)
  native, _ = ref.render(codes)
  os.environ['DREAMERV3_CONV_IMPL'] = 'reference'
  import importlib
  import embodied.jax.nets as nets
  importlib.reload(nets)
  assert nets._USE_REFERENCE_CONV
  fallback = rc.GoalRenderer(task, arm, compute_dtype='float32')
  ref_img, _ = fallback.render(codes)
  os.environ.pop('DREAMERV3_CONV_IMPL', None)
  importlib.reload(nets)
  mx, mean = _diff(native, ref_img)
  print(f'  conv native vs reference : max {mx:3d}/255  mean {mean:.4f}')
  assert mx <= 2, f'conv implementations disagree by {mx}/255'
  return ref


def test_dtype_agrees(task, arm, f32):
  """float32 (demo) vs bfloat16 (training compute dtype)."""
  bf16 = rc.GoalRenderer(task, arm, compute_dtype='bfloat16')
  codes = _codes(f32)
  a, _ = f32.render(codes)
  b, _ = bf16.render(codes)
  mx, mean = _diff(a, b)
  print(f'  float32 vs bfloat16      : max {mx:3d}/255  mean {mean:.4f}')
  assert mean < 2.0, f'dtype changes the frame by {mean:.2f}/255 on average'


def random_pc1_baseline(classes, dim, trials=20000, seed=0):
  """PC1 variance fraction for ``classes`` random points in R^dim.

  Eight points span at most a 7-dimensional affine subspace, so one axis
  captures a large share of their spread by construction -- 0.41 on average for
  8 points in R^8. Any claim about the codebook being a line has to clear this.
  """
  rng = np.random.RandomState(seed)
  fracs = np.empty(trials)
  for i in range(trials):
    x = rng.randn(classes, dim)
    sv = np.linalg.svd(x - x.mean(0), compute_uv=False)
    fracs[i] = sv[0] ** 2 / (sv ** 2).sum()
  return fracs.mean(), np.percentile(fracs, 95)


def test_codebook_is_a_line(renderer):
  """How close each block's entries are to collinear, against random points.

  This is what makes extrapolating along the path meaningful for our arm, and
  it is not uniform across environments: cartpole sits near 0.99, the
  locomotion tasks lower, with individual blocks on cheetah down at the random
  level. The assertion is only that the arm as a whole beats the random p95.
  """
  table = renderer.codebook()
  if table is None:
    print('  codebook                 : none (Director has no embedding)')
    return
  fracs = []
  for block in table:
    centered = block - block.mean(0)
    sv = np.linalg.svd(centered, compute_uv=False)
    fracs.append(sv[0] ** 2 / (sv ** 2).sum())
  fracs = np.array(fracs)
  mean, p95 = random_pc1_baseline(*table.shape[1:])
  print(f'  codebook PC1 var fraction: min {fracs.min():.3f} '
        f'mean {fracs.mean():.3f}  (random {mean:.3f}, p95 {p95:.3f})')
  assert fracs.mean() > p95, (
      f'codebook is no more line-like than {table.shape[1]} random points')


def test_extrapolation_rule(renderer):
  """The fitted-line rule, on the three things it has to get right.

  It must leave the trained range exactly as it was (integers still land on
  codebook entries), join onto it without a jump, and outside it move one mean
  segment length per class along the block's fitted direction.
  """
  if not renderer.is_vq:
    print('  extrapolation rule       : one-hot extension (Director has no line)')
    return
  table = renderer.codebook()
  L, C = renderer.blocks, renderer.classes
  emb = lambda t: rc.class_embeddings(
      np.asarray(t, np.float32), table, renderer.line_dirs, renderer.line_steps)

  ints = np.tile(np.arange(C, dtype=np.float32), (L, 1)).T    # (C, L)
  got = emb(ints)
  want = np.stack([table[:, j] for j in range(C)])            # (C, L, D)
  worst = np.abs(got - want).max()
  print(f'  integers -> codebook     : max error {worst:.2e}')
  assert worst < 1e-5, 'the new rule moved the trained classes'

  edge = np.abs(emb(np.full((1, L), -1e-4)) - table[:, 0]).max()
  edge = max(edge, np.abs(emb(np.full((1, L), C - 1 + 1e-4))
                          - table[:, C - 1]).max())
  print(f'  joins at the ends        : max gap {edge:.2e}')
  assert edge < 1e-3, 'the extrapolated path does not meet the codebook path'

  out = emb(np.stack([np.full(L, t) for t in (-4, -3, 8, 9, 10)]))
  steps = np.linalg.norm(np.diff(out[:2], axis=0), axis=-1)[0]
  want_step = renderer.line_steps
  print(f'  step outside             : {steps.mean():.4f} vs mean segment '
        f'{want_step.mean():.4f}')
  assert np.allclose(steps, want_step, atol=1e-4), (
      'extrapolated steps are not the mean segment length')

  moved = out[-1] - out[-2]
  cos = (moved * renderer.line_dirs).sum(-1) / (
      np.linalg.norm(moved, axis=-1) + 1e-9)
  print(f'  direction outside        : cosine to fitted line '
        f'{cos.min():.4f}..{cos.max():.4f}')
  assert cos.min() > 0.999, 'extrapolation is not following the fitted line'


def test_encoder_decoder_match(task, arm):
  """Decode a code, re-encode the resulting deter, count blocks that come back.

  This is ``_code_diag``'s round-trip on random codes, so it is not expected to
  be perfect -- a random code is off the manifold of codes real states produce,
  and the decoder was never trained to make those re-encode. What it does show
  is coupling: an encoder and decoder that came from the same checkpoint score
  far above the 1/C chance rate, whereas a mismatched pair scores at chance.
  That is what catches loading same-shaped weights from the wrong run.
  """
  import jax.numpy as jnp
  import ninjax as nj
  import jax

  r = rc.GoalRenderer(task, arm, compute_dtype='float32')
  model = r.model

  def roundtrip(weights):
    goal = model.goal_dec(weights, 1).pred()
    enc = model.goal_enc(goal, 1)
    dist = enc['skill'] if isinstance(enc, dict) else enc
    return jnp.argmax(dist.pred(), -1)

  codes = _codes(r, n=64)
  w = jnp.asarray(rc.class_weights(codes, r.classes))
  state = nj.init(roundtrip)({}, w, seed=0)
  params, _ = rc._load_params(runsmod.ckpt_path(task, arm), state)
  pure = nj.pure(roundtrip)
  back = np.asarray(jax.jit(lambda s, x: pure(s, x)[1])(
      jax.device_put(params), w))
  agree = (back == codes.astype(np.int64)).mean()
  chance = 1.0 / r.classes
  print(f'  code -> deter -> code    : {agree:.1%} of blocks '
        f'(chance {chance:.1%})')
  assert agree > 2 * chance, (
      f'round-trip {agree:.1%} is near the {chance:.1%} chance rate: '
      'goal_enc and goal_dec look like they came from different runs')


def _rank(x):
  order = np.argsort(np.argsort(x))
  return (order - order.mean()) / (order.std() + 1e-9)


def test_values_read_the_agent(renderer):
  """The critic readout, on the properties that catch a mis-wired head.

  Nothing here re-derives a value -- the heads are the checkpoint's own, called
  on the checkpoint's own ``feat``. What is worth asserting is that each label
  is attached to the head it names, which is exactly the mistake a dict of five
  similar-looking MLPs invites:

  * they must disagree; three names over one head would read as one column
    repeated three times, and nothing else in the demo would show it;
  * the task critic must rank states the way the reward head does, since one is
    a discounted sum of the other, while the exploration critic -- trained on
    the goal autoencoder's own error -- has no reason to and mostly does not;
  * the worker critic must actually consume the goal it is handed, since it is
    the only head whose input is a pair and the only one the demo can ask about
    two codes at once.

  How *much* it prefers standing on its goal is reported rather than asserted.
  It should prefer it -- the cosine reward is maximal there -- but by how much
  is a property of the arm and of how off-manifold a hand-built state is, not
  of the wiring, and on Director the margin is thin.
  """
  if not renderer.has_values:
    print('  critics                  : none (not an HRL checkpoint)')
    return
  codes = _codes(renderer, n=64)
  other = _codes(renderer, n=64, seed=1)
  vals = renderer.values(codes)
  assert all(np.isfinite(v).all() for v in vals.values()), 'a head returned NaN'
  spread = {k: float(v.std()) for k, v in vals.items()}
  print('  critics vary across codes:', ' '.join(
      f'{k} {v:.3g}' for k, v in sorted(spread.items())))
  names = ['mgr_extr', 'mgr_expl', 'wkr_goal']
  # ``cont`` is left out on purpose: on a task that never terminates it is
  # pinned near 1 everywhere, and that is the head working, not failing.
  assert min(spread[k] for k in (*names, 'reward')) > 1e-4, (
      'a head is constant across codes, so it is not reading the state')

  for i, a in enumerate(names):
    for b in names[i + 1:]:
      assert np.abs(vals[a] - vals[b]).max() > 1e-3, f'{a} and {b} are one head'

  extr = float((_rank(vals['mgr_extr']) * _rank(vals['reward'])).mean())
  expl = float((_rank(vals['mgr_expl']) * _rank(vals['reward'])).mean())
  print(f'  rank corr with reward    : task {extr:+.2f}  explore {expl:+.2f}')
  assert extr > 0.3, (
      f'the task critic ranks states {extr:+.2f} against the reward head; it '
      'is not the extrinsic critic')

  self_goal = vals['wkr_goal']
  cross = renderer.values(codes, other)['wkr_goal']
  moved = float(np.abs(self_goal - cross).mean())
  assert moved > 1e-3, (
      'the worker critic ignores the goal it is given, so the second code is '
      'not reaching its input')
  same = renderer.values(codes, codes)['wkr_goal']
  assert np.abs(same - self_goal).max() < 1e-4, (
      'passing the code as its own goal differs from the default, so the two '
      'halves of the input are not being filled from where they should be')
  share = float((self_goal > cross).mean())
  print(f'  worker prefers its goal  : {share:.0%} of 64 pairs '
        f'({self_goal.mean():.1f} vs {cross.mean():.1f}, '
        f'moves {moved:.1f} on average)')


def test_goal_distance_search(renderer):
  """Asking for a goal-space distance has to actually deliver it.

  The walk is only useful if the number it reports is the number you get, so
  each result is re-measured from its own code, and a distance no ray can reach
  has to come back flagged rather than quietly short.
  """
  rng = np.random.default_rng(0)
  span = (-4, renderer.classes - 1 + 4)
  code = np.full(renderer.blocks, 3.0, np.float32)
  base = renderer.goal_vectors(code)[0]
  worst = 0.0
  for target in (0.5, 2.0, 5.0, 8.0):
    out = renderer.code_at_goal_distance(code, target, span, rng)
    got = float(np.linalg.norm(
        renderer.goal_vectors(np.asarray(out['code']))[0] - base))
    assert not out['capped'], f'{target} should be reachable, got {got:.3f}'
    assert abs(got - out['dist']) < 1e-3, 'the reported distance is not the code'
    worst = max(worst, abs(got - target) / target)
  print(f'  goal-distance walk       : worst miss {worst:.2%} of the target')
  assert worst < 0.02, 'the walk is not landing on the distance it was asked for'
  far = renderer.code_at_goal_distance(code, 1e4, span, rng)
  print(f'  furthest this arm reaches: {far["dist"]:.2f} '
        f'(a random pair sits at {renderer.goal_scale()["span"]:.2f})')
  assert far['capped'], 'an unreachable distance was reported as reached'


def main():
  task = sys.argv[1] if len(sys.argv) > 1 else 'dmc_cartpole_swingup'
  for arm in runsmod.ARMS:
    print(f'{task} / {arm} ({runsmod.info(task, arm)["exp"]})')
    f32 = test_conv_impls_agree(task, arm)
    test_dtype_agrees(task, arm, f32)
    test_codebook_is_a_line(f32)
    test_extrapolation_rule(f32)
    test_values_read_the_agent(f32)
    test_goal_distance_search(f32)
    test_encoder_decoder_match(task, arm)
    print()
  print('all checks passed')


if __name__ == '__main__':
  main()
