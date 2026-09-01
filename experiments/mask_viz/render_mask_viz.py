#!/usr/bin/env python3
"""Regenerate the policy mask-viz panels from a finished checkpoint.

The panel logged during training as ``epstats/policy_mask_viz_image`` is built
inside ``policy()`` and gated by ``agent.report_mask_viz``, which was turned off
for the later runs. Nothing about it depends on training state, so it can be
rebuilt after the fact: restore the checkpoint, flip the flag on, roll the
policy out, and keep ``log/mask_viz_image`` from each step.

Each frame is three rows, upscaled by ``report_mask_viz_scale``:

    row 1   the observation
    row 2   the running goal code Z_t, one cell per (block, class), with the
            blocks the manager CHANGED at the last switch tinted yellow and the
            ones it re-emitted unchanged left grey -- this is the implicit
            sparsity, not a mask
    row 3   the active goal, decoded back to pixels

Writes a GIF of the episode, a contact sheet of evenly-spaced frames, and the
raw frames as .npz so a figure can be built later without another rollout.

Needs a GPU for the rollout (MUJOCO_GL=egl).

  python -u render_mask_viz.py --run_dir /work/.../e534_... --out out/e534
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

from functools import partial as bind

import elements
import embodied
import numpy as np

import dreamerv3.main as m
from diag_goal_struct_corr import load_config


def build(run_dir, ckpt_path=None, platform=None):
  config = load_config(run_dir)
  # load_config turns the diagnostics off for the geometry tool; this one wants
  # the mask-viz panel on and the goal image rendered.
  flat = config.flat
  flat.update({
      'agent.report_mask_viz': True,
      'agent.policy_goal_image': True,
      'agent.policy_struct_diag': False,
  })
  if platform:
    # CPU rollouts are viable and often faster to obtain than a GPU slot: this
    # is a single-env, few-hundred-step rollout, not training. pinpad is a pure
    # numpy gridworld with no MuJoCo at all; DMC needs MUJOCO_GL=osmesa for
    # software rendering instead of egl.
    flat.update({'jax.platform': platform,
                 'jax.policy_devices': [0], 'jax.train_devices': [0]})
    config = elements.Config(flat)
  config = elements.Config(flat)
  agent = m.make_agent(config)
  cp = elements.Checkpoint(directory=elements.Path(config.logdir) / 'ckpt')
  cp.agent = agent
  if ckpt_path:
    cp.load(path=elements.Path(ckpt_path), keys=['agent'])
  else:
    cp.load(keys=['agent'])
  import jax
  jax.config.update('jax_transfer_guard', 'allow')
  return config, agent


def rollout(config, agent, steps, key='log/mask_viz_image'):
  driver = embodied.Driver([bind(m.make_env, config, 0)], parallel=False)
  driver.reset(agent.init_policy)
  frames, scores, seen = [], [], set()

  def grab(tran, worker):
    if key in tran:
      frames.append(np.asarray(tran[key]).copy())
    for k in tran:
      seen.add(k)
    if tran.get('is_last'):
      scores.append(float(tran.get('score', float('nan'))))

  driver.on_step(grab)
  driver(agent.policy, steps=steps)
  driver.close()
  return frames, scores, seen


def contact_sheet(frames, n=12, cols=4):
  """Evenly spaced frames tiled into one image."""
  if not frames:
    return None
  idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
  sel = [frames[i] for i in idx]
  h, w = sel[0].shape[:2]
  rows = int(np.ceil(len(sel) / cols))
  sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
  for i, f in enumerate(sel):
    r, c = divmod(i, cols)
    sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f[..., :3]
  return sheet


def write_mp4(arr, out, fps):
  """H.264 via the ffmpeg binary imageio_ffmpeg vendors -- there is no system
  ffmpeg on these nodes. yuv420p needs even dimensions; the panel is 768x256 so
  it already is, but pad rather than let the encoder fail on a future layout."""
  import imageio.v2 as imageio
  frames = arr[..., :3]
  h, w = frames.shape[1:3]
  if h % 2 or w % 2:
    pad = ((0, 0), (0, h % 2), (0, w % 2), (0, 0))
    frames = np.pad(frames, pad, mode='edge')
  path = str(out) + '.mp4'
  writer = imageio.get_writer(
      path, fps=fps, codec='libx264', quality=8,
      macro_block_size=None, ffmpeg_params=['-pix_fmt', 'yuv420p'])
  try:
    for f in frames:
      writer.append_data(np.ascontiguousarray(f))
  finally:
    writer.close()
  return path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--ckpt_path', default=None)
  ap.add_argument('--platform', default=None,
                  help='override jax.platform, e.g. cpu (no GPU slot needed)')
  ap.add_argument('--steps', type=int, default=520)
  ap.add_argument('--fps', type=int, default=20)
  ap.add_argument('--gif', action='store_true',
                  help='also write the (much larger) GIF')
  ap.add_argument('--from_npz', default=None,
                  help='re-encode an existing <out>.npz instead of rolling out '
                       'again; needs no GPU and no checkpoint')
  a = ap.parse_args()

  if a.from_npz:
    arr = np.load(a.from_npz)['frames']
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print('re-encoded', write_mp4(arr, out, a.fps), '<-', a.from_npz)
    return

  config, agent = build(a.run_dir, a.ckpt_path, a.platform)
  name = pathlib.Path(a.run_dir.rstrip('/')).name
  print(f'=== {name} | task={config.task}')
  frames, scores, seen = rollout(config, agent, a.steps)
  if not frames:
    cand = sorted(k for k in seen if 'viz' in k or 'log/' in k)
    print('NO FRAMES. keys the policy did emit:')
    for k in cand[:30]:
      print('   ', k)
    raise SystemExit('mask_viz_image not produced -- see keys above')

  arr = np.stack(frames)
  print(f'{len(arr)} frames of {arr.shape[1:]}, episode scores: '
        f'{["%.1f" % s for s in scores]}')
  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(str(out) + '.npz', frames=arr,
                      task=str(config.task), run=name,
                      scores=np.array(scores, np.float32))

  try:
    print('wrote', write_mp4(arr, out, a.fps))
  except Exception as exc:
    print('mp4 encode failed (%r); .npz still written' % (exc,))

  sheet = contact_sheet(list(arr))
  try:
    from PIL import Image
    Image.fromarray(sheet).save(str(out) + '_sheet.png')
    print('wrote', str(out) + '_sheet.png')
    if a.gif:
      imgs = [Image.fromarray(f[..., :3]) for f in arr]
      imgs[0].save(str(out) + '.gif', save_all=True, append_images=imgs[1:],
                   duration=int(1000 / a.fps), loop=0, optimize=True)
      print('wrote', str(out) + '.gif')
  except Exception as exc:
    print('PIL unavailable or failed (%r); .npz still written' % (exc,))


if __name__ == '__main__':
  main()
