"""Animate two `sim_rollout.py` recordings side by side (MP4 + GIF).

Each column is one agent: a side view of the robot (wheel, phone body,
bumper stops, ground marks that scroll as it drives), and below it the tilt
and, for the command task, the forward speed and turn rate against the
command. 16:9, 30 fps, real time.

  python tools/robot/dev/sim_video.py A.npz B.npz OUT --labels "Flat DreamerV3" "HRL (OG, tuned)"
"""

import argparse
import math
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms

COLOURS = ['#000000', '#0072B2']
CMD = '#E69F00'
GRID = '#d5d5d2'
WHEEL_R = 0.03      # m
BODY_H = 0.15       # m, the phone standing in its cradle
BODY_W = 0.075
FPS = 30


def load(path):
  d = np.load(path, allow_pickle=True)
  return {k: d[k] for k in d.files}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('a')
  parser.add_argument('b')
  parser.add_argument('out')
  parser.add_argument('--labels', nargs=2, default=['Flat DreamerV3', 'HRL'])
  parser.add_argument('--title', default='')
  parser.add_argument('--gif', action='store_true')
  args = parser.parse_args()
  runs = [load(args.a), load(args.b)]
  task = str(runs[0]['task'])
  hz = float(runs[0]['hz'])
  n = min(len(r['t']) for r in runs)
  stride = max(1, int(round(hz / FPS)))
  frames = range(0, n, stride)
  command = task == 'command'

  plt.rcParams.update({'font.size': 13})
  fig = plt.figure(figsize=(16, 9), dpi=120, facecolor='white')
  rows = 3 if command else 2
  ratios = [1.5, 1, 1] if command else [1.6, 1]
  gs = fig.add_gridspec(rows, 2, height_ratios=ratios, hspace=0.35,
                        wspace=0.12, left=0.06, right=0.98, top=0.88,
                        bottom=0.07)
  title = args.title or ('Balance at 65 Hz, pushed at 8, 14, 20, 26 s'
                         if not command else
                         'Command following at 65 Hz, fixed command schedule')
  fig.suptitle(title, fontsize=17)
  clock = fig.text(0.5, 0.925, '', ha='center', fontsize=14, color='#404040')
  artists = []
  for col, (run, label, colour) in enumerate(zip(runs, args.labels, COLOURS)):
    t = run['t'][:n]
    theta = np.degrees(run['theta'][:n])
    ax = fig.add_subplot(gs[0, col])
    ax.set_title(label, fontsize=17, color=colour)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.02, 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot([-1, 1], [0, 0], color='#404040', lw=2)
    marks = [ax.plot([], [], color='#8a8a85', lw=1.5)[0] for _ in range(8)]
    wheel = patches.Circle((0, WHEEL_R), WHEEL_R, fc='#d5d5d2', ec='#404040',
                           lw=2)
    spoke, = ax.plot([], [], color='#404040', lw=2)
    body = patches.FancyBboxPatch(
        (-BODY_W / 2, WHEEL_R), BODY_W, BODY_H,
        boxstyle='round,pad=0,rounding_size=0.008', fc=colour, ec='#202020',
        lw=1.5, alpha=0.9)
    screen = patches.Rectangle((-BODY_W / 2 + 0.008, WHEEL_R + 0.02),
                               BODY_W - 0.016, BODY_H - 0.04, fc='#56B4E9',
                               alpha=0.35, lw=0)
    ax.add_patch(body)
    ax.add_patch(screen)
    ax.add_patch(wheel)
    info = ax.text(-0.29, 0.19, '', ha='left', va='top', fontsize=13,
                   color='#404040', family='monospace')
    push = ax.text(0.12, 0.12, '', ha='center', fontsize=16, color='#D55E00',
                   fontweight='bold')
    pushes = [(float(tp), float(k)) for tp, k in zip(run['t'], run['pushed']) if k]
    if command:
      heading = ax.annotate('', xy=(0.22, 0.13), xytext=(0.22, 0.13),
                            arrowprops=dict(arrowstyle='->', lw=2.5,
                                            color=colour))
      ax.text(0.22, 0.19, 'heading\n(top view)', ha='center', va='top', fontsize=11,
              color='#404040')
    else:
      heading = None

    tax = fig.add_subplot(gs[1, col])
    tax.plot(t, theta, color=colour, lw=1.6)
    tax.axhline(0, color='#8a8a85', lw=0.8)
    for lim in (run['bumper_lo'], run['bumper_hi']):
      tax.axhline(np.degrees(lim), color='#D55E00', lw=0.8, ls=':')
    tax.set_xlim(0, t[-1])
    tax.set_ylim(-9, 11)
    tax.grid(color=GRID, lw=0.7)
    if col == 0:
      tax.set_ylabel('tilt (deg)', fontsize=13)
    cursors = [tax.axvline(0, color='#404040', lw=1)]
    if command:
      sax = fig.add_subplot(gs[2, col])
      fwd = -0.5 * (run['v_l'][:n] + run['v_r'][:n]) / float(run['command_speed'])
      turn = 0.5 * (run['v_l'][:n] - run['v_r'][:n]) / float(run['command_turn'])
      sax.plot(t, run['cmd_f'][:n], color=CMD, lw=2, label='forward command')
      sax.plot(t, fwd, color=colour, lw=1.5, label='forward speed')
      sax.plot(t, run['cmd_t'][:n], color=CMD, lw=2, ls='--',
               label='turn command')
      sax.plot(t, turn, color=colour, lw=1.5, ls='--', label='turn rate')
      sax.set_xlim(0, t[-1])
      sax.set_ylim(-1, 1)
      sax.grid(color=GRID, lw=0.7)
      sax.set_xlabel('time (s)', fontsize=13)
      if col == 0:
        sax.set_ylabel('speed (÷ top)', fontsize=13)
        sax.legend(fontsize=10, frameon=False, ncol=2, loc='lower left')
      cursors.append(sax.axvline(0, color='#404040', lw=1))
    else:
      tax.set_xlabel('time (s)', fontsize=13)
    artists.append(dict(run=run, ax=ax, marks=marks, wheel=wheel, spoke=spoke,
                        body=body, screen=screen, info=info, push=push,
                        heading=heading, cursors=cursors, pushes=pushes))

  def draw(i):
    clock.set_text(f't = {i / hz:4.1f} s')
    out = [clock]
    for a in artists:
      run = a['run']
      th = float(run['theta'][i])
      x = float(run['x'][i])
      # Scrolling ground marks: the camera follows the robot.
      for k, m in enumerate(a['marks']):
        gx = (k * 0.1 - x) % 0.8 - 0.4
        m.set_data([gx, gx - 0.02], [0, -0.015])
      roll = -x / WHEEL_R
      a['spoke'].set_data([0, WHEEL_R * math.cos(roll)],
                          [WHEEL_R, WHEEL_R + WHEEL_R * math.sin(roll)])
      # Positive tilt is a backward lean: the body tips left (forward = right).
      rot = transforms.Affine2D().rotate_around(0, WHEEL_R, th) + a['ax'].transData
      a['body'].set_transform(rot)
      a['screen'].set_transform(rot)
      win = slice(max(0, i - int(hz)), i + 1)
      a['info'].set_text(f'tilt    {math.degrees(th):+5.1f}°\n'
                         f'reward  {np.mean(run["reward"][win]):+5.2f} /step')
      now = [k for tp, k in a['pushes'] if 0 <= i / hz - tp < 0.8]
      a['push'].set_text(('← push' if now[0] > 0 else 'push →') if now else '')
      if a['heading'] is not None:
        yaw = float(run['yaw'][i])
        a['heading'].xy = (0.22 + 0.04 * math.cos(yaw), 0.11 + 0.04 * math.sin(yaw))
        a['heading'].set_position((0.22, 0.11))
      for c in a['cursors']:
        c.set_xdata([i / hz, i / hz])
      out += [a['body'], a['screen'], a['spoke'], a['info'], a['push'],
              *a['marks'], *a['cursors']]
    return out

  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
  import imageio_ffmpeg
  matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
  anim.save(out.with_suffix('.mp4'), writer=animation.FFMpegWriter(
      fps=FPS, bitrate=4000, extra_args=['-pix_fmt', 'yuv420p']))
  print('wrote', out.with_suffix('.mp4'))
  draw(frames[len(frames) // 3])
  fig.savefig(out.with_suffix('.png'), dpi=120, facecolor='white')
  if args.gif:
    anim2 = animation.FuncAnimation(fig, draw, frames=range(0, n, stride * 2))
    anim2.save(out.with_suffix('.gif'), writer=animation.PillowWriter(fps=FPS // 2),
               dpi=60)
    print('wrote', out.with_suffix('.gif'))


if __name__ == '__main__':
  main()
