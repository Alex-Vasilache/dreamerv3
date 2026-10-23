"""3D replay of two `sim_rollout.py` recordings, side by side, in MuJoCo.

Recordings of the MuJoCo robot (env.robot.sim_model mujoco) carry the body's
full state and are replayed exactly, on the same model. Recordings of the
planar model (pendulum) are replayed kinematically on a stand-in 3D robot: the
body pitches by the recorded tilt, each wheel turns by its own travelled
distance, and the robot drives and turns on a checkered floor by integrating
the two wheel speeds. One camera per robot follows it at a fixed world angle.
1920x1080, 30 fps, real time. --meshes draws the MuJoCo robot with the real
body, phone and wheel meshes (tools/robot/dev/fetch_spr_meshes.sh).

Needs MuJoCo rendering (MUJOCO_GL=egl on a GPU node) and an ffmpeg binary
(--ffmpeg, e.g. from imageio-ffmpeg).

  MUJOCO_GL=egl python tools/robot/dev/sim_video3d.py A.npz B.npz OUT.mp4 \
      --labels "Flat DreamerV3" "HRL (OG, tuned)" --ffmpeg /path/to/ffmpeg
"""

import argparse
import math
import subprocess

import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from embodied.envs import robot_mujoco  # noqa: E402

W, H = 1920, 1080
HEAD = 110
VIEW_W, VIEW_H = W // 2, H - HEAD
FPS = 30
R = 0.03          # wheel radius, m
TRACK = 0.12      # wheel separation, m
FONT = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'
FONT_M = '/usr/share/fonts/dejavu/DejaVuSansMono.ttf'


def xml(rgb):
  r, g, b = rgb
  return f"""
<mujoco>
  <visual>
    <global offwidth="{VIEW_W}" offheight="{VIEW_H}"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.4 0.4 0.4" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.95 0.95 0.93" rgb2="0.84 0.85 0.83"/>
    <material name="floor" texture="grid" texrepeat="4 4" texuniform="true"
              reflectance="0" specular="0" shininess="0"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1"
             rgb2="0.85 0.88 0.9" width="64" height="64"/>
  </asset>
  <worldbody>
    <geom type="plane" size="200 200 0.1" material="floor"/>
    <body name="robot" pos="0 0 {R}">
      <freejoint/>
      <light mode="trackcom" pos="0.4 -0.4 1.2" dir="-0.3 0.3 -1"
             diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1" castshadow="true"/>
      <camera name="follow" mode="trackcom" pos="0.25 -0.85 0.32"
              xyaxes="{cam_axes()}"/>
      <!-- base holding the axle and the bumpers front and back -->
      <geom type="box" size="0.022 0.05 0.018" pos="0 0 0.008" rgba="0.25 0.25 0.25 1"/>
      <geom type="capsule" fromto="0.03 -0.04 -0.004 0.03 0.04 -0.004" size="0.005" rgba="0.84 0.37 0 1"/>
      <geom type="capsule" fromto="-0.03 -0.04 -0.004 -0.03 0.04 -0.004" size="0.005" rgba="0.84 0.37 0 1"/>
      <!-- the phone, upright, screen facing forward -->
      <geom type="box" size="0.006 0.038 0.075" pos="0 0 0.1" rgba="{r} {g} {b} 1"/>
      <geom type="box" size="0.0005 0.033 0.068" pos="0.0062 0 0.1" rgba="0.34 0.71 0.91 0.9"/>
      <!-- push arrows, shown only while a push acts -->
      <geom name="push_back" type="capsule" fromto="0.20 0 0.16 0.04 0 0.16" size="0.008" rgba="0.84 0.37 0 0" contype="0" conaffinity="0"/>
      <geom name="push_back_tip" type="sphere" pos="0.04 0 0.16" size="0.016" rgba="0.84 0.37 0 0" contype="0" conaffinity="0"/>
      <geom name="push_fwd" type="capsule" fromto="-0.20 0 0.16 -0.04 0 0.16" size="0.008" rgba="0.84 0.37 0 0" contype="0" conaffinity="0"/>
      <geom name="push_fwd_tip" type="sphere" pos="-0.04 0 0.16" size="0.016" rgba="0.84 0.37 0 0" contype="0" conaffinity="0"/>
      <body name="wheel_l" pos="0 {TRACK / 2} 0">
        <joint name="wl" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="{R} 0.008" euler="90 0 0" rgba="0.15 0.15 0.15 1"/>
        <geom type="box" size="{R * 0.8} 0.0085 0.004" rgba="0.75 0.75 0.75 1"/>
      </body>
      <body name="wheel_r" pos="0 {-TRACK / 2} 0">
        <joint name="wr" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="{R} 0.008" euler="90 0 0" rgba="0.15 0.15 0.15 1"/>
        <geom type="box" size="{R * 0.8} 0.0085 0.004" rgba="0.75 0.75 0.75 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def cam_axes():
  pos = np.array([0.25, -0.85, 0.32])
  target = np.array([0.0, 0.0, 0.07])
  f = target - pos
  f /= np.linalg.norm(f)
  x = np.cross(f, [0, 0, 1])
  x /= np.linalg.norm(x)
  y = np.cross(-f, x)
  return ' '.join(f'{v:.4f}' for v in (*x, *y))


def quat(yaw, pitch):
  qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
  qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
  w1, x1, y1, z1 = qz
  w2, x2, y2, z2 = qy
  return np.array([
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def trajectory(run):
  hz = float(run['hz'])
  # Forward is negative wheel speed on this robot; mm/s -> m per step.
  dl = -np.asarray(run['v_l']) * 1e-3 / hz
  dr = -np.asarray(run['v_r']) * 1e-3 / hz
  yaw = np.cumsum((dr - dl) / TRACK)
  ds = 0.5 * (dl + dr)
  px = np.cumsum(ds * np.cos(yaw))
  py = np.cumsum(ds * np.sin(yaw))
  return dict(px=px, py=py, yaw=yaw, wl=np.cumsum(dl) / R, wr=np.cumsum(dr) / R)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('a')
  parser.add_argument('b')
  parser.add_argument('out')
  parser.add_argument('--labels', nargs=2, default=['Flat DreamerV3', 'HRL'])
  parser.add_argument('--title', default='')
  parser.add_argument('--ffmpeg', default='ffmpeg')
  parser.add_argument('--meshes', default=None,
                      help='directory of the robot STLs (fetch_spr_meshes.sh)')
  args = parser.parse_args()
  runs = [dict(np.load(p, allow_pickle=True)) for p in (args.a, args.b)]
  colours = [(0.1, 0.1, 0.1), (0.0, 0.45, 0.70)]
  hexes = ['#000000', '#0072B2']
  task = str(runs[0]['task'])
  hz = float(runs[0]['hz'])
  n = min(len(r['t']) for r in runs)
  stride = max(1, int(round(hz / FPS)))
  title = args.title or ('Balance at 65 Hz, pushed at 8, 14, 20, 26 s'
                         if task != 'command' else
                         'Command following at 65 Hz, fixed command schedule')

  scenes = []
  for run, rgb in zip(runs, colours):
    if 'qpos' in run:
      lo, hi = float(run['bumper_lo']), float(run['bumper_hi'])
      model = mujoco.MjModel.from_xml_string(robot_mujoco.model_xml(
          bumper_back=hi, bumper_front=-lo, visual=rgb, meshdir=args.meshes))
    else:
      model = mujoco.MjModel.from_xml_string(xml(rgb))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, VIEW_H, VIEW_W)
    geoms = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, k)
             for k in ('push_back', 'push_back_tip', 'push_fwd', 'push_fwd_tip')}
    pushes = [(float(t), float(k)) for t, k in zip(run['t'], run['pushed']) if k]
    scenes.append(dict(run=run, model=model, data=data, renderer=renderer,
                       traj=trajectory(run), geoms=geoms, pushes=pushes))

  f_title = ImageFont.truetype(FONT, 38)
  f_clock = ImageFont.truetype(FONT, 26)
  f_label = ImageFont.truetype(FONT_B, 36)
  f_info = ImageFont.truetype(FONT_M, 26)
  ffmpeg = subprocess.Popen(
      [args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', args.out],
      stdin=subprocess.PIPE)
  still = None
  for frame, i in enumerate(range(0, n, stride)):
    t = i / hz
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((W // 2, 22), title, font=f_title, fill='black', anchor='mt')
    draw.text((W // 2, 72), f't = {t:4.1f} s', font=f_clock, fill='#404040',
              anchor='mt')
    for col, (s, label, hexc) in enumerate(zip(scenes, args.labels, hexes)):
      run, tr, m, d = s['run'], s['traj'], s['model'], s['data']
      theta = float(run['theta'][i])
      if 'qpos' in run:
        d.qpos[:] = run['qpos'][i]
      else:
        d.qpos[:3] = [tr['px'][i], tr['py'][i], R]
        # Positive tilt is a forward lean: the top tips toward +x.
        d.qpos[3:7] = quat(tr['yaw'][i], theta)
        d.qpos[7] = tr['wl'][i] - theta
        d.qpos[8] = tr['wr'][i] - theta
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      mujoco.mj_camlight(m, d)
      now = [k for tp, k in s['pushes'] if 0 <= t - tp < 0.6]
      for name, gid in s['geoms'].items():
        show = now and ((now[0] > 0) == name.startswith('push_back'))
        m.geom_rgba[gid, 3] = 1.0 if show else 0.0
      s['renderer'].update_scene(d, camera='follow')
      view = Image.fromarray(s['renderer'].render())
      x0 = col * VIEW_W
      canvas.paste(view, (x0, HEAD))
      draw.text((x0 + VIEW_W // 2, HEAD + 18), label, font=f_label, fill=hexc,
                anchor='mt')
      win = slice(max(0, i - int(hz)), i + 1)
      lines = [f'tilt    {math.degrees(theta):+5.1f} deg',
               f'reward  {np.mean(run["reward"][win]):+5.2f} /step']
      if task == 'command':
        fwd = -0.5 * (run['v_l'][i] + run['v_r'][i]) / float(run['command_speed'])
        turn = 0.5 * (run['v_l'][i] - run['v_r'][i]) / float(run['command_turn'])
        lines += [f'command fwd {run["cmd_f"][i]:+.2f}  turn {run["cmd_t"][i]:+.2f}',
                  f'actual  fwd {fwd:+.2f}  turn {turn:+.2f}']
      y = HEAD + VIEW_H - 30 - 34 * len(lines)
      for line in lines:
        draw.text((x0 + 30, y), line, font=f_info, fill='#202020')
        y += 34
      if now:
        draw.text((x0 + VIEW_W - 30, HEAD + 80), 'push', font=f_label,
                  fill='#D55E00', anchor='rt')
    draw.line([(VIEW_W, HEAD), (VIEW_W, H)], fill='#d5d5d2', width=3)
    ffmpeg.stdin.write(np.asarray(canvas).tobytes())
    if frame == int(10.6 * FPS):
      still = canvas.copy()
  ffmpeg.stdin.close()
  ffmpeg.wait()
  if still is not None:
    still.save(args.out.rsplit('.', 1)[0] + '.png')
  print('wrote', args.out)


if __name__ == '__main__':
  main()
