"""The simulated robots against what the real one does.

Signs and stops as measured on the robot: positive tilt is a forward lean,
forward motor output (negative wheel speed) tips the body back, and it rests
on its front bumper at bumper_hi and on its back bumper at bumper_lo.
"""

import math
import pathlib

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from embodied.envs import robot_mujoco, robot_sim  # noqa: E402

HZ = 65.0
MESHES = pathlib.Path.home() / '.cache' / 'spr_meshes'


def make(model):
  cls = robot_mujoco.RobotMujoco if model == 'mujoco' else robot_sim.RobotSim
  return cls('balance', hz=HZ, discrete=False, symmetric=False, sim_noise=0.0)


def run(env, actions):
  thetas = []
  for act in actions:
    env.pending = np.asarray(act, np.float64)
    sensors, _ = env._recv()
    thetas.append(sensors['theta'])
  return np.array(thetas)


@pytest.mark.parametrize('model', ['mujoco', 'pendulum'])
def test_forward_output_tips_the_body_back(model):
  env = make(model)
  env.theta, env.omega = env.theta_zero, 0.0
  env.theta_eq = env.theta_zero
  run(env, [(0, 0)] * 2)
  thetas = run(env, [(0.8, 0.8)] * 3)  # a tick of pipeline, then the wheels
  assert env.speed.mean() < 0          # forward
  assert thetas[-1] < env.theta_zero - math.radians(0.5)


@pytest.mark.parametrize('model', ['mujoco', 'pendulum'])
@pytest.mark.parametrize('side', ['front', 'back'])
def test_it_comes_to_rest_on_each_bumper_at_its_measured_tilt(model, side):
  env = make(model)
  stop = env.bumper_hi if side == 'front' else env.bumper_lo
  env.theta = env.theta_zero + 0.5 * stop
  env.omega = 0.0
  thetas = run(env, [(0, 0)] * int(3 * HZ))
  assert abs(thetas[-1] - env.theta_zero - stop) < math.radians(0.3)


@pytest.mark.skipif(not MESHES.exists(), reason='tools/robot/dev/fetch_spr_meshes.sh')
@pytest.mark.parametrize('side', ['front', 'back'])
def test_the_drawn_body_touches_the_floor_on_its_bumpers(side):
  env = make('mujoco')
  stop = env.bumper_hi if side == 'front' else env.bumper_lo
  env.theta = env.theta_zero + 0.5 * stop
  run(env, [(0, 0)] * int(3 * HZ))
  model = mujoco.MjModel.from_xml_string(robot_mujoco.model_xml(
      -env.bumper_lo, env.bumper_hi, visual=(0, 0, 0), meshdir=str(MESHES)))
  data = mujoco.MjData(model)
  data.qpos[:] = env.data.qpos
  mujoco.mj_forward(model, data)
  lowest = np.inf
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    m = model.geom_dataid[g]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m)
    if 'wheel' in name:
      continue
    v0, nv = model.mesh_vertadr[m], model.mesh_vertnum[m]
    verts = model.mesh_vert[v0:v0 + nv]
    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    lowest = min(lowest, world[:, 2].min())
  assert abs(lowest) < 0.001   # m
