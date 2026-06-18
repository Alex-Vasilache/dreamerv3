"""Smoke test: goal_deter is held constant while goal_code Z is unchanged.

Runs a short policy rollout (no training) with masked + variable-duration goals,
then asserts decoded goal vectors do not drift across held steps.
"""
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent / 'dreamerv3'
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))

import elements
import embodied
import jax
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main as m


def make_config():
  defaults = yaml.YAML(typ='safe').load(
      (folder / 'configs.yaml').read_text())['defaults']
  config = elements.Config(defaults)
  config = config.update({
      'logdir': '/tmp/smoke_goal_hold',
      'task': 'dmc_cartpole_swingup',
      'env.dmc.image': True,
      'env.dmc.proprio': True,
      'env.dmc.size': [32, 32],
      'agent.enc.simple.outer': True,
      'agent.dec.simple.outer': True,
      'agent.variable_goal_length': True,
      'agent.goal_duration_min': 1,
      'agent.goal_duration_max': 8,
      'agent.mask_sparsity_mode': 'prob',
      'agent.mask_sparsity_target': 0.3,
      'agent.mgr_cond_goalcode': True,
      'jax.platform': 'cuda',
      'jax.prealloc': False,
  })
  for name in ('size6m', 'debug', 'masked_goals'):
    block = yaml.YAML(typ='safe').load(
        (folder / 'configs.yaml').read_text())[name]
    config = config.update(block)
  return config


def main():
  config = make_config()
  agent = m.make_agent(config)
  driver = embodied.Driver([bind(m.make_env, config, 0)], parallel=False)
  driver.reset(agent.init_policy)

  prev_code = prev_goal = None
  held_checks = 0
  violations = 0

  def check_hold(tran, worker):
    nonlocal prev_code, prev_goal, held_checks, violations
    carry = jax.tree.map(
        lambda x: x[0] if isinstance(x, list) else x, driver.carry)
    enc, dyn, dec, prevact, mgr_skill, mgr_step = agent.model._unpack_carry(
        carry)
    code = np.asarray(mgr_skill['goal_code'][0])
    goal = np.asarray(mgr_skill['goal_deter'][0])
    if prev_code is not None and np.allclose(code, prev_code, atol=0, rtol=0):
      held_checks += 1
      if not np.allclose(goal, prev_goal, atol=1e-6, rtol=1e-5):
        violations += 1
        print(f'HOLD VIOLATION at step {worker}: max_delta='
              f'{np.abs(goal - prev_goal).max():.6g}')
    prev_code, prev_goal = code, goal

  driver.on_step(check_hold)
  driver(agent.policy, steps=400)

  print(f'goal_hold: steps=400 held_checks={held_checks} violations={violations}')
  if held_checks < 50:
    raise SystemExit(f'TOO FEW held steps ({held_checks}); duration sampling may be broken')
  if violations:
    raise SystemExit(f'goal_deter drifted on {violations}/{held_checks} held steps')
  print('GOAL_HOLD OK')


if __name__ == '__main__':
  main()
