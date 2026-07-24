"""Smoke test: held goal does not drift across non-switch steps.

Runs a short policy rollout (no training) with masked + variable-duration goals,
then asserts that on every step where the decoded goal vector (``goal_deter``) is
held, the cached *rendered goal image* (``goal_img_*``) is held bit-for-bit too.
The image cache and the deter cache share the same refresh condition, so a held
deter must imply a held frame; this is what stabilises the policy mask_viz panel
even while the image decoder keeps training online.
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

  prev_goal = None
  prev_imgs = None
  held_checks = 0
  deter_violations = 0
  img_violations = 0

  def img_frames(mgr_skill):
    return {k: np.asarray(v[0]) for k, v in mgr_skill.items()
            if k.startswith('goal_img_')}

  def check_hold(tran, worker):
    nonlocal prev_goal, prev_imgs, held_checks
    nonlocal deter_violations, img_violations
    carry = jax.tree.map(
        lambda x: x[0] if isinstance(x, list) else x, driver.carry)
    enc, dyn, dec, prevact, mgr_skill, mgr_step = agent.model._unpack_carry(
        carry)
    goal = np.asarray(mgr_skill['goal_deter'][0])
    imgs = img_frames(mgr_skill)
    # A held step = the cached deter is unchanged (no manager switch / reset).
    if prev_goal is not None and np.allclose(goal, prev_goal, atol=0, rtol=0):
      held_checks += 1
      if not np.allclose(goal, prev_goal, atol=1e-6, rtol=1e-5):
        deter_violations += 1
      for k, v in imgs.items():
        if not np.array_equal(v, prev_imgs[k]):
          img_violations += 1
          print(f'IMG HOLD VIOLATION at step {worker} ({k}): changed '
                f'{int((v != prev_imgs[k]).sum())} px on a held step')
    prev_goal, prev_imgs = goal, imgs

  driver.on_step(check_hold)
  driver(agent.policy, steps=400)

  print(f'goal_hold: steps=400 held_checks={held_checks} '
        f'deter_violations={deter_violations} img_violations={img_violations} '
        f'img_keys={sorted((prev_imgs or {}).keys())}')
  if held_checks < 50:
    raise SystemExit(f'TOO FEW held steps ({held_checks}); duration sampling may be broken')
  if not prev_imgs:
    raise SystemExit('no goal_img_* cache in carry; image-hold fix not active')
  if deter_violations:
    raise SystemExit(f'goal_deter drifted on {deter_violations}/{held_checks} held steps')
  if img_violations:
    raise SystemExit(f'goal image drifted on {img_violations} held steps')
  print('GOAL_HOLD OK')


if __name__ == '__main__':
  main()
