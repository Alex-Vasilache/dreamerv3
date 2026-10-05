"""The driver's `executed/` channel: an env reporting the action it really ran.

A robot with an onboard controller decides for itself and tells us after the
fact. What the world model must learn from is what the wheels did, not what the
actor asked for -- so an env may return `executed/<key>` and the driver lifts it
over the policy's own action when it assembles the transition.

Without this the on-device policy would train the world model on actions that
were computed but never applied, which is the kind of mismatch that produces a
plausible-looking learning curve and a robot that does not improve.
"""

import embodied
import numpy as np
from embodied.envs import dummy


class ExecutedEnv(embodied.Env):
  """A dummy env that overrides the action with one of its own choosing."""

  def __init__(self, length=10, report=True):
    self._env = dummy.Dummy('disc', length=length)
    self._report = report
    self.executed = None

  @property
  def obs_space(self):
    space = dict(self._env.obs_space)
    if self._report:
      space['executed/act_disc'] = elements_space()
    return space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    if self._report:
      # Deliberately different from whatever the policy asked for, so the test
      # cannot pass by coincidence.
      self.executed = np.int32((int(action['act_disc']) + 1) % 5)
      obs['executed/act_disc'] = self.executed
    return obs


def elements_space():
  import elements
  return elements.Space(np.int32, (), 0, 5)


class Agent:

  def init_policy(self, batch_size):
    return ()

  def policy(self, carry, obs, **kwargs):
    batch = len(obs['is_first'])
    acts = {'act_disc': np.zeros(batch, np.int32), 'act_cont': np.zeros((batch, 6), np.float32)}
    return carry, acts, {}


class TestExecutedChannel:

  def test_executed_overrides_the_policy_action(self):
    agent = Agent()
    env = ExecutedEnv()
    driver = embodied.Driver([lambda: env], parallel=False)
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    # The policy always asks for 0; the env always executes something else.
    assert any(int(t['act_disc']) != 0 for t in seq)
    assert all(int(t['act_disc']) in (0, 1, 2, 3, 4) for t in seq)

  def test_executed_key_is_not_left_in_the_transition(self):
    agent = Agent()
    driver = embodied.Driver([lambda: ExecutedEnv()], parallel=False)
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    assert all('executed/act_disc' not in t for t in seq)

  def test_absent_channel_leaves_the_policy_action_alone(self):
    agent = Agent()
    driver = embodied.Driver(
        [lambda: ExecutedEnv(report=False)], parallel=False)
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    assert all(int(t['act_disc']) == 0 for t in seq)

  def test_executed_is_masked_on_the_last_step(self):
    """`acts` are zeroed on is_last; the executed action must be too, or the
    final transition of every episode carries an action that was never run."""
    agent = Agent()
    driver = embodied.Driver([lambda: ExecutedEnv()], parallel=False)
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    last = [t for t in seq if bool(t['is_last'])]
    assert last, 'episode never ended'
    assert all(int(t['act_disc']) == 0 for t in last)
