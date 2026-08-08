"""What is and is not reproducible given `--seed`.

Written to answer "do we even have determinism with the same seed?". The answer
is no, and these tests pin down each independent reason so that fixing one does
not create the impression that the whole chain is fixed.

Every test here is a statement about the CURRENT behaviour. A test that starts
failing means someone made the run more reproducible, which is good -- update
the test and the corresponding row in `docs/VERIFICATION.md`.
"""

import elements
import numpy as np
import pytest
import ruamel.yaml as yaml

import embodied

CONFIGS = elements.Path(__file__).parent.parent.parent / 'dreamerv3/configs.yaml'


def load_defaults():
  return yaml.YAML(typ='safe').load(CONFIGS.read())['defaults']


class TestEnvSeeding:
  """DMC environments never receive a seed."""

  def test_dmc_ctor_does_not_take_or_forward_a_seed(self):
    import inspect

    from embodied.envs import dmc
    sig = inspect.signature(dmc.DMC.__init__)
    assert 'seed' not in sig.parameters, (
        'DMC now accepts a seed; wire it into suite.load(task_kwargs=...) and '
        'update docs/VERIFICATION.md')
    src = inspect.getsource(dmc.DMC.__init__)
    assert 'task_kwargs' not in src, (
        'suite.load now gets task_kwargs; check whether `random` is set')

  def test_only_dmlab_opts_into_use_seed(self):
    """`make_env` seeds an env only when its config block sets `use_seed`."""
    env = load_defaults()['env']
    opted_in = {k for k, v in env.items() if isinstance(v, dict) and v.get('use_seed')}
    assert opted_in == {'dmlab'}, (
        f'expected only dmlab to opt into seeding, got {opted_in}')
    assert 'use_seed' not in env['dmc']

  def test_two_identical_dmc_envs_start_from_different_states(self):
    """The empirical consequence: same task, same code path, different rollout."""
    from embodied.envs import dmc
    try:
      a = dmc.DMC('hopper_hop', repeat=1, size=(16, 16), image=False, proprio=True)
      b = dmc.DMC('hopper_hop', repeat=1, size=(16, 16), image=False, proprio=True)
    except Exception as e:  # no GL backend on this node
      pytest.skip(f'cannot construct DMC here: {e}')
    act = {k: np.zeros(v.shape, v.dtype) for k, v in a.act_space.items()}
    act['reset'] = np.bool_(True)
    obs_a, obs_b = a.step(act), b.step(act)
    obs_a = {k: np.asarray(v) for k, v in obs_a.items()}
    obs_b = {k: np.asarray(v) for k, v in obs_b.items()}
    keys = [k for k in obs_a if obs_a[k].dtype.kind == 'f' and obs_a[k].size > 1]
    assert keys, 'expected at least one float observation to compare'
    same = all(np.allclose(obs_a[k], obs_b[k]) for k in keys)
    assert not same, (
        'two fresh DMC envs produced identical initial states; either dm_control '
        'started seeding by default or DMC now forwards a seed')


class TestScheduleSeeding:
  """Schedules that gate RNG-consuming work are wall-clock, not step-based."""

  def test_report_save_and_log_all_use_the_wall_clock(self):
    import inspect

    from embodied.run import train
    src = inspect.getsource(train)
    for name in ('should_log', 'should_report', 'should_save'):
      line = next(l for l in src.splitlines() if l.strip().startswith(f'{name} ='))
      assert 'LocalClock' in line, (
          f'{name} is no longer wall-clock: {line.strip()}')

  def test_should_train_is_step_based_and_therefore_reproducible(self):
    import inspect

    from embodied.run import train
    src = inspect.getsource(train)
    line = next(l for l in src.splitlines() if l.strip().startswith('should_train ='))
    assert 'Ratio' in line, line.strip()


class TestJaxDeterminism:

  def test_deterministic_flag_is_off_by_default(self):
    assert load_defaults()['jax']['deterministic'] is False, (
        'jax.deterministic changed; XLA reductions may now be reproducible')


class TestWhatSeedDoesControl:
  """The seed is not inert -- it does drive parameter initialization."""

  def test_agent_receives_the_config_seed(self):
    import inspect

    from dreamerv3 import main
    src = inspect.getsource(main.make_agent)
    assert 'seed=config.seed' in src.replace(' ', ''), (
        'the agent no longer receives config.seed')
