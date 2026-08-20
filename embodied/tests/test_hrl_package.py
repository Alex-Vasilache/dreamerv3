"""Structural guards for the ``dreamerv3.hrl`` package split.

``agent.py`` used to be one 4.5k-line module; its helpers now live in
``dreamerv3.hrl``. Nothing else in the suite imports every submodule, so a name
that silently fails to make the move (or a re-export that points at nothing)
would only surface at training time. These tests are cheap and catch exactly
that class of packaging mistake.
"""
import importlib
import pkgutil

import pytest

import dreamerv3.hrl as hrl

# Kept in sync with dreamerv3/hrl/ by test_no_unlisted_submodules below.
# `goal_ae`, `lipschitz` and `vq` arrived with the SOM-VAE / LipVQ goal
# autoencoder and were never added here, so that test had been failing on the
# branch since those modules landed.
SUBMODULES = [
    'explore',
    'goal_ae',
    'goals',
    'heads',
    'lipschitz',
    'losses',
    'manager',
    'reporting',
    'tensors',
    'video',
    'vq',
]


def test_every_submodule_imports():
  for name in SUBMODULES:
    importlib.import_module(f'dreamerv3.hrl.{name}')


def test_no_unlisted_submodules():
  """``SUBMODULES`` must stay in sync with what is actually on disk."""
  found = {m.name for m in pkgutil.iter_modules(hrl.__path__)}
  assert found == set(SUBMODULES), (sorted(found), sorted(SUBMODULES))


@pytest.mark.parametrize('name', hrl.__all__)
def test_all_exports_resolve(name):
  """Every name in ``__all__`` is importable and not accidentally ``None``.

  Guards the failure mode this split actually hit: a helper that fell between
  two extraction ranges, leaving ``__init__`` re-exporting a name no submodule
  defined any more.
  """
  assert hasattr(hrl, name), f'dreamerv3.hrl.__all__ lists missing name {name!r}'
  assert getattr(hrl, name) is not None


def test_agent_module_imports_and_exposes_public_api():
  agent = importlib.import_module('dreamerv3.agent')
  assert hasattr(agent, 'Agent')
  for method in ('policy', 'train', 'loss', 'report', 'init_policy'):
    assert callable(getattr(agent.Agent, method, None)), method


def test_agent_mixin_methods_are_all_reachable():
  """Every method the mixins provide must survive the MRO on ``Agent``."""
  from dreamerv3.agent import Agent
  from dreamerv3.hrl.goals import GoalCodeMixin
  from dreamerv3.hrl.manager import ManagerMixin
  from dreamerv3.hrl.reporting import ReportMixin

  for mixin in (GoalCodeMixin, ManagerMixin, ReportMixin):
    for name, value in vars(mixin).items():
      if name.startswith('__') or not callable(value):
        continue
      assert hasattr(Agent, name), f'{mixin.__name__}.{name} not reachable on Agent'


def test_removed_mechanisms_are_gone():
  """The superseded goal-masking / var-K branches must not creep back in.

  Only implicit sparsity (``goal_soft_reuse_adapt``) and block-pooled variable-K
  survive; these names all belonged to mechanisms that were removed.
  """
  import dreamerv3.agent as agent

  for name in ('mask_perblock_advantages', 'mask_logp_perblock_time',
               'bernoulli_entropy', 'bernoulli_kl',
               'aggregate_mgr_extr_rew_variable_fullres'):
    assert not hasattr(agent, name), f'{name} should have been removed'
  for name in ('_edit_running_goal', '_select_topk_mask', '_select_sparsemax_mask',
               '_delta_combine_skill', '_mask_prob', '_mask_logit',
               '_decode_goal_no_decoder_grad', '_mgr_goal_q_inp'):
    assert not hasattr(agent.Agent, name), f'Agent.{name} should have been removed'
