"""Director hierarchy helpers: trajectory assembly and split/abstract."""

from __future__ import annotations

import jax.numpy as jnp

from . import director_rewards as dr
from . import director_trajectories as dtr

f32 = jnp.float32


def feat2tensor(deter, stoch):
  """Flatten RSSM stoch for policy/critic input."""
  flat = stoch.reshape(*stoch.shape[:-2], -1)
  return jnp.concatenate([deter.astype(f32), flat.astype(f32)], -1)


def build_imag_traj_for_hierarchy(
    *,
    deter_bt,
    stoch_bt,
    act_primitives,
    skill_idx,
    goal_bt,
    cont_bt,
    rew_wm_pred,
    expl_rew_bt,
    goal_reward_name,
    enc_fwd,
):
  """Assemble `traj` dict (B, T, ...) before split/abstract (Director `train_jointly`)."""
  context = dr.broadcast_context(deter_bt)
  rew_goal = dr.goal_reward(
      goal_reward_name,
      deter_bt,
      goal_bt,
      skill_idx.astype(f32),
      context,
      enc_fwd=enc_fwd,
  )
  rew_extr = rew_wm_pred[:, :-1]
  assert rew_extr.shape == expl_rew_bt.shape == rew_goal.shape, (
      rew_extr.shape, expl_rew_bt.shape, rew_goal.shape)
  return dict(
      deter=deter_bt,
      stoch=stoch_bt,
      action=act_primitives,
      skill=skill_idx,
      goal=goal_bt,
      cont=cont_bt,
      reward_extr=rew_extr,
      reward_expl=expl_rew_bt,
      reward_goal=rew_goal,
  )


def split_and_abstract(traj, k, discount):
  return dtr.split_traj(traj, k, discount), dtr.abstract_traj(traj, k, discount)
