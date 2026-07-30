"""Hierarchical (Director-style) machinery for the DreamerV3 agent.

The manager emits a discrete goal code -- and, under ``variable_goal_length``, a
hold duration -- which a goal autoencoder decodes into a ``deter`` target for the
worker policy. This package holds the parts of that objective that are pure
functions of rollout tensors, so ``dreamerv3.agent.Agent`` is left holding only
the network wiring and the forward passes:

``tensors``
    Manager credit assignment: fixed-K block pooling and the variable-K
    switch-mask machinery (segment ids, pooled rewards/continuations, packing
    into a static buffer, and the reductions that keep the padded axis
    equivalent to fixed-K's exactly-sized one).
``heads``
    Reading multi-head policy distributions (log-probs, entropies, event
    alignment) in the shapes the actor-critic losses expect.
``losses``
    The actor-critic losses themselves, for the manager, the worker, and the
    flat (non-HRL) agent, plus ``lambda_return``.
``video``
    Image/video composition helpers for report and episode panels.
"""

from .heads import (
    align_skill_events,
    head_entropy_perdim_time,
    head_entropy_time,
    head_logp_time,
    manager_reinforce_policy,
    policy_time_slice,
)
from .losses import (
    imag_loss,
    imag_loss_mgr,
    imag_loss_wkr,
    lambda_return,
    repl_loss,
)
from .tensors import (
    aggregate_mgr_cont,
    aggregate_mgr_cont_variable,
    aggregate_mgr_extr_rew,
    aggregate_mgr_extr_rew_variable,
    decision_mean_rescale,
    downsample_at_switch_mask,
    downsample_manager_states,
    forward_fill_packed,
    goal_reward_cosine_max,
    imag_reward_pad,
    masked_cumprod,
    mgr_as_dict,
    pairwise_cosmax,
    patch_trailing_replay_state,
    relabel_truncated_last_duration,
    truncated_last_decision_mask,
    skill_switch,
    switch_valid_mask,
    variable_block_director_tensors,
    variable_segment_ids,
    worker_split_window,
)
from .video import resize_frames, tb_video_grid, vec_to_tb_rgb

__all__ = [
    'aggregate_mgr_cont',
    'aggregate_mgr_cont_variable',
    'aggregate_mgr_extr_rew',
    'aggregate_mgr_extr_rew_variable',
    'align_skill_events',
    'decision_mean_rescale',
    'downsample_at_switch_mask',
    'downsample_manager_states',
    'forward_fill_packed',
    'goal_reward_cosine_max',
    'head_entropy_perdim_time',
    'head_entropy_time',
    'head_logp_time',
    'imag_loss',
    'imag_loss_mgr',
    'imag_loss_wkr',
    'imag_reward_pad',
    'lambda_return',
    'manager_reinforce_policy',
    'masked_cumprod',
    'mgr_as_dict',
    'pairwise_cosmax',
    'patch_trailing_replay_state',
    'policy_time_slice',
    'relabel_truncated_last_duration',
    'truncated_last_decision_mask',
    'repl_loss',
    'resize_frames',
    'skill_switch',
    'switch_valid_mask',
    'tb_video_grid',
    'variable_block_director_tensors',
    'variable_segment_ids',
    'vec_to_tb_rgb',
    'worker_split_window',
]
