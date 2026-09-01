"""Goal-code encoding/decoding and the goal-derived reward signals.

The goal autoencoder maps an RSSM ``deter`` state to a discrete ``(L, C)`` skill
code and back. The manager picks codes; the worker is rewarded for driving
``deter`` toward the decoded goal (``goal_reward_cosine_max``), and the manager
gets an exploration bonus from the autoencoder's own reconstruction error.
"""
import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax.nets as nn

from .tensors import goal_reward_cosine_max, imag_reward_pad

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)


class GoalCodeMixin:
  """Goal-code helpers mixed into ``dreamerv3.agent.Agent``."""

  def _running_goal_code(self, skills):
    """The skill code that decodes to the worker goal (the emitted skill).

    ``goal_code`` is still accepted so callers can pass a synthetic seed dict
    (e.g. the current-state encoding used to prime the manager's conditioning
    channel at an episode / imagination boundary)."""
    if isinstance(skills, dict) and 'goal_code' in skills:
      return skills['goal_code']
    return skills['skill'] if isinstance(skills, dict) else skills

  def _encode_goal_code(self, deter, bdims):
    """Encode a ``deter`` vector into a sampled skill code (goal-VAE encoder)."""
    enc = sample(self.goal_enc(sg(deter), bdims))
    return sg(enc['skill'] if isinstance(enc, dict) else enc)

  def _goal_from_skill(self, skills, bdims=1):
    """Decode the running goal code to a ``deter`` goal vector (single batch dim)."""
    return self.goal_dec(self._running_goal_code(skills), bdims).pred()

  def _held_goal_deter(self, mgr_skill, update, reset, cached_goal):
    """Decode Z to deter; refresh on manager switch or episode reset, else hold."""
    fresh = sg(self._goal_from_skill(jax.tree.map(sg, mgr_skill), bdims=1))
    refresh = jnp.logical_or(update, reset)
    rshape = refresh.reshape(refresh.shape + (1,) * (fresh.ndim - refresh.ndim))
    return jnp.where(rshape, fresh, cached_goal)

  def _goals_from_skills(self, skills, bdims=2):
    """Batch-decode running goal codes to goal vectors (Director ``dec.mode()``)."""
    s = self._running_goal_code(skills)
    bshape = s.shape[:bdims]
    flat = s.reshape((-1, *s.shape[bdims:]))
    goals = self.goal_dec(flat, 1).pred()
    return goals.reshape(bshape + goals.shape[1:])

  def _feat_from_goal(self, goal):
    """RSSM feat for image decode from a proposed ``deter`` goal vector."""
    goal = nn.cast(goal)
    logit = self.dyn._prior(goal)
    stoch = nn.cast(self.dyn._dist(logit).pred())
    return dict(deter=goal, stoch=stoch, logit=logit)

  def _wkr_goal_reward(self, goals, imgfeat):
    """Worker goal reward: ``cosine_max`` between ``goal`` and ``deter`` (Director)."""
    feat = sg(self.feat2deter(imgfeat))
    goal = sg(goals)
    cos = goal_reward_cosine_max(goal, feat)
    return imag_reward_pad(cos[:, 1:])

  def _mgr_expl_reward(self, imgfeat):
    """Manager exploration reward: ``elbo_reward`` with ``adver_impl=squared``.

    Uses per-step goal VAE recon ``((dec.mode() - feat)^2).mean(-1)``; our
    encoder/decoder do not take a separate ``context`` input like Director.
    Returns dense rewards of shape (B, T).
    """
    deter = sg(self.feat2deter(imgfeat))
    encoded = self.goal_enc(deter, 2)
    skill = sample(encoded)
    s = skill['skill'] if isinstance(skill, dict) else skill
    pred = self.goal_dec(s, 2).pred()
    sq = ((pred - deter) ** 2).mean(-1)
    return sq

  def _code_ids(self, feat):
    """Deterministic goal-code index per state, ``(..., L)``.

    ARGMAX, not a sample. ``_mgr_expl_reward`` samples because it wants the
    reconstruction of a drawn code; counting wants the opposite -- the same
    state must always land in the same cell, or the counts measure sampling
    noise instead of visitation. This is also why a badly-reconstructing
    encoder is still usable here: a count needs a consistent LABEL, not an
    accurate description of the state.
    """
    enc = self.goal_enc(sg(self.feat2deter(feat)), 2)
    dist = enc['skill'] if isinstance(enc, dict) else enc
    return jnp.argmax(dist.pred(), -1)

  def _mgr_novel_reward(self, feat):
    """Manager novelty reward: ``1/sqrt(n+eps)`` on the decaying code counts.

    Independent of ``_mgr_expl_reward`` and carried by its own critic. The two
    measure different things -- reconstruction error says the goal autoencoder
    cannot represent this state, the count says the agent has not been here
    lately -- and a state can be either one without being the other. Returns
    dense rewards of shape (B, T).
    """
    return self.code_counts.novelty(self._code_ids(feat))

  def _mgr_novel_update(self, repfeat):
    """Fold the real replay batch into the count memory. Training only.

    Only REAL states update the memory. The reward is queried on the imagined
    rollout too, but imagined states are the world model's guesses -- if those
    counted as visits the manager could suppress its own bonus by imagining its
    way through a region it has never actually reached.
    """
    self.code_counts.update(self._code_ids(repfeat))

  def _mgr_select_update(self, repfeat, weight):
    """Fold the manager's PROPOSED goals into the selection table (UCB only).

    ``downsample=True`` returns one skill per K steps, i.e. one entry per
    manager DECISION rather than one per env step, which is the unit a UCB count
    has to be in. The draws must be SAMPLES for the counts to be on the same
    distribution the candidates come from -- see the comment at the call.

    NOTE these are the manager's untilted SAMPLES, not the UCB-tilted choice
    made during collection: ``_manager_skills_on_sequence`` calls ``_emit_manager``
    without ``explore``, and deliberately so -- running the tilt inside the
    training scan would break the env-rollout-only scoping that keeps the
    REINFORCE gradient unbiased, and would read the selection table in the same
    step that writes it. The consequence is that the table tracks what the
    policy favours rather than what the tilt chose. The main trap still closes
    (the policy's own favourite gets counted, so the manager cannot fixate on
    it), and because the policy is trained on tilted data the loop closes with
    one update of delay.

    These are the goals the current manager would choose at these replay states,
    not the goals actually chosen when the data was collected -- the behaviour
    policy's choices live in the agent carry and never reach replay. For
    discouraging what the manager keeps proposing *now*, the recomputed version
    is arguably the better signal anyway.
    """
    # SAMPLE, not mode. This is not cosmetic: the UCB candidates are policy
    # samples, so the table has to be built from the same distribution or the
    # counts never land where the candidates look. Measured with mode counting
    # (e660-e667, abandoned at ~100k): the table collapsed to ~20 effective
    # cells, candidate bonus sat at 0.72 and the flip rate was pinned at 76%
    # from 12.5k decisions to 500k -- no annealing whatsoever, which is the very
    # pathology the cumulative table exists to avoid. With sampling the bonus
    # falls 0.31 -> 0.08 and the flip rate 93% -> 32% over a 4M run.
    skills = self._manager_skills_on_sequence(
        repfeat, downsample=True, deterministic=False)
    onehot = skills['skill'] if isinstance(skills, dict) else skills
    self.code_counts.update_selected(jnp.argmax(onehot, -1), weight)

  def _code_diag(self, repfeat):
    """Offline diagnostics on the goal-code structure (report_code_diag).

    Tests whether the running-goal-code overwrite is on a meaningful manifold:
      (1) code volatility -- per-block change rate of deterministic codes along
          real trajectories (consecutive steps; host computes the K-step too);
      (2) on-manifold splice test -- take a real code, overwrite a random subset
          of k blocks with blocks from another real state, decode and re-encode,
          and measure the round-trip block mismatch + goal drift as k grows. A
          clean per-block code round-trips (mismatch~0); off-manifold splices do
          not. Returns raw arrays; the driver computes the summary tables host-side."""
    L = int(self.skill_shape[0])
    C = int(self.skill_shape[-1])
    deter = sg(self.feat2deter(repfeat))                  # (B, T, D)
    B, T, D = deter.shape

    def code_onehot(d, bdims):
      enc = self.goal_enc(sg(d), bdims)
      dist = enc['skill'] if isinstance(enc, dict) else enc
      return dist.pred()                                  # (..., L, C) one-hot
    def code_ids(d, bdims):
      return jnp.argmax(code_onehot(d, bdims), -1).astype(i32)

    # (1) deterministic code ids along real trajectories.
    real_ids = code_ids(deter, 2)                         # (B, T, L)

    # (2) splice test on the flattened pool of real codes.
    flat_oh = code_onehot(deter, 2).reshape((B * T, L, C))
    flat_ids = jnp.argmax(flat_oh, -1)                    # (N, L)
    partner = jnp.roll(flat_oh, 1, axis=0)                # a *different* real code
    rk = jax.random.uniform(nj.seed(), (B * T, L))
    rank = jnp.argsort(jnp.argsort(rk, -1), -1)           # random 0..L-1 per row

    real_goal = self.goal_dec(flat_oh, 1).pred()          # (N, D)
    real_norm = jnp.linalg.norm(real_goal, axis=-1) + 1e-8
    rt_real = (code_ids(real_goal, 1) != flat_ids).astype(f32).mean()

    mism, drift = [], []
    for k in range(1, L + 1):
      gate = (rank < k)[..., None]                        # overwrite k random blocks
      spliced = jnp.where(gate, partner, flat_oh)
      goal = self.goal_dec(spliced, 1).pred()             # (N, D)
      rt = code_ids(goal, 1)
      mism.append((rt != jnp.argmax(spliced, -1)).astype(f32).mean())
      drift.append((jnp.linalg.norm(goal - real_goal, axis=-1) / real_norm).mean())

    return {
        'codediag/real_ids': real_ids,                    # (B, T, L)
        'codediag/rt_mismatch_real': rt_real,             # scalar baseline
        'codediag/rt_mismatch_by_k': jnp.stack(mism),     # (L,)
        'codediag/goal_drift_by_k': jnp.stack(drift),     # (L,)
        'codediag/manager_freq': jnp.asarray(
            int(getattr(self, 'manager_sample_freq', 1)), i32),
    }
