"""``Agent.report``: eval metrics, open-loop video, and goal-proposal panels."""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

import embodied.jax.nets as nn

from .video import resize_frames, tb_video_grid, vec_to_tb_rgb

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
mode = lambda xs: jax.tree.map(lambda x: x.pred(), xs)
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)


class ReportMixin:
  """Reporting/visualization logic mixed into ``dreamerv3.agent.Agent``."""

  def _goal_img_cache(self, batch_size):
    """Zeros for the held decoded-goal *image* cache (one uint8 frame per imgkey).

    Holding the deter goal vector is not enough for a stable panel: the image
    decoder / dynamics prior keep training during online rollout, so a fresh
    decode of the same held goal drifts every step. We cache the rendered frame in
    the carry and refresh it only on a manager switch / episode reset. Empty unless
    the policy goal-image panels are active; gated identically to the render path in
    ``policy`` so the carry pytree matches what ``policy`` writes back."""
    cache = {}
    if self.dec.imgkeys and bool(getattr(self.config, 'policy_goal_image', True)):
      for k in self.dec.imgkeys:
        shp = tuple(int(x) for x in self.obs_space[k].shape)
        cache[f'goal_img_{k}'] = jnp.zeros((batch_size, *shp), jnp.uint8)
    return cache

  def _video(self, video_bthwc):
    """Apply config-driven time/space stride before flattening for video logs."""
    ts = int(getattr(self.config, 'report_video_time_stride', 1))
    ss = int(getattr(self.config, 'report_video_space_stride', 1))
    return tb_video_grid(video_bthwc, time_stride=ts, space_stride=ss)

  def _decode_images_uint8(self, dec_carry, feat, reset):
    _, _, recons = self.dec(dec_carry, feat, reset, training=False)
    return {
        k: jnp.clip(recons[k].pred() * 255, 0, 255).astype(jnp.uint8)
        for k in self.dec.imgkeys}

  def _report_impl_videos(self, repfeat, prevact, dec_carry, impl, RB, T):
    """Director ``report_worker``: [initial | proposed goal | worker rollout]."""
    metrics = {}
    if not self.dec.imgkeys:
      return metrics
    horizon = int(getattr(self.config, 'worker_report_horizon', 32))
    t0 = min(4, max(T - 1, 0))
    horizon = min(horizon, max(T - t0 - 1, 1))
    length = 1 + horizon

    start_feat = jax.tree.map(lambda x: x[:, t0], repfeat)
    goal = self._propose_goal(start_feat, impl)
    dyn_start = dict(
        deter=nn.cast(start_feat['deter']),
        stoch=nn.cast(start_feat['stoch']))
    _, imgfeat, _ = self.dyn.imagine(
        dyn_start, self._worker_policy_fixed_goal(goal), horizon, training=False)

    reset1 = jnp.zeros((RB, 1), bool)
    reset_len = jnp.zeros((RB, length), bool)
    start_dec = self._decode_images_uint8(dec_carry, start_feat, reset1)
    goal_dec = self._decode_images_uint8(
        dec_carry, self._feat_from_goal(goal), reset1)
    roll_feat = concat([
        jax.tree.map(lambda x: x[:, None], start_feat), imgfeat], 1)
    roll_dec = self._decode_images_uint8(dec_carry, roll_feat, reset_len)

    def tile_rb(u8, length):
      if u8.ndim == 5:
        return jnp.repeat(u8, length, axis=1)
      return jnp.repeat(u8[:, None], length, axis=1)

    for key in self.dec.imgkeys:
      init_u8 = tile_rb(start_dec[key], length)
      targ_u8 = tile_rb(goal_dec[key], length)
      roll_u8 = roll_dec[key]
      if roll_u8.ndim == 4:
        roll_u8 = jnp.repeat(roll_u8[:, None], length, axis=1)
      video = jnp.concatenate([init_u8, targ_u8, roll_u8], axis=3)
      metrics[f'impl_{impl}/{key}'] = self._video(video)
    return metrics

  def report(self, carry, data):
    """Eval metrics, open-loop video, goal VAE panels, and Director goal-proposal videos."""
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    enc_carry, dyn_carry, dec_carry, _, mgr_skill, mgr_step = self._unpack_carry(
        carry)
    wm_carry = (enc_carry, dyn_carry, dec_carry)
    B, T = obs['is_first'].shape
    RB = min(int(getattr(self.config, 'report_max_rows', 6)), B)
    metrics = {}

    _, (new_carry, entries, outs, mets) = self.loss(
        wm_carry, obs, prevact, training=False)
    metrics.update(mets)
    if self.use_hrl and bool(getattr(self.config, 'report_code_diag', False)):
      metrics.update(self._code_diag(outs['repfeat']))
    rep = jax.tree.map(lambda x: x[:RB, :T], outs['repfeat'])
    reset_s = obs['is_first'][:RB, :T]
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)

    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, c: self.loss(
              c, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, wm_carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop: observe first half, imagine second half (baseline Dreamer report).
    obs_rb = jax.tree.map(lambda x: x[:RB], obs)
    prevact_rb = jax.tree.map(lambda x: x[:RB], prevact)
    tokens_rb = jax.tree.map(lambda x: x[:RB], outs['tokens'])
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:, T // 2:], xs)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(tokens_rb), firsthalf(prevact_rb),
        firsthalf(obs_rb['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact_rb), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs_rb['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs_rb['is_first'])),
        training=False)
    for key in self.dec.imgkeys:
      assert obs_rb[key].dtype == jnp.uint8
      true = obs_rb[key]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)
      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)
      metrics[f'openloop/{key}'] = self._video(video)

    if not self.use_hrl:
      enc_carry_n, dyn_carry_n, dec_carry_n = new_carry
      carry = self._pack_carry(
          enc_carry_n, dyn_carry_n, dec_carry_n,
          {k: data[k][:, -1] for k in self.act_space},
          mgr_skill, mgr_step)
      return carry, metrics

    # Goal VAE on replay features (train-time encoder path).
    deter_feat = sg(self.feat2deter(rep))
    encoded_goal = self.goal_enc(deter_feat, 2)
    skill_s = sample(encoded_goal)
    skill_code_s = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
    pred_deter = nn.cast(self.goal_dec(skill_code_s, 2).pred())
    feat_goal = self._feat_from_goal(pred_deter)
    _, _, recons_goal = self.dec(dec_carry, feat_goal, reset_s, training=False)

    # Manager-proposed goals over the report sequence (K-step skill hold).
    want_mgr_recon = bool(getattr(self.config, 'report_mgr_recon', False))
    want_skill_viz = bool(getattr(self.config, 'report_skill_viz', True))
    want_goal_enc_viz = bool(getattr(self.config, 'report_goal_enc_viz', True))
    need_mgr = (want_mgr_recon or want_skill_viz or
                bool(getattr(self.config, 'report_vec_viz', False)))
    if need_mgr:
      mgr_skills = self._manager_skills_on_sequence(rep)
      mgr_skills.pop('countdown', None)
      mgr_goals = sg(self._goals_from_skills(mgr_skills, bdims=2))
      mgr_goal_feat = self._feat_from_goal(mgr_goals)
      _, _, recons_mgr = self.dec(dec_carry, mgr_goal_feat, reset_s, training=False)
    else:
      mgr_skills = mgr_goals = mgr_goal_feat = recons_mgr = None

    # Optional dense vec→RGB visualisations of latent vectors and skills.
    # Off by default — they are debug-grade and slow down the video pipeline.
    if bool(getattr(self.config, 'report_vec_viz', False)):
      metrics['goal/deter_feat'] = self._video(vec_to_tb_rgb(deter_feat))
      metrics['goal/decoded_deter'] = self._video(vec_to_tb_rgb(pred_deter))
      sk = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
      if sk.ndim == 3:
        metrics['goal/skill_sampled'] = self._video(vec_to_tb_rgb(sk))
      else:
        metrics['goal/skill_sampled'] = self._video(
            jnp.repeat((sk * 255).astype(jnp.uint8)[..., None], 3, axis=-1))
      m_sk = mgr_skills['skill']
      if m_sk.ndim == 3:
        metrics['goal/mgr_skill'] = self._video(vec_to_tb_rgb(m_sk))
      else:
        metrics['goal/mgr_skill'] = self._video(
            jnp.repeat((m_sk * 255).astype(jnp.uint8)[..., None], 3, axis=-1))
      metrics['goal/mgr_proposed_deter'] = self._video(vec_to_tb_rgb(mgr_goals))

    # VAE / manager goal reconstruction panels. ``goal/image_{key}`` (true frames
    # only) was a duplicate of the leftmost column here — dropped.
    for key in self.dec.imgkeys:
      true = obs[key][:RB, :T]
      pred_g = jnp.clip(recons_goal[key].pred() * 255, 0, 255).astype(jnp.uint8)
      metrics[f'goal/recon_{key}'] = self._video(
          jnp.concatenate([true, pred_g, ((i32(pred_g) - i32(true) + 255) // 2).astype(np.uint8)], 2))
      if want_mgr_recon and recons_mgr is not None:
        pred_m = jnp.clip(recons_mgr[key].pred() * 255, 0, 255).astype(jnp.uint8)
        metrics[f'goal/mgr_recon_{key}'] = self._video(
            jnp.concatenate([true, pred_m, ((i32(pred_m) - i32(true) + 255) // 2).astype(np.uint8)], 2))

    # 3-row manager skill video: skill heatmap | deter heatmap | decoded goal image.
    # With masked goals the skill row is the *running* goal code Z (the accumulated
    # command after block-overwrite); the decoded goal row is its decode.
    if want_skill_viz and need_mgr and self.dec.imgkeys:
      sk = self._running_goal_code(mgr_skills)
      sk_flat = sk.reshape(*sk.shape[:2], -1)  # (RB, T, L*C)
      skill_rgb = vec_to_tb_rgb(sk_flat)  # (RB, T, h_sk, w_sk, 3) uint8
      deter_rgb = vec_to_tb_rgb(deter_feat)  # (RB, T, h_d, w_d, 3) uint8
      for key in self.dec.imgkeys:
        H_img, W_img = int(obs[key].shape[2]), int(obs[key].shape[3])
        skill_row = resize_frames(skill_rgb, H_img, W_img)   # (RB, T, H, W, 3)
        deter_row = resize_frames(deter_rgb, H_img, W_img)   # (RB, T, H, W, 3)
        goal_row = jnp.clip(recons_mgr[key].pred() * 255, 0, 255).astype(jnp.uint8)
        C_img = int(obs[key].shape[4])
        if C_img == 1:
          goal_row = goal_row[..., :1]
          skill_row = skill_row[..., :1]
          deter_row = deter_row[..., :1]
        panel = jnp.concatenate([skill_row, deter_row, goal_row], axis=2)
        metrics[f'skill_viz/{key}'] = self._video(panel)

    # 5-row goal encoder video: og image | encoded deter | encoded goal | decoded deter | decoded image.
    if want_goal_enc_viz and self.dec.imgkeys:
      sk_enc = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
      sk_flat_enc = (sk_enc.reshape(*sk_enc.shape[:2], -1)
                     if sk_enc.ndim == 4 else sk_enc)  # (RB, T, L*C) or (RB, T, D)
      enc_deter_rgb = vec_to_tb_rgb(deter_feat)    # input to goal encoder
      enc_goal_rgb = vec_to_tb_rgb(sk_flat_enc)    # encoded skill
      dec_deter_rgb = vec_to_tb_rgb(pred_deter)    # decoded deter
      for key in self.dec.imgkeys:
        H_img, W_img = int(obs[key].shape[2]), int(obs[key].shape[3])
        C_img = int(obs[key].shape[4])
        og_u8 = obs[key][:RB, :T]
        enc_deter_row = resize_frames(enc_deter_rgb, H_img, W_img)
        enc_goal_row = resize_frames(enc_goal_rgb, H_img, W_img)
        dec_deter_row = resize_frames(dec_deter_rgb, H_img, W_img)
        dec_img_row = jnp.clip(recons_goal[key].pred() * 255, 0, 255).astype(jnp.uint8)
        if C_img == 1:
          enc_deter_row = enc_deter_row[..., :1]
          enc_goal_row = enc_goal_row[..., :1]
          dec_deter_row = dec_deter_row[..., :1]
        panel = jnp.concatenate(
            [og_u8, enc_deter_row, enc_goal_row, dec_deter_row, dec_img_row], axis=2)
        metrics[f'goal_enc_viz/{key}'] = self._video(panel)

    # NOTE: the masked / overwrite-goal panel moved to the online policy() path so
    # it spans an entire episode (logged as epstats/policy_mask_viz_{key}), mirroring
    # epstats/policy_image_with_goal. Its 3 rows (each upscaled 10x) are obs | goal
    # code with blocks CHANGED vs the previous goal in yellow, unchanged blocks
    # white/grayscale | active goal image. Tracked via the sticky
    # ``last_change_mask`` for masked AND plain-Director goals alike (a plain
    # manager reproducing a block's previous class shows white there too). See
    # ``mask_viz_on`` in ``policy``. ``report_mask_viz`` gates it.

    # Director-style: [initial | proposed goal | worker rollout] per proposal
    # mode. Disabled by default (lowest-value-per-encoding-cost panel); set
    # ``report_impl_videos_enabled: True`` to render the impls listed in
    # ``report_impl_videos``.
    if bool(getattr(self.config, 'report_impl_videos_enabled', False)):
      impls = tuple(getattr(self.config, 'report_impl_videos', ['manager']))
    else:
      impls = ()
    for impl in impls:
      metrics.update(self._report_impl_videos(
          rep, prevact, dec_carry, impl, RB, T))

    enc_carry, dyn_carry, dec_carry = new_carry
    carry = self._pack_carry(
        enc_carry, dyn_carry, dec_carry,
        {k: data[k][:, -1] for k in self.act_space},
        mgr_skill, mgr_step)
    return carry, metrics
