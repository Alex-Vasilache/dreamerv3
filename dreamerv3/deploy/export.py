"""Turn learner params into the .npz the phone runs.

Shared by `tools/robot/dev/export_policy.py` (offline, from a checkpoint) and by the
online actor, which exports in-process every time it installs new weights so
the bridge has something to push down. Keeping one implementation matters: the
manifest is a contract, and a phone fed a manifest built two different ways is
a bug nobody would find from the outside.
"""

import io
import json

import numpy as np

# Everything the acting path reads. Anything outside these prefixes is
# optimizer state, the decoder, or the critic -- all learner-only.
KEEP_PREFIXES = ('enc/', 'dyn/', 'pol/')
# ...except the RSSM prior, which only imagination uses. Dropping it saves
# 460KB and makes it obvious that this file cannot dream.
DROP_PREFIXES = ('dyn/prior', 'dyn/priorlogit')
# With emotion=True the file also carries what deploy/emotion.py reads: the
# prior (surprise, novelty, empowerment) and the critic and reward heads
# (valence). About 1 MB more as int8.
# cval/ is the fear critic (agent.fear), when the run has one.
EMOTION_PREFIXES = ('dyn/prior', 'val/', 'rew/', 'cval/')


# dtype='int8': kernels of at least this many entries go as int8 codes with
# one float16 scale per output channel; everything smaller (biases, norm
# scales, the 512-entry action head) goes as float16. Measured on e9662's
# weights (2026-11-03), stepping from the float32 policy's own state: int8
# per channel moves the action by 0.0015 on average (p99 0.005) against a
# policy stddev of at least 0.1, at half the bytes of float16; the action head
# alone accounts for most of that, hence the size floor. int4 was 20x worse.
INT8_MIN_SIZE = 4096
SCALE_SUFFIX = '.scale'


def quantize_int8(v):
  """Symmetric per-output-channel int8: (codes, float16 scales).

  The input axis is the second to last for every kernel here, plain
  (in, out) and blocked (blocks, in, out) alike, so one scale covers each
  output column. The scales are rounded to float16 before the codes are taken,
  so codes * scales is exactly what the phone reconstructs.
  """
  v = np.asarray(v, np.float32)
  amax = np.abs(v).max(axis=-2, keepdims=True)
  scale = (np.maximum(amax, 1e-12) / 127.0).astype(np.float16)
  codes = np.clip(np.round(v / scale.astype(np.float32)), -127, 127)
  return codes.astype(np.int8), scale


def filter_params(params, dtype='float32', emotion=False):
  """The acting params, as `dtype`. For 'int8' a quantized kernel `k` is
  stored as int8 codes under `k` and its scales under `k + SCALE_SUFFIX`."""
  keep = {}
  for k, v in params.items():
    wanted = k.startswith(KEEP_PREFIXES) and not k.startswith(DROP_PREFIXES)
    if not (wanted or (emotion and k.startswith(EMOTION_PREFIXES))):
      continue
    v = np.asarray(v, np.float32)
    if dtype == 'int8':
      if v.ndim >= 2 and v.size >= INT8_MIN_SIZE:
        keep[k], keep[k + SCALE_SUFFIX] = quantize_int8(v)
      else:
        keep[k] = v.astype(np.float16)
    else:
      keep[k] = v.astype(dtype)
  if not keep:
    raise ValueError(
        f'No policy params found. Keys looked like: {sorted(params)[:5]}')
  return keep


def veckeys(config):
  # Balance and drive are proprio-only; 'target' appears for the track task.
  keys = ['orientation', 'wheels']
  # The task rides in the top-level `task: robot_<task>`; env.robot has none.
  task = str(config.task).split('_', 1)[-1]
  if task == 'track':
    keys.append('target')
  if task == 'command':
    keys.append('command')
  if float(config.env.robot.get('pos_tau', 0.0)):
    keys.append('position')
  return keys


def build_meta(config, params):
  """The architecture facts the numpy policy needs.

  Layer counts and widths are counted off the parameters rather than read from
  the config: the config carries defaults the checkpoint may not have been
  built with (MLPHead defaults to five layers, this policy has three), and the
  weights are what the phone will actually run.
  """
  rssm = config.agent.dyn.rssm
  enc = config.agent.enc.simple

  def count(fmt):
    n = 0
    while fmt.format(i=n) in params:
      n += 1
    return n

  gru = params['dyn/dyngru/kernel']
  blocks = int(gru.shape[0])
  deter = int(params['dyn/dyngru/bias'].shape[0] // 3)
  classes = int(rssm.classes)
  stoch = int(params['dyn/obslogit/bias'].shape[0] // classes)
  act_dim = int(params['dyn/dynin2/kernel'].shape[0])
  obs_keys = sorted(veckeys(config))
  in_dim = int(params['enc/mlp0/kernel'].shape[0])
  assert in_dim == 2 * len(obs_keys), (
      f'encoder takes {in_dim} inputs but {obs_keys} imply {2 * len(obs_keys)}')

  meta = dict(
      deter=deter,
      stoch=stoch,
      classes=classes,
      blocks=blocks,
      unimix=float(rssm.unimix),
      obslayers=count('dyn/obs{i}/kernel'),
      dynlayers=count('dyn/dynhid{i}/kernel'),
      enc_layers=count('enc/mlp{i}/kernel'),
      pol_layers=count('pol/mlp/linear{i}/kernel'),
      symlog=bool(enc.symlog),
      obs_keys=obs_keys,
      act_key='drive',
      act_dim=act_dim,
      # bounded_normal: stddev = (hi-lo)*sigmoid(raw+2)+lo.
      minstd=float(config.agent.policy.get('minstd', 1.0)),
      maxstd=float(config.agent.policy.get('maxstd', 1.0)),
      # The observation scaling from embodied/envs/robot.py:_obs. It travels
      # with the weights because the phone has to reproduce it exactly: a
      # policy fed differently-scaled inputs than it was trained on is wrong in
      # a way nothing downstream can detect. One source of truth, shipped.
      obs_scale=dict(
          theta_zero=float(config.env.robot.theta_zero),
          obs_theta_scale=float(config.env.robot.obs_theta_scale),
          obs_rate_scale=float(config.env.robot.obs_rate_scale),
          obs_wheel_scale=float(config.env.robot.obs_wheel_scale),
          obs_clip=float(config.env.robot.obs_clip),
          # The leaky wheel distance (robot.py:track_position), when observed.
          **(dict(
              pos_tau=float(config.env.robot.pos_tau),
              obs_pos_scale=float(config.env.robot.obs_pos_scale),
              pos_jump=float(config.env.robot.pos_jump),
              hz=float(config.env.robot.hz),
          ) if float(config.env.robot.get('pos_tau', 0.0)) else {}),
      ),
  )
  meta['signature'] = '|'.join(
      f'{k}={meta[k]}' for k in sorted(meta) if k != 'signature')
  return meta


def pack(config, params, dtype='float32', emotion=False):
  """Return (npz_bytes, meta) for the policy params in `params`. The phone
  loads any float dtype, and int8 codes with their scales, into float32, so
  `dtype` only sets the bytes sent. int8 is also deflated: the codes compress
  by ~7%, float32 and float16 by nothing worth the time."""
  arrays, meta = pack_arrays(config, params, dtype, emotion)
  return encode(meta, arrays, compress=(dtype == 'int8')), meta


def pack_arrays(config, params, dtype='float32', emotion=False):
  """The arrays `pack` would send, and their meta, not yet encoded."""
  keep = filter_params(params, dtype, emotion)
  meta = build_meta(config, keep)
  if emotion:
    # The critic's discount, for the TD error: 1 - 1/horizon steps.
    r = config.env.robot
    horizon = (float(r.horizon_s) * float(r.hz) if float(r.get('horizon_s', 0))
               else float(config.agent.horizon))
    meta['emotion'] = dict(gamma=1.0 - 1.0 / horizon)
    meta['signature'] = '|'.join(
        f'{k}={meta[k]}' for k in sorted(meta) if k != 'signature')
  return keep, meta


# Delta pushes. A delta names the push it builds on (`__base__`) and holds,
# for every array the base has in the same shape and dtype, only how it
# changed: int8 codes as their difference mod 256 (`<key>@d`, uint8), float16
# arrays as the XOR of their bits (`<key>@x`, uint16), both zero where nothing
# changed. Everything else goes whole. The phone rebuilds the full push
# exactly (numpy_policy.expand_delta), so deltas never accumulate error. On
# e10100's weights (2026-11-03) half the int8 codes move between pushes 2 s
# apart, nearly all by one step, which deflates to ~1/3 of the full push
# (553 KB vs 1.3 MB for pushes 6 s apart).
DELTA_INT8 = '@d'
DELTA_XOR = '@x'


def encode(meta, arrays, stamp=None, base=None, compress=True):
  """npz bytes for `arrays`. With `stamp`, the push records its own stamp
  (`__stamp__`), which a later delta checks its base against. With `base` =
  (base_stamp, base_arrays), a delta against those arrays."""
  out = {'__meta__': np.array(json.dumps(meta))}
  if stamp is not None:
    out['__stamp__'] = np.array(str(stamp))
  if base is not None:
    out['__base__'] = np.array(str(base[0]))
  for k, v in arrays.items():
    old = None if base is None else base[1].get(k)
    if old is not None and old.shape == v.shape and old.dtype == v.dtype:
      if v.dtype == np.int8:
        out[k + DELTA_INT8] = v.view(np.uint8) - old.view(np.uint8)
        continue
      if v.dtype == np.float16:
        out[k + DELTA_XOR] = v.view(np.uint16) ^ old.view(np.uint16)
        continue
    out[k] = v
  buffer = io.BytesIO()
  (np.savez_compressed if compress else np.savez)(buffer, **out)
  return buffer.getvalue()
