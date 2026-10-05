"""Turn learner params into the .npz the phone runs.

Shared by `tools/export_policy.py` (offline, from a checkpoint) and by the
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


def filter_params(params):
  keep = {}
  for k, v in params.items():
    if not k.startswith(KEEP_PREFIXES) or k.startswith(DROP_PREFIXES):
      continue
    keep[k] = np.asarray(v, np.float32)
  if not keep:
    raise ValueError(
        f'No policy params found. Keys looked like: {sorted(params)[:5]}')
  return keep


def veckeys(config):
  # Balance and drive are proprio-only; 'target' appears for the track task.
  keys = ['orientation', 'wheels']
  if getattr(config.env.robot, 'task', None) == 'track':
    keys.append('target')
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
      ),
  )
  meta['signature'] = '|'.join(
      f'{k}={meta[k]}' for k in sorted(meta) if k != 'signature')
  return meta


def pack(config, params):
  """Return (npz_bytes, meta) for the policy params in `params`."""
  keep = filter_params(params)
  meta = build_meta(config, keep)
  buffer = io.BytesIO()
  np.savez(buffer, __meta__=np.array(json.dumps(meta)), **keep)
  return buffer.getvalue(), meta
