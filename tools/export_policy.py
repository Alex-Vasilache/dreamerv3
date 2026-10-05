"""Turn a learner policy checkpoint into the compact .npz the phone runs.

The pickles the bridge already ships are ~33MB, but four fifths of that is Adam
moment state for parts the actor never touches. The acting path -- encoder,
RSSM, actor head -- is 1.58M parameters, so the file the phone needs is ~6.3MB.
That matters twice over: it is what crosses the WiFi on every weight update,
and the phone has to parse it between control ticks.

  .venv/bin/python tools/export_policy.py \
      --checkpoint ~/logdir/robot_local/online_shared/policy/policy_*.pkl \
      --out /tmp/policy.npz

It also accepts a training checkpoint directory (`logdir/ckpt/<stamp>`), which
is what you have on Saion:

  .venv/bin/python tools/export_policy.py --checkpoint <ckpt-dir> --out ...

Run `tools/test_numpy_policy.py` after any change here; the manifest this
writes is the contract the on-device policy reads.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import ruamel.yaml as yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import elements  # noqa: E402

from dreamerv3.deploy import export as exportlib  # noqa: E402


def load_params(path):
  """Read params from a policy .pkl or a checkpoint directory."""
  path = pathlib.Path(path)
  if path.is_dir():
    blob = (path / 'agent.pkl').read_bytes()
    data = _loads(blob)
  else:
    data = _loads(path.read_bytes())
  # Policy pickles wrap params in {'params': ..., 'counters': ...}; checkpoint
  # payloads vary, so accept either shape.
  if isinstance(data, dict) and 'params' in data:
    data = data['params']
  return elements.tree.flatdict(data)[0] if not _is_flat(data) else data


def _is_flat(data):
  return isinstance(data, dict) and all(
      not isinstance(v, dict) for v in data.values())


def _loads(blob):
  # numpy renamed `numpy.core` to `numpy._core` in 2.0, and this file is read
  # on whichever side happens to have it: the learner writes on Saion's numpy,
  # the export may run on the MacBook's. Map in whichever direction the local
  # install actually needs rather than assuming one of them.
  class Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
      for src, dst in (('numpy.core', 'numpy._core'),
                       ('numpy._core', 'numpy.core')):
        if module.startswith(src):
          try:
            return super().find_class(module, name)
          except (ModuleNotFoundError, AttributeError):
            return super().find_class(module.replace(src, dst, 1), name)
      return super().find_class(module, name)

  import io
  return Unpickler(io.BytesIO(blob)).load()


def flatten(tree, prefix=''):
  out = {}
  for k, v in tree.items():
    key = f'{prefix}{k}'
    if isinstance(v, dict):
      out.update(flatten(v, key + '/'))
    else:
      out[key] = v
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--configs', nargs='+', default=['robot_daydreamer'])
  args = ap.parse_args()

  import dreamerv3.main as main_mod
  configs = yaml.YAML(typ='safe').load(
      elements.Path(main_mod.folder / 'configs.yaml').read())
  config = elements.Config(configs['defaults'])
  for name in args.configs:
    config = config.update(configs[name])

  raw = load_params(args.checkpoint)
  params = flatten(raw) if not _is_flat(raw) else dict(raw)

  blob, meta = exportlib.pack(config, params)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_bytes(blob)

  keep = exportlib.filter_params(params)
  total = sum(v.size for v in keep.values())
  print(f'wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, '
        f'{total:,} params, {len(keep)} arrays)')
  print('meta:', json.dumps(meta, indent=2))


if __name__ == '__main__':
  main()
