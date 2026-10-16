"""Offline module reset: splice fresh-init params into a run's latest checkpoint.

The in-run ``run.reset_regex`` path re-calls ``Agent._init_params`` after the
train step is compiled, which trips ninjax's optimizer-state key check. This
does the same reset offline instead: take a step-0 checkpoint of the same
config/seed (``--fresh``), replace every parameter (incl. optimizer state,
normalizers, slow targets) whose name matches ``--regex`` in the run's latest
checkpoint, and write it back (original kept as agent.pkl.orig). Counters are
untouched, so the run resumes at its own step with the module re-initialised.

Usage: python tools/splice_reset.py --run_dir <dir> --fresh <fresh logdir> --regex <re>
"""
import argparse, pathlib, pickle, re, shutil


def latest(ckptdir):
  ckptdir = pathlib.Path(ckptdir)
  return ckptdir / (ckptdir / 'latest').read_text().strip()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--fresh', required=True)
  ap.add_argument('--regex', required=True)
  args = ap.parse_args()
  target = latest(pathlib.Path(args.run_dir) / 'logdir' / 'ckpt') / 'agent.pkl'
  fresh = latest(pathlib.Path(args.fresh) / 'ckpt') / 'agent.pkl'
  backup = target.with_suffix('.pkl.orig')
  if not backup.exists():
    shutil.copy2(target, backup)
  data = pickle.load(open(backup, 'rb'))
  init = pickle.load(open(fresh, 'rb'))['params']
  pattern = re.compile(args.regex)
  keys = sorted(k for k in data['params'] if pattern.search(k))
  assert keys, 'regex matches nothing'
  for k in keys:
    assert k in init, k
    assert init[k].shape == data['params'][k].shape, (k, init[k].shape, data['params'][k].shape)
    data['params'][k] = init[k]
  pickle.dump(data, open(target, 'wb'))
  (pathlib.Path(args.run_dir) / 'logdir' / 'reset_done').write_text(
      f'offline splice from {fresh}\ncounters={data["counters"]}\n' + '\n'.join(keys) + '\n')
  print(f'{args.run_dir}: reset {len(keys)} params, counters {data["counters"]}')


if __name__ == '__main__':
  main()
