"""How much of the goal autoencoder's rising error is ``deter`` changing scale?

For each milestone checkpoint, encodes a fixed set of pinpad probe trajectories
(see ``code_drift.py``) and reports the scale of ``deter`` and the goal AE's
reconstruction of those states, both absolute (summed squared error, what the
training metric ``goal/rec_mean`` reports) and relative (fraction of the
states' variance left unexplained), plus cosine between state and reconstruction.

Usage: python tools/deter_scale.py --run <run_dir> [--run ...] [--steps 500000,...]
"""
import argparse, os, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import code_drift as cd


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run', action='append', required=True)
  ap.add_argument('--steps', default='')
  ap.add_argument('--platform', default='cpu')
  ap.add_argument('--out', default='')
  args = ap.parse_args()
  want = [int(float(s)) for s in args.steps.split(',')] if args.steps else None
  print(f"{'run':6s} {'step':>8s} | {'|deter|':>8s} {'dim std':>8s} {'dim mean|':>9s} | "
        f"{'rec SSE':>8s} {'unexpl.var':>10s} {'cos':>5s} | {'std-profile corr vs first':>25s}")
  results = []
  for run in args.run:
    mdl = cd.Model(run, args.platform)
    probe = cd.probe_data(mdl.task)
    jnp = mdl.jnp
    mdir = pathlib.Path(run) / 'logdir' / 'ckpt_milestones'
    steps = sorted(int(p.name) for p in mdir.iterdir() if (p / 'done').exists())
    steps = [s for s in steps if want is None or s in want]
    first_std = None
    for s in steps:
      mdl.init_and_load(str(mdir / f'{s:012d}' / 'agent.pkl'), probe)
      ds = []
      for lo in range(0, cd.B, 8):
        obs, prevact, reset = mdl._obs(probe, lo, lo + 8)
        d, _ = mdl.run('encode', obs, prevact, reset)
        ds.append(np.asarray(d))
      deter = np.concatenate(ds).reshape(-1, ds[0].shape[-1])
      code = mdl.run('state_codes', jnp.asarray(deter))
      rec = np.asarray(mdl.run('decode', code))
      err = ((rec - deter) ** 2).sum(-1)
      var = ((deter - deter.mean(0)) ** 2).sum(-1)
      cos = (rec * deter).sum(-1) / (np.linalg.norm(rec, axis=-1) * np.linalg.norm(deter, axis=-1) + 1e-12)
      std = deter.std(0)
      first_std = std if first_std is None else first_std
      corr = np.corrcoef(first_std, std)[0, 1]
      print(f"{os.path.basename(run)[:5]:6s} {s:8d} | {np.linalg.norm(deter, axis=-1).mean():8.2f} "
            f"{std.mean():8.3f} {np.abs(deter.mean(0)).mean():9.3f} | {err.mean():8.1f} "
            f"{err.mean() / var.mean():10.3f} {cos.mean():5.2f} | {corr:25.2f}", flush=True)
      results.append(dict(run=os.path.basename(run)[:5].rstrip('_'), step=s,
                          norm=float(np.linalg.norm(deter, axis=-1).mean()),
                          sse=float(err.mean()), unexpl=float(err.mean() / var.mean()),
                          cos=float(cos.mean()), corr=float(corr)))
      if args.out:
        import json; json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
  main()
