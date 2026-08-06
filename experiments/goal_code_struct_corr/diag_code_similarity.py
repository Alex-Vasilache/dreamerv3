"""Code-space geometry under four similarity definitions, for a trained run.

``goal/struct_corr_hard`` scores two goal codes by the fraction of blocks that
pick the same entry. That is the only definition available for the Director
head, whose entry indices are arbitrary labels, but it discards the geometry a
SOM-ordered codebook creates: one ring step and half a ring apart score alike.

This computes, from a checkpoint, the correlation between goal-space geometry
and code-space geometry under:

  hamming   fraction of blocks picking the same entry (== goal/struct_corr_hard)
  onehot    cosine_max between flattened one-hot codes (identical to hamming;
            kept as a check that the unified definition is backward compatible)
  embed     cosine_max between the gathered embeddings, i.e. the vector the
            decoder is actually handed (== goal/struct_corr_code)
  ring      circular index distance min(|i-j|, C-|i-j|), meaningful only when a
            ring exists

States are collected exactly as diag_goal_struct_corr.py does: a live rollout
with agent.policy_struct_diag, thinned per worker by ``stride``.

  python -u diag_code_similarity.py --run_dir /work/.../e415_..._j4676001
"""
import argparse, pathlib, sys
root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
from functools import partial as bind
import elements, embodied, numpy as np, jax, jax.numpy as jnp, ninjax as nj
import dreamerv3.main as m
from diag_goal_struct_corr import load_config, pairwise_cosmax, pearson_offdiag


def code_sims(ids, z_q, onehot, C):
    """The four code-space similarity matrices for one batch."""
    diff = np.abs(ids[:, None, :] - ids[None, :, :])
    return {
        'hamming': 1.0 - (ids[:, None, :] != ids[None, :, :]).mean(-1),
        'onehot': pairwise_cosmax(onehot.reshape(len(ids), -1)),
        'embed': pairwise_cosmax(z_q.reshape(len(ids), -1)),
        'ring': 1.0 - np.minimum(diff, C - diff).mean(-1) / (C // 2),
    }


def run(run_dir, stride=8, seed=0, n_batches=6, n_envs=8):
    config = load_config(run_dir)
    L, C = config.agent.skill_shape
    impl = str(getattr(config.agent, 'goal_ae_impl', 'director'))
    n_states = int(config.batch_size) * int(config.batch_length)
    steps = n_envs * stride * (-(-n_batches * n_states // n_envs))
    agent = m.make_agent(config)
    cp = elements.Checkpoint(directory=elements.Path(config.logdir) / 'ckpt')
    cp.agent = agent
    cp.load(keys=['agent'])
    # embodied sets jax_transfer_guard='disallow'; this script pulls arrays back
    # to the host for the numpy similarity computations.
    jax.config.update('jax_transfer_guard', 'allow')

    driver = embodied.Driver([bind(m.make_env, config, i) for i in range(n_envs)],
                             parallel=(n_envs > 1))
    driver.reset(agent.init_policy)
    by_env = [[] for _ in range(n_envs)]
    driver.on_step(lambda tran, w: 'log/struct_diag_deter' in tran and
                   by_env[w].append(np.asarray(tran['log/struct_diag_deter']).copy()))
    driver(agent.policy, steps=steps)
    driver.close()
    deters = np.concatenate([np.stack(d, 0)[::stride] for d in by_env if d], 0)

    model = agent.model
    def encode(d):
        if impl == 'vq':
            z_e = model.goal_enc.latent(d, 1)
            q = model.goal_dec.codebook.quantize(z_e)
            return q['ids'], q['z_q'], q['onehot']
        enc = model.goal_enc(d, 1)
        oh = (enc['skill'] if isinstance(enc, dict) else enc).pred()
        return jnp.argmax(oh, -1), oh, oh
    out = {k: [] for k in ('hamming', 'onehot', 'embed', 'ring')}
    for b in range(n_batches):
        d = deters[b * n_states:(b + 1) * n_states]
        if len(d) < n_states:
            break
        ids, z_q, oh = [np.asarray(x) for x in
                        nj.pure(encode)(agent.params, jnp.asarray(d))[1]]
        sd = pairwise_cosmax(d)
        for k, sc in code_sims(ids, z_q, oh, C).items():
            out[k].append(pearson_offdiag(sd, sc))
    name = pathlib.Path(run_dir.rstrip('/')).name
    print(f'\n=== {name}  impl={impl}  batches={len(out["hamming"])} '
          f'x {n_states} states ===')
    for k in ('hamming', 'onehot', 'embed', 'ring'):
        v = np.array(out[k])
        print(f'   corr[{k:8s}] = {v.mean():.3f} +/- {v.std():.3f}')
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', nargs='+', required=True)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--n_batches', type=int, default=6)
    p.add_argument('--n_envs', type=int, default=8)
    a = p.parse_args()
    for rd in a.run_dir:
        run(rd, stride=a.stride, n_batches=a.n_batches, n_envs=a.n_envs)
