# Working HRL cartpole run (reinforce / adaptive actent)

Validated on Saion, June 2026. SLURM script: `run_v3_cartpole_adent_std01expl_6m.sbatch`.
Smoke (1M params): `run_v3_cartpole_adent_std01expl.sbatch`.

## SLURM

- Partition: `gpu`, 1× `gpu:v100`, 4 cores, 96G RAM, 24h wall
- Logs: `/work/DoyaU/vasilache/work/slurm_logs/%x-%j.{out,err}`

## Python / DreamerV3 flags

```
--configs size6m
--task dmc_cartpole_swingup
--env.dmc.image True
--env.dmc.proprio True
--env.dmc.size 32 32
--agent.enc.simple.outer True
--agent.dec.simple.outer True
--agent.mgr_expl_weight 0.1
--agent.mgr_retnorm.impl meanstd
--online_learning False
--online_actor_cpu False
--run.envs 16
--run.steps 4000000
--logger.outputs jsonl,scope,wandb
--logger.wandb_mode online
--logger.wandb_entity vasilache
--jax.platform cuda
--jax.prealloc False
```

DreamerV4 toggles (PMPO, RMS loss-norm, single-rollout) are off on this branch.
`AGENT_FLAGS` in the sbatch script is empty (defaults + adaptive manager actent).

## A/B (not the working line)

`run_v3_cartpole_adent_perc1expl_6m.sbatch`: `mgr_expl_weight 1.0`, `mgr_retnorm.impl perc`.
