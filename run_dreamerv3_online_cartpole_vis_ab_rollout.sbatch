#!/bin/bash -l
#SBATCH -J dreamerv3_vis_ab_rollout
#SBATCH -p gpu
#SBATCH -t 0-4:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#
# A/B: same as run_dreamerv3_online_cartpole_vis.sbatch + ONLY use_single_rollout.

set -euo pipefail
mkdir -p /work/DoyaU/vasilache/work/slurm_logs
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_online_vis1m32_1env_ab_rollout)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
export MUJOCO_GL=egl
cd "$RUN_DIR"
python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size1m \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --batch_size 8 \
  --batch_length 32 \
  --online_learning True \
  --online_actor_cpu True \
  --run.envs 1 \
  --run.steps 1100000 \
  --logger.outputs jsonl,scope,tensorboard \
  --jax.platform cuda \
  --jax.prealloc False \
  --agent.use_single_rollout True
exit $?
