#!/bin/bash
#SBATCH -J h0313_eval
#SBATCH -p short-a100
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 0-01:30:00
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x_%A_%a.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x_%A_%a.err
# h0313: evaluate the 4M checkpoint of a finished (archived) run, greedy or sampled.
# Params file (PARAMS), one line per array task:  <bucket run basename> <greedy True|False>
#   GREEDY=True  -> agent.eval_greedy: manager goal and worker action = distribution modes
#   GREEDY=False -> same eval loop, sampled like training (the matched control)
# Output: /work/DoyaU/vasilache/work/h0313_eval/<run>_<greedy|sample>/logdir/metrics.jsonl
set -euo pipefail
PARAMS="${PARAMS:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/sbatch/h0313_eval_params.txt}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS")
RUN=$(echo "$LINE" | awk '{print $1}'); GREEDY=$(echo "$LINE" | awk '{print $2}')
SRC=/bucket/DoyaU/vasilache/bucket/results/dreamerv3/$RUN
CKPT=$SRC/logdir/ckpt_milestones/${CKPT_STEP:-000004000000}
[ -f "$CKPT/agent.pkl" ] || { echo "no checkpoint $CKPT"; exit 1; }
# Run name: e1234_<task>_<cfg-cfg-...>_s<seed>_j...; task is pinpad_* or dmc_<a>_<b>.
REST=${RUN#*_}
case "$REST" in
  pinpad_*) TASK=$(echo "$REST" | cut -d_ -f1-2) ;;
  dmc_*)    TASK=$(echo "$REST" | cut -d_ -f1-3) ;;
esac
CFG=${REST#${TASK}_}; CFG=${CFG%%_s[0-9]*}; CONFIG=$(echo "$CFG" | tr '-' ' ')
SEED=$(echo "$RUN" | sed -E 's/.*_s([0-9]+)_j.*/\1/')
TAG=$([ "$GREEDY" = True ] && echo greedy || echo sample)
OUT=/work/DoyaU/vasilache/work/h0313_eval/${RUN}_${TAG}
mkdir -p "$OUT"; rm -rf "$OUT/logdir"
echo "RUN=$RUN TASK=$TASK SEED=$SEED GREEDY=$GREEDY CONFIG=$CONFIG" | tee "$OUT/job.env"
ENV_FLAGS=()
case "$TASK" in
  dmc_*) ENV_FLAGS=(--env.dmc.image True --env.dmc.proprio False --env.dmc.size 64 64) ;;
esac
# Pin pad episodes are 2000 steps, DMC 1000: 10 episodes per env x 4 envs.
STEPS=${STEPS:-$([ "${TASK%%_*}" = pinpad ] && echo 80000 || echo 40000)}
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
[ "${KEEP_REF_CONV:-0}" = 1 ] || unset DREAMERV3_CONV_IMPL   # V100: keep the reference conv (native needs Route A)
python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --script eval_only \
  --logdir "$OUT/logdir" \
  --seed "$((SEED + 1000))" \
  --configs $CONFIG \
  --task "$TASK" \
  "${ENV_FLAGS[@]}" \
  --run.from_checkpoint "$CKPT" \
  --run.steps "$STEPS" \
  --run.envs 4 \
  --run.log_every 60 \
  --run.log_video False \
  --agent.report False \
  --agent.eval_greedy "$GREEDY" \
  --logger.outputs jsonl \
  --jax.platform cuda \
  --jax.prealloc False
