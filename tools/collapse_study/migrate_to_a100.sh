#!/bin/bash
# Move running run_benchmark.sbatch jobs onto gpu-a100, resuming from their own
# RUN_DIR (checkpoint + replay; at most save_every=15 min of training is lost).
# Usage: migrate_to_a100.sh <slurm_job_id> [...]   (array tasks as 4742472_3)
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
cd "$REPO"
for jid in "$@"; do
  envf=$(grep -l "^JOB=" /work/DoyaU/vasilache/work/e1*/job.env | while read f; do
    grep -q "^JOB=$(squeue -h -j "$jid" -o %A)$" "$f" && echo "$f"; done | head -1)
  [ -n "$envf" ] || { echo "$jid: job.env not found"; continue; }
  . <(grep -E '^(EXP_TAG|CONFIG|TASK|SEED|RUN_DIR|WANDB_PROJECT)=' "$envf" | sed 's/^\([A-Z_]*\)=\(.*\)$/\1="\2"/')
  scancel "$jid"
  while [ -n "$(squeue -h -j "$jid" 2>/dev/null)" ]; do sleep 5; done
  new=$(sbatch --parsable -p gpu-a100 --gres=gpu:a100:1 -c 16 --mem=128G -J "${EXP_TAG}_a100" \
    --export=ALL,CONFIG="$CONFIG",TASK="$TASK",SEED="$SEED",EXP_TAG="$EXP_TAG",RUN_DIR="$RUN_DIR",WANDB_PROJECT="$WANDB_PROJECT" \
    sbatch/run_benchmark.sbatch)
  echo "$jid -> $new  $EXP_TAG $TASK s$SEED  $RUN_DIR"
done
