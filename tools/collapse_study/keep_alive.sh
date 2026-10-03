#!/bin/bash
# Resubmit runs that stopped before their target step (e.g. host OOM at 64G:
# the in-RAM replay passes 64G near 3.6M pinpad steps). Resumes from the run's
# own RUN_DIR with more memory, on gpu-a100 if a slot is free, else P100/V100.
# Usage: nohup keep_alive.sh <target_step> <exp> [...] &
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
TARGET=$1; shift
STATE=job_logs/keep_alive_state.tsv; touch "$STATE"
alive() { [ -n "$1" ] && [ -n "$(squeue -h -j "$1" 2>/dev/null)" ]; }
count() { squeue -u "$USER" -h -p "$1" -t RUNNING,PENDING | wc -l; }
while :; do
  pending=0
  for e in "$@"; do
    d=$(ls -d /work/DoyaU/vasilache/work/${e}_* 2>/dev/null | head -1); [ -n "$d" ] || continue
    step=$(tail -1 "$d/logdir/metrics.jsonl" 2>/dev/null | python3 -c 'import sys,json;print(int(json.loads(sys.stdin.read())["step"]))' 2>/dev/null || echo 0)
    [ "$step" -ge "$TARGET" ] && continue
    pending=1
    j1=$(grep -h '^JOB=' "$d/job.env" | tail -1 | cut -d= -f2); j2=$(awk -v e="$e" '$1==e{j=$2} END{print j}' "$STATE")
    # Any of my jobs named "${e}_*" (e.g. a pending migration) also counts:
    # job.env only learns a job id once that job starts.
    named=$(squeue -u "$USER" -h -o '%j' | grep -c "^${e}_")
    { alive "$j1" || alive "$j2" || [ "$named" -gt 0 ]; } && continue
    . <(grep -E '^(EXP_TAG|CONFIG|TASK|SEED|RUN_DIR|WANDB_PROJECT)=' "$d/job.env" | sed 's/^\([A-Z_]*\)=\(.*\)$/\1="\2"/')
    if [ "$(count gpu-a100)" -lt 8 ]; then res=(-p gpu-a100 --gres=gpu:a100:1 -c 16 --mem=128G)
    elif [ "$(count gpu-p100)" -lt 8 ]; then res=(-p gpu-p100 --gres=gpu:p100:1 --nodelist='saion-gpu[11-14]' -c 8 --mem=120G)
    else res=(-p gpu-v100 --gres=gpu:v100:1 -c 8 --mem=120G); fi
    new=$(sbatch --parsable "${res[@]}" -t 2-00:00:00 -J "${EXP_TAG}_k" \
      --export=ALL,CONFIG="$CONFIG",TASK="$TASK",SEED="$SEED",EXP_TAG="$EXP_TAG",RUN_DIR="$RUN_DIR",WANDB_PROJECT="$WANDB_PROJECT" \
      sbatch/run_benchmark.sbatch)
    printf '%s\t%s\t%s\t%s\n' "$e" "$new" "${res[1]}" "$(date '+%F %T') step=$step" >> "$STATE"
    echo "$(date '+%F %T') resubmit $e at step $step -> $new on ${res[1]}"
  done
  [ "$pending" = 0 ] && { echo "$(date '+%F %T') all at target"; exit 0; }
  sleep 300
done
