#!/bin/bash -l
# Generate and submit the 2026-09-04 benchmark matrix.
#
#   3 configs (director, som_lip, dreamerv3)
# x 6 tasks   (cheetah, pinpad6, hopper, pinpad5, cartpole, pinpad4)
# x 3 seeds   = 54 runs, one GPU each.
#
# ORDERING. Task index is seed-major, then environment, then config, so the
# array's natural index order is exactly the requested sequence:
#
#   0: director cheetah s0     1: som_lip cheetah s0     2: dreamerv3 cheetah s0
#   3: director pinpad6 s0     4: som_lip pinpad6 s0     5: dreamerv3 pinpad6 s0
#   ...                       17: dreamerv3 pinpad4 s0
#  18: director cheetah s1     ...                       53: dreamerv3 pinpad4 s2
#
# SLURM starts array tasks in index order when resources allow, so a whole seed
# sweep is in flight before any seed-1 run begins, and within an environment the
# priority is director -> som_lip -> dreamerv3.
#
# CONCURRENCY. `%N` throttles the array. Our association quota is 8 GPUs on each
# of gpu-a100 / gpu-v100 / gpu-p100 (24 total), and the array is capped below
# that so the cluster keeps headroom. Jobs are submitted to all three partitions
# at once; SLURM places each task wherever a slot frees first, which keeps the
# order sensible even while the A100s are busy.
#
# P100 SAFETY. Only saion-gpu[11-14] have been upgraded to el8; the rest still
# run glibc 2.17 and segfault on Python 3.11 at import. `--exclude` keeps tasks
# off them.
#
# Usage:
#   bash sbatch/submit_benchmark.sh            # submit
#   DRY_RUN=1 bash sbatch/submit_benchmark.sh  # print the matrix and exit

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS_ORDER=(director som_lip dreamerv3)
TASKS_ORDER=(dmc_cheetah_run pinpad_six dmc_hopper_hop pinpad_five dmc_cartpole_swingup pinpad_four)
SEEDS=(0 1 2)

FIRST_EXP="${FIRST_EXP:-735}"
PARAMS="${PARAMS:-$PWD/sbatch/bench_params.txt}"
MAXCONC="${MAXCONC:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-dreamerv3-bench-2026-09}"
PARTITIONS="${PARTITIONS:-gpu-a100,gpu-v100,gpu-p100}"
EXCLUDE="${EXCLUDE:-saion-gpu[01-10]}"
GPUS_PER_RUN="${GPUS_PER_RUN:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48G}"
WALLTIME="${WALLTIME:-2-00:00:00}"

: > "$PARAMS"
i=0
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS_ORDER[@]}"; do
    for cfg in "${CONFIGS_ORDER[@]}"; do
      exp="e$((FIRST_EXP + i))"
      printf '%s %s %s %s\n' "$cfg" "$task" "$seed" "$exp" >> "$PARAMS"
      i=$((i + 1))
    done
  done
done
N=$i
echo "wrote $N tasks to $PARAMS  (e${FIRST_EXP}..e$((FIRST_EXP + N - 1)))"
echo "first 6:"; head -6 "$PARAMS" | nl -ba -w3 -s': ' | sed 's/^/  /'

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] would submit: --array=0-$((N - 1))%${MAXCONC} -p $PARTITIONS"
  exit 0
fi

JOB=$(sbatch --parsable \
  -J bench \
  -p "$PARTITIONS" \
  --exclude="$EXCLUDE" \
  --gres=gpu:"$GPUS_PER_RUN" \
  -c "$CPUS" \
  --mem="$MEM" \
  -t "$WALLTIME" \
  --array=0-$((N - 1))%${MAXCONC} \
  --export=ALL,PARAMS="$PARAMS",WANDB_PROJECT="$WANDB_PROJECT" \
  sbatch/run_benchmark.sbatch)

echo "submitted array job $JOB  ($N tasks, max $MAXCONC concurrent)"
mkdir -p job_logs
LOG="job_logs/bench_$(date +%Y%m%d_%H%M%S).tsv"
{
  printf 'array_job\ttask_idx\texp\tconfig\ttask\tseed\n'
  n=0
  while read -r cfg task seed exp; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB" "$n" "$exp" "$cfg" "$task" "$seed"
    n=$((n + 1))
  done < "$PARAMS"
} > "$LOG"
echo "logged to $LOG"
