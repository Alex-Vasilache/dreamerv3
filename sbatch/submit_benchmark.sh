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
OUTDIR="${OUTDIR:-$PWD/sbatch}"
WANDB_PROJECT="${WANDB_PROJECT:-dreamerv3-bench-2026-09}"
WALLTIME="${WALLTIME:-2-00:00:00}"

# MEASURED 2026-09-04 with the size6m configs (3.94M params), one GPU per run:
#
#   A100-80GB  native conv     24.2 env fps   2.4 GB   ~12.6 h per 1.1M steps
#   V100-16GB  reference conv   7.8 env fps   4.2 GB   ~39 h
#   P100-16GB  reference conv   7.2 env fps   3.8 GB   ~42 h
#
# Two consequences. (1) One GPU per run is enough everywhere -- the old
# multi-V100 scripts used 4 GPUs because the 107M `director_match` model needed
# ~59 GB; at 3.94M a single 16 GB card is 4x oversized. (2) A GPU cannot be
# shared between runs even though 2.4 GB of an 80 GB A100 is nothing: this
# cluster exposes `GresTypes = gpu,mic` with no MPS or shard type, so SLURM
# hands out whole GPUs.
#
# Work is therefore split across partitions in proportion to throughput, so all
# three finish at about the same wall-clock time rather than the P100s trailing
# by days: 12 cells to A100, 4 to V100, 2 to P100.
#
# The unit of assignment is a CELL -- the three algorithms for one
# (environment, seed) -- not an individual run. Assigning runs individually is
# what the first submission did (index mod 9) and it silently correlated
# hardware with algorithm: algorithm is index mod 3, so mod 9 sent every P100
# task to dreamerv3 and every V100 task to director/som_lip. Hardware does not
# change a result at a fixed step budget, but it does change numerics -- the
# som_lip bfloat16 NaN below appeared on V100 and not on A100 -- and a
# confound between arm and card is not worth the convenience. Keeping a cell
# together also means the three arms being compared always ran on identical
# hardware.
#
# Concurrency 8 + 8 + 4 = 20 of our 24-GPU quota, leaving 4 P100 slots free.
A100_CONC="${A100_CONC:-8}"
V100_CONC="${V100_CONC:-8}"
P100_CONC="${P100_CONC:-4}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48G}"
# Only saion-gpu[11-14] are upgraded to el8; the rest still run glibc 2.17 and
# segfault on Python 3.11 at import.
P100_NODES="${P100_NODES:-saion-gpu[11-14]}"

ALL="$OUTDIR/bench_params_all.txt"
: > "$ALL"
i=0
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS_ORDER[@]}"; do
    for cfg in "${CONFIGS_ORDER[@]}"; do
      printf '%s %s %s e%s\n' "$cfg" "$task" "$seed" "$((FIRST_EXP + i))" >> "$ALL"
      i=$((i + 1))
    done
  done
done
N=$i
echo "matrix: $N runs, e${FIRST_EXP}..e$((FIRST_EXP + N - 1))"

# Split whole cells (3 consecutive runs = one env+seed, all three algorithms),
# round-robin over a 6-cell pattern weighted by throughput: A100 A100 A100 A100
# V100 P100 -> 12/4/2 cells = 36/12/6 runs. Order inside each slice is
# preserved, so the priority sequence still holds within a partition.
A="$OUTDIR/bench_params_a100.txt"; : > "$A"
V="$OUTDIR/bench_params_v100.txt"; : > "$V"
P="$OUTDIR/bench_params_p100.txt"; : > "$P"
# 9-cell pattern: 6 A100, 2 V100, 1 P100. Over 18 cells that is 12/4/2 cells
# = 36/12/6 runs, which is the throughput-weighted split
# (8x24.2 : 8x7.8 : 4x7.2 = 68% : 22% : 10%) and lands all three partitions at
# ~57-63 h: A100 36/8 x 12.6, V100 12/8 x 39, P100 6/4 x 42.
NALGO=${#CONFIGS_ORDER[@]}
n=0
while read -r line; do
  cell=$((n / NALGO))
  case $((cell % 9)) in
    0|1|2|3|4|5) echo "$line" >> "$A" ;;
    6|7)         echo "$line" >> "$V" ;;
    8)           echo "$line" >> "$P" ;;
  esac
  n=$((n + 1))
done < "$ALL"
NA=$(wc -l < "$A"); NV=$(wc -l < "$V"); NP=$(wc -l < "$P")
echo "split: a100=$NA v100=$NV p100=$NP"
echo "first 6 overall:"; head -6 "$ALL" | nl -ba -w3 -s': ' | sed 's/^/  /'

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] a100 0-$((NA-1))%$A100_CONC | v100 0-$((NV-1))%$V100_CONC | p100 0-$((NP-1))%$P100_CONC"
  exit 0
fi

mkdir -p job_logs
LOG="job_logs/bench_$(date +%Y%m%d_%H%M%S).tsv"
printf 'partition\tarray_job\ttask_idx\texp\tconfig\ttask\tseed\n' > "$LOG"

submit_one() {
  local name=$1 part=$2 gres=$3 conc=$4 params=$5 count=$6 extra=$7
  [ "$count" -gt 0 ] || { echo "$name: nothing to submit"; return; }
  # shellcheck disable=SC2086
  local jid
  jid=$(sbatch --parsable -J "bench_$name" -p "$part" --gres="$gres" \
        -c "$CPUS" --mem="$MEM" -t "$WALLTIME" \
        --array=0-$((count - 1))%"$conc" $extra \
        --export=ALL,PARAMS="$params",WANDB_PROJECT="$WANDB_PROJECT" \
        sbatch/run_benchmark.sbatch)
  echo "$name: array $jid, $count tasks, max $conc concurrent"
  local k=0
  while read -r cfg task seed exp; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$jid" "$k" "$exp" "$cfg" "$task" "$seed" >> "$LOG"
    k=$((k + 1))
  done < "$params"
}

submit_one a100 gpu-a100 gpu:a100:1 "$A100_CONC" "$A" "$NA" ""
submit_one v100 gpu-v100 gpu:v100:1 "$V100_CONC" "$V" "$NV" ""
submit_one p100 gpu-p100 gpu:p100:1 "$P100_CONC" "$P" "$NP" "--nodelist=$P100_NODES"

echo "logged to $LOG"
