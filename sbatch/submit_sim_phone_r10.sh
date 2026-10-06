#!/bin/bash
# Round 10: does the real-time recipe (`sim_phone_fast`) carry over to other
# tasks, or is it tuned to cartpole swingup? Base `sim_phone` vs the recipe on
# three more DMC tasks, real time, one seed each.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r10.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:00:00"
submit() {
  local name=$1 configs=$2; shift 2
  for _ in $(seq 1 360); do
    if out=$(CONFIGS="$configs" sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$configs $*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$configs $*" | tee -a "$LOG"
}
for task in pendulum_swingup cartpole_balance reacher_easy; do
  short=${task//_/}
  submit e${N:?}_${short}_rt50_base "sim_phone" --task dmc_$task; N=$((N+1))
  submit e${N}_${short}_rt12_recipe "sim_phone sim_phone_fast" --task dmc_$task; N=$((N+1))
done
