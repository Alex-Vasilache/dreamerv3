#!/bin/bash
# Round 13: pendulum_swingup is sparse; one seed of the 25 Hz recipe never
# found the swing-up (score 0 at 29 min) while 12.5 Hz recipe (9.6) and the
# 50 Hz base (18.7) did. Second seeds before reading anything into it.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r13.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-0:45:00"
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
T="--task dmc_pendulum_swingup --seed 1"
submit e1194_pendulumswingup_rt50_base_seed1 "sim_phone" $T
submit e1195_pendulumswingup_rt12_recipe_seed1 "sim_phone sim_phone_fast" $T
submit e1196_pendulumswingup_rt25_recipe_seed1 "sim_phone sim_phone_fast" $T --env.dmc.repeat 4 --env.realtime_hz 25 --agent.horizon 50
