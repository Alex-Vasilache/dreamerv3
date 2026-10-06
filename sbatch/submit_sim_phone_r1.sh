#!/bin/bash
# Round 1 of the real-time cartpole sweep (config `sim_phone`): which knobs
# move score-per-minute-of-robot-time. All on upgraded P100s so learner speed
# is comparable across arms. Retries while the Slurm controller is down.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r1.tsv
mkdir -p job_logs
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10]"

submit() {
  local name=$1; shift
  local out
  for _ in $(seq 1 360); do
    if out=$(sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"
      return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$*" | tee -a "$LOG"
}

submit e1113_cartpole_rt50
submit e1114_cartpole_rt50_sync5 --online_sync_every 5
submit e1115_cartpole_rt25 --env.dmc.repeat 4 --env.realtime_hz 25
submit e1116_cartpole_rt12 --env.dmc.repeat 8 --env.realtime_hz 12.5
submit e1117_cartpole_rt100 --env.dmc.repeat 1 --env.realtime_hz 100
submit e1118_cartpole_rt50_tr2048 --run.train_ratio 2048
submit e1119_cartpole_rt50_tr128 --run.train_ratio 128
submit e1120_cartpole_rt12_sync5 --env.dmc.repeat 8 --env.realtime_hz 12.5 --online_sync_every 5
