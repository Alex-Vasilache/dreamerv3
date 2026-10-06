#!/bin/bash
# Round 6 of the real-time cartpole sweep: tune at 12.5 Hz, the rate the robot
# runs at today (10.8 Hz) until the firmware fix lands. Round 5: at a 2 s
# horizon + 32x64, 12.5 Hz holds 800 at 23.7 min (36.0 with the default
# horizon) against 16.6 at 50 Hz.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r6.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:00:00"
R12="--run.train_ratio 8192 --env.dmc.repeat 8 --env.realtime_hz 12.5 --batch_size 32"
submit() {
  local name=$1; shift
  for _ in $(seq 1 360); do
    if out=$(sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$*" | tee -a "$LOG"
}
submit e1151_cartpole_rt12_h25_b32x64_seed1 $R12 --agent.horizon 25 --seed 1
submit e1152_cartpole_rt12_h12_b32x64 $R12 --agent.horizon 12
submit e1153_cartpole_rt12_h50_b32x64 $R12 --agent.horizon 50
submit e1154_cartpole_rt12_h25_b32x64_imag32 $R12 --agent.horizon 25 --agent.imag_length 32
submit e1155_cartpole_rt12_h25_b32x64_units128 $R12 --agent.horizon 25 --.*\\.units 128
submit e1156_cartpole_rt12_h25_b32x64_lr3e4 $R12 --agent.horizon 25 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4
submit e1157_cartpole_rt25_h50_b32x64_seed1 --run.train_ratio 8192 --env.dmc.repeat 4 --env.realtime_hz 25 --batch_size 32 --agent.horizon 50 --seed 1
