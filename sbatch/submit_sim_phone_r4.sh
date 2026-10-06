#!/bin/bash
# Round 4 of the real-time cartpole sweep. Round 3: horizon 100 + batch 32x64
# (14.0 min to hold 800) and horizon 100 + lr 3e-4 (16.7) beat the base
# (25.5 mean of 4 seeds); replicate them, and ask whether control rate matters
# once the return horizon is held fixed in seconds (2 s), which is the
# question for the robot at 10-25 Hz.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r4.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:15:00"
B="--run.train_ratio 8192"
LR="--agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4"
submit() {
  local name=$1; shift
  for _ in $(seq 1 360); do
    if out=$(sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch $B "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$*" | tee -a "$LOG"
}
submit e1137_cartpole_rt50_h100_b32x64_seed1 --agent.horizon 100 --batch_size 32 --seed 1
submit e1138_cartpole_rt50_h100_b32x64_seed2 --agent.horizon 100 --batch_size 32 --seed 2
submit e1139_cartpole_rt50_h100_lr3e4_seed1 --agent.horizon 100 $LR --seed 1
submit e1140_cartpole_rt50_h100_lr3e4_seed2 --agent.horizon 100 $LR --seed 2
submit e1141_cartpole_rt50_h100_b32x64_lr3e4_seed1 --agent.horizon 100 --batch_size 32 $LR --seed 1
submit e1142_cartpole_rt50_h100_b64x64 --agent.horizon 100 --batch_size 64
submit e1143_cartpole_rt100_h200_b32x64 --env.dmc.repeat 1 --env.realtime_hz 100 --agent.horizon 200 --batch_size 32
submit e1144_cartpole_rt25_h50_b32x64 --env.dmc.repeat 4 --env.realtime_hz 25 --agent.horizon 50 --batch_size 32
submit e1145_cartpole_rt12_h25_b32x64 --env.dmc.repeat 8 --env.realtime_hz 12.5 --agent.horizon 25 --batch_size 32
