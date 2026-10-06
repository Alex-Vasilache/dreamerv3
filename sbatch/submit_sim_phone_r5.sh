#!/bin/bash
# Round 5 of the real-time cartpole sweep: refine the round-4 recipe
# (horizon 100 + batch 32x64, 16.6 min mean to hold 800).
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r5.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:00:00"
R="--run.train_ratio 8192 --agent.horizon 100"
submit() {
  local name=$1; shift
  for _ in $(seq 1 360); do
    if out=$(sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch $R "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$*" | tee -a "$LOG"
}
submit e1146_cartpole_rt50_h100_b64x64_seed1 --batch_size 64 --seed 1
submit e1147_cartpole_rt50_h100_b64x64_seed2 --batch_size 64 --seed 2
submit e1148_cartpole_rt50_h100_b32x64_imag32 --batch_size 32 --agent.imag_length 32
submit e1149_cartpole_rt50_h150_b32x64 --batch_size 32 --agent.horizon 150
submit e1150_cartpole_rt50_h100_b32x64_units128 --batch_size 32 --.*\\.units 128
