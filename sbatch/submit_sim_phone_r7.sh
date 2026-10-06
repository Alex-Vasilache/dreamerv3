#!/bin/bash
# Round 7 of the real-time cartpole sweep. Round 6: at 12.5 Hz, lr 3e-4 held
# 800 at 12.8 min and MLP units 128 at 15.0 -- both beat the 50 Hz recipe
# (16.6). At a low control rate the learner does ~4x more updates per sample,
# so it can afford a faster lr and more capacity. Replicate and combine.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r7.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:00:00"
B="--run.train_ratio 8192 --batch_size 32"
R12="$B --env.dmc.repeat 8 --env.realtime_hz 12.5 --agent.horizon 25"
R25="$B --env.dmc.repeat 4 --env.realtime_hz 25 --agent.horizon 50"
R50="$B --agent.horizon 100"
LR3="--agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4"
LR10="--agent.opt.lr 1e-3 --agent.ac_opt.lr 1e-3"
U128="--.*\.units 128"
U256="--.*\.units 256"
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
submit e1158_cartpole_rt12_lr3e4_seed1 $R12 $LR3 --seed 1
submit e1159_cartpole_rt12_lr3e4_seed2 $R12 $LR3 --seed 2
submit e1160_cartpole_rt12_lr3e4_units128 $R12 $LR3 $U128
submit e1161_cartpole_rt12_lr1e3 $R12 $LR10
submit e1162_cartpole_rt12_lr3e4_units256 $R12 $LR3 $U256
submit e1163_cartpole_rt25_lr3e4_units128 $R25 $LR3 $U128
submit e1164_cartpole_rt50_lr3e4_units128 $R50 $LR3 $U128
submit e1165_cartpole_rt12_units128_seed1 $R12 $U128 --seed 1
