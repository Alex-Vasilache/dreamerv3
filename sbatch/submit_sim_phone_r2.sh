#!/bin/bash
# Round 2 of the real-time cartpole sweep: round 1 showed every arm is bound
# by learner throughput, so these vary what one wall-clock second of GPU buys.
# Base: 50 Hz, sync 30 s, train_ratio 2048 (saturated), size1m 16x64.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r2.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:15:00"
B="--run.train_ratio 2048"
submit() {
  local name=$1; shift
  out=$(sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch $B "$@" 2>&1)
  echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"
}
submit e1121_cartpole_rt50_seed1 --seed 1
submit e1122_cartpole_rt50_lr3e4 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4
submit e1123_cartpole_rt50_b16x32 --batch_length 32
submit e1124_cartpole_rt50_b8x64 --batch_size 8
submit e1125_cartpole_rt50_deter256 --agent.dyn.rssm.deter 256
submit e1126_cartpole_rt50_h100 --agent.horizon 100
submit e1127_cartpole_rt50_lr3e4_b16x32 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4 --batch_length 32
submit e1128_cartpole_rt50_seed2 --seed 2
