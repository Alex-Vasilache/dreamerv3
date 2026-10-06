#!/bin/bash
# Round 3 of the real-time cartpole sweep. Round 2: horizon 100 was fastest,
# and a smaller batch did not buy more updates/s -- the learner is per-update
# overhead bound -- so a larger batch may be free. Retries while the Slurm
# controller is down.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r3.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-1:15:00"
B="--run.train_ratio 8192"
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
submit e1129_cartpole_rt50_h100_seed1 --agent.horizon 100 --seed 1
submit e1130_cartpole_rt50_h100_seed2 --agent.horizon 100 --seed 2
submit e1131_cartpole_rt50_b32x64 --batch_size 32
submit e1132_cartpole_rt50_b64x64 --batch_size 64
submit e1133_cartpole_rt50_h100_b32x64 --agent.horizon 100 --batch_size 32
submit e1134_cartpole_rt50_h50 --agent.horizon 50
submit e1135_cartpole_rt50_h100_lr3e4 --agent.horizon 100 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4
submit e1136_cartpole_rt50_h100_b32x64_lr3e4 --agent.horizon 100 --batch_size 32 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4
