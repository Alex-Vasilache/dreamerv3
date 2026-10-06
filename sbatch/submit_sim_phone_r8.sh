#!/bin/bash
# Round 8 of the real-time cartpole sweep. Round 7: at 12.5 Hz, lr 3e-4 +
# MLP units 128/256 held 800 at 9.1/8.9 min; 25 Hz + the same at 9.9; lr 3e-4
# alone 12.6 (3 seeds). Replicate, push capacity, and run at 10 Hz (robot today).
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r8.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-0:50:00"
B="--run.train_ratio 8192 --batch_size 32 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4"
R12="$B --env.dmc.repeat 8 --env.realtime_hz 12.5 --agent.horizon 25"
R25="$B --env.dmc.repeat 4 --env.realtime_hz 25 --agent.horizon 50"
R10="$B --env.dmc.repeat 10 --env.realtime_hz 10 --agent.horizon 20"
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
submit e1166_cartpole_rt12_lr3e4_u128_seed1 $R12 $U128 --seed 1
submit e1167_cartpole_rt12_lr3e4_u256_seed1 $R12 $U256 --seed 1
submit e1168_cartpole_rt12_lr3e4_u256_seed2 $R12 $U256 --seed 2
submit e1169_cartpole_rt25_lr3e4_u128_seed1 $R25 $U128 --seed 1
submit e1170_cartpole_rt25_lr3e4_u256 $R25 $U256
submit e1171_cartpole_rt12_lr3e4_u256_deter1024 $R12 $U256 --agent.dyn.rssm.deter 1024
submit e1172_cartpole_rt10_lr3e4_u256 $R10 $U256
submit e1173_cartpole_rt12_lr5e4_u256 $R12 $U256 --agent.opt.lr 5e-4 --agent.ac_opt.lr 5e-4
