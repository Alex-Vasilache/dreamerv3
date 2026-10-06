#!/bin/bash
# Round 9 of the real-time cartpole sweep: robot-facing checks of the low-rate
# recipe (12.5 Hz, 32x64, horizon 25, lr 3e-4, MLP units 256; 10.5 min mean to
# hold 800 over 3 seeds). Does the weight-push interval matter now that
# learning is fast, and does lr 3e-4 stay stable over a long run?
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r9.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10]"
RECIPE="--run.train_ratio 8192 --batch_size 32 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4 --env.dmc.repeat 8 --env.realtime_hz 12.5 --agent.horizon 25 --.*\.units 256"
submit() {
  local name=$1 wall=$2; shift 2
  for _ in $(seq 1 360); do
    if out=$(sbatch -J "$name" $P100 -t "$wall" sbatch/run_sim_phone.sbatch $RECIPE "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$*" | tee -a "$LOG"
}
submit e1174_cartpole_rt12_recipe_sync5 0-0:45:00 --online_sync_every 5
submit e1175_cartpole_rt12_recipe_sync10 0-0:45:00 --online_sync_every 10
submit e1176_cartpole_rt12_recipe_sync60 0-0:45:00 --online_sync_every 60
submit e1177_cartpole_rt12_recipe_long_seed3 0-3:00:00 --seed 3
submit e1178_cartpole_rt10_recipe_seed1 0-0:45:00 --env.dmc.repeat 10 --env.realtime_hz 10 --agent.horizon 20 --seed 1
submit e1179_cartpole_rt12_recipe_b16x64 0-0:45:00 --batch_size 16
