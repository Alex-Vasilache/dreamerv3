#!/bin/bash
# Round 12: 12.5 Hz is too coarse for reacher_easy (fails with default or
# recipe knobs), while the same knobs at 25 Hz hold 800 at 14.4. Check that
# the 25 Hz variant of the recipe also keeps the gains on the other tasks.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r12.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-0:45:00"
R25="--env.dmc.repeat 4 --env.realtime_hz 25 --agent.horizon 50"
submit() {
  local name=$1; shift
  for _ in $(seq 1 360); do
    if out=$(CONFIGS="sim_phone sim_phone_fast" sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$*" | tee -a "$LOG"
}
submit e1190_pendulumswingup_rt25_recipe --task dmc_pendulum_swingup $R25
submit e1191_cartpolebalance_rt25_recipe --task dmc_cartpole_balance $R25
submit e1192_reachereasy_rt25_recipe_seed1 --task dmc_reacher_easy $R25 --seed 1
submit e1193_cartpole_rt25_recipe_seed2 --task dmc_cartpole_swingup $R25 --seed 2
