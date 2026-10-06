#!/bin/bash
# Round 14: after the literature survey. Robustness of the recipe to
# actuation delay (the robot has some), which half of the lr bump matters,
# more width, exploration, and a third seed of the 25 Hz variant.
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r14.tsv
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
submit e1215_cartpole_rt12_recipe_delay1 --env.action_delay 1
submit e1216_cartpole_rt25_recipe_delay1 $R25 --env.action_delay 1
submit e1217_cartpole_rt12_recipe_u512 --.*\.units 512
submit e1218_cartpole_rt12_recipe_lrmodelonly --agent.ac_opt.lr 1e-4
submit e1219_cartpole_rt12_recipe_lraconly --agent.opt.lr 1e-4
submit e1220_cartpole_rt25_recipe_seed3 $R25 --seed 3
submit e1221_cartpole_rt12_recipe_actent1e3 --agent.imag_loss.actent 1e-3
submit e1222_cartpole_rt50_recipe_h100 --env.dmc.repeat 2 --env.realtime_hz 50 --agent.horizon 100
