#!/bin/bash
# Round 11: the recipe failed on reacher_easy (251 at 29 min; base 50 Hz held
# 800 at 14.0). Is it the 12.5 Hz control rate or the lr/width/horizon knobs?
set -u
cd "$(dirname "$0")/.."
LOG=job_logs/sim_phone_r11.tsv
P100="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] -t 0-0:50:00"
KNOBS="--run.train_ratio 8192 --batch_size 32 --agent.opt.lr 3e-4 --agent.ac_opt.lr 3e-4 --.*\.units 256"
submit() {
  local name=$1 configs=$2; shift 2
  for _ in $(seq 1 360); do
    if out=$(CONFIGS="$configs" sbatch -J "$name" $P100 sbatch/run_sim_phone.sbatch "$@" 2>&1); then
      echo -e "$name\t${out##* }\t$configs $*" | tee -a "$LOG"; return 0
    fi
    sleep 20
  done
  echo -e "$name\tFAILED\t$configs $*" | tee -a "$LOG"
}
submit e1186_reachereasy_rt50_knobs "sim_phone" --task dmc_reacher_easy $KNOBS --agent.horizon 100
submit e1187_reachereasy_rt25_knobs "sim_phone" --task dmc_reacher_easy $KNOBS --agent.horizon 50 --env.dmc.repeat 4 --env.realtime_hz 25
submit e1188_reachereasy_rt12_defaultknobs "sim_phone" --task dmc_reacher_easy --env.dmc.repeat 8 --env.realtime_hz 12.5
submit e1189_reachereasy_rt12_recipe_seed1 "sim_phone sim_phone_fast" --task dmc_reacher_easy --seed 1
