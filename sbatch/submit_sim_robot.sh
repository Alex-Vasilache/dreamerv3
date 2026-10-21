#!/bin/bash
# Launch a round of the simulated-robot sweep (docs/SIM_ROBOT.md): each line
# of PARAMS is one run,
#
#   <exp> <arm> <task: balance|command> <hz: 25|65> <gpu: v100|p100> [main.py flags...]
#
# with <arm> 'flat' (DreamerV3), 'hrl' (the OG Director) or 'hrlfast'
# (hrl + hrl_robot_fast, the tuned goal-only recipe). Flags after the
# GPU go to main.py, e.g. --agent.manager_sample_freq 16 --seed 1.
#
#   PARAMS=sbatch/sim_robot_r1.txt sbatch/submit_sim_robot.sh
#   MINUTES=75 PARAMS=... sbatch/submit_sim_robot.sh   # wall limit per run
set -euo pipefail
cd "$(dirname "$0")/.."
PARAMS=${PARAMS:?set PARAMS to a params file}
MINUTES=${MINUTES:-75}
LOG=job_logs/$(basename "${PARAMS%.txt}").tsv
mkdir -p job_logs

while read -r exp arm task hz gpu extra; do
  [[ -z "$exp" || "$exp" == \#* ]] && continue
  configs="robot_daydreamer robot_fast robot_${hz}hz"
  flags=""
  if [ "$task" = command ]; then
    configs="$configs robot_command robot_command_lean"
    # robot_command_lean's 5 s return horizon is 325 steps at 65 Hz.
    flags="--agent.horizon $((5 * hz))"
  fi
  case $arm in
    flat) ;;
    hrl) configs="director director_og $configs hrl_robot" ;;
    hrlfast) configs="director director_og $configs hrl_robot hrl_robot_fast" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  configs="$configs sim_robot sim_${hz}hz"
  case $gpu in
    v100) gflags="-p gpu-v100 --gres=gpu:v100:1 --exclude=saion-gpu20" ;;
    p100) gflags="-p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01,02,04,06,07-10]" ;;
    a100) gflags="-p gpu-a100 --gres=gpu:a100:1" ;;
  esac
  name="${exp}_simrobot_${task}${hz}_${arm}"
  id=$(CONFIGS="$configs" sbatch --parsable -J "$name" -t "0-0:${MINUTES}:00" \
       $gflags sbatch/run_sim_phone.sbatch $flags $extra)
  printf '%s\t%s\t%s\t%s\t%s\n' "$exp" "$id" "$name" "$configs" "$flags $extra" \
    | tee -a "$LOG"
done < "$PARAMS"
