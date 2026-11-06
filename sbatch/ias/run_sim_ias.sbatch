#!/bin/bash
#SBATCH -J sim_ias
#SBATCH -t 0-1:15:00
#SBATCH -c 3
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -C rtx3090|a5000
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
# The simulated robot (configs robot robot_sim) on the IAS cluster, TU Darmstadt,
# as an array over the lines of $PARAMS (format of sbatch/submit_sim_robot.sh:
# <exp> flat balance <hz> <gpu> [main.py flags]; <gpu> ignored). Submitted by
# sbatch/ias/submit_sim_ias.sh, which sets CODE (a code snapshot) and PARAMS.
set -uo pipefail
CODE="${CODE:?}"; PARAMS="${PARAMS:?}"; RUNS="${RUNS:-$HOME/robot_runs}"
VENV="${VENV:-$HOME/envs/dreamerv3}"
line=$(grep -v -e '^#' -e '^\s*$' "$PARAMS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r exp arm task hz gpu extra <<< "$line"
[[ $arm == flat && $task == balance ]] || { echo "only flat balance: $line"; exit 1; }
name="${exp}_simrobot_${task}${hz}_${arm}"
RUN_DIR="$RUNS/${name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
printf 'RUN_DIR=%s\nARGS=%s\nNODE=%s\nGPU=%s\nJOB=%s_%s\n' "$RUN_DIR" "$extra" "$(hostname)" \
  "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" > "$RUN_DIR/job.env"
echo "[sim] $name node=$(hostname) run_dir=$RUN_DIR args=$extra"
cd "$RUN_DIR"
"$VENV/bin/python" -u "$CODE/dreamerv3/main.py" --logdir "$RUN_DIR/logdir" \
  --configs robot robot_sim --jax.platform cuda --env.robot.hz "$hz" $extra
exit $?
