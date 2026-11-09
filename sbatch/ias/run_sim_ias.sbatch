#!/bin/bash
#SBATCH -J sim_ias
#SBATCH -t 0-2:30:00
#SBATCH -c 3
#SBATCH --mem=7G
#SBATCH --gres=gpu:1
#SBATCH -C rtx3090|a5000
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
# The simulated robot (configs robot robot_sim) on the IAS cluster, TU Darmstadt,
# as an array over the lines of $PARAMS (format of sbatch/submit_sim_robot.sh:
# <exp> flat <balance|still|command> <hz> <gpu> [main.py flags]; <gpu> ignored). Submitted by
# sbatch/ias/submit_sim_ias.sh, which sets CODE (a code snapshot) and PARAMS.
set -uo pipefail
CODE="${CODE:?}"; PARAMS="${PARAMS:?}"; RUNS="${RUNS:-$HOME/robot_runs}"
VENV="${VENV:-$HOME/envs/dreamerv3}"
line=$(grep -v -e '^#' -e '^\s*$' "$PARAMS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r exp arm task hz gpu extra <<< "$line"
[[ $arm == flat ]] || { echo "only flat arms: $line"; exit 1; }
case $task in
  balance) configs="robot robot_sim" ;;
  still) configs="robot robot_still robot_sim" ;;
  command) configs="robot robot_command robot_sim" ;;
  *) echo "unknown task: $line"; exit 1 ;;
esac
name="${exp}_simrobot_${task}${hz}_${arm}"
RUN_DIR="$RUNS/${name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
# The run writes to the node's local /tmp (replay, checkpoints, experience are
# rewritten all the time; IAS asks to keep that churn out of home, 2026-11-09).
# On exit, everything except replay and checkpoints is copied back, plus the
# last milestone snapshot.
WORK="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${name}_${SLURM_JOB_ID}"
mkdir -p "$WORK/logdir"
copy_back() {
  kill "${sync_pid:-0}" 2>/dev/null
  rsync -a --exclude replay --exclude ckpt --exclude ckpt_milestones "$WORK/logdir/" "$RUN_DIR/logdir/"
  last=$(ls -d "$WORK"/logdir/ckpt_milestones/* 2>/dev/null | tail -1)
  [ -n "$last" ] && mkdir -p "$RUN_DIR/logdir/ckpt_milestones" && cp -r "$last" "$RUN_DIR/logdir/ckpt_milestones/"
  rm -rf "$WORK"
}
trap copy_back EXIT
trap "exit 143" TERM INT
# Copy the logdir back every 5 min as well, so a run killed before the final
# copy finishes (time limit, OOM) still keeps its experience (11-10: round X
# lost 33 of 40 runs that way). Experience chunks are written once, so each
# file is copied once.
( while sleep 300; do
    rsync -a --exclude replay --exclude ckpt --exclude ckpt_milestones "$WORK/logdir/" "$RUN_DIR/logdir/" 2>/dev/null
  done ) &
sync_pid=$!
printf 'RUN_DIR=%s\nARGS=%s\nNODE=%s\nGPU=%s\nJOB=%s_%s\n' "$RUN_DIR" "$extra" "$(hostname)" \
  "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" > "$RUN_DIR/job.env"
echo "[sim] $name node=$(hostname) run_dir=$RUN_DIR args=$extra"
cd "$WORK"
"$VENV/bin/python" -u "$CODE/dreamerv3/main.py" --logdir "$WORK/logdir" \
  --configs $configs --jax.platform cuda --env.robot.hz "$hz" $extra
rc=$?
exit $rc
