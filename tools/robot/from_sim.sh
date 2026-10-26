#!/usr/bin/env bash
# Use an agent trained in the MuJoCo sim on the real robot.
#
#   tools/robot/from_sim.sh try <name> <sim run dir> [step]
#       Put the sim policy on the phone's Dreamer Player list, as
#       save_policy.sh does for a real run: no learning, just see what it does.
#   CONFIGS="robot robot_command" tools/robot/from_sim.sh train <sim run dir> [step] [name] [main.py flags]
#       Start a real training run (tools/robot/train.sh) whose agent begins
#       with the sim run's weights -- world model, actor, critic and
#       optimizer state; replay and step counters start empty.
#
# <sim run dir> is on the trainer host (e.g. a run archived in the bucket).
# [step] picks a snapshot from its logdir/ckpt_milestones/ (env steps:
# 19500 = 5 min at 65 Hz; 'latest', the default, is the final checkpoint).
# The sim run's configs must match CONFIGS apart from robot_sim (same task,
# same network); train refuses a run of another task.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE=${1:?usage: from_sim.sh try <name> <sim run dir> [step] | train <sim run dir> [step] [name] [flags]}
shift

ckpt_dir() {  # ckpt_dir RUN STEP: the checkpoint directory on the trainer host
  local run=$1 step=$2
  if [ "$step" = latest ]; then
    echo "$run/logdir/ckpt/$(trainer_sh "cat '$run/logdir/ckpt/latest'")"
  else
    printf '%s/logdir/ckpt_milestones/%012d\n' "$run" "$step"
  fi
}

case $MODE in
  try)
    NAME=${1:?name}; RUN=${2:?sim run dir}; STEP=${3:-latest}
    CKPT=$(ckpt_dir "$RUN" "$STEP")
    trainer_sh "test -f '$CKPT/agent.pkl'" || robot_die "no checkpoint at $CKPT"
    OUT="$HOME/logdir/robot_policies"
    mkdir -p "$OUT"
    tmp=$(mktemp -d)
    trainer_get "$CKPT/agent.pkl" "$tmp/agent.pkl"
    trainer_get "$RUN/logdir/config.yaml" "$tmp/config.yaml"
    cd "$ROBOT_ROOT"
    "$(robot_python)" - "$tmp" "$OUT" "$NAME" "$RUN" "$STEP" <<'PY'
import json, pathlib, sys, time
import elements, ruamel.yaml as yaml
sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path.cwd() / 'tools' / 'robot' / 'dev'))
from dreamerv3.deploy import export
from export_policy import load_params, flatten, _is_flat
from embodied.envs import robot

tmp, out, name, run, step = sys.argv[1:]
tmp, out = pathlib.Path(tmp), pathlib.Path(out)
config = elements.Config(yaml.YAML(typ='safe').load((tmp / 'config.yaml').read_text()))
raw = load_params(tmp)
params = dict(raw) if _is_flat(raw) else flatten(raw)
blob, meta = export.pack(config, params)
(out / f'{name}.npz').write_bytes(blob)
r = config.env.robot
# As a training run's own sidecar (robot.py _settings). `length` must be the
# episode in steps: the Player starts the policy's memory afresh every
# `length` steps, and the config's 0 ("from episode_s") reset it every step.
hz = float(r.hz)
side = dict(
    name=name, run=run, job=pathlib.Path(run).name, stamp=step,
    published=time.strftime('%Y-%m-%d %H:%M'),
    task=config.task.removeprefix('robot_'), hz=hz,
    length=int(r.length) or int(round(hz * float(r.episode_s))),
    sim=True, sim_surface=str(r.sim_surface),
    **{k: r[k] for k in robot.PLAYER_SETTINGS if k in r})
(out / f'{name}.json').write_text(json.dumps(side, indent=1))
print(f'{name}: {len(blob) / 1e6:.1f} MB, {config.task} policy from the sim '
      f'({r.sim_surface}, {step}) of {run}')
PY
    DEST=$PHONE_FILES/policies
    adb shell mkdir -p "$DEST"
    adb push "$OUT/$NAME.npz" "$OUT/$NAME.json" "$DEST/" >/dev/null
    echo "saved to $OUT and the phone ($DEST); pick '$NAME' in the Dreamer Player"
    ;;
  train)
    RUN=${1:?sim run dir}; shift
    STEP=latest
    if [ $# -gt 0 ] && [[ $1 =~ ^([0-9]+|latest)$ ]]; then STEP=$1; shift; fi
    CKPT=$(ckpt_dir "$RUN" "$STEP")
    trainer_sh "test -f '$CKPT/agent.pkl'" || robot_die "no checkpoint at $CKPT"
    # The agent's network follows the task (the command adds an input), so
    # the sim run must have trained the same task as this one.
    WANT=robot_balance
    [[ " ${CONFIGS:-robot} " == *" robot_command "* ]] && WANT=robot_command
    GOT=$(trainer_sh "sed -n 's/^task: *//p' '$RUN/logdir/config.yaml'")
    [ "$GOT" = "$WANT" ] || robot_die "the sim run trained $GOT, CONFIGS asks for $WANT"
    NAME=from_sim
    if [ $# -gt 0 ] && [[ $1 != -* ]]; then NAME=$1; shift; fi
    echo "== warm start from $CKPT"
    exec "$ROBOT_TOOLS/train.sh" "$NAME" --run.from_checkpoint "$CKPT" "$@"
    ;;
  *) robot_die "mode must be try or train, not '$MODE'" ;;
esac
