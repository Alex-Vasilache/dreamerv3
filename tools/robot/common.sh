# Sourced by the robot tools: loads tools/robot/robot.env and gives the
# scripts one way to run commands and read files on the machine the learner
# runs on, whether that is this one (ROBOT_TRAINER=local) or a Slurm cluster
# over ssh (ROBOT_TRAINER=slurm).

ROBOT_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_ROOT="$(cd "$ROBOT_TOOLS/../.." && pwd)"
APP=jp.oist.abcvlib.dreamerBridge
PHONE_FILES=/sdcard/Android/data/$APP/files

# robot.env fills only what the environment has not set, so a one-off
# `ROBOT_TRAINER=local tools/robot/train.sh` beats the file. ROBOT_PROFILE=<p>
# first loads robot.<p>.env (also gitignored), so one computer can keep
# settings for more than one cluster: `ROBOT_PROFILE=ias tools/robot/demo.sh`.
robot_load_env() {
  local line key
  while IFS= read -r line || [ -n "$line" ]; do
    [[ $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[1]}
    [ -n "${!key+set}" ] && continue
    eval "export $key=${BASH_REMATCH[2]}"
  done < "$1"
}
if [ -n "${ROBOT_PROFILE:-}" ]; then
  [ -f "$ROBOT_TOOLS/robot.$ROBOT_PROFILE.env" ] \
    || { echo "error: ROBOT_PROFILE=$ROBOT_PROFILE but tools/robot/robot.$ROBOT_PROFILE.env is missing" >&2; exit 1; }
  robot_load_env "$ROBOT_TOOLS/robot.$ROBOT_PROFILE.env"
fi
[ -f "$ROBOT_TOOLS/robot.env" ] && robot_load_env "$ROBOT_TOOLS/robot.env"

ROBOT_TRAINER=${ROBOT_TRAINER:-local}
ROBOT_RUNS=${ROBOT_RUNS:-$HOME/logdir/robot}
ROBOT_SETUP=${ROBOT_SETUP:-}
ROBOT_NUMBERS_CMD=${ROBOT_NUMBERS_CMD:-}
# Slurm off the cluster's network (e.g. eduroam): relay the phone's link
# through this computer and an ssh forward (train.sh).
ROBOT_RELAY=${ROBOT_RELAY:-}
RELAY_PID=${TMPDIR:-/tmp}/robot_relay.pid
RELAY_LOG=${TMPDIR:-/tmp}/robot_relay.log

robot_die() { echo "error: $*" >&2; exit 1; }

# End the relay's ssh loops (train.sh writes one pid per line: the control
# link and each weight stream) and the ssh each is running.
relay_stop() {
  [ -f "$RELAY_PID" ] || return 0
  local pid
  while read -r pid; do
    [ -n "$pid" ] || continue
    pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
  done <"$RELAY_PID"
  rm -f "$RELAY_PID"
}

robot_need() {  # robot_need VAR...: fail with a pointer to robot.env
  for v in "$@"; do
    [ -n "${!v:-}" ] || robot_die "$v is not set. Set it in tools/robot/robot.env (see robot.env.example)."
  done
}

case $ROBOT_TRAINER in
  local) ;;
  slurm) robot_need ROBOT_SSH ROBOT_REMOTE_CODE ROBOT_RUNS ;;
  *) robot_die "ROBOT_TRAINER must be local or slurm, not '$ROBOT_TRAINER'" ;;
esac

# Run a shell command on the trainer host.
trainer_sh() {
  if [ "$ROBOT_TRAINER" = slurm ]; then
    ssh -o BatchMode=yes "$ROBOT_SSH" "$1"
  else
    bash -c "$1"
  fi
}

# Copy a file from the trainer host to a local path.
trainer_get() {
  if [ "$ROBOT_TRAINER" = slurm ]; then
    scp -q "$ROBOT_SSH:$1" "$2"
  else
    cp "$1" "$2"
  fi
}

# The run dir of the most recently started robot job.
latest_run_dir() {
  trainer_sh "sed -n 's/^RUN_DIR=//p' '$ROBOT_RUNS/robot_endpoints/latest' 2>/dev/null"
}

# This computer's LAN address, the one the phone can reach.
local_ip() {
  if command -v ipconfig >/dev/null && [ "$(uname)" = Darwin ]; then
    for i in en0 en1 en2; do ipconfig getifaddr $i 2>/dev/null && return; done
  fi
  # Linux: the source address of the default route.
  ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p'
}

robot_python() {
  if [ -x "$ROBOT_ROOT/.venv/bin/python" ]; then
    echo "$ROBOT_ROOT/.venv/bin/python"
  else
    command -v python3 || command -v python
  fi
}
