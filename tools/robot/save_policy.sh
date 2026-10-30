#!/usr/bin/env bash
# Save a robot run's current policy under a name, for the Dreamer Player app.
#
#   tools/robot/save_policy.sh balance_table              # the latest robot job
#   tools/robot/save_policy.sh balance_carpet <run dir>   # a given run (on the trainer host)
#
# Takes the newest weights the run's learner published, packs them into the
# phone's .npz (dreamerv3/deploy/export.py, with the run's own config), writes
# <name>.json beside it (source run, control rate, episode length, reward
# settings, so the player can pace itself and score steps the same way), keeps
# both in ~/logdir/robot_policies/ and pushes them to the phone's policy list.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NAME=${1:?usage: tools/robot/save_policy.sh <name> [run dir]}
RUN=${2:-$(latest_run_dir)}
[ -n "$RUN" ] || robot_die "no running robot job; pass its run dir"
ROOT=$ROBOT_ROOT
OUT="$HOME/logdir/robot_policies"
mkdir -p "$OUT"
cd "$ROOT"
tmp=$(mktemp -d)

STAMP=$(trainer_sh "cat $RUN/logdir/online_shared/policy/latest")
trainer_get "$RUN/logdir/online_shared/policy/policy_$STAMP.pkl" "$tmp/policy.pkl"
trainer_get "$RUN/logdir/config.yaml" "$tmp/config.yaml"
# The job name (e.g. e1294_robot_25hz) is the run dir's name minus its date.
JOB=$(basename "$RUN" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')
# Run dirs from before 2026-10-16 had generic names: find the job by its log.
if [[ ! $JOB =~ [0-9]+hz ]] && [ "$ROBOT_TRAINER" = slurm ]; then
  JOB=$(trainer_sh "grep -l '$RUN' $ROBOT_RUNS/slurm_logs/*.out 2>/dev/null | tail -1 | xargs -r basename" || true)
  JOB=${JOB%.out}
fi

"$(robot_python)" - "$tmp" "$OUT" "$NAME" "$RUN" "$STAMP" "${JOB%.out}" <<'PY'
import json, pathlib, sys, time
import elements, ruamel.yaml as yaml
sys.path.insert(0, str(pathlib.Path.cwd()))
from dreamerv3.deploy import export
from embodied.envs.robot import _load_policy_pickle

tmp, out, name, run, stamp, job = sys.argv[1:]
tmp, out = pathlib.Path(tmp), pathlib.Path(out)
config = elements.Config(yaml.YAML(typ='safe').load((tmp / 'config.yaml').read_text()))
data = _load_policy_pickle((tmp / 'policy.pkl').read_bytes())
params = data['params'] if isinstance(data, dict) and 'params' in data else data
blob, meta = export.pack(config, params)
(out / f'{name}.npz').write_bytes(blob)
r = config.env.robot
keys = ('theta_zero', 'theta_lo', 'theta_hi', 'theta_sigma', 'speed_scale',
        'drift_penalty', 'drift_clip', 'wheel_penalty', 'rate_penalty',
        'action_rate_penalty', 'command_scale', 'length')
side = dict(
    name=name, run=run, job=job, stamp=stamp,
    published=time.strftime('%Y-%m-%d %H:%M', time.localtime(int(stamp) / 1e9)),
    task=config.task, hz=float(r['hz']) if 'hz' in r else None,
    **{k: r[k] for k in keys})
(out / f'{name}.json').write_text(json.dumps(side, indent=1))
print(f'{name}: {len(blob) / 1e6:.1f} MB, weights of {side["published"]} from {run}')
PY

# The rate the run trained at: the job name's tag (..._25hz) if it has one,
# else the config's env.robot.hz (sim jobs are named e.g. ..._command65),
# else 25 for runs from before the key.
HZ=$(echo "$JOB" | grep -oE '[0-9]+hz' | grep -oE '[0-9]+' || true)
python3 - "$OUT/$NAME.json" "$HZ" <<'PY'
import json, sys
p, hz = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d['hz'] = float(hz or d['hz'] or 25)
json.dump(d, open(p, 'w'), indent=1)
PY

DEST=$PHONE_FILES/policies
adb shell mkdir -p "$DEST"
adb push "$OUT/$NAME.npz" "$OUT/$NAME.json" "$DEST/" >/dev/null
echo "saved to $OUT and the phone ($DEST)"
