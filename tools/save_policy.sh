#!/usr/bin/env bash
# Save a robot run's current policy under a name, for the Dreamer Player app.
#
#   tools/save_policy.sh balance_table                      # the latest robot job
#   tools/save_policy.sh balance_carpet <run dir on /work>  # a given run
#
# Takes the newest weights the run's learner published, packs them into the
# phone's .npz (dreamerv3/deploy/export.py, with the run's own config), writes
# <name>.json beside it (source run, control rate, episode length, reward
# settings, so the player can pace itself and score steps the same way), keeps
# both in ~/logdir/robot_policies/ and pushes them to the phone's policy list.
set -euo pipefail
NAME=${1:?usage: tools/save_policy.sh <name> [run dir on Saion]}
RUN=${2:-$(ssh saion 'sed -n "s/^RUN_DIR=//p" /work/DoyaU/vasilache/work/robot_endpoints/latest')}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HOME/logdir/robot_policies"
APP=jp.oist.abcvlib.dreamerBridge
mkdir -p "$OUT"
cd "$ROOT"
tmp=$(mktemp -d)

STAMP=$(ssh saion "cat $RUN/logdir/online_shared/policy/latest")
scp -q "saion:$RUN/logdir/online_shared/policy/policy_$STAMP.pkl" "$tmp/policy.pkl"
scp -q "saion:$RUN/logdir/config.yaml" "$tmp/config.yaml"
JOB=$(ssh saion "grep -l '$RUN' /work/DoyaU/vasilache/work/slurm_logs/*.out 2>/dev/null | tail -1 | xargs -r basename" || true)

"$ROOT/.venv/bin/python" - "$tmp" "$OUT" "$NAME" "$RUN" "$STAMP" "${JOB%.out}" <<'PY'
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
    task=config.task, hz=None, **{k: r[k] for k in keys})
(out / f'{name}.json').write_text(json.dumps(side, indent=1))
print(f'{name}: {len(blob) / 1e6:.1f} MB, weights of {side["published"]} from {run}')
PY

# The rate the run trained at is what the phone was told, in the run's own
# trainer.json; the job name carries it too (..._25hz).
HZ=$(echo "$JOB" | grep -oE '[0-9]+hz' | grep -oE '[0-9]+' || true)
python3 - "$OUT/$NAME.json" "${HZ:-25}" <<'PY'
import json, sys
p, hz = sys.argv[1], float(sys.argv[2])
d = json.load(open(p)); d['hz'] = hz
json.dump(d, open(p, 'w'), indent=1)
PY

DEST=/sdcard/Android/data/$APP/files/policies
adb shell mkdir -p "$DEST"
adb push "$OUT/$NAME.npz" "$OUT/$NAME.json" "$DEST/" >/dev/null
echo "saved to $OUT and the phone ($DEST)"
