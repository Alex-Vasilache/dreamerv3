#!/bin/bash -l
# Smoke test for `run.save_every_steps` -- the permanent milestone checkpoints
# written to logdir/ckpt_milestones/ on an absolute step grid.
#
# Tiny debug model on a real DMC task, so it needs a GL backend and a couple of
# minutes on a GPU node (the `intel` CPU partition queues for hours). Uses a
# 200-step grid instead of the real 500k.
#
# Usage:
#   sbatch sbatch/run_smoke_ckpt_milestones.sbatch
#   REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3 sbatch sbatch/run_smoke_ckpt_milestones.sbatch
#SBATCH -J smoke_ckpt_milestones
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -t 0-00:20:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
# The `dummy` env's 2-D int observation trips an unrelated shape assert in the
# HRL loss, so smoke on a real (tiny) DMC task instead.
export MUJOCO_GL=egl

REPO="${REPO:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3}"
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/smoke_ckpt_milestones_j${SLURM_JOB_ID:-manual}}"
LOGDIR="$RUN_DIR/logdir"
EVERY=200
STEPS=1000
mkdir -p "$RUN_DIR"
cd "$REPO"

echo "REPO=$REPO LOGDIR=$LOGDIR EVERY=$EVERY STEPS=$STEPS"

python -u dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs debug \
  --task dmc_cartpole_swingup \
  --jax.platform cuda \
  --run.steps "$STEPS" \
  --run.save_every_steps "$EVERY"

echo "---- ckpt_milestones ----"
ls -la "$LOGDIR/ckpt_milestones" || true

# Every multiple of EVERY up to STEPS must have a complete (`done`) folder, and
# the rolling ckpt/ must still be there untouched.
python - "$LOGDIR" "$EVERY" "$STEPS" <<'EOF'
import pathlib
import pickle
import sys

logdir, every, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
milestones = logdir / 'ckpt_milestones'
expected = [i * every for i in range(1, steps // every + 1)]
failed = []
for want in expected:
    folder = milestones / f'{want:012d}'
    if not (folder / 'done').exists():
        failed.append(f'missing or incomplete: {folder}')
        continue
    if not (folder / 'agent.pkl').exists():
        failed.append(f'no agent.pkl in {folder}')
    if (folder / 'replay.pkl').exists():
        failed.append(f'replay was saved into milestone {folder}')
    step = pickle.loads((folder / 'step.pkl').read_bytes())
    if not (want <= int(step) < want + every):
        failed.append(f'{folder} holds step {int(step)}, expected ~{want}')
if not (logdir / 'ckpt' / 'latest').exists():
    failed.append('rolling ckpt/latest is missing')
extra = [p.name for p in sorted(milestones.glob('*'))
         if p.name not in {f'{w:012d}' for w in expected}]
if extra:
    failed.append(f'unexpected milestone folders: {extra}')
if failed:
    print('MILESTONE SMOKE FAILED')
    print('\n'.join(failed))
    sys.exit(1)
print(f'MILESTONE SMOKE PASSED: {len(expected)} milestones at {expected}')
EOF

echo "Run dir: $RUN_DIR (delete after inspecting)"
exit $?
