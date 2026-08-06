#!/bin/bash -l
#SBATCH -J smoke_goal_ae
#SBATCH -p gpu-a100
#SBATCH -t 0:50:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Pre-launch smoke test for the goal-AE ablation: runs every arm end to end
# through the real env + replay + driver loop (which the pytest suites do not
# cover) at `debug` scale for a few hundred steps, and checks that the new
# train/goal/* metrics actually reach metrics.jsonl.
#
#   sbatch sbatch/run_smoke_goal_ae_ablation.sbatch
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

STEPS="${STEPS:-400}"
ARMS="${ARMS:-director vq som lipvq som_lipvq som_lipvq_mgr}"
BASE="${BASE:-/work/DoyaU/vasilache/work/smoke_goal_ae_j${SLURM_JOB_ID}}"
mkdir -p "$BASE"

fail=0
for arm in $ARMS; do
  case "$arm" in
    director)  cfg="" ;;
    vq)        cfg="goal_vq" ;;
    som)       cfg="goal_som" ;;
    lipvq)     cfg="goal_lipvq" ;;
    som_lipvq) cfg="goal_som_lipvq" ;;
    som_lipvq_mgr) cfg="goal_som_lipvq_mgr" ;;
  esac
  echo "=============== ARM=$arm (configs: debug $cfg) ==============="
  logdir="$BASE/$arm"
  python -u "$REPO/dreamerv3/main.py" \
    --logdir "$logdir" \
    --configs debug ${cfg} \
    --task dmc_hopper_hop \
    --run.steps "$STEPS" \
    --run.envs 2 \
    --jax.platform cuda \
    --jax.prealloc False \
    --logger.outputs jsonl \
    --agent.report False
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "ARM=$arm FAILED (exit $rc)"; fail=1; continue
  fi
  python - "$logdir" "$arm" <<'PY'
import json, sys
logdir, arm = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f'{logdir}/metrics.jsonl') if '"step"' in l]
assert rows, f'{arm}: no metrics rows'
last = rows[-1]
print(f'{arm}: {len(rows)} rows, last step {last["step"]}')
want = [] if arm == 'director' else [
    'train/goal/rec_q', 'train/goal/codebook', 'train/goal/commit',
    'train/goal/quant_err', 'train/goal/used_frac', 'train/goal/perplexity']
missing = [k for k in want if not any(k in r for r in rows)]
assert not missing, f'{arm}: missing metrics {missing}'
for k, v in last.items():
    if isinstance(v, float):
        assert v == v and abs(v) != float('inf'), f'{arm}: {k} = {v}'
seen = {k: r[k] for r in rows for k in r if k.startswith('train/goal/')}
print(f'{arm}: goal metrics ->', {k: round(v, 4) for k, v in sorted(seen.items())})
PY
  [ $? -ne 0 ] && fail=1
done

echo "=============== SMOKE RESULT: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ==============="
exit $fail
