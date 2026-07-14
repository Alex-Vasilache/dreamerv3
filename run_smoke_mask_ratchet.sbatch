#!/bin/bash -l
#SBATCH -J smoke_mask_ratchet
#SBATCH -p gpu-v100
#SBATCH -t 0-00:10:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
MAIN=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py
BASE="$(dreamerv3_mktemp_run_dir smoke_mask_ratchet)"
echo "BASE=$BASE"

COMMON=(
  --task dmc_cartpole_swingup
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32
  --agent.enc.simple.outer True --agent.dec.simple.outer True
  --run.steps 1000 --agent.imag_length 16 --agent.report False
  --run.log_every 1
  --jax.platform cuda --jax.prealloc False
  --logger.outputs jsonl
)

# Leg A: bare ratchet on the default 'prob' sparsity mode, fast vel so it
# visibly moves within 1000 smoke steps -- validates AutoAdapt(target=) override
# + Ratchet wiring in isolation.
python -u "$MAIN" --logdir "$BASE/ratchet_prob" --configs size6m masked_goals debug "${COMMON[@]}" \
  --agent.mask_sparsity_mode prob \
  --agent.mask_sparsity_target 0.3 \
  --agent.mask_sparsity_target_init 1.0 \
  --agent.mask_sparsity_target_vel 0.05

# Leg B: full e204-style recipe combo (prob_entropy mask + var-K lagrangian
# duration + struct adapt + worker countdown conditioning + mask ratchet) --
# the exact flag set the queued acrobot/hopper/cheetah/cartpole rescue runs use.
python -u "$MAIN" --logdir "$BASE/full_recipe" --configs size6m masked_goals variable_goals debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 16 \
  --agent.goal_duration_target 8 \
  --agent.goal_duration_reg 0.0 --agent.goal_duration_adapt False \
  --agent.goal_duration_lagrange True --agent.goal_duration_lagrange_max 5.0 \
  --agent.mask_topk 0 \
  --agent.mask_sparsity_mode prob_entropy \
  --agent.mask_sparsity_max 5.0 \
  --agent.mask_sparsity_target 0.3 \
  --agent.mask_sparsity_target_init 1.0 \
  --agent.mask_sparsity_target_vel 0.05 \
  --agent.mgr_cond_goalcode True \
  --agent.goal_struct_weight 200.0 \
  --agent.goal_struct_adapt True \
  --agent.goal_struct_adapt_target 0.006 \
  --agent.worker_timed_goals True \
  --agent.mgr_reward_agg mean

echo "=== metric keys present + ratchet trajectory (ratchet_prob) ==="
python - "$BASE/ratchet_prob/metrics.jsonl" <<'PY'
import json, sys
rows=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if line:
        try: rows.append(json.loads(line))
        except: pass
key='train/goal/mask_sparsity_target_now'
vals=[r[key] for r in rows if key in r]
print('FOUND ' + key if vals else 'MISSING ' + key)
if vals:
    print('first=%.4f last=%.4f n=%d' % (vals[0], vals[-1], len(vals)))
    assert vals[-1] < 1.0 - 1e-6, 'ratchet did not move down from init 1.0'
    assert vals[-1] >= 0.3 - 1e-6, 'ratchet overshot target'
    if len(vals) > 1:
        assert vals[0] >= vals[-1] - 1e-6, 'ratchet moved upward'
    print('RATCHET_OK')
else:
    raise SystemExit('no target_now rows logged')
PY

echo "=== metric keys present (full_recipe) ==="
python - "$BASE/full_recipe/metrics.jsonl" <<'PY'
import json, sys
keys=set()
last={}
for line in open(sys.argv[1]):
    line=line.strip()
    if line:
        try:
            d=json.loads(line); keys|=set(d.keys()); last.update(d)
        except: pass
want=['train/goal/mask_sparsity_target_now','train/goal/mask_prob_mean',
      'train/mgr_duration_lagrange_scale_mean','train/goal/struct_adapt_scale',
      'train/mgr_extr_rew']
for k in want:
    print(('FOUND ' if k in keys else 'MISSING ')+k)
print('final target_now =', last.get('train/goal/mask_sparsity_target_now'))
PY

echo "ALL LEGS DONE"
exit 0
