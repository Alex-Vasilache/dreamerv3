#!/bin/bash -l
#SBATCH -J smoke_vargoals
#SBATCH -p gpu-v100
#SBATCH -t 0-01:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

MAIN=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py
BASE="${BASE:-$(dreamerv3_mktemp_run_dir smoke_vargoals)}"
mkdir -p "$BASE"
echo "BASE=$BASE"

# Tiny debug nets, CPU jax, short run. ``debug`` overrides batch/size; we add a
# moderate imag_length and a small duration_max so several switches occur per
# rollout. ``set -e`` aborts on the first failure. Each leg runs a handful of
# train+report cycles and exits.
# NOTE: ``--configs`` is per-leg (the config-block names must travel WITH the
# --configs flag, not as trailing positionals). COMMON holds only scalar flags.
COMMON=(
  --task dmc_cartpole_swingup
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32
  --agent.enc.simple.outer True --agent.dec.simple.outer True
  --run.steps 800
  --agent.imag_length 16
  --logger.outputs jsonl
  --jax.platform cpu
)

echo "=================== LEG 1: flag OFF (fixed K, no-regression) ==================="
python -u "$MAIN" --logdir "$BASE/off" --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length False
echo "LEG 1 OK"

echo "=================== LEG 2: variable goal length, PLAIN goals ==================="
python -u "$MAIN" --logdir "$BASE/var_plain" --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 6
echo "LEG 2 OK"

echo "=================== LEG 2b: variable goal length + block manager rewards ======"
python -u "$MAIN" --logdir "$BASE/var_plain_blockrew" --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.variable_goal_block_rew True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 6
echo "LEG 2b OK"

echo "=================== LEG 3: variable goal length, MASKED goals (e17 recipe) ====="
python -u "$MAIN" --logdir "$BASE/var_masked" \
  --configs size6m debug masked_goals "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 6 \
  --agent.mask_sparsity_mode prob --agent.mask_sparsity_target 0.3 \
  --agent.mgr_cond_goalcode True
echo "LEG 3 OK"

echo "=================== LEG 4: block-rew + SUM agg + duration near imag_length ====="
# Regression test for relabel_truncated_last_duration: goal_duration_max ==
# imag_length forces frequent horizon-truncated holds (the manager routinely
# samples durations that can't fully fit in the 16-step imagined rollout), and
# variable_goal_block_rew + mgr_reward_agg=sum is exactly the combination that
# fed a truncated-but-mislabeled reward into REINFORCE before the fix. Also
# exercises the Lagrangian duration prior (the plain_vark launch recipe) since
# that reads the duration distribution directly, unaffected by relabeling, and
# should keep training as normal alongside it.
python -u "$MAIN" --logdir "$BASE/var_plain_blockrew_sum_trunc" \
  --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.variable_goal_block_rew True \
  --agent.mgr_reward_agg sum \
  --agent.goal_duration_min 1 --agent.goal_duration_max 16 \
  --agent.goal_duration_target 8.0 \
  --agent.goal_duration_reg 0.0 --agent.goal_duration_adapt False \
  --agent.goal_duration_lagrange True
echo "LEG 4 OK"

echo "=================== LEG 5: same as LEG 4 but relabel fix OFF (ablation) ========"
python -u "$MAIN" --logdir "$BASE/var_plain_blockrew_sum_trunc_norelabel" \
  --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.variable_goal_block_rew True \
  --agent.goal_duration_relabel_truncated False \
  --agent.mgr_reward_agg sum \
  --agent.goal_duration_min 1 --agent.goal_duration_max 16 \
  --agent.goal_duration_target 8.0 \
  --agent.goal_duration_reg 0.0 --agent.goal_duration_adapt False \
  --agent.goal_duration_lagrange True
echo "LEG 5 OK"

echo "ALL SMOKE LEGS PASSED"
exit 0
