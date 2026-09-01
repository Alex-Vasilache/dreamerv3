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
# --run.log_every 1 overrides the debug default of 5s. The 400-step loop can
# finish inside one flush interval, so the metrics assertion below was passing
# or failing on timing rather than on the arm: a healthy run would write zero
# rows and be reported as FAIL.
#
#   sbatch sbatch/run_smoke_goal_ae_ablation.sbatch
set -uo pipefail

# Overridable so the audit worktree can smoke-test its own code.
REPO="${REPO:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3}"
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

STEPS="${STEPS:-400}"
# Override the partition on the sbatch line (-p gpu-v100 --gres=gpu:v100:1) when
# the a100s are full. PLATFORM=cpu falls back to no GPU at all; debug scale is
# small enough that it still exercises the env/replay/driver loop.
PLATFORM="${PLATFORM:-cuda}"
ARMS="${ARMS:-director vq som lipvq som_lipvq som_lipvq_mgr som_lipvq_smooth som_line som_lipvq_line som_lipvq_line_smooth som_lipvq_line_mgr som_orig_line som_orig_lipvq_line som_orig_line_adapt som_orig_lipvq_line_paper}"
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
    som_lipvq_smooth) cfg="goal_som_lipvq_smooth" ;;
    som_line) cfg="goal_som_line" ;;
    som_line_poisson) cfg="goal_som_line mgr_poisson" ;;
    som_lipvq_line_prod_poisson) cfg="goal_som_lipvq_line_prod mgr_poisson" ;;
    som_lipvq_line_relu_gaussian) cfg="goal_som_lipvq_line_prod lip_relu mgr_gaussian" ;;
    som_lipvq_line_relu_gaussian_eps) cfg="goal_som_lipvq_line_prod lip_relu mgr_gaussian mgr_explore" ;;
    som_lipvq_line) cfg="goal_som_lipvq_line" ;;
    som_lipvq_line_smooth) cfg="goal_som_lipvq_line_smooth" ;;
    som_lipvq_line_mgr) cfg="goal_som_lipvq_line_mgr" ;;
    som_orig_line) cfg="goal_som_orig_line" ;;
    som_orig_lipvq_line) cfg="goal_som_orig_lipvq_line" ;;
    som_orig_line_adapt) cfg="goal_som_orig_line_adapt" ;;
    som_orig_lipvq_line_paper) cfg="goal_som_orig_lipvq_line_paper" ;;
    lipvq_prod)               cfg="goal_lipvq_prod" ;;
    som_lipvq_line_prod)      cfg="goal_som_lipvq_line_prod" ;;
    som_orig_lipvq_line_prod) cfg="goal_som_orig_lipvq_line_prod" ;;
    # Without this, an arm missing from the list above leaves `cfg` unset and
    # `set -u` kills the whole script at the echo below with "cfg: unbound
    # variable" -- naming neither the arm nor the real problem.
    *) echo "unknown ARM=$arm; add it to the case above" >&2; fail=1; continue ;;
  esac
  echo "=============== ARM=$arm (configs: debug $cfg) ==============="
  logdir="$BASE/$arm"
  python -u "$REPO/dreamerv3/main.py" \
    --logdir "$logdir" \
    --configs debug ${cfg} \
    --task dmc_hopper_hop \
    --run.steps "$STEPS" \
    --run.envs 2 \
    --run.log_every 1 \
    --jax.platform "$PLATFORM" \
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
if 'poisson' in arm or 'gaussian' in arm:
    # The manager head is reparameterized, so re-check the two things that go
    # silently wrong when a head stops looking like OneHot: the adaptive
    # entropy regularizer skipping it (e478 -- mgr_ent_loss identically 0),
    # and the entropy pinning at the tie floor because tau_min is too high.
    ent = [r[k] for r in rows for k in r if k.endswith('mgr_ent_norm_skill_mean')]
    assert ent, f'{arm}: manager entropy never regularized (the e478 failure)'
    assert all(0.0 < e <= 1.0 for e in ent), f'{arm}: normalized entropy {ent}'
    print(f'{arm}: mgr normalized entropy {ent[0]:.3f} -> {ent[-1]:.3f}')
    mult = [r[k] for r in rows for k in r if k.endswith('mgr_actent_skill_scale_mean')]
    assert mult, f'{arm}: entropy adapter produced no multiplier'
    print(f'{arm}: actent multiplier {mult[0]:.4g} -> {mult[-1]:.4g}')
    loss = [r[k] for r in rows for k in r if k.endswith('train/mgr_ent_loss')]
    assert loss and any(v != 0.0 for v in loss), (
        f'{arm}: mgr_ent_loss identically zero -- the e478 failure')
for k, v in last.items():
    # timer/* legitimately carries inf: `min` over a stage that never ran in
    # this window (e.g. checkpoint_save) is reported as +inf by design.
    if isinstance(v, float) and not k.startswith('timer/'):
        assert v == v and abs(v) != float('inf'), f'{arm}: {k} = {v}'
seen = {k: r[k] for r in rows for k in r if k.startswith('train/goal/')}
print(f'{arm}: goal metrics ->', {k: round(v, 4) for k, v in sorted(seen.items())})
PY
  [ $? -ne 0 ] && fail=1
done

echo "=============== SMOKE RESULT: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ==============="
exit $fail
