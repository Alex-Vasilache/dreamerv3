#!/bin/bash
# Sweep the commitment weight alpha for the PURE SOM-VAE arm on hopper.
#
# The arm is motivation.tex Eq. 12 as written -- both reconstructions, no
# straight-through estimator, one undivided commitment term, open-path codebook,
# no Lipschitz -- and alpha is the only thing that varies.
#
# Why sweep it. At the SOM-VAE default alpha=1.0 the commitment term pulls the
# encoder toward the codebook harder than the reconstruction pushes it to stay
# informative. Measured max-gradient ratio on the encoder at initialization:
#
#     alpha    1.0    0.5    0.25   0.1    0.05
#     rec/com  0.87   1.73   3.47   8.65   17.30
#
# Only alpha=1 has the commitment winning, and that is the configuration that
# collapsed over 577k steps in e415's first launch (perplexity 1.57 -> 1.28 of a
# possible 8, reconstruction error rising 5.1 -> 35.7). Lowering alpha should
# stop that, but not monotonically: at some point the commitment term stops
# tying the encoder to the codebook at all, so a sweet spot is expected rather
# than "smaller is better".
#
# alpha=0.25 is already running as e486 and is not repeated here.
#
# Reading the result. Rank first on codebook health, which separates within
# ~100k steps -- tools/watch_arm_checkpoints.py reports perplexity and
# used_frac, and an arm falling toward perplexity 1 can be cancelled early to
# free its slot. Then rank the survivors on task return at 400k+. Any winner
# should be confirmed on a second seed before being adopted, since one hopper
# seed is noisy.
#
#   sbatch/submit_alpha_sweep.sh            # submit; they pend until slots free
#   DRY_RUN=1 sbatch/submit_alpha_sweep.sh  # print without submitting
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TASK="${TASK:-dmc_hopper_hop}"
SEED="${SEED:-0}"
ARM="${ARM:-som_orig_line}"
LOG="job_logs/alpha_sweep_$(date +%Y%m%d_%H%M%S).tsv"

# exp_tag:alpha
RUNS=(
  "e488:1.0"     # the control: the SOM-VAE default, expected to collapse
  "e489:0.5"
  "e490:0.1"
  "e491:0.05"
  "e492:0.025"
)

mkdir -p job_logs
printf 'exp\talpha\tjobid\tarm\ttask\tseed\n' > "$LOG"
for entry in "${RUNS[@]}"; do
  exp="${entry%%:*}"; alpha="${entry##*:}"
  name="${exp}_hopper_${ARM}_a${alpha}_s${SEED}"
  cmd=(sbatch --parsable -J "$name"
       --export="ALL,EXP_TAG=${exp},TASK=${TASK},ARM=${ARM},SEED=${SEED},COMMIT_SCALE=${alpha}"
       sbatch/run_v3_goal_ae_ablation_big_a100.sbatch)
  if [ -n "${DRY_RUN:-}" ]; then
    echo "would submit: ${cmd[*]}"
    continue
  fi
  jid="$("${cmd[@]}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$exp" "$alpha" "$jid" "$ARM" "$TASK" "$SEED" >> "$LOG"
  echo "submitted $exp  alpha=$alpha  job $jid"
done
[ -n "${DRY_RUN:-}" ] || echo "log: $LOG"
