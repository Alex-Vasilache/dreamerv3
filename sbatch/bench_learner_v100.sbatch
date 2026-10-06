#!/bin/bash -l
#SBATCH -J bench_learner
#SBATCH -p gpu-v100
#SBATCH -t 0-00:30:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Learner throughput by shape and batch size on a V100, so the network can be
# sized against the machine that will actually train it. The MacBook's answer
# does not carry over: measured there, batch 16 -> 256 bought only 12% more
# throughput because the M5 is FLOP-bound at this size, while a V100 has enough
# parallel headroom that it may well be launch-bound instead.
set -uo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
CODE="${CODE:-/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3}"
OUT="${OUT:-/work/DoyaU/vasilache/work/bench_learner_v100.txt}"
cd "$CODE"
{
  echo "=== V100 $(hostname) $(date -Is)"
  python -u tools/bench_learner.py --mode learner \
    --deters 256 512 --units 256 512 \
    --batches 16 64 256 1024 --seconds 12 --actor_hz 40
} > "$OUT" 2>&1
echo "wrote $OUT"
