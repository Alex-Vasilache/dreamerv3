#!/bin/bash -l
#SBATCH -J smoke_freeze_grad
#SBATCH -p gpu-v100
#SBATCH -t 0-00:05:00
#SBATCH -c 2
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/test_freeze_grad.py
