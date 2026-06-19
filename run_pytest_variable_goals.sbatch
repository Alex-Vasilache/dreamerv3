#!/bin/bash -l
#SBATCH -J pytest_vargoals
#SBATCH -p gpu-v100
#SBATCH -t 0-00:15:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
python -m pytest embodied/tests/test_variable_goals.py -v -rA
exit $?
