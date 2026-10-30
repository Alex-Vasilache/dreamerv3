#!/bin/bash
#SBATCH -p gpu-p100 -x saion-gpu[01-10] -c 2 --mem=12G -t 0-2 -o /work/DoyaU/vasilache/work/sysid/r6/logs/%x_%j.out
# cl.sh RUN OUT MINUTES MODE variant...: a trained policy closed loop in sim variants (CPU).
# CMDS=rec.csv replays a robot recording's stick; CODE= the code tree (default: robot checkout).
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh >/dev/null 2>&1
export JAX_PLATFORMS=cpu MUJOCO_GL=egl
cd ${CODE:-/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3}
RUN=$1; OUT=$2; MIN=$3; MODE=$4; shift 4
V=(); for v in "$@"; do V+=(--variant "$v"); done
C=(); [ -n "$CMDS" ] && C=(--commands $CMDS)
[ -n "$CKPT" ] && C+=(--ckpt $CKPT)
python tools/robot/dev/sim_closed_loop.py $RUN /work/DoyaU/vasilache/work/sysid/r6/$OUT --minutes $MIN --mode $MODE "${C[@]}" "${V[@]}" 2>&1 | grep -v -i "warn\|unstable\|FLOPS\|Memory\|compil\|Report"
