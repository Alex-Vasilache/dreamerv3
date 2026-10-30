Round-6 analysis scripts (docs/robot/SIM_COMMAND.md §11), copied from the
working dir `/work/DoyaU/vasilache/work/sysid/r6/` where they run (paths are
hard-coded to it and to its `ev/` results).

- `sweep.py`, `loss.py`, `cl.sh`, `ol.sh`, `resid.py`, `realstats.py`: closed-loop sysid
  (replay recorded policies with their recorded stick in sim variants, score against the robot).
- `evalpol.py TAG CODE FLOOR RUN...` (env MINUTES, CKPT): fitted sim + 9 robustness variants;
  `evsum.py TAG` and `armtab.py` summarise; `ckptsweep.sh` evaluates the last saved checkpoints.
- `fig_sysid.py`, `fig_train.py`, `fig_ckpt.py`: the §11 figures. `swa.py`: weight averaging (negative result).
