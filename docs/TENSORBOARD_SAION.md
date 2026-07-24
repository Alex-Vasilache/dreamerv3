# TensorBoard (Saion login node)

Run on the **login node**, not inside Slurm. The unit TensorBoard install lives under **`/apps/unit/DoyaU/vasilache/apps/rl_env`** (not `.../vasilache/rl_env`).

## Exact command (verified, Saion login)

Background process; logs under the run directory. Replace `LOGDIR` / `PORT` for other runs:

```bash
source /etc/profile.d/modules.sh
module load python/3.7.3

LOGDIR=/work/DoyaU/vasilache/work/dreamerv3_online_vis1m32_1env_v4_20260515_140552_kIEXlI/
PORT=6006
TBLOG="${LOGDIR%/}/tensorboard_${PORT}.log"

nohup tensorboard --logdir "$LOGDIR" --port "$PORT" --host 0.0.0.0 --load_fast=false \
  >"$TBLOG" 2>&1 &
echo "TB_PID=$!  log=$TBLOG  open http://127.0.0.1:${PORT}/"
```

You may see a harmless `openmpi.gcc` module warning; TensorBoard still starts. Use the URL from the `echo` line (default port 6006).

## Single run (foreground, `apps/rl_env`)

Alternative if you prefer the unit venv only (no `module load`):

```bash
source "/apps/unit/DoyaU/vasilache/apps/rl_env/bin/activate" && tensorboard \
  --logdir "/work/DoyaU/vasilache/work/dreamerv3_online_vis1m32_1env_v4_20260515_140552_kIEXlI/" \
  --port 6006 --host 0.0.0.0 --load_fast=false
```

Then open `http://127.0.0.1:6006/` (or SSH port-forward that port; match `--port` if you change it).

`--logdir` may be the run directory (as above) or `.../<RUN_ID>/logdir/`; TensorBoard recurses under the path you give.

## Wrapper scripts (same flags)

- One run, defaults: `./tensorboard_login.sh` (override `LOGDIR`, `PORT`, `BACKGROUND`).
- **All baseline (v3 defaults) + v4 + single-flag A/Bs** under `dreamerv3_online_vis1m32_1env_*` (sorted newest-first, every run that has a `logdir/`):

  ```bash
  cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
  ./tensorboard_all_last_runs.sh
  ```

  Newest run **per** variant only (one baseline, one `ab_pmpo`, one `ab_rms`, one `ab_rollout`, one `v4`):

  ```bash
  TB_ONE_PER_VARIANT=1 ./tensorboard_all_last_runs.sh
  ```

  Background: `PORT=6007 BACKGROUND=1 ./tensorboard_all_last_runs.sh`

## Common mistake

Use **`--load_fast=false`** exactly once. Avoid pasted typos like `load_fast=falset` or duplicated `--port` / `--host`.
